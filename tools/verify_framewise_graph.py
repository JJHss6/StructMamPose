"""Check the framewise graph rewrite against the original computation.

Run from any directory with the project's Python environment.  All checks use
CPU double precision and do not need the CUDA selective-scan extension.
"""

import argparse
import copy
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lib.model.csms6s import CrossScan_plus_poselimbs
from lib.model.mambablocks_my import FramewiseGraphConditioning


PARENT_JOINTS = [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]


def reference_scan(x):
    """The original four directions, with ordinary PyTorch autograd."""
    horizontal = (x + x[..., PARENT_JOINTS]).flatten(2)
    vertical = x.transpose(2, 3).flatten(2)
    return torch.stack(
        (horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1
    )


def reference_graph(module, x):
    """Execute both full B*T graph branches and all four projections."""
    batch, frames, joints, _ = x.shape
    adjacency = module.adj.expand(batch, frames, joints, joints)
    graph_features = module.gcn(adjacency @ x)
    topology_features = module.proj_adj(adjacency)

    def scan_and_project(features):
        sequences = CrossScan_plus_poselimbs.apply(features.permute(0, 3, 1, 2))
        projected = module.proj_gcn(sequences.permute(0, 1, 3, 2))
        return projected.permute(0, 1, 3, 2)

    return scan_and_project(graph_features) + scan_and_project(topology_features)


def check_scan_gradients():
    scan = CrossScan_plus_poselimbs.apply
    root_example = torch.zeros(1, 1, 1, 17, dtype=torch.double, requires_grad=True)
    root_grad, = torch.autograd.grad(scan(root_example)[:, 0].sum(), root_example)
    assert root_grad[0, 0, 0, 0].item() == 4.0, root_grad

    examples = (
        torch.randn(1, 3, 2, 17, dtype=torch.double).requires_grad_(),
        torch.randn(2, 3, 5, 34, dtype=torch.double)[..., ::2].requires_grad_(),
    )
    assert not examples[1].is_contiguous()
    for x in examples:
        actual, expected = scan(x), reference_scan(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        weights = torch.randn_like(actual)
        # Separate directions catch missing parent gradients that can otherwise
        # be obscured when horizontal and vertical gradients are combined.
        for direction in (*range(4), None):
            selection = slice(None) if direction is None else direction
            actual_grad, = torch.autograd.grad(
                actual[:, selection], x, weights[:, selection], retain_graph=True
            )
            expected_grad, = torch.autograd.grad(
                expected[:, selection], x, weights[:, selection], retain_graph=True
            )
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-13, atol=1e-13)

    x = torch.randn(1, 1, 2, 17, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(scan, (x,))
    assert torch.autograd.gradgradcheck(scan, (x,))
    print("PASS scan: parent-root gradient 4; all directions; gradcheck/gradgradcheck")


def check_graph_case(module, batch, frames, noncontiguous=False):
    reference = copy.deepcopy(module)
    dim = module.gcn[0].in_features
    if noncontiguous:
        backing = torch.randn(batch, frames, 17, dim * 2, dtype=torch.double)
        x = backing[..., ::2].requires_grad_()
        reference_x = backing.clone()[..., ::2].requires_grad_()
        assert not x.is_contiguous() and not reference_x.is_contiguous()
    else:
        x = torch.randn(batch, frames, 17, dim, dtype=torch.double, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()

    actual = module(x)
    expected = reference_graph(reference, reference_x)
    state_dim = module.proj_gcn[-1].out_features
    assert actual.shape == (batch, 4, state_dim, frames * 17)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)

    names, parameters = zip(*module.named_parameters())
    reference_names, reference_parameters = zip(*reference.named_parameters())
    assert names == reference_names
    weights = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (x, *parameters), weights)
    reference_gradients = torch.autograd.grad(
        expected, (reference_x, *reference_parameters), weights
    )
    max_gradient_error = 0.0
    for name, actual_grad, expected_grad in zip(
        ("input", *names), gradients, reference_gradients
    ):
        assert torch.isfinite(actual_grad).all(), name
        try:
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-9, atol=1e-10)
        except AssertionError as error:
            raise AssertionError(f"Gradient mismatch: {name}") from error
        max_gradient_error = max(
            max_gradient_error, (actual_grad - expected_grad).abs().max().item()
        )

    layout = "strided" if noncontiguous else "contiguous"
    dims = (
        module.gcn[0].in_features,
        module.gcn[0].out_features,
        module.gcn[-1].out_features,
        module.proj_gcn[0].out_features,
    )
    print(
        f"PASS graph {dims}, B={batch}, T={frames}, {layout}: "
        f"output error={(actual - expected).abs().max().item():.3e}, "
        f"input/all-{len(names)}-parameter gradient error={max_gradient_error:.3e}"
    )


def check_graph_equivalence():
    # Compact graph widths, a different graph output width, and original widths.
    for hidden, output, projection in ((68, 128, 34), (80, 96, 40), (128, 128, 64)):
        module = FramewiseGraphConditioning(
            dim=128, hidden_dim=hidden, output_dim=output,
            projection_dim=projection, state_dim=16, num_joints=17,
        ).double()
        for batch, frames, noncontiguous in ((1, 1, False), (2, 5, False), (2, 5, True)):
            check_graph_case(module, batch, frames, noncontiguous)


def check_configuration(config_path, expected_params):
    from lib.utils.learning import load_backbone
    from lib.utils.tools import get_config

    config_path = Path(config_path).expanduser()
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    model = load_backbone(get_config(str(config_path)))
    blocks = [*model.STEblocks, *model.TTEblocks]
    assert len(blocks) == 20, len(blocks)
    from lib.model.grouped_posemamba import FrameContextGraph

    graph_modes = {"original": 0, "equivalent": 0, "frame_context": 0}
    for index, block in enumerate(blocks):
        op = block.op
        assert op.graph_pool_stride is None, f"Block {index} uses the pooled/lite graph path"
        if op.graph_equivalent:
            if isinstance(op.graph_conditioning, FrameContextGraph):
                graph_modes["frame_context"] += 1
            else:
                assert isinstance(op.graph_conditioning, FramewiseGraphConditioning), index
                graph_modes["equivalent"] += 1
        else:
            assert hasattr(op, "gcn") and hasattr(op.gcn, "gcn"), index
            assert hasattr(op, "proj_gcn") and hasattr(op, "proj_adj"), index
            assert getattr(op.gcn, "limb_mapping", None) is None, index
            graph_modes["original"] += 1
    actual_params = sum(parameter.numel() for parameter in model.parameters())
    assert actual_params == expected_params, (actual_params, expected_params)
    print(
        f"PASS configuration {config_path.name}: 20 framewise graph blocks "
        f"(original={graph_modes['original']}, equivalent={graph_modes['equivalent']}, "
        f"frame_context={graph_modes['frame_context']}); "
        f"full Params={actual_params:,}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "pose3d" / "PoseMamba_train_h36m_S.yaml",
    )
    parser.add_argument("--expected-params", type=int, default=2_226_246)
    parser.add_argument("--skip-config", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.set_num_threads(1)
    check_scan_gradients()
    check_graph_equivalence()
    if not args.skip_config:
        check_configuration(args.config, args.expected_params)
    print("All framewise graph checks passed.")


if __name__ == "__main__":
    main()
