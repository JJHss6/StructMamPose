"""Independent numerical/gradient checks for the anatomical FFN implementation."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import torch
from torch import nn
from torch.nn import functional as F

from lib.model.anatomical_ffn import AnatomicalFFN, ANATOMICAL_JOINT_GROUPS
from lib.model.mambablocks_my import Mlp
from lib.model.PoseMamba import PoseMamba


EXPECTED_GROUPS = ((0,), (1, 4), (2, 5), (3, 6), (7,), (8,), (9,), (10,),
                   (11, 14), (12, 15), (13, 16))
EXPECTED_MAPPING = (0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 9, 10)


def reference_forward(x, *parameters):
    """Eleven explicit dense FFNs, with anatomical slots restored independently."""
    parts, order = [], []
    for group, joints in enumerate(EXPECTED_GROUPS):
        weight1, bias1, weight2, bias2 = parameters[4 * group:4 * group + 4]
        hidden = F.gelu(F.linear(x[:, :, list(joints)], weight1, bias1))
        parts.append(F.linear(hidden, weight2, bias2))
        order.extend(joints)
    inverse = torch.tensor(order, device=x.device).argsort()
    return torch.cat(parts, dim=2).index_select(2, inverse)


def close(actual, expected, dtype):
    torch.testing.assert_close(actual, expected,
                               rtol=1e-11 if dtype == torch.float64 else 1e-4,
                               atol=1e-12 if dtype == torch.float64 else 2e-5)
    return float((actual - expected).abs().max())


def check_reference(dtype):
    module = AnatomicalFFN(64, 192, dtype=dtype)
    # Distinct parameters across groups ensure this test detects incorrect
    # joint ordering and incorrect gradient accumulation between bilateral pairs.
    x = torch.randn(2, 3, 17, 128, dtype=dtype)[..., ::2].requires_grad_()
    assert not x.is_contiguous()
    reference_x = x.detach().clone().requires_grad_()
    parameters = tuple(module.parameters())
    reference_parameters = tuple(parameter.detach().clone().requires_grad_() for parameter in parameters)
    output = module(x)
    expected = reference_forward(reference_x, *reference_parameters)
    forward_error = close(output, expected, dtype)
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad((output * upstream).sum(), (x,) + parameters)
    reference_gradients = torch.autograd.grad((expected * upstream).sum(),
                                             (reference_x,) + reference_parameters)
    gradient_errors = [close(actual, reference, dtype)
                       for actual, reference in zip(gradients, reference_gradients)]
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    # Selecting one joint may update only its own anatomical weight set.
    module.zero_grad(set_to_none=True)
    probe = torch.randn(1, 2, 17, 64, dtype=dtype)
    module(probe)[:, :, 1].sum().backward()
    for group, ffn in enumerate(module.ffns):
        for parameter in ffn.parameters():
            assert parameter.grad is not None
            if group == 1:
                assert torch.count_nonzero(parameter.grad).item() > 0
            else:
                assert torch.count_nonzero(parameter.grad).item() == 0

    # The corresponding left/right joints share weights, not their features.
    probe[:, :, 4] = probe[:, :, 1]
    paired = module(probe)
    torch.testing.assert_close(paired[:, :, 1], paired[:, :, 4], rtol=0, atol=0)
    probe[:, :, 4] += .75
    separate = module(probe)
    assert not torch.equal(separate[:, :, 1], separate[:, :, 4])
    return dict(forward_max_abs=forward_error, gradient_max_abs=max(gradient_errors),
                gradient_errors=gradient_errors, independent_group_gradients=True,
                bilateral_weight_sharing_without_feature_pooling=True)


def check_joint_order():
    module = AnatomicalFFN(3, 5, out_features=2, dtype=torch.float64)
    assert module.joint_groups == EXPECTED_GROUPS == ANATOMICAL_JOINT_GROUPS
    assert tuple(module.joint_to_group.tolist()) == EXPECTED_MAPPING
    with torch.no_grad():
        for group, ffn in enumerate(module.ffns):
            ffn.fc1.weight.zero_()
            ffn.fc1.bias.zero_()
            ffn.fc2.weight.zero_()
            ffn.fc2.bias.copy_(torch.tensor([group, group + .25], dtype=torch.float64))
    output = module(torch.randn(2, 4, 17, 3, dtype=torch.float64))
    expected = torch.tensor([[group, group + .25] for group in EXPECTED_MAPPING], dtype=torch.float64)
    torch.testing.assert_close(output, expected.expand(2, 4, 17, 2), rtol=0, atol=0)
    return True


def check_template_and_serialization():
    results = {}
    for dtype in (torch.float32, torch.float64):
        template = Mlp(64, 192, act_layer=nn.GELU, drop=.25).to(dtype=dtype).eval()
        template.act = nn.GELU(approximate='tanh').eval()
        template.fc1.weight.requires_grad_(False)
        template_state = {key: value.detach().clone() for key, value in template.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        module = AnatomicalFFN.from_mlp(template)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert not any(child.training for child in module.modules())
        assert all(ffn.act.approximate == 'tanh' and ffn.drop.p == .25 for ffn in module.ffns)
        assert all(not ffn.fc1.weight.requires_grad and ffn.fc2.weight.requires_grad for ffn in module.ffns)
        assert len({parameter.data_ptr() for parameter in module.parameters()}) == 44
        assert not ({parameter.data_ptr() for parameter in template.parameters()}
                    & {parameter.data_ptr() for parameter in module.parameters()})
        assert all(torch.equal(value, template.state_dict()[key]) for key, value in template_state.items())
        inputs = torch.randn(1, 5, 17, 64, dtype=dtype)
        with torch.no_grad():
            output = module(inputs)
            template_error = close(output, template(inputs), dtype)
        with tempfile.TemporaryDirectory(prefix='anatomical-ffn-') as directory:
            path = Path(directory) / 'state.pt'
            torch.save(module.state_dict(), path)
            restored = AnatomicalFFN(64, 192, drop=.25, dtype=dtype).eval()
            for ffn in restored.ffns:
                ffn.act = nn.GELU(approximate='tanh').eval()
            restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
            assert all(parameter.dtype == dtype and parameter.device.type == 'cpu'
                       for parameter in restored.parameters())
            assert restored.joint_to_group.dtype == torch.long
            with torch.no_grad():
                torch.testing.assert_close(restored(inputs), output, rtol=0, atol=0)
        # Both original dropout positions are retained, including output bias.
        dropped = AnatomicalFFN(64, 192, drop=1., dtype=dtype).train()
        assert torch.count_nonzero(dropped(inputs)).item() == 0
        dropped.eval()
        assert torch.count_nonzero(dropped(inputs)).item() > 0
        results[str(dtype)] = dict(template_max_abs=template_error, rng_preserved=True,
                                   save_load_exact=True, dropout_positions_preserved=True)
    return results


def check_compute():
    module = AnatomicalFFN(64, 192).eval()
    inputs = torch.zeros(1, 243, 17, 64)
    records = []
    original_linear = F.linear
    def audited_linear(inputs, weight, bias=None):
        output = original_linear(inputs, weight, bias)
        records.append((tuple(inputs.shape), tuple(weight.shape), output.numel() * weight.shape[1]))
        return output
    with patch.object(F, 'linear', audited_linear), torch.no_grad():
        output = module(inputs)
    assert len(records) == 22
    assert output.shape == (1, 243, 17, 64)
    macs = sum(row[2] for row in records)
    assert macs == module.profile_macs((inputs,)) == 101523456
    params = sum(parameter.numel() for parameter in module.parameters())
    assert params == 273152
    from thop import profile
    thop_macs, thop_params = profile(module, inputs=(inputs,), verbose=False)
    assert (int(thop_params), int(thop_macs)) == (params, macs)
    return dict(parameters=params, macs=macs, linear_calls=records, default_THOP_exact=True)


def check_integration():
    options = dict(num_frame=3, embed_dim_ratio=64, depth=10, mlp_ratio=3.,
                   graph_hidden_dim=128, ffn_hidden_dim=192)
    torch.manual_seed(104)
    shared = PoseMamba(**options)
    torch.manual_seed(104)
    anatomical = PoseMamba(**options, anatomical_ffn_blocks=[9])
    assert anatomical.anatomical_ffn_blocks == [9]
    assert sum(isinstance(module, AnatomicalFFN) for module in anatomical.modules()) == 1
    assert isinstance(anatomical.STEblocks[9].mlp, AnatomicalFFN)
    assert sum(parameter.numel() for parameter in anatomical.parameters()) == 2805382 + 64 * (3 - 243)
    shared_state = shared.state_dict()
    for name, value in anatomical.state_dict().items():
        if not name.startswith('STEblocks.9.mlp.'):
            assert torch.equal(value, shared_state[name]), name
    for block in list(anatomical.STEblocks) + list(anatomical.TTEblocks):
        assert block.op.gcn.gcn[0].weight.shape == (128, 128)
        assert block.op.gcn.gcn[3].weight.shape == (128, 128)
        assert block.op.in_proj.weight.shape == (256, 64)
        assert block.op.out_proj.weight.shape == (64, 128)
        if not isinstance(block.mlp, AnatomicalFFN):
            assert block.mlp.fc1.weight.shape == (192, 64)
            assert block.mlp.fc2.weight.shape == (64, 192)
    for invalid in ([9, 9], [-1], [10], [True], '9'):
        try:
            PoseMamba(**options, anatomical_ffn_blocks=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid block selection accepted: {invalid!r}')
    for conflicting in ({'lite_graph': True}, {'graph_equivalent': True}, {'num_joints': 16}):
        try:
            PoseMamba(**options, anatomical_ffn_blocks=[9], **conflicting)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Incompatible anatomical configuration accepted: {conflicting}')
    # No-selection default must preserve historical initial weights and counts.
    legacy_options = dict(num_frame=3, embed_dim_ratio=64, depth=10, mlp_ratio=2.)
    torch.manual_seed(105)
    legacy = PoseMamba(**legacy_options)
    torch.manual_seed(105)
    legacy_explicit = PoseMamba(**legacy_options, anatomical_ffn_blocks=[])
    assert sum(parameter.numel() for parameter in legacy.parameters()) == 2391942 + 64 * (3 - 243)
    assert all(torch.equal(value, legacy_explicit.state_dict()[name])
               for name, value in legacy.state_dict().items())
    return dict(selected_STE=[9], full_parameters_at_243=2805382,
                unchanged_GCN_and_SSM=True, untouched_modules_initialization_exact=True,
                default_legacy_constructor_unchanged=True, invalid_options_rejected=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(24023)
    report = dict(status='passed', reference_checks={str(dtype): check_reference(dtype)
                  for dtype in (torch.float64, torch.float32)}, joint_order_exact=check_joint_order(),
                  template_and_serialization=check_template_and_serialization(),
                  compute=check_compute(), integration=check_integration(),
                  GPU_latency='not_measured', pose_accuracy='not_evaluated')
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
