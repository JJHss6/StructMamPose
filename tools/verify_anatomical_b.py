"""Verify anatomical B architecture, declared THOP scope, training and reload.

The shape-only selective scan is used exclusively for operation counting.
Training and inference checks use the differentiable CPU scan recurrence.
"""

import argparse
import io
import json
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn

from lib.model.anatomical_ffn import AnatomicalFFN
from lib.model.csms6s import SelectiveScanCore
from lib.model.mambablocks_my import Mlp
from lib.model.PoseMamba import PoseMamba
from lib.utils.learning import load_backbone
from lib.utils.model_logging import REPORTING_NOTE, log_model_complexity
from lib.utils.model_profile import profile_graph_conditioning_excluded
from lib.utils.scan_reference import reference_scan_context
from lib.utils.tools import get_config
from tools.verify_original_posemamba import check_parent_gradient


EXPECTED = dict(thop_params=4958350, macs=15195734784,
                baseline_params=8396750, baseline_macs=49819463424,
                full_parameters=9781710,
                excluded_graph_params=3438400, excluded_graph_macs=34623728640)
EXPECTED_GROUPS = ((0,), (1, 4), (2, 5), (3, 6), (7,), (8,), (9,), (10,),
                   (11, 14), (12, 15), (13, 16))


def linear_shape(module, source, target):
    assert isinstance(module, nn.Linear), type(module)
    assert (module.in_features, module.out_features) == (source, target)


def check_architecture(model):
    assert model.anatomical_ffn_blocks == [9]
    assert len(model.STEblocks) == len(model.TTEblocks) == 10
    assert [name for name, module in model.named_modules()
            if isinstance(module, AnatomicalFFN)] == ['STEblocks.9.mlp']
    linear_shape(model.Spatial_patch_to_embedding, 2, 128)
    assert model.Spatial_pos_embed.shape == (1, 17, 128)
    assert model.Temporal_pos_embed.shape == (1, 243, 128)
    shared_count = 0
    for block in list(model.STEblocks) + list(model.TTEblocks):
        op = block.op
        assert not op.graph_equivalent and op.graph_pool_stride is None
        assert getattr(op.gcn, 'limb_mapping', None) is None
        linear_shape(op.gcn.gcn[0], 256, 256)
        assert isinstance(op.gcn.gcn[1], nn.SiLU)
        assert isinstance(op.gcn.gcn[2], nn.LayerNorm)
        assert op.gcn.gcn[2].normalized_shape == (256,)
        linear_shape(op.gcn.gcn[3], 256, 256)
        linear_shape(op.proj_gcn[0], 256, 128)
        linear_shape(op.proj_gcn[3], 128, 16)
        linear_shape(op.proj_adj, 17, 256)
        linear_shape(op.in_proj, 128, 512)
        linear_shape(op.out_proj, 256, 128)
        assert op.x_proj_weight.shape == (4, 40, 256)
        assert op.dt_projs_weight.shape == (4, 256, 8)
        assert op.dt_projs_bias.shape == (4, 256)
        assert op.A_logs.shape == (1024, 16) and op.Ds.shape == (1024,)
        assert op.conv2d.in_channels == op.conv2d.out_channels == op.conv2d.groups == 256
        assert op.conv2d.kernel_size == (3, 3)
        if not isinstance(block.mlp, AnatomicalFFN):
            shared_count += 1
            assert isinstance(block.mlp, Mlp)
            linear_shape(block.mlp.fc1, 128, 288)
            linear_shape(block.mlp.fc2, 288, 128)
    assert shared_count == 19
    selected = model.STEblocks[9].mlp
    assert selected.joint_groups == EXPECTED_GROUPS
    assert isinstance(selected.ffns, nn.ModuleList) and len(selected.ffns) == 11
    for ffn in selected.ffns:
        assert isinstance(ffn, Mlp)
        linear_shape(ffn.fc1, 128, 504)
        linear_shape(ffn.fc2, 504, 128)
        assert isinstance(ffn.act, nn.GELU) and isinstance(ffn.drop, nn.Dropout)
    assert len({parameter.data_ptr() for parameter in selected.parameters()}) == 44
    assert sum(parameter.numel() for parameter in selected.parameters()) == 1426216
    assert sum(parameter.numel() for parameter in model.parameters()) == EXPECTED['full_parameters']
    linear_shape(model.proj_side[0], 2, 128)
    linear_shape(model.proj_side[2], 128, 128)
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        linear_shape(getattr(model.cross_attn, name), 128, 128)


def snapshot(model):
    return dict(state={name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
                modes={name: module.training for name, module in model.named_modules()},
                flags={name: parameter.requires_grad for name, parameter in model.named_parameters()},
                hooks={name: dict(module._forward_hooks) for name, module in model.named_modules()},
                rng=torch.get_rng_state().clone(), threads=torch.get_num_threads())


def assert_unchanged(model, saved):
    assert saved['state'].keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, saved['state'][name], rtol=0, atol=0, msg=name)
    assert saved['modes'] == {name: module.training for name, module in model.named_modules()}
    assert saved['flags'] == {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    assert saved['hooks'] == {name: dict(module._forward_hooks) for name, module in model.named_modules()}
    assert torch.equal(saved['rng'], torch.get_rng_state())
    assert saved['threads'] == torch.get_num_threads()


def check_counts_and_logging(model, config):
    saved = snapshot(model)
    with patch.object(SelectiveScanCore, 'apply', staticmethod(lambda u, *_: u)):
        report = profile_graph_conditioning_excluded(model, (torch.zeros(1, 243, 17, 2),))
    assert {key: report[key] for key in EXPECTED} == EXPECTED, report
    assert len(report['graph_regions']) == 60
    assert_unchanged(model, saved)

    class Log:
        def __init__(self):
            self.lines = []

        def info(self, message):
            self.lines.append(message)

        def warning(self, message):
            raise AssertionError(message)

    log = Log()
    measured = log_model_complexity(model, config, log)
    assert log.lines == [REPORTING_NOTE, 'Params=4.958350 M | MACs=15.195734784 G'], log.lines
    assert all('Partial' not in line and 'Equivalent deployment' not in line for line in log.lines)
    assert measured['deploy'] is None
    assert {key: measured['reported'][key] for key in EXPECTED} == EXPECTED
    assert_unchanged(model, saved)
    return report, log.lines


def check_real_training(model, config):
    torch.manual_seed(79)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    drop_probs = {module: module.drop_prob for module in model.modules() if hasattr(module, 'drop_prob')}
    inputs = torch.randn(2, 3, 17, 2, requires_grad=True)
    target = torch.randn(2, 3, 17, 3)
    selected = model.STEblocks[9].mlp
    before = [ffn.fc1.weight.detach().clone() for ffn in selected.ffns]
    try:
        for module in drop_probs:
            module.drop_prob = 0.0
        with reference_scan_context():
            prediction = model(inputs)
            loss = (prediction - target).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        for name, parameter in selected.named_parameters():
            assert parameter.grad.abs().sum().item() > 0, name
        optimizer.step()
    finally:
        for module, probability in drop_probs.items():
            module.drop_prob = probability
    for old, ffn in zip(before, selected.ffns):
        assert not torch.equal(old, ffn.fc1.weight)
    assert not torch.equal(selected.ffns[0].fc1.weight, selected.ffns[1].fc1.weight)

    model.eval()
    checkpoint = io.BytesIO()
    torch.save({'model_pos': model.state_dict()}, checkpoint)
    checkpoint.seek(0)
    restored = load_backbone(config).cpu().float().eval()
    restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model_pos'], strict=True)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0, msg=name)
    with reference_scan_context(), torch.no_grad():
        expected, actual = model(inputs[:1].detach()), restored(inputs[:1].detach())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return float(loss.detach()), len(before)


def check_full_clip(model):
    observed, handles = [], []
    for block in list(model.STEblocks) + list(model.TTEblocks):
        handles.append(block.op.gcn.register_forward_pre_hook(
            lambda _module, inputs: observed.append(list(inputs[0].shape))))
    try:
        with reference_scan_context(), torch.no_grad():
            output = model(torch.randn(1, 243, 17, 2))
    finally:
        for handle in handles:
            handle.remove()
    assert tuple(output.shape) == (1, 243, 17, 3) and torch.isfinite(output).all()
    assert observed == [[1, 243, 17, 256]] * 20, observed
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'pose3d/PoseMamba_train_h36m_B.yaml')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--full-clip', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(97)
    config = get_config(str(args.config))
    model = load_backbone(config).cpu().float()
    check_architecture(model)
    counts, lines = check_counts_and_logging(model, config)
    check_parent_gradient()
    print('PASS B architecture, exact statistics, two-line logging and parent gradient', flush=True)

    original = PoseMamba(num_frame=243, embed_dim_ratio=128, depth=10, mlp_ratio=2)
    assert not original.anatomical_ffn_blocks
    assert sum(parameter.numel() for parameter in original.parameters()) == 8265158
    del original
    loss, updated_groups = check_real_training(model, config)
    print('PASS actual CPU-scan backward, eleven AdamW group updates and strict checkpoint reload', flush=True)
    observed = check_full_clip(model) if args.full_clip else None
    if args.full_clip:
        print('PASS actual CPU-scan 243-frame forward; every graph receives all frames', flush=True)
    report = dict(status='passed', config=str(args.config.resolve()), input_shape=[1, 243, 17, 2],
                  statistics=counts, startup_log=lines, parent_gradient=4,
                  original_b_without_override_full_params=8265158,
                  shared_ffn_hidden_dim=288, shared_ffn_blocks=19,
                  anatomical_ffn_hidden_dim=504, anatomical_ste_blocks=[9],
                  anatomical_groups_updated=updated_groups, real_cpu_scan_backward=True,
                  gradient_check_loss=loss, stochastic_depth_restored=True,
                  profile_weights_modes_flags_hooks_rng_threads_unchanged=True,
                  checkpoint_reload_exact=True, full_clip_real_cpu_scan=args.full_clip,
                  full_clip_graph_input_shapes=observed,
                  accuracy_evaluated=False, gpu_training_speed_measured=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
