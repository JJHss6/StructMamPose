"""Check the anatomical S architecture, exact accounting, gradients and reload.

CUDA scan is substituted only for THOP shape counting. Numerical checks use
an actual differentiable CPU selective-scan recurrence, never that placeholder.
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
from lib.model.PoseMamba import PoseMamba
from lib.model.posemamba_deploy import convert_to_deploy
from lib.utils.learning import load_backbone
from lib.utils.model_logging import collect_model_complexity, log_model_complexity
from lib.utils.model_profile import profile_graph_conditioning_excluded
from lib.utils.scan_reference import reference_scan_context
from lib.utils.tools import get_config
from tools.verify_original_posemamba import check_parent_gradient


def shape(linear, dims):
    assert isinstance(linear, nn.Linear)
    assert (linear.in_features, linear.out_features) == dims


def check_architecture(model):
    assert model.anatomical_ffn_blocks == [9]
    assert len(model.STEblocks) == len(model.TTEblocks) == 10
    assert [name for name, m in model.named_modules() if isinstance(m, AnatomicalFFN)] == ['STEblocks.9.mlp']
    for name, block in list(model.STEblocks.named_children()) + list(model.TTEblocks.named_children()):
        op = block.op
        assert not op.graph_equivalent and op.graph_pool_stride is None
        shape(op.gcn.gcn[0], (128, 128))
        shape(op.gcn.gcn[3], (128, 128))
        shape(op.proj_gcn[0], (128, 64))
        shape(op.proj_gcn[3], (64, 16))
        shape(op.proj_adj, (17, 128))
        shape(op.in_proj, (64, 256))
        shape(op.out_proj, (128, 64))
        assert op.x_proj_weight.shape == (4, 36, 128)
        assert op.dt_projs_weight.shape == (4, 128, 4)
        assert op.A_logs.shape == (512, 16)
        assert op.conv2d.groups == 128
        if not isinstance(block.mlp, AnatomicalFFN):
            shape(block.mlp.fc1, (64, 192))
            shape(block.mlp.fc2, (192, 64))
    mlp = model.STEblocks[9].mlp
    assert len(mlp.ffns) == 11
    for ffn in mlp.ffns:
        shape(ffn.fc1, (64, 192))
        shape(ffn.fc2, (192, 64))
    assert sum(p.numel() for p in mlp.parameters()) == 273152
    assert sum(p.numel() for p in model.parameters()) == 2805382
    shape(model.proj_side[0], (2, 64))
    shape(model.proj_side[2], (64, 64))
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        shape(getattr(model.cross_attn, name), (64, 64))


def check_counts(model, config):
    saved = {name: value.clone() for name, value in model.state_dict().items()}
    modes = {name: module.training for name, module in model.named_modules()}
    rng = torch.get_rng_state().clone()
    with patch.object(SelectiveScanCore, 'apply', staticmethod(lambda u, *_: u)):
        report = profile_graph_conditioning_excluded(model, (torch.zeros(1, 243, 17, 2),))
    assert report['thop_params'] == 1294662 and report['macs'] == 4353346944, report
    assert report['baseline_params'] == 2194822 and report['baseline_macs'] == 13543334784, report
    assert report['excluded_graph_params'] == 900160 and report['excluded_graph_macs'] == 9189987840, report
    assert report['full_parameters'] == 2805382
    assert torch.equal(rng, torch.get_rng_state())
    assert modes == {name: module.training for name, module in model.named_modules()}
    assert saved.keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        assert torch.equal(saved[name], value), name
    class Log:
        def __init__(self): self.lines = []
        def info(self, text): self.lines.append(text)
        def warning(self, text): raise AssertionError(text)
    log = Log()
    log_model_complexity(model, config, log)
    assert any('1.294662' in line and '4.353346944' in line for line in log.lines), log.lines
    assert all('Equivalent deployment' not in line for line in log.lines), log.lines
    for name, value in model.state_dict().items():
        assert torch.equal(saved[name], value), name
    return report, log.lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT / 'pose3d/PoseMamba_train_h36m_S.yaml')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--full-clip', action='store_true', help='Also run the real CPU scan for all 243 frames.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    config = get_config(str(args.config))
    model = load_backbone(config).cpu().float()
    check_architecture(model)
    counts, lines = check_counts(model, config)
    check_parent_gradient()
    print('PASS architecture, THOP scope, logging and parent gradient', flush=True)

    # Verify the old no-option constructor remains the original S architecture.
    original = PoseMamba(num_frame=243, embed_dim_ratio=64, depth=10, mlp_ratio=2)
    assert not original.anatomical_ffn_blocks
    assert sum(p.numel() for p in original.parameters()) == 2391942
    del original
    try:
        convert_to_deploy(model.eval())
    except ValueError as error:
        assert 'anatomical FFNs' in str(error)
    else:
        raise AssertionError('Legacy exporter must not silently mis-convert the anatomical model.')

    torch.manual_seed(31)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    # Force every residual branch to participate in this numerical gradient
    # check; restore the configured stochastic-depth probabilities afterward.
    drop_probs = {module: module.drop_prob for module in model.modules()
                  if hasattr(module, 'drop_prob')}
    for module in drop_probs:
        module.drop_prob = 0.0
    inputs = torch.randn(2, 3, 17, 2)
    target = torch.randn(2, 3, 17, 3)
    before = model.STEblocks[9].mlp.ffns[0].fc1.weight.detach().clone()
    with reference_scan_context():
        prediction = model(inputs)
        loss = (prediction - target).square().mean()
        loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    for name, p in model.STEblocks[9].mlp.named_parameters():
        assert p.grad.abs().sum().item() > 0, name
    optimizer.step()
    for module, probability in drop_probs.items():
        module.drop_prob = probability
    assert not torch.equal(before, model.STEblocks[9].mlp.ffns[0].fc1.weight)
    assert not torch.equal(model.STEblocks[9].mlp.ffns[0].fc1.weight, model.STEblocks[9].mlp.ffns[1].fc1.weight)
    print('PASS real CPU-scan backward and AdamW update in every anatomical group', flush=True)

    buffer = io.BytesIO()
    torch.save({'model_pos': model.state_dict()}, buffer)
    buffer.seek(0)
    reloaded = load_backbone(config).eval()
    reloaded.load_state_dict(torch.load(buffer, map_location='cpu')['model_pos'], strict=True)
    model.eval()
    with torch.no_grad(), reference_scan_context():
        expected = model(inputs[:1])
        actual = reloaded(inputs[:1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print('PASS strict training-checkpoint reload, identical output', flush=True)

    if args.full_clip:
        graph_shapes, handles = [], []
        for block in list(model.STEblocks) + list(model.TTEblocks):
            handles.append(block.op.gcn.register_forward_pre_hook(lambda _module, x: graph_shapes.append(list(x[0].shape))))
        try:
            with torch.no_grad(), reference_scan_context():
                full = model(torch.randn(1, 243, 17, 2))
        finally:
            for handle in handles: handle.remove()
        assert tuple(full.shape) == (1, 243, 17, 3) and torch.isfinite(full).all()
        assert graph_shapes == [[1, 243, 17, 128]] * 20, graph_shapes
        print('PASS real CPU-scan 243-frame forward; all 20 GCNs receive every frame', flush=True)

    report = {'status': 'passed', 'input_shape': [1, 243, 17, 2], 'statistics': counts,
              'startup_log': lines, 'parent_gradient': 4, 'real_cpu_scan_backward': True,
              'all_anatomical_groups_updated': True, 'stochastic_depth_disabled_for_gradient_check': True, 'checkpoint_reload_exact': True,
              'full_clip_real_cpu_scan': args.full_clip, 'cuda_available': torch.cuda.is_available(),
              'accuracy_evaluated': False, 'gpu_training_speed_measured': False}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
