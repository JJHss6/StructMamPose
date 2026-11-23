"""Verify the current original S and its equivalent online inference export.

Uses the true PyTorch selective-scan recurrence for small CPU numerical checks.
The scan shape placeholder is used only for the separately labelled MAC counts.
This is a random-weight regression check, not a training or MPJPE evaluation.
"""

import argparse
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn

from lib.model.csms6s import CrossScan_plus_poselimbs, SelectiveScanCore
from lib.utils.learning import load_backbone
from lib.utils.model_logging import log_model_complexity
from lib.utils.tools import get_config


def _rng():
    return (pickle.dumps(random.getstate()), pickle.dumps(np.random.get_state()),
            torch.get_rng_state().clone())


def _assert_rng(before):
    after = _rng()
    assert before[:2] == after[:2], 'Python or NumPy RNG changed.'
    assert torch.equal(before[2], after[2]), 'Torch CPU RNG changed.'


def _globals():
    return (torch.get_num_threads(), torch.is_grad_enabled(), torch.cuda.is_initialized(),
            torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
            torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
            torch.backends.cuda.matmul.allow_tf32,
            torch.are_deterministic_algorithms_enabled(), torch.get_default_dtype())


def _snapshot(model):
    return {
        'state': {name: value.detach().clone() for name, value in model.state_dict().items()},
        'modes': {name: module.training for name, module in model.named_modules()},
        'flags': {name: value.requires_grad for name, value in model.named_parameters()},
        'gradients': {name: None if value.grad is None else value.grad.detach().clone()
                      for name, value in model.named_parameters()},
        'hooks': {name: (len(module._forward_hooks), len(module._forward_pre_hooks),
                         len(module._backward_hooks)) for name, module in model.named_modules()},
    }


def _assert_snapshot(before, model):
    after = _snapshot(model)
    for field in ('modes', 'flags', 'hooks'):
        assert before[field] == after[field], field
    for field in ('state', 'gradients'):
        assert before[field].keys() == after[field].keys(), field
        for name, value in before[field].items():
            actual = after[field][name]
            assert (actual is None if value is None else
                    actual is not None and torch.equal(value, actual)), (field, name)


def _linear(layer, incoming, outgoing):
    assert type(layer) is nn.Linear
    assert (layer.in_features, layer.out_features) == (incoming, outgoing)


def _architecture(model):
    assert len(model.STEblocks) == len(model.TTEblocks) == 10
    assert model.Spatial_pos_embed.shape == (1, 17, 64)
    assert model.Temporal_pos_embed.shape == (1, 243, 64)
    _linear(model.Spatial_patch_to_embedding, 2, 64)
    for block in (*model.STEblocks, *model.TTEblocks):
        op = block.op
        assert not op.graph_equivalent and op.graph_pool_stride is None
        assert getattr(op.gcn, 'limb_mapping', None) is None
        _linear(op.gcn.gcn[0], 128, 128)
        _linear(op.gcn.gcn[3], 128, 128)
        _linear(op.proj_gcn[0], 128, 64)
        _linear(op.proj_gcn[3], 64, 16)
        _linear(op.proj_adj, 17, 128)
        _linear(block.mlp.fc1, 64, 128)
        _linear(block.mlp.fc2, 128, 64)
        _linear(op.in_proj, 64, 256)
        _linear(op.out_proj, 128, 64)
        assert op.x_proj_weight.shape == (4, 36, 128)
        assert op.dt_projs_weight.shape == (4, 128, 4)
        assert op.A_logs.shape == (512, 16) and op.Ds.shape == (512,)


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, message):
        self.lines.append(message)

    def warning(self, message):
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=PROJECT_ROOT / 'pose3d/PoseMamba_train_h36m_S.yaml')
    parser.add_argument('--verify-frames', type=int, default=3)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not 1 <= args.verify_frames <= 243:
        parser.error('--verify-frames must be between 1 and 243.')
    config_path = args.config.expanduser()
    if not config_path.is_absolute() and not config_path.is_file():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if args.report is not None:
        args.report = args.report.expanduser().resolve()
        if (args.report == config_path or
                (args.report.exists() and config_path.exists()
                 and args.report.samefile(config_path))):
            parser.error('--report must not overwrite the source config.')
    config = get_config(str(config_path))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model = load_backbone(config).float()
        _architecture(model)
        # Nondefault learned normalization parameters exercise the affine folding.
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, nn.LayerNorm) and module.elementwise_affine:
                    module.weight.uniform_(.8, 1.2)
                    module.bias.uniform_(-.1, .1)
            model.Spatial_pos_embed.normal_(0, .05)
            model.Temporal_pos_embed.normal_(0, .05)
        model.train()
        model.STEblocks[0].eval()
        parameters = list(model.parameters())
        parameters[0].requires_grad_(False)
        parameters[1].grad = torch.full_like(parameters[1], .125)

        # Intentionally before imports of deployment/profiling modules: cover their
        # first lazy import as well as the actual startup-statistics calculation.
        cold = 'lib.model.posemamba_online' not in sys.modules
        before = _snapshot(model)
        before_rng, before_globals = _rng(), _globals()
        scan_apply = SelectiveScanCore.apply
        log = _Log()
        startup = log_model_complexity(model, config, log)
        _assert_snapshot(before, model)
        _assert_rng(before_rng)
        assert _globals() == before_globals
        assert SelectiveScanCore.apply == scan_apply
        assert len(log.lines) == 2
        assert not startup.get('profiling_error')

        from lib.model.posemamba_deploy import load_deploy_model, save_deploy_model
        from lib.model.posemamba_online import convert_to_online_deploy
        from lib.utils.model_profile import profile_online_reported, profile_posemamba
        from lib.utils.scan_reference import reference_scan_context

        # Record compatibility with the intentionally retained original backward.
        # This is not a claim that its parent-feature derivative is mathematically correct.
        x = torch.zeros(1, 1, 1, 17, requires_grad=True)
        root_gradient, = torch.autograd.grad(CrossScan_plus_poselimbs.apply(x)[:, 0].sum(), x)
        assert root_gradient[0, 0, 0, 0].item() == 1.0

        model.eval()
        before = _snapshot(model)
        before_rng = _rng()
        deployment = convert_to_online_deploy(model)
        _assert_snapshot(before, model)
        _assert_rng(before_rng)
        assert deployment.online_factorized and deployment.architecture == 'original_s'
        assert all(not parameter.requires_grad for parameter in deployment.parameters())
        assert not {p.data_ptr() for p in model.parameters()}.intersection(
            p.data_ptr() for p in deployment.parameters())

        shape = torch.zeros(1, 243, 17, 2)
        with patch.object(SelectiveScanCore, 'apply', staticmethod(lambda u, *_: u)):
            source_macs, source_params = profile_posemamba(model, (shape,), verbose=False)
            deploy_macs, deploy_params = profile_posemamba(deployment, (shape,), verbose=False)
            reported = profile_online_reported(deployment, (shape,), verbose=False)
        assert (int(source_params), int(source_macs)) == (1781382, 12866511744)
        assert sum(p.numel() for p in model.parameters()) == 2391942
        assert (int(deploy_params), int(deploy_macs)) == (1697578, 5768135962)
        assert sum(p.numel() for p in deployment.parameters()) == 2308138
        assert (reported['thop_params'], reported['macs']) == (1202218, 4393846042)
        assert startup['deploy']['thop_params'] == reported['thop_params']
        assert startup['deploy']['macs'] == reported['macs']

        checks = []
        with tempfile.TemporaryDirectory(prefix='posemamba-online-') as temporary:
            path = Path(temporary) / 'deployment.pt'
            metadata = {'purpose': 'random_initialization_regression'}
            save_deploy_model(deployment, path, metadata=metadata)
            reloaded = load_deploy_model(path, map_location='cpu')
            original_state, reloaded_state = deployment.state_dict(), reloaded.state_dict()
            assert original_state.keys() == reloaded_state.keys()
            assert all(torch.equal(value, reloaded_state[name]) for name, value in original_state.items())
            assert reloaded.online_factorized and reloaded.export_metadata == metadata
            with reference_scan_context(), torch.no_grad():
                for batch, frames in dict.fromkeys(((1, 1), (1, args.verify_frames), (2, args.verify_frames))):
                    inputs = torch.randn(batch, frames, 17, 2)
                    expected, actual, restored = model(inputs), deployment(inputs), reloaded(inputs)
                    assert torch.isfinite(actual).all()
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    assert torch.equal(actual, restored)
                    checks.append({'batch': batch, 'frames': frames,
                                   'max_abs_error': (actual - expected).abs().max().item(),
                                   'mean_abs_error': (actual - expected).abs().mean().item()})
        _assert_snapshot(before, model)
        report = {
            'status': 'passed', 'source': 'random_initialization', 'accuracy_evaluation': 'not_performed',
            'architecture': 'original_s', 'numerical_backend': 'real_cpu_selective_scan',
            'numerical_checks': checks, 'rtol': 1e-4, 'atol': 1e-5,
            'profiling_input': [1, 243, 17, 2], 'profiling_scan': 'shape_only',
            'source_thop_params': int(source_params), 'source_macs': int(source_macs),
            'deploy_thop_params': int(deploy_params), 'deploy_macs': int(deploy_macs),
            'reported': reported, 'reload_state_exact': True, 'reload_output_exact': True,
            'source_state_unchanged': True, 'logging_cold_import_checked': cold,
            'logging_rng_preserved': {'python': True, 'numpy': True, 'torch': True},
            'logging_modes_gradients_flags_hooks_preserved': True,
            'logging_global_settings_preserved': True, 'original_scan_root_gradient': 1.0,
            'startup_log': log.lines,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    main()
