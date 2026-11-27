"""Export the current original PoseMamba-S to its equivalent online deployment."""

import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from lib.model.csms6s import SelectiveScanCore
from lib.model.deploy_s.posemamba_deploy import load_deploy_model, save_deploy_model
from lib.model.deploy_s.posemamba_online import convert_to_online_deploy
from lib.utils.learning import load_backbone
from lib.utils.model_logging_s import reporting_note_for, uses_online_equivalent_scope
from lib.utils.model_profile_s import (
    deploy_cache_tensors, profile_online_reported, profile_posemamba,
)
from lib.utils.scan_reference import reference_scan_context
from lib.utils.tools import get_config


def _existing_path(value, relative_to_project=False):
    path = Path(value).expanduser()
    if relative_to_project and not path.is_absolute() and not path.is_file():
        path = PROJECT_ROOT / path
    return path.resolve()


def _same_path(first, second):
    return first == second or (first.exists() and second.exists() and first.samefile(second))


def _checkpoint_state(path, trusted=False):
    kwargs = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        kwargs['weights_only'] = not trusted
    elif not trusted:
        raise ValueError('This PyTorch cannot load safely. Use --trusted-checkpoint only for a trusted training checkpoint.')
    try:
        checkpoint = torch.load(path, **kwargs)
    except pickle.UnpicklingError as error:
        if trusted:
            raise
        raise ValueError(
            'Safe checkpoint loading failed. Older PyTorch versions cannot read '
            'Python floats or NumPy metadata in train.py checkpoints this way. '
            'For a checkpoint from a trusted source, explicitly add --trusted-checkpoint. '
            'This permits pickle deserialization; model weights will still be loaded strictly.'
        ) from error
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a state dict or contain model_pos/state_dict.')
    state = checkpoint.get('model_pos', checkpoint.get('state_dict', checkpoint))
    if not isinstance(state, dict) or not state:
        raise ValueError('Checkpoint does not contain a nonempty model state dict.')
    cleaned = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('The model state dict must contain only named tensors.')
        name = name[7:] if name.startswith('module.') else name
        if name in cleaned:
            raise ValueError('Duplicate parameter name after stripping module.: ' + name)
        cleaned[name] = value
    return cleaned


def _clone_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _assert_state_equal(expected, model):
    actual = model.state_dict()
    if expected.keys() != actual.keys():
        raise AssertionError('State-dict keys changed.')
    for name, value in expected.items():
        if not torch.equal(value, actual[name].detach().cpu()):
            raise AssertionError('State-dict tensor changed: ' + name)


def _error(reference, actual, rtol, atol):
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise AssertionError('Numerical verification produced a nonfinite output.')
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    difference = (actual - reference).abs()
    return {'max_abs_error': difference.max().item(),
            'mean_abs_error': difference.mean().item()}


def _shape_only_scan(u, *_):
    # Used ONLY for THOP's module accounting, never for numerical equivalence.
    return u


def _counts(model, inputs):
    with patch.object(SelectiveScanCore, 'apply', staticmethod(_shape_only_scan)):
        macs, params = profile_posemamba(model, inputs=(inputs,), verbose=False)
    return {'thop_params': int(params), 'thop_macs': int(macs),
            'full_parameter_elements': sum(p.numel() for p in model.parameters()),
            'trainable_parameter_elements': sum(p.numel() for p in model.parameters()
                                                if p.requires_grad)}


def _cache_counts(model):
    caches = deploy_cache_tensors(model)
    return {'tensor_count': len(caches),
            'floating_elements': sum(b.numel() for b in caches.values() if b.is_floating_point()),
            'bytes': sum(b.numel() * b.element_size() for b in caches.values()),
            'names': list(caches)}


def _validate_current_config(config):
    """Keep this user-facing entry point on the current dense S configuration."""
    required = {'backbone': 'PoseMamba', 'maxlen': 243, 'num_joints': 17, 'dim_feat': 64, 'depth': 10, 'mlp_ratio': 2}
    for (name, expected) in required.items():
        if getattr(config, name, None) != expected:
            raise ValueError(f'Current original S requires {name}={expected!r}.')
    if getattr(config, 'ffn_hidden_dim', None) not in (None, 128):
        raise ValueError('Original S requires FFN width 128.')
    if not uses_online_equivalent_scope(None, config):
        raise ValueError('This exporter requires the current online_equivalent_partial scope.')


def _model_description(model, config):
    return {
        'name': 'posemamba_original_s',
        'trunk_width': int(config.dim_feat),
        'frames': int(config.maxlen),
        'joints': int(config.num_joints),
        'block_pairs': int(config.depth),
        'mlp_ratio': float(config.mlp_ratio),
        'graph_hidden_dim': 128,
        'ffn_hidden_dim': 128,
        'checkpoint_compatibility': 'Requires original dense S: GCN 128->128->128 and shared FFN 64->128->64.',
    }


def main():
    parser = argparse.ArgumentParser(
        description='Export the current original PoseMamba-S with numerical and serialization checks.')
    parser.add_argument('--config', default='pose3d/PoseMamba_train_h36m_S.yaml')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', help='Checkpoint matching the selected architecture; loaded strictly.')
    source.add_argument('--random-init', action='store_true',
                        help='Export a randomly initialized example, without any accuracy claim.')
    parser.add_argument('--trusted-checkpoint', action='store_true',
                        help='Allow pickle loading of a trusted training checkpoint containing optimizer/NumPy metadata.')
    parser.add_argument('--output', required=True, help='Destination deployment checkpoint.')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--verify-frames', type=int, default=3,
                        help='Numerical verification length; CPU default is 3, supports 243.')
    parser.add_argument('--report', help='Optional JSON verification report.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.trusted_checkpoint and not args.checkpoint:
        parser.error('--trusted-checkpoint requires --checkpoint.')

    config_path = _existing_path(args.config, relative_to_project=True)
    checkpoint_path = _existing_path(args.checkpoint) if args.checkpoint else None
    output_path = _existing_path(args.output)
    report_path = _existing_path(args.report) if args.report else None
    if not config_path.is_file():
        parser.error('Config does not exist: ' + str(config_path))
    if checkpoint_path is not None and not checkpoint_path.is_file():
        parser.error('Checkpoint does not exist: ' + str(checkpoint_path))
    protected_paths = [config_path] + ([checkpoint_path] if checkpoint_path else [])
    for path in (output_path, report_path):
        if path is not None and any(_same_path(path, protected) for protected in protected_paths):
            parser.error('Output/report must not overwrite the source checkpoint or config.')
    if report_path is not None and _same_path(output_path, report_path):
        parser.error('--report and --output must be different files.')
    config = get_config(str(config_path))
    try:
        _validate_current_config(config)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.verify_frames <= config.maxlen:
        parser.error('--verify-frames must be between 1 and the configured maxlen.')
    if config.maxlen != 243 or config.num_joints != 17:
        parser.error('This export validation expects a 243-frame, 17-joint configuration.')
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device_name == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA was requested but is unavailable.')
    device = torch.device(device_name)
    torch.manual_seed(args.seed)
    if device_name == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        # FP32 equivalence should not depend on reduced-precision CUDA matmul.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_num_threads(1)

    model = load_backbone(config).float().to(device)
    model_description = _model_description(model, config)
    if checkpoint_path is not None:
        # Never silently adapt checkpoint shapes or discard model weights.
        state = _checkpoint_state(checkpoint_path, trusted=args.trusted_checkpoint)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as error:
            parser.error(
                f'Checkpoint does not match {model_description["name"]}. '
                f'{model_description["checkpoint_compatibility"]} '
                f'{str(error).splitlines()[0]}'
            )
    else:
        print('Source: random initialization; this is an export check, not a trained model or accuracy result.',
              flush=True)
    print(f'Architecture: {model_description["name"]}; config: {config_path}', flush=True)
    original_state = _clone_state(model)
    original_modes = {module: module.training for module in model.modules()}
    original_grad_flags = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = None
    temporary_report = None
    try:
        model.eval()
        if not uses_online_equivalent_scope(model, config):
            raise ValueError('This exporter requires the current online-equivalent scope.')
        deploy = convert_to_online_deploy(model).to(device).eval()
        rtol, atol = 1e-4, 1e-5
        cases = list(dict.fromkeys(((1, 1), (1, args.verify_frames), (2, args.verify_frames))))
        checks = []
        examples = []
        context = reference_scan_context() if device_name == 'cpu' else nullcontext()
        with context, torch.no_grad():
            for batch, frames in cases:
                inputs = torch.randn(batch, frames, config.num_joints, 2, device=device)
                reference = model(inputs)
                actual = deploy(inputs)
                errors = _error(reference, actual, rtol, atol)
                checks.append({'batch': batch, 'frames': frames, **errors})
                examples.append((inputs, reference, actual))
                print(f'Numerical check B={batch}, T={frames}: max_abs={errors["max_abs_error"]:.9g}', flush=True)

        profile_inputs = torch.zeros(1, 243, 17, 2, device=device)
        original_counts = _counts(model, profile_inputs)
        deploy_counts = _counts(deploy, profile_inputs)
        with patch.object(SelectiveScanCore, 'apply', staticmethod(_shape_only_scan)):
            reported_counts = profile_online_reported(deploy, (profile_inputs,), verbose=False)
        cache = _cache_counts(deploy)
        source_info = {'kind': 'random_init' if args.random_init else 'checkpoint',
                       'checkpoint': str(checkpoint_path) if checkpoint_path else None,
                       'checkpoint_loading': ('trusted_pickle' if args.trusted_checkpoint else 'weights_only')
                       if checkpoint_path else None}
        if checkpoint_path is not None:
            digest = hashlib.sha256()
            with checkpoint_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            source_info['sha256'] = digest.hexdigest()
        metadata = {'source': source_info, 'seed': args.seed, 'config': str(config_path),
                    'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    'architecture': deploy.architecture,
                    'equivalent_conversion': {'fold_affine': deploy.fold_affine,
                                              'fuse_attention': deploy.fuse_attention,
                                              'advanced': deploy.advanced,
                                              'online_factorized': deploy.online_factorized},
                    'reported_scope': reported_counts['scope'],
                    'reporting_note': reporting_note_for(deploy, config),
                    'model_configuration': model_description,
                    'accuracy_evaluation': 'not_performed', 'mpjpe': None}
        descriptor, name = tempfile.mkstemp(prefix=output_path.name + '.', suffix='.tmp', dir=output_path.parent)
        os.close(descriptor)
        temporary_output = Path(name)
        save_deploy_model(deploy, str(temporary_output), metadata=metadata)
        reloaded = load_deploy_model(str(temporary_output), map_location=device_name).to(device).eval()
        _assert_state_equal(_clone_state(deploy), reloaded)
        if (reloaded.architecture != deploy.architecture
                or reloaded.construction_info != deploy.construction_info
                or reloaded.fold_affine != deploy.fold_affine
                or reloaded.fuse_attention != deploy.fuse_attention
                or reloaded.advanced != deploy.advanced
                or reloaded.online_factorized != deploy.online_factorized):
            raise AssertionError('Deployment architecture changed during checkpoint reload.')
        if reloaded.export_metadata != json.loads(json.dumps(metadata)):
            raise AssertionError('Deployment source/config metadata changed during checkpoint reload.')
        context = reference_scan_context() if device_name == 'cpu' else nullcontext()
        with context, torch.no_grad():
            for check, (inputs, reference, previous) in zip(checks, examples):
                actual = reloaded(inputs)
                check['reload_vs_original'] = _error(reference, actual, rtol, atol)
                check['reload_vs_export'] = _error(previous, actual, 0.0, 0.0)

        # Conversion, profiling and serialization must leave the training model intact.
        for module, training in original_modes.items():
            module.training = training
        _assert_state_equal(original_state, model)
        if original_grad_flags != {name: p.requires_grad for name, p in model.named_parameters()}:
            raise AssertionError('The source model parameter training flags changed.')
        if any(module.training != training for module, training in original_modes.items()):
            raise AssertionError('The source model training modes were not restored.')

        report = {**metadata, 'status': 'passed', 'output': str(output_path), 'device': device_name,
                  'dtype': 'float32', 'verification_backend': 'pytorch_reference_scan' if device_name == 'cpu' else 'cuda_scan',
                  'rtol': rtol, 'atol': atol, 'numerical_checks': checks,
                  'profiling': {'input_shape': list(profile_inputs.shape), 'scan': 'shape_only',
                                'accounting': 'THOP_with_online_deployment_hooks_and_explicit_partial_scope',
                                'limitations': 'Selective scan and unhooked functional tensor operations are not counted.',
                                'training': original_counts,
                                'original': original_counts, 'deploy': deploy_counts,
                                'deployment_reported': reported_counts},
                  'deployment_cache': cache, 'reload_state_exact': True, 'reload_output_exact': True,
                  'reload_architecture_exact': True, 'reload_metadata_exact': True,
                  'source_model_state_unchanged': True, 'source_model_modes_restored': True,
                  'source_parameter_grad_flags_unchanged': True}
        if report_path is not None:
            descriptor, name = tempfile.mkstemp(prefix=report_path.name + '.', suffix='.tmp', dir=report_path.parent)
            os.close(descriptor)
            temporary_report = Path(name)
            temporary_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        os.replace(temporary_output, output_path)
        temporary_output = None
        if report_path is not None:
            os.replace(temporary_report, report_path)
            temporary_report = None
        print(json.dumps(report, indent=2, ensure_ascii=False))
    finally:
        for module, training in original_modes.items():
            module.training = training
        for path in (temporary_output, temporary_report):
            if path is not None and path.exists():
                path.unlink()


if __name__ == '__main__':
    main()
