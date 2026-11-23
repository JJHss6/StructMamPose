"""One-time statistics with an explicitly declared MAC reporting scope."""

import copy
from unittest.mock import patch

import torch


REPORTING_NOTE = (
    "We follow PoseMamba’s THOP-based profiling framework and report parameter "
    "and MAC counts excluding state-space parameterization modules."
)

LEGACY_REPORTING_NOTE = (
    "We follow PoseMamba’s THOP-based framework for counting parameters and MACs, "
    "excluding Mamba’s selective parameter (Δ, B, C) generation and the first encoding "
    "projection layer in the graph-conditioned parameter generation front end."
)


def _is_anatomical(model):
    source = model.module if isinstance(model, torch.nn.DataParallel) else model
    return bool(getattr(source, 'anatomical_ffn_blocks', ()))


def reporting_note_for(model):
    """Keep the two distinct reporting scopes attached to their own models."""
    return REPORTING_NOTE if _is_anatomical(model) else LEGACY_REPORTING_NOTE


def _profile_counts(model, inputs):
    # Lazy import: unavailable optional profiling dependencies must not stop training.
    from .model_profile import profile_posemamba

    macs, params = profile_posemamba(model, inputs=(inputs,), verbose=False)
    return {
        'thop_params': int(params), 'macs': int(macs),
        'full_parameters': sum(parameter.numel() for parameter in model.parameters()),
    }


def collect_model_complexity(model, config):
    """Count CPU copies only; never convert the model used by the optimizer.

    The scan placeholder is restricted to counting. These are architecture
    statistics, not a numerical-equivalence check or a latency measurement.
    """
    from lib.model.csms6s import SelectiveScanCore
    from lib.model.posemamba_deploy import convert_to_deploy
    from .model_profile import (
        deploy_cache_storage, profile_graph_conditioning_excluded, profile_posemamba_reported,
    )

    source = model.module if isinstance(model, torch.nn.DataParallel) else model
    shape = (1, int(config.maxlen), int(source.Spatial_pos_embed.shape[1]),
             int(source.Spatial_patch_to_embedding.in_features))
    report = {'input_shape': list(shape), 'training': None, 'deploy': None,
              'deploy_unavailable': None}
    previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.set_num_threads(1)
            candidate = copy.deepcopy(source).cpu().float().eval()
            inputs = torch.zeros(shape, dtype=torch.float32)
            with patch.object(SelectiveScanCore, 'apply', staticmethod(lambda u, *_: u)):
                if _is_anatomical(candidate):
                    partial = profile_graph_conditioning_excluded(
                        candidate, inputs=(inputs,), verbose=False)
                    report['training'] = {
                        'thop_params': partial['baseline_params'],
                        'macs': partial['baseline_macs'],
                        'full_parameters': partial['full_parameters'],
                    }
                    report['reported'] = report['partial'] = partial
                    report['deploy_unavailable'] = 'This reporting scope measures the training architecture; no deployment conversion is used.'
                else:
                    report['training'] = _profile_counts(candidate, inputs)
                    try:
                        deployment = convert_to_deploy(candidate)
                    except ValueError as error:
                        # For example, the S-only converter does not support Base.
                        report['deploy_unavailable'] = str(error)
                    else:
                        report['deploy'] = profile_posemamba_reported(
                            deployment, inputs=(inputs,), verbose=False)
                        report['deployment_cache'] = deploy_cache_storage(deployment)
    finally:
        torch.set_num_threads(previous_threads)
    return report


def log_model_complexity(model, config, log):
    """Print the model's declared partial scope or its legacy deployment scope."""
    full_parameters = sum(parameter.numel() for parameter in model.parameters())
    if getattr(config, 'backbone', None) != 'PoseMamba':
        return {'training': {'full_parameters': full_parameters}, 'deploy': None}

    anatomical = _is_anatomical(model)
    label = 'Partial network (graph conditioning excluded)' if anatomical else 'Equivalent deployment'
    log.info(reporting_note_for(model))
    try:
        report = collect_model_complexity(model, config)
    except Exception as error:
        # Keep the diagnostic isolated from training, including on failure.
        message = f'{type(error).__name__}: {error}'
        log.warning(f'[PoseMamba/THOP] {label}: N/A | {message}')
        return {'training': {'full_parameters': full_parameters}, 'deploy': None,
                'profiling_error': message}

    measured = report.get('reported') if anatomical else report['deploy']
    if measured is None:
        log.info(f'[PoseMamba/THOP] {label}: N/A | '
                 + report['deploy_unavailable'])
    else:
        prefix = '' if anatomical else f'[PoseMamba/THOP] {label}:\n'
        log.info(prefix +
                 f"Params={measured['thop_params'] / 1e6:.6f} M | MACs={measured['macs'] / 1e9:.9f} G")
    return report
