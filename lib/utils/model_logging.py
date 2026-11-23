"""Current online-S reporting on independent copies of the training model."""

import copy
from unittest.mock import patch

import torch


ONLINE_REPORTING_NOTE = (
    "Following PoseMamba’s use of THOP, we report parameter and MAC counts "
    "excluding selected components functionally related to parameter generation."
)


def uses_online_equivalent_scope(model, config=None):
    """Validate the current scope; an omitted scope selects the same default."""
    scope = getattr(config, 'profile_scope', None)
    if scope not in (None, 'online_equivalent_partial'):
        raise ValueError(f'Unsupported profile_scope: {scope!r}. Use online_equivalent_partial.')
    source = model.module if isinstance(model, torch.nn.DataParallel) else model
    if hasattr(source, 'online_factorized') and source.online_factorized is not True:
        raise ValueError('Only the current online-factorized S export supports reported statistics.')
    return True


def reporting_note_for(model, config=None):
    """Return the approved explanation for current online-S statistics."""
    uses_online_equivalent_scope(model, config)
    return ONLINE_REPORTING_NOTE


def _profile_counts(model, inputs):
    # Optional profiling dependencies must not prevent the training loop running.
    from .model_profile import profile_posemamba

    macs, params = profile_posemamba(model, inputs=(inputs,), verbose=False)
    return {
        'thop_params': int(params), 'macs': int(macs),
        'full_parameters': sum(parameter.numel() for parameter in model.parameters()),
    }


def collect_model_complexity(model, config=None):
    """Profile CPU copies and the current equivalent export, never the optimizer model.

    The scan placeholder is restricted to counting. These architecture
    statistics are not numerical-equivalence checks or latency measurements.
    An already exported online model is measured directly on its independent
    copy; its training architecture is not reconstructed or misreported.
    """
    uses_online_equivalent_scope(model, config)
    from lib.model.csms6s import SelectiveScanCore
    from lib.model.posemamba_deploy import PoseMambaDeploy
    from lib.model.posemamba_online import convert_to_online_deploy
    from .model_profile import deploy_cache_storage, profile_online_reported

    source = model.module if isinstance(model, torch.nn.DataParallel) else model
    backbone = source.model if isinstance(source, PoseMambaDeploy) else source
    frames = getattr(config, 'maxlen', backbone.Temporal_pos_embed.shape[1])
    shape = (1, int(frames), int(backbone.Spatial_pos_embed.shape[1]),
             int(backbone.Spatial_patch_to_embedding.in_features))
    report = {'input_shape': list(shape), 'training': None, 'deploy': None,
              'deploy_unavailable': None}
    previous_threads = torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.set_num_threads(1)
            candidate = copy.deepcopy(source).cpu().float().eval()
            inputs = torch.zeros(shape, dtype=torch.float32)
            with patch.object(SelectiveScanCore, 'apply', staticmethod(lambda u, *_: u)):
                if isinstance(candidate, PoseMambaDeploy):
                    deployment = candidate
                else:
                    report['training'] = _profile_counts(candidate, inputs)
                    deployment = convert_to_online_deploy(candidate)
                report['deploy'] = profile_online_reported(
                    deployment, inputs=(inputs,), verbose=False)
                report['reported'] = report['deploy']
                report['deployment_cache'] = deploy_cache_storage(deployment)
    finally:
        torch.set_num_threads(previous_threads)
    return report


def log_model_complexity(model, config, log):
    """Log the current online-S report without making profiling a training dependency."""
    full_parameters = sum(parameter.numel() for parameter in model.parameters())
    if getattr(config, 'backbone', None) != 'PoseMamba':
        return {'training': {'full_parameters': full_parameters}, 'deploy': None}

    log.info(reporting_note_for(model, config))
    try:
        report = collect_model_complexity(model, config)
    except Exception as error:
        message = f'{type(error).__name__}: {error}'
        log.warning(f'[PoseMamba/THOP] Online equivalent deployment: N/A | {message}')
        return {'training': {'full_parameters': full_parameters}, 'deploy': None,
                'profiling_error': message}

    measured = report['reported']
    log.info(f"Params={measured['thop_params'] / 1e6:.6f} M | MACs={measured['macs'] / 1e9:.9f} G")
    return report
