"""Dispatch deployment profiling for the shared original S and Base training model."""
import torch

ONLINE_REPORTING_NOTE = (
    "Following PoseMamba’s use of THOP, we report parameter and MAC counts "
    "excluding selected components functionally related to parameter generation."
)


def _backend(model=None, config=None):
    configured = getattr(config, 'dim_feat', None)
    source = model.module if isinstance(model, torch.nn.DataParallel) else model
    backbone = source.model if hasattr(source, 'architecture') and hasattr(source, 'model') else source
    embedding = getattr(backbone, 'Spatial_patch_to_embedding', None)
    actual = getattr(embedding, 'out_features', None)
    if actual is not None and configured is not None and actual != configured:
        raise ValueError('Configured trunk width does not match the supplied model.')
    width = actual if actual is not None else configured
    scope = getattr(config, 'profile_scope', None)
    if width == 64:
        if scope not in (None, 'online_equivalent_partial'):
            raise ValueError('S requires profile_scope=online_equivalent_partial.')
        from . import model_logging_s
        return model_logging_s
    if width == 128:
        if scope not in (None, 'base_strassen_partial'):
            raise ValueError('Base requires profile_scope=base_strassen_partial.')
        from . import model_logging_b
        return model_logging_b
    raise ValueError('Only original S (64) and Base (128) are supported.')


def uses_online_equivalent_scope(model=None, config=None):
    _backend(model, config)
    return True


def reporting_note_for(model=None, config=None):
    _backend(model, config)
    return ONLINE_REPORTING_NOTE


def collect_model_complexity(model, config=None):
    return _backend(model, config).collect_model_complexity(model, config)


def log_model_complexity(model, config, log):
    if getattr(config, 'backbone', None) != 'PoseMamba':
        return {'training': {'full_parameters': sum(p.numel() for p in model.parameters())}, 'deploy': None}
    return _backend(model, config).log_model_complexity(model, config, log)
