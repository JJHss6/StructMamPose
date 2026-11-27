"""THOP accounting with exact counts for equivalent graph rewrites.

The graph hook includes the functional affine compositions used by the optimized
implementation. Other operators retain the project's original THOP rules:
adjacency aggregation, attention matmul, SSM einsum, selective scan, and tensor
operations remain omitted. This is not a complete count of model arithmetic.
"""

import torch
from torch import nn
from thop import profile

from lib.model.deploy_s.posemamba_deploy import DeployGraphConditioning
from lib.model.deploy_s.deploy_coordinates import FirstInputProjection
from lib.model.deploy_s.posemamba_online import OnlineFactorizedGraph, OnlineWeightComposer
from lib.model.deploy_s.deploy_graph_side import (
    AdvancedGraphConditioning, AdvancedSideProjection, AdvancedCrossAttention,
)


def count_advanced_module(module, inputs, output):
    """Count a complete advanced leaf, including functional arithmetic/parameters."""
    module.total_ops += module.total_ops.new_tensor([module.profile_macs(inputs)])
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())


def count_first_input_projection(module, inputs, output):
    # In addition to the 2 -> 256 Linear, explicitly count variance statistics
    # conservatively at the existing no-affine LN cost (2 * 64 per token), and
    # the 256 output scalings. Constant additions retain the existing convention.
    tokens = inputs[0].numel() // 2
    macs = tokens * (2 * 256 + 2 * 64 + 256)
    module.total_ops += module.total_ops.new_tensor([macs])
    # offset is a direct Parameter, not a Linear child; it must be included.
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())






def _deploy_graph_macs(module, inputs):
    batch, frames, joints, dim = inputs.shape
    hidden = module.first[0].out_features
    projection = module.merged.out_features
    state = module.tail[2].out_features
    norm_ops = 4 if module.first[2].elementwise_affine else 2
    tokens = batch * frames * joints
    return tokens * (
        hidden * dim + hidden * norm_ops
        + hidden * projection
        + 2 * projection * (4 + state)
    )


def count_deploy_graph_conditioning(module, inputs, output):
    """Count the exported graph, including its separate parent-bias correction.

    The graph no longer executes the frozen topology branch or composes weights
    at runtime. Its cached topology outputs remain part of checkpoint storage,
    reported separately by ``deploy_cache_storage``. Freezing a parameter does
    not remove it from this parameter count.
    """
    macs = _deploy_graph_macs(module, inputs[0])
    module.total_ops += module.total_ops.new_tensor([macs])
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())




def deploy_cache_tensors(model):
    """Identify all learned deployment caches, including first-layer/root constants."""
    caches = {}
    for prefix, module in model.named_modules():
        names = ()
        if isinstance(module, (DeployGraphConditioning, AdvancedGraphConditioning, OnlineFactorizedGraph)):
            names = ('static_horizontal', 'static_vertical')
        elif isinstance(module, FirstInputProjection):
            names = ('joint_numerator',)
        elif isinstance(module, AdvancedCrossAttention):
            names = ('root_output',)
        for name in names:
            caches[f'{prefix}.{name}' if prefix else name] = getattr(module, name)
    return caches


def deploy_cache_storage(model):
    """Return derived inference constants separately from registered parameters."""
    caches = deploy_cache_tensors(model).values()
    return {'elements': sum(cache.numel() for cache in caches),
            'bytes': sum(cache.numel() * cache.element_size() for cache in caches)}


def posemamba_custom_ops():
    return {FirstInputProjection: count_first_input_projection, OnlineFactorizedGraph: count_advanced_module, AdvancedGraphConditioning: count_advanced_module, AdvancedSideProjection: count_advanced_module, AdvancedCrossAttention: count_advanced_module, DeployGraphConditioning: count_deploy_graph_conditioning}


class _ProfileRoot(nn.Module):
    """Ensure a custom-hooked model is also counted as a THOP child/leaf."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model(*inputs)


def profile_posemamba(model, inputs, **kwargs):
    """Return (MACs, THOP Params) without double-counting graph submodules.

    The wrapper also makes direct profiling of either graph module safe:
    some THOP versions always traverse the root's children despite its hook.
    """
    custom_ops = posemamba_custom_ops()
    custom_ops.update(kwargs.pop('custom_ops', {}) or {})
    wrapper = _ProfileRoot(model)
    wrapper.training = model.training
    # THOP versions can leave counter buffers on modules without registered
    # hooks. Restore pre-profile buffers so profiling does not alter checkpoints.
    modules = tuple(wrapper.modules())
    training = {module: module.training for module in modules}
    counters = {
        module: {name: module._buffers[name] for name in ('total_ops', 'total_params')
                 if name in module._buffers}
        for module in modules
    }
    nonpersistent_counters = {
        module: module._non_persistent_buffers_set.intersection(('total_ops', 'total_params'))
        for module in modules
    }
    forward_hooks = {module: set(module._forward_hooks) for module in modules}
    try:
        return profile(wrapper, inputs=inputs, custom_ops=custom_ops, **kwargs)
    finally:
        for module, previous in counters.items():
            for name in ('total_ops', 'total_params'):
                module._buffers.pop(name, None)
                module._non_persistent_buffers_set.discard(name)
            module._buffers.update(previous)
            module._non_persistent_buffers_set.update(nonpersistent_counters[module])
            module.training = training[module]
            # THOP leaves its hooks installed if the profiled forward raises.
            for key in set(module._forward_hooks).difference(forward_hooks[module]):
                del module._forward_hooks[key]


def profile_online_reported(model, inputs, *, verbose=False):
    """Count an explicit subset of the online-composed inference model.

    Both Params and MACs exclude every online graph's first Linear and its
    OnlineWeightComposer. GCN second weights, LN-affine preparation, effective
    per-token projection and graph tails remain counted. This is not an
    exclusion of all conditioning, nor the complete deployment cost.

    Exclusions come from actual forward calls and a parameter-identity union.
    The raw baseline includes online matrix/vector products, gamma scaling and
    the functional effective-map application through the complete graph hook.
    Existing hooks, mode/counter state and CPU/CUDA RNG are preserved, including
    when model execution or validation raises.
    """
    regions = {}
    found_graphs = 0
    for (name, module) in model.named_modules():
        if isinstance(module, OnlineFactorizedGraph):
            if type(module) is not OnlineFactorizedGraph:
                raise ValueError('Online reporting does not support graph subclasses with different execution.')
            module.validate_structure()
            found_graphs += 1
            candidates = (('first.0', 'graph_input_projection', module.first[0]), ('composer', 'weight_composer', module.composer))
            for (suffix, kind, region) in candidates:
                if kind == 'graph_input_projection' and type(region) is not nn.Linear or (kind == 'weight_composer' and type(region) is not OnlineWeightComposer):
                    raise ValueError(f'Unsupported online exclusion module: {name}.{suffix}')
                regions.setdefault(id(region), dict(name=f'{name}.{suffix}' if name else suffix, kind=kind, module=region, calls=0, macs=0))
        elif isinstance(module, (AdvancedGraphConditioning, DeployGraphConditioning)):
            raise ValueError('Online reporting requires all graph paths to use OnlineFactorizedGraph.')
    if not found_graphs:
        raise ValueError('No supported online graph modules were found for this reporting scope.')
    devices = sorted({tensor.device.index for tensor in list(model.parameters()) + list(model.buffers()) + [value for value in inputs if isinstance(value, torch.Tensor)] if tensor.device.type == 'cuda'})
    handles = []
    try:
        for region in regions.values():

            def measure(module, arguments, output, region=region):
                if region['kind'] == 'graph_input_projection':
                    macs = output.numel() * module.in_features
                else:
                    macs = module.profile_macs(arguments)
                region['calls'] += 1
                region['macs'] += int(macs)
            handles.append(region['module'].register_forward_hook(measure))
        with torch.random.fork_rng(devices=devices):
            (macs, params) = profile_posemamba(model, inputs, verbose=verbose)
        excluded_parameters = {}
        breakdown = {}
        details = []
        for region in regions.values():
            if not region['calls']:
                raise ValueError(f"Online exclusion module did not execute: {region['name']}")
            unique_parameters = {id(parameter): parameter for parameter in region['module'].parameters()}
            excluded_parameters.update(unique_parameters)
            item = dict(name=region['name'], kind=region['kind'], calls=region['calls'], params=sum((parameter.numel() for parameter in unique_parameters.values())), macs=region['macs'])
            details.append(item)
            category = breakdown.setdefault(region['kind'], dict(parameters={}, macs=0, calls=0))
            category['parameters'].update(unique_parameters)
            category['macs'] += item['macs']
            category['calls'] += item['calls']
        excluded_params = sum((parameter.numel() for parameter in excluded_parameters.values()))
        excluded_macs = sum((region['macs'] for region in regions.values()))
        (baseline_params, baseline_macs) = (int(params), int(macs))
        if excluded_params > baseline_params or excluded_macs > baseline_macs:
            raise ValueError('Online exclusions exceed the measured THOP baseline.')
        result = {'scope': 'partial_online_deployment_first_gcn_and_weight_composer_excluded', 'thop_params': baseline_params - excluded_params, 'macs': baseline_macs - excluded_macs, 'baseline_params': baseline_params, 'baseline_macs': baseline_macs, 'excluded_parameters': excluded_params, 'excluded_macs': excluded_macs, 'full_parameters': sum((parameter.numel() for parameter in model.parameters())), 'graph_count': found_graphs, 'excluded_regions': details}
        for (kind, category) in breakdown.items():
            result[f'excluded_{kind}_params'] = sum((p.numel() for p in category['parameters'].values()))
            result[f'excluded_{kind}_macs'] = category['macs']
            result[f'{kind}_calls'] = category['calls']
        return result
    finally:
        for handle in handles:
            handle.remove()
