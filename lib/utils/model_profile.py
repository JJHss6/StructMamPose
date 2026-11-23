"""Legacy THOP accounting with exact counts for equivalent graph rewrites.

The graph hook includes the functional affine compositions used by the optimized
implementation. Other operators retain the project's original THOP rules:
adjacency aggregation, attention matmul, SSM einsum, selective scan, and tensor
operations remain omitted. This is not a complete count of model arithmetic.
"""

import torch
from torch import nn
from thop import profile

from lib.model.graph_conditioning import FramewiseGraphConditioning
from lib.model.grouped_posemamba import GroupedLinear
from lib.model.posemamba_deploy import DeployFrameContextGraph, DeployGraphConditioning
from lib.model.deploy_coordinates import FirstInputProjection
from lib.model.deploy_graph_side import (
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


def count_grouped_linear(module, inputs, output):
    """Count the actual grouped weight blocks used by the functional einsum."""
    macs = output.numel() * (module.in_features // module.groups)
    module.total_ops += module.total_ops.new_tensor([macs])
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())


def count_framewise_graph_conditioning(module, inputs, output):
    """Count the entire graph module once, including its registered parameters.

    THOP treats a custom-hooked child module as a leaf when summing results, so
    its Linear/LayerNorm child hooks must not be added to this total again.
    Weight and bias compositions are computed once per forward, and the static
    adjacency branch is computed once per joint, independent of batch/frames.
    """
    batch, frames, joints, dim = inputs[0].shape
    hidden = module.gcn[0].out_features
    graph_out = module.gcn[3].out_features
    projection = module.proj_gcn[0].out_features
    state = module.proj_gcn[3].out_features
    tokens = batch * frames * joints

    macs = (
        tokens * hidden * (dim + 4)  # First GCN Linear and affine LayerNorm.
        + tokens * hidden * projection  # Apply the composed affine map.
        + 2 * tokens * projection * (4 + state)  # Two distinct nonlinear tails.
        + projection * graph_out * hidden  # Compose the two Linear weights.
        + projection * graph_out  # Compose the intermediate bias.
        + joints * (
            joints * graph_out + 2 * projection * (graph_out + 4 + state)
        )  # Static adjacency embedding and its two nonlinear projections.
    )
    module.total_ops += module.total_ops.new_tensor([macs])
    # Include the original GCN/output projection weights even though the fused
    # forward accesses them functionally instead of calling their Linear modules.
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())


def _deploy_graph_macs(module, inputs):
    batch, frames, joints, dim = inputs.shape
    hidden = module.first[0].out_features
    groups = getattr(module.first[0], 'groups', 1)
    projection = module.merged.out_features
    state = module.tail[2].out_features
    norm_ops = 4 if module.first[2].elementwise_affine else 2
    tokens = batch * frames * joints
    return tokens * (
        hidden * dim // groups + hidden * norm_ops
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


def count_deploy_frame_context_graph(module, inputs, output):
    """Count the graph and one context MLP evaluation per frame as one leaf.

    THOP must not add the nested base graph or context Linear hooks a second
    time. The context output has 17 different joint vectors, but its weights
    are applied once per frame rather than once per joint or scan direction.
    """
    batch, frames = inputs[0].shape[:2]
    first, last = module.context[0], module.context[2]
    context_macs = batch * frames * (
        first.in_features * first.out_features + last.in_features * last.out_features
    )
    macs = _deploy_graph_macs(module.base, inputs[0]) + context_macs
    module.total_ops += module.total_ops.new_tensor([macs])
    module.total_params[0] = sum(parameter.numel() for parameter in module.parameters())


def deploy_cache_tensors(model):
    """Identify all learned deployment caches, including first-layer/root constants."""
    caches = {}
    for prefix, module in model.named_modules():
        names = ()
        if isinstance(module, (DeployGraphConditioning, AdvancedGraphConditioning)):
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
    return {
        FirstInputProjection: count_first_input_projection,
        AdvancedGraphConditioning: count_advanced_module,
        AdvancedSideProjection: count_advanced_module,
        AdvancedCrossAttention: count_advanced_module,
        GroupedLinear: count_grouped_linear,
        FramewiseGraphConditioning: count_framewise_graph_conditioning,
        DeployGraphConditioning: count_deploy_graph_conditioning,
        DeployFrameContextGraph: count_deploy_frame_context_graph,
    }


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


def profile_posemamba_reported(model, inputs, *, verbose=False):
    """Report a declared subset of legacy THOP MACs, keeping every parameter.

    ``baseline_macs`` is the unchanged result of ``profile_posemamba``. The
    reported ``macs`` excludes only actual calls to each supported graph's first
    Linear projection. These projections still exist and execute in the model;
    this function changes the reporting scope, not the architecture or runtime.
    The legacy exclusions documented above are not subtracted a second time.

    Equivalent graph modules are supported, including the nested base inside
    ``DeployFrameContextGraph``. Unsupported graph projections and models with
    no executed eligible projection raise instead of producing a misleadingly
    labelled result. Temporary measurement hooks are removed even on failure.
    """
    projections = []
    seen = set()
    for name, module in model.named_modules():
        if isinstance(module, (AdvancedGraphConditioning, DeployGraphConditioning)):
            projection = module.first[0]
        elif isinstance(module, FramewiseGraphConditioning):
            projection = module.gcn[0]
        else:
            continue
        if not isinstance(projection, (nn.Linear, GroupedLinear)):
            raise ValueError(
                f'Unsupported graph input projection at {name or "<root>"}: '
                f'{type(projection).__name__}. Expected Linear or GroupedLinear.'
            )
        if id(projection) not in seen:
            seen.add(id(projection))
            projections.append(projection)
    if not projections:
        raise ValueError('No supported graph input projections were found for this reporting scope.')

    measured = {'macs': 0, 'calls': 0}
    handles = []

    def measure_projection(module, _inputs, output):
        groups = module.groups if isinstance(module, GroupedLinear) else 1
        measured['macs'] += output.numel() * (module.in_features // groups)
        measured['calls'] += 1

    try:
        for projection in projections:
            handles.append(projection.register_forward_hook(measure_projection))
        baseline_macs, params = profile_posemamba(model, inputs, verbose=verbose)
        if measured['calls'] == 0:
            raise ValueError('No supported graph input projections executed for the supplied inputs.')
        baseline_macs = int(baseline_macs)
        excluded_macs = measured['macs']
        if excluded_macs > baseline_macs:
            raise ValueError('Measured graph projection MACs exceed the baseline THOP count.')
        return {
            'scope': 'thop_excluding_graph_input_projections',
            'thop_params': int(params),
            'macs': baseline_macs - excluded_macs,
            'baseline_macs': baseline_macs,
            'excluded_graph_input_projection_macs': excluded_macs,
            'full_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'projection_calls': measured['calls'],
        }
    finally:
        for handle in handles:
            handle.remove()


def profile_graph_conditioning_excluded(model, inputs, *, verbose=False):
    """Return an explicitly partial count for an unconverted framewise graph.

    Both Params and MACs exclude ``gcn``, ``proj_gcn`` and ``proj_adj``. The
    baseline is still the existing THOP scope, which already omits functional
    selective-parameter generation/scan and adjacency/attention matmuls. These
    exclusions change only the reported subset: the graph still executes.

    MACs come from THOP's actual accumulated layer information, so repeated
    dynamic/static projection calls are all included. Parameters are a union
    by identity, counted once even when used by multiple calls or regions.
    Equivalent, pooled, and deployed graph paths are rejected.
    """
    regions = {}
    for name, module in model.named_modules():
        if not hasattr(module, 'in_proj') or not any(
            hasattr(module, field) for field in ('gcn', 'graph_conditioning', 'proj_gcn')
        ):
            continue
        if (getattr(module, 'graph_equivalent', False)
                or getattr(module, 'graph_pool_stride', None) is not None
                or not all(hasattr(module, field) for field in ('gcn', 'proj_gcn', 'proj_adj'))):
            raise ValueError('Graph-conditioning exclusion requires the unconverted full-frame graph path.')
        graph = getattr(module.gcn, 'gcn', None)
        projection = module.proj_gcn
        if (getattr(module.gcn, 'limb_mapping', None) is not None
                or not isinstance(graph, nn.Sequential) or len(graph) != 4
                or not isinstance(graph[0], nn.Linear) or not isinstance(graph[1], nn.SiLU)
                or not isinstance(graph[2], nn.LayerNorm) or not isinstance(graph[3], nn.Linear)
                or not isinstance(projection, nn.Sequential) or len(projection) != 4
                or not isinstance(projection[0], nn.Linear)
                or not isinstance(projection[1], nn.LayerNorm)
                or not isinstance(projection[2], nn.SiLU)
                or not isinstance(projection[3], nn.Linear)
                or not isinstance(module.proj_adj, nn.Linear)):
            raise ValueError('Graph-conditioning exclusion requires the original two-Linear graph/projection structure.')
        for field in ('gcn', 'proj_gcn', 'proj_adj'):
            region = getattr(module, field)
            regions.setdefault(id(region), dict(name=f'{name}.{field}' if name else field,
                                                kind=field, module=region, calls=0))
    if not regions:
        raise ValueError('No unconverted graph-conditioning modules were found.')

    handles = []
    devices = sorted({tensor.device.index for tensor in list(model.parameters()) + list(model.buffers())
                      + [value for value in inputs if isinstance(value, torch.Tensor)]
                      if tensor.device.type == 'cuda'})

    def flatten_layer_info(info, prefix=''):
        result = {}
        for name, (macs, params, children) in info.items():
            path = f'{prefix}.{name}' if prefix else name
            result[path] = (macs, params)
            result.update(flatten_layer_info(children, path))
        return result

    try:
        for region in regions.values():
            def count_call(_module, _inputs, _output, region=region):
                region['calls'] += 1
            handles.append(region['module'].register_forward_hook(count_call))
        with torch.random.fork_rng(devices=devices):
            macs, params, info = profile_posemamba(
                model, inputs, verbose=verbose, ret_layer_info=True)
        layers = flatten_layer_info(info)
        excluded_parameters = {}
        details = []
        for region in regions.values():
            if not region['calls']:
                raise ValueError(f'Graph-conditioning region did not execute: {region["name"]}')
            key = 'model.' + region['name']  # profile_posemamba's _ProfileRoot wrapper.
            if key not in layers:
                raise ValueError(f'THOP did not expose graph-conditioning region: {region["name"]}')
            for parameter in region['module'].parameters():
                excluded_parameters[id(parameter)] = parameter
            details.append(dict(name=region['name'], kind=region['kind'], calls=region['calls'],
                                macs=int(layers[key][0]),
                                params=sum(parameter.numel() for parameter in region['module'].parameters())))
        excluded_params = sum(parameter.numel() for parameter in excluded_parameters.values())
        excluded_macs = sum(region['macs'] for region in details)
        baseline_params, baseline_macs = int(params), int(macs)
        if excluded_params > baseline_params or excluded_macs > baseline_macs:
            raise ValueError('Graph-conditioning exclusions exceed the measured THOP baseline.')
        return {
            'scope': 'partial_network_graph_conditioning_excluded',
            'thop_params': baseline_params - excluded_params,
            'macs': baseline_macs - excluded_macs,
            'baseline_params': baseline_params,
            'baseline_macs': baseline_macs,
            'excluded_graph_params': excluded_params,
            'excluded_graph_macs': excluded_macs,
            'full_parameters': sum(parameter.numel() for parameter in model.parameters()),
            'graph_regions': details,
        }
    finally:
        for handle in handles:
            handle.remove()
