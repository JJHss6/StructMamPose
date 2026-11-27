"""Inference-only equivalent exports for original PoseMamba S.

Training weights are left untouched. The exported graph composes adjacent
linear maps, reuses reversed directions, and stores the two static topology
encodings. Its cached encodings must be counted separately from parameters.
"""

import copy
import json
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn


_FORMAT = 'posemamba.deploy'
_FORMAT_VERSION = 6
_PARENTS = (0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16)
_EDGES = ((0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
          (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
          (12, 13), (14, 15), (15, 16))


def _require(condition, message):
    if not condition:
        raise ValueError('PoseMamba S deployment export: ' + message)


def _linear(module, in_features, out_features, name, bias=True):
    expected_shape = (out_features, in_features)
    _require(isinstance(module, nn.Linear) and module.in_features == in_features
             and module.out_features == out_features and tuple(module.weight.shape) == expected_shape
             and ((module.bias is not None) == bias),
             f'{name} must be Linear({in_features}, {out_features}, bias={bias}).')


def _norm(module, width, name):
    _require(isinstance(module, nn.LayerNorm)
             and tuple(module.normalized_shape) == (width,)
             and module.elementwise_affine and module.bias is not None,
             f'{name} must be an affine LayerNorm({width}).')


def _without_affine(norm):
    return nn.LayerNorm(norm.normalized_shape, eps=norm.eps, elementwise_affine=False)


def _fold_norm_into_linear(norm, linear):
    """Keep normalization/epsilon; absorb only its channel affine transform."""
    _require(isinstance(linear, nn.Linear) and linear.bias is not None,
             'normalization folding requires an existing Linear bias.')
    with torch.no_grad():
        linear.bias.add_(linear.weight @ norm.bias)
        linear.weight.mul_(norm.weight.unsqueeze(0))
    return _without_affine(norm)


def _check_topology(adj, parents=None):
    expected = 2 * torch.eye(17, device=adj.device, dtype=adj.dtype)
    for first, second in _EDGES:
        expected[first, second] = expected[second, first] = 1
    _require(torch.equal(adj, expected), 'the original fixed 17-joint adjacency is required.')
    if parents is not None:
        _require(torch.equal(parents, torch.tensor(_PARENTS, device=parents.device)),
                 'the original parent-joint order is required.')


def _check_graph_parts(first, last, projection, proj_adj):
    _require(isinstance(first, nn.Sequential) and len(first) == 3 and isinstance(projection, nn.Sequential) and (len(projection) == 4), 'the original graph mapping activations and normalizations are required.')
    hidden = getattr(first[0], 'out_features', None)
    _require(type(hidden) is int and hidden > 0, 'GCN hidden width must be a positive integer.')
    _require(hidden == 128, 'The original full-width GCN is required.')
    _linear(first[0], 128, hidden, 'GCN first layer')
    _require(isinstance(first[1], nn.SiLU), 'GCN activation must be SiLU.')
    _norm(first[2], hidden, 'GCN normalization')
    _linear(last, hidden, 128, 'GCN last layer')
    _linear(projection[0], 128, 64, 'graph projection first layer')
    _norm(projection[1], 64, 'graph projection normalization')
    _require(isinstance(projection[2], nn.SiLU), 'graph projection activation must be SiLU.')
    _linear(projection[3], 64, 16, 'graph projection last layer')
    _linear(proj_adj, 17, 128, 'static adjacency projection')
    return hidden


def _check_original_graph(op):
    _require(not getattr(op, 'graph_equivalent', False) and getattr(op, 'graph_pool_stride', None) is None, 'use the original full-frame graph, without pooling or prior equivalent conversion.')
    _require(hasattr(op, 'gcn') and hasattr(op.gcn, 'gcn') and hasattr(op, 'proj_gcn') and hasattr(op, 'proj_adj'), 'the original GCN and graph projections are missing.')
    (graph, projection) = (op.gcn.gcn, op.proj_gcn)
    _require(isinstance(graph, nn.Sequential) and len(graph) == 4 and isinstance(projection, nn.Sequential) and (len(projection) == 4), 'the original four-layer graph mappings are required.')
    hidden = _check_graph_parts(graph[:3], graph[3], projection, op.proj_adj)
    _check_topology(op.gcn.adj)
    return hidden




class DeployGraphConditioning(nn.Module):
    """Frozen framewise graph map with persistent learned topology encodings."""

    def __init__(self, first, merged, tail, horizontal_correction, static_horizontal, static_vertical, adj, parents, fold_affine):
        super().__init__()
        self.first = first
        self.merged = merged
        self.tail = tail
        self.horizontal_correction = nn.Parameter(horizontal_correction, requires_grad=False)
        self.register_buffer('static_horizontal', static_horizontal)
        self.register_buffer('static_vertical', static_vertical)
        self.register_buffer('adj', adj, persistent=False)
        self.register_buffer('parents', parents, persistent=False)
        self.fold_affine = bool(fold_affine)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_op(cls, op, fold_affine=True):
        """Convert a supported evaluated S graph without changing its weights."""
        _require(not any((module.training for module in op.modules())), 'call eval() before converting the graph.')
        _require(hasattr(op, 'gcn') and hasattr(op.gcn, 'gcn'), 'missing the source graph.')
        _check_original_graph(op)
        (graph, projection) = (op.gcn.gcn, op.proj_gcn)
        return cls._from_parts(graph[:3], graph[3], projection, op.proj_adj, op.gcn.adj, fold_affine)

    @classmethod
    def _from_parts(cls, source_first, source_last, projection, proj_adj, adj, fold_affine):
        first = copy.deepcopy(source_first)
        tail = copy.deepcopy(projection[1:])
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            weight = projection[0].weight @ source_last.weight
            bias = projection[0].weight @ source_last.bias + projection[0].bias
            if fold_affine:
                bias = bias + weight @ source_first[2].bias
                weight = weight * source_first[2].weight.unsqueeze(0)
                first[2] = _without_affine(source_first[2])
            merged = nn.Linear(weight.shape[1], 64).to(device=weight.device, dtype=weight.dtype)
            merged.weight.copy_(weight)
            merged.bias.copy_(bias)
            parents = torch.tensor(_PARENTS, device=weight.device, dtype=torch.long)
            topology = proj_adj(adj)
            static_horizontal = projection(topology + topology.index_select(0, parents))
            static_vertical = projection(topology)
        return cls(first, merged, tail, projection[0].bias.detach().clone(), static_horizontal.detach().clone(), static_vertical.detach().clone(), adj.detach().clone(), parents, fold_affine)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (17, 128):
            raise ValueError('Deployment graph expects [batch, frames, 17, 128].')
        q = self.merged(self.first(self.adj @ x))
        return self._finish(q)

    def _finish(self, q):
        horizontal = self.tail(q + q.index_select(-2, self.parents) - self.horizontal_correction) + self.static_horizontal
        vertical = self.tail(q) + self.static_vertical
        horizontal = horizontal.flatten(1, 2).transpose(1, 2)
        vertical = vertical.transpose(1, 2).flatten(1, 2).transpose(1, 2)
        return torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)




class DeployCrossAttention(nn.Module):
    """Single-head inference attention with equivalent Q/K and V/output maps."""

    def __init__(self, query, value, norm):
        super().__init__()
        self.query, self.value, self.norm = query, value, norm
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_attention(cls, attention):
        _require(not any(module.training for module in attention.modules()), 'call eval() before attention export.')
        _require(attention.num_heads == 1, 'fused attention requires one head.')
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            _linear(getattr(attention, name), 64, 64, 'attention ' + name)
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            query = nn.Linear(64, 64).to(attention.q_proj.weight)
            value = nn.Linear(64, 64).to(attention.v_proj.weight)
            query.weight.copy_(attention.k_proj.weight.T @ attention.q_proj.weight)
            query.bias.copy_(attention.k_proj.weight.T @ attention.q_proj.bias)
            value.weight.copy_(attention.out_proj.weight @ attention.v_proj.weight)
            value.bias.copy_(attention.out_proj.weight @ attention.v_proj.bias + attention.out_proj.bias)
        return cls(query, value, copy.deepcopy(attention.norm))

    def forward(self, x, y, attn_mask=None):
        if x.shape != y.shape or x.ndim != 4 or x.shape[-2:] != (17, 64):
            raise ValueError('Deployment attention expects matching [batch, frames, 17, 64] inputs.')
        # Retain the singleton head axis so the original attention-mask
        # broadcasting convention, including [B,1,T,T], is preserved.
        query = self.query(x).flatten(2).unsqueeze(1)
        key = y.flatten(2).unsqueeze(1)
        value = self.value(y).flatten(2).unsqueeze(1)
        # The original key bias contributes a row constant and cancels under
        # softmax. Attention rows sum to one, permitting V/output bias fusion.
        scores = query @ key.transpose(-2, -1) / (17 * 64) ** .5
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        return self.norm((weights @ value).squeeze(1).reshape_as(x))


def _inspect_original_model(model):
    from ..PoseMamba import PoseMamba
    from ..csms6s import CrossScan_plus_poselimbs, CrossMerge_plus_poselimbs
    _require(isinstance(model, PoseMamba), 'expected a PoseMamba model; unwrap DataParallel first.')
    _require(not any((module.training for module in model.modules())), 'call model.eval() before export.')
    _linear(model.Spatial_patch_to_embedding, 2, 64, 'input embedding')
    _require(tuple(model.Spatial_pos_embed.shape) == (1, 17, 64), 'expected 17 joints and trunk width 64.')
    _require(model.Temporal_pos_embed.ndim == 3 and model.Temporal_pos_embed.shape[0] == 1 and (model.Temporal_pos_embed.shape[2] == 64), 'unexpected temporal positional embedding.')
    frames = int(model.Temporal_pos_embed.shape[1])
    _require(frames > 0 and model.block_depth == 10 and (len(model.STEblocks) == len(model.TTEblocks) == 10), 'the original 10 spatial + 10 temporal blocks are required.')
    for (index, block) in enumerate(list(model.STEblocks) + list(model.TTEblocks)):
        name = f'block {index}'
        op = block.op
        _require(_check_original_graph(op) == 128, name + ' requires the original full-width GCN.')
        _norm(block.norm, 64, name + ' SSM input normalization')
        _norm(block.norm2, 64, name + ' FFN normalization')
        _linear(block.mlp.fc1, 64, 128, name + ' FFN first layer')
        _linear(block.mlp.fc2, 128, 64, name + ' FFN last layer')
        _require(isinstance(block.mlp.act, nn.GELU) and block.mlp.act.approximate == 'none', name + ' requires the original GELU FFN.')
        _linear(op.in_proj, 64, 256, name + ' SSM input projection', bias=False)
        _linear(op.out_proj, 128, 64, name + ' SSM output projection', bias=False)
        _require(tuple(op.x_proj_weight.shape) == (4, 36, 128) and tuple(op.dt_projs_weight.shape) == (4, 128, 4) and (tuple(op.dt_projs_bias.shape) == (4, 128)) and (tuple(op.A_logs.shape) == (512, 16)) and (tuple(op.Ds.shape) == (512,)), name + ' requires unchanged SSM inner=128, state=16, dt_rank=4.')
        _require(not op.channel_first and (not op.disable_z) and (not op.disable_z_act) and isinstance(op.act, nn.SiLU) and isinstance(op.out_act, nn.Identity), name + ' has an unsupported SSM activation or gate configuration.')
        _norm(op.out_norm, 128, name + ' SSM output normalization')
        conv = op.conv2d
        _require(isinstance(conv, nn.Conv2d) and conv.in_channels == conv.out_channels == 128 and (conv.groups == 128) and (conv.kernel_size == (3, 3)) and (conv.stride == (1, 1)) and (conv.padding == (1, 1)) and (conv.dilation == (1, 1)) and (conv.bias is not None), name + ' requires the original depthwise 3x3 convolution.')
        keywords = getattr(op.forward_core, 'keywords', {})
        _require(keywords.get('CrossScan') is CrossScan_plus_poselimbs and keywords.get('CrossMerge') is CrossMerge_plus_poselimbs, name + ' requires the original four pose-limb scan directions.')
    for name in ('Spatial_norm', 'Temporal_norm'):
        _norm(getattr(model, name), 64, name)
    for name in ('head_1', 'head_2'):
        head = getattr(model, name)
        _require(isinstance(head, nn.Sequential) and len(head) == 2, name + ' is unsupported.')
        _norm(head[0], 64, name + ' normalization')
        _linear(head[1], 64, 3, name + ' output')
    _linear(model.proj_side[0], 2, 64, 'side input projection')
    _require(isinstance(model.proj_side[1], nn.SiLU), 'side activation must be SiLU.')
    _linear(model.proj_side[2], 64, 64, 'side output projection')
    _norm(model.proj_side[3], 64, 'side normalization')
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        _linear(getattr(model.cross_attn, name), 64, 64, 'cross attention ' + name)
    _norm(model.cross_attn.norm, 64, 'cross attention normalization')
    _require(model.cross_attn.num_heads == 1, 'cross attention must retain its original single head.')
    expected_parameters = 2391942
    _require(sum((parameter.numel() for parameter in model.parameters())) == expected_parameters + 64 * (frames - 243), 'unexpected additional, removed, or shared parameters in the S model.')
    floating = [tensor for tensor in model.state_dict().values() if tensor.is_floating_point()]
    _require(len({(tensor.dtype, tensor.device) for tensor in floating}) == 1, 'export requires a single floating dtype/device for the original model.')
    info = dict(num_frame=frames, num_joints=17, in_chans=2, embed_dim_ratio=64, depth=10, mlp_ratio=2.0, ffn_hidden_dim=128)
    return info


class PoseMambaDeploy(nn.Module):
    """Frozen inference wrapper; resume training from the original checkpoint."""

    def __init__(self, model, construction_info, normalization_eps, fold_affine=True,
                 fuse_attention=True, advanced=False):
        super().__init__()
        self.model = model
        self._construction_info = copy.deepcopy(construction_info)
        self._normalization_eps = copy.deepcopy(normalization_eps)
        self.architecture = construction_info.get('architecture', 'original_s')
        self.fold_affine = bool(fold_affine)
        self.fuse_attention = bool(fuse_attention)
        self.advanced = bool(advanced)
        self.online_factorized = False
        self.export_metadata = {}
        self.requires_grad_(False)
        self.train(False)

    @property
    def construction_info(self):
        return copy.deepcopy(self._construction_info)

    def train(self, mode=True):
        if mode:
            raise RuntimeError('PoseMambaDeploy is inference-only. Train the original model and export it again.')
        return super().train(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def convert_to_deploy(model, fold_affine=True, fuse_attention=True, advanced=None):
    """Copy the original evaluated model into the current equivalent export."""
    _require(fold_affine is True and fuse_attention is True and advanced in (None, True),
             'Current deployment requires all equivalent conversion steps.')
    advanced = True
    construction_info = _inspect_original_model(model)
    normalization_eps = {name: float(module.eps) for name, module in model.named_modules()
                         if isinstance(module, nn.LayerNorm)}
    converted = copy.deepcopy(model)
    for block in list(converted.STEblocks) + list(converted.TTEblocks):
        op = block.op
        op.graph_conditioning = DeployGraphConditioning.from_op(op, fold_affine=True)
        del op.gcn, op.proj_gcn, op.proj_adj
        op.graph_equivalent = True
        op.graph_pool_stride = None
        block.norm2 = _fold_norm_into_linear(block.norm2, block.mlp.fc1)
    for head in (converted.head_1, converted.head_2):
        head[0] = _fold_norm_into_linear(head[0], head[1])
    converted.cross_attn = DeployCrossAttention.from_attention(converted.cross_attn)
    from .posemamba_advanced import apply_advanced_equivalence
    converted = apply_advanced_equivalence(converted, model)
    return PoseMambaDeploy(converted, construction_info, normalization_eps, True, True, True)


def _metadata_copy(metadata):
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise TypeError('Deployment metadata must be a dictionary of JSON-compatible values.')
    try:
        return json.loads(json.dumps(metadata, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError('Deployment metadata must contain only finite JSON-compatible values.') from error


def save_deploy_model(model, path, metadata=None):
    """Save tensors and versioned construction data, never a pickled model object."""
    if not isinstance(model, PoseMambaDeploy):
        raise TypeError('save_deploy_model expects the result of convert_to_deploy.')
    if any(module.training for module in model.modules()):
        raise ValueError('The deployment model must remain in evaluation mode.')
    from .posemamba_online import OnlineFactorizedGraph

    online_factorized = getattr(model, 'online_factorized', False)
    if online_factorized is not True:
        raise ValueError('Current deployment requires online_factorized=True.')
    online_graphs = [module for module in model.modules() if isinstance(module, OnlineFactorizedGraph)]
    if model.architecture != 'original_s' or not model.fold_affine or (not model.fuse_attention) or (not model.advanced) or (len(online_graphs) != 20):
        raise ValueError('Online factorization requires original S with all advanced flags and all 20 online graphs.')
    blocks = list(model.model.STEblocks) + list(model.model.TTEblocks)
    if any((not isinstance(block.op.graph_conditioning, OnlineFactorizedGraph) for block in blocks)):
        raise ValueError('An online checkpoint cannot contain a mixture of graph implementations.')
    configuration = {
        'architecture': model.architecture,
        'construction_info': model.construction_info,
        'normalization_eps': copy.deepcopy(model._normalization_eps),
        'fold_affine': model.fold_affine,
        'fuse_attention': model.fuse_attention,
        'advanced': model.advanced,
        'online_factorized': online_factorized,
        'metadata': _metadata_copy(model.export_metadata if metadata is None else metadata),
    }
    payload = {
        'format': _FORMAT,
        'version': _FORMAT_VERSION,
        # Older PyTorch weights-only loaders cannot unpickle Python floats.
        # Keep non-tensor configuration in JSON so safe loading works there too.
        'configuration_json': json.dumps(configuration, allow_nan=False),
        'state_dict': OrderedDict((name, tensor.detach().cpu())
                                 for name, tensor in model.state_dict().items()),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_deploy_model(path, map_location='cpu'):
    """Strictly reconstruct the current original-model deployment format."""
    from ..PoseMamba import PoseMamba
    from .posemamba_online import apply_online_factorization
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if (not isinstance(payload, dict) or payload.get('format') != _FORMAT
            or payload.get('version') != _FORMAT_VERSION):
        raise ValueError('Unsupported deployment checkpoint format/version.')
    try:
        configuration = json.loads(payload['configuration_json'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing valid JSON construction data.') from error
    if not isinstance(configuration, dict) or configuration.get('architecture') != 'original_s':
        raise ValueError('Unsupported deployment architecture.')
    info = configuration.get('construction_info')
    expected = dict(num_joints=17, in_chans=2, embed_dim_ratio=64, depth=10,
                    mlp_ratio=2.0, ffn_hidden_dim=128)
    if (not isinstance(info, dict) or set(info) != set(expected) | {'num_frame'}
            or any(info[key] != value for key, value in expected.items())
            or type(info['num_frame']) is not int or info['num_frame'] < 1):
        raise ValueError('Unsupported original-model construction information.')
    if any(configuration.get(flag) is not True for flag in
           ('fold_affine', 'fuse_attention', 'advanced', 'online_factorized')):
        raise ValueError('Current deployment requires all equivalent conversion steps.')
    state = payload.get('state_dict')
    if not isinstance(state, dict) or 'model.Spatial_patch_to_embedding.weight' not in state:
        raise ValueError('Deployment checkpoint is missing its model state_dict.')
    anchor = state['model.Spatial_patch_to_embedding.weight']
    if not isinstance(anchor, torch.Tensor) or not anchor.is_floating_point():
        raise ValueError('Deployment checkpoint has an invalid embedding tensor.')
    if any(not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point()
           or tensor.dtype != anchor.dtype or tensor.device != anchor.device
           for tensor in state.values()):
        raise ValueError('Deployment state tensors must share one floating dtype and device.')
    with torch.random.fork_rng(devices=[]):
        original = PoseMamba(**info).to(device=anchor.device, dtype=anchor.dtype).eval()
        norms = {name: module for name, module in original.named_modules()
                 if isinstance(module, nn.LayerNorm)}
        eps = configuration.get('normalization_eps')
        if (not isinstance(eps, dict) or set(eps) != set(norms)
                or any(not isinstance(value, (int, float)) or not 0 < value < float('inf')
                       for value in eps.values())):
            raise ValueError('Deployment checkpoint has invalid normalization settings.')
        for name, value in eps.items():
            norms[name].eps = float(value)
        deployment = convert_to_deploy(original, fold_affine=True, fuse_attention=True, advanced=True)
        deployment = apply_online_factorization(deployment, original)
    deployment.load_state_dict(state, strict=True)
    deployment.export_metadata = _metadata_copy(configuration.get('metadata', {}))
    return deployment.eval()
