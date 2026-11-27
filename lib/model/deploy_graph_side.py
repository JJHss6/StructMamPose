"""Inference coordinate reductions for PoseMamba's graph and side-view path.

These transformations retain the original normalization dimensions and epsilon.
Only redundant linear coordinates are removed; all nonlinearities remain. The
side-view specialization relies on PoseMamba's root-relative input: joint zero
is identically zero before ``proj_side``.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F

def _require(condition, message):
    if not condition:
        raise ValueError('Advanced PoseMamba deployment: ' + message)

def _require_eval(module):
    _require(not any((child.training for child in module.modules())), 'the source module must be in evaluation mode.')

def _linear_from(weight, bias):
    """Construct an independent Linear without advancing the caller's RNG."""
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        result = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None, device='cpu', dtype=weight.dtype).to(weight.device)
        result.weight.copy_(weight)
        if bias is not None:
            result.bias.copy_(bias)
    return result

def _input_tensor(inputs):
    return inputs[0] if isinstance(inputs, (tuple, list)) else inputs

def _pad_reference_coordinate(tensor):
    return F.pad(tensor, (0, 1))

class AdvancedGraphConditioning(nn.Module):
    """The same two-layer graph in (hidden-1)-input / 63-output coordinates."""

    def __init__(self, first, merged, tail, horizontal_correction, static_horizontal, static_vertical, adj, parents):
        super().__init__()
        (self.first, self.merged, self.tail) = (first, merged, tail)
        self.horizontal_correction = nn.Parameter(horizontal_correction, requires_grad=False)
        self.register_buffer('static_horizontal', static_horizontal)
        self.register_buffer('static_vertical', static_vertical)
        self.register_buffer('adj', adj, persistent=False)
        self.register_buffer('parents', parents, persistent=False)
        self.fold_affine = True
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_graph(cls, graph):
        from .posemamba_deploy import DeployGraphConditioning
        _require(isinstance(graph, DeployGraphConditioning), 'expected an existing equivalent DeployGraphConditioning.')
        _require_eval(graph)
        hidden = graph.first[0].out_features
        _require(graph.fold_affine and (not graph.first[2].elementwise_affine) and (tuple(graph.first[2].normalized_shape) == (hidden,)), 'graph input normalization must already have its affine transform folded.')
        _require(isinstance(graph.first[0], nn.Linear) and graph.first[0].in_features == 256 and (hidden > 0), 'a dense 256-to-hidden graph input is required.')
        _require(tuple(graph.merged.weight.shape) == (128, hidden) and graph.merged.bias is not None and (tuple(graph.horizontal_correction.shape) == (128,)), 'expected a hidden-to-128 composed graph projection.')
        _require(tuple(graph.tail[0].normalized_shape) == (128,) and graph.tail[0].elementwise_affine and isinstance(graph.tail[1], nn.SiLU) and (tuple(graph.tail[2].weight.shape) == (16, 128)) and (graph.tail[2].bias is not None), 'the original LN128/SiLU/Linear128-to-16 graph tail is required.')
        with torch.no_grad():
            rows = graph.merged.weight[:127] - graph.merged.weight[127:]
            weight = rows[:, :-1] - rows[:, -1:]
            bias = graph.merged.bias[:127] - graph.merged.bias[127]
            correction = graph.horizontal_correction[:127] - graph.horizontal_correction[127]
            merged = _linear_from(weight, bias)
            tail = copy.deepcopy(graph.tail)
            final_bias = graph.tail[2].bias
            tail[2] = _linear_from(graph.tail[2].weight, None)
            horizontal = graph.static_horizontal + final_bias
            vertical = graph.static_vertical + final_bias
        return cls(copy.deepcopy(graph.first), merged, tail, correction.detach().clone(), horizontal.detach().clone(), vertical.detach().clone(), graph.adj.detach().clone(), graph.parents.detach().clone())

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (17, 256):
            raise ValueError('Advanced graph expects [batch, frames, 17, 256].')
        features = self.first(self.adj @ x)
        q = self.merged(features[..., :-1])
        horizontal_q = q + q.index_select(-2, self.parents) - self.horizontal_correction
        horizontal = self.tail(_pad_reference_coordinate(horizontal_q)) + self.static_horizontal
        vertical = self.tail(_pad_reference_coordinate(q)) + self.static_vertical
        horizontal = horizontal.flatten(1, 2).transpose(1, 2)
        vertical = vertical.transpose(1, 2).flatten(1, 2).transpose(1, 2)
        return torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)

    def profile_macs(self, inputs):
        """Whole-leaf MACs under the project's existing THOP accounting scope."""
        x = _input_tensor(inputs)
        tokens = x.shape[0] * x.shape[1] * 17
        (first, merged, tail) = (self.first[0], self.merged, self.tail)
        return tokens * (first.in_features * first.out_features + 2 * first.out_features + merged.in_features * merged.out_features + 2 * tail[2].in_features * (4 + tail[2].out_features))

class AdvancedSideProjection(nn.Module):
    """Root-relative side MLP; return the un-affined LN128 output for attention."""

    def __init__(self, first, activation, last, norm):
        super().__init__()
        (self.first, self.activation, self.last, self.norm) = (first, activation, last, norm)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (17, 2):
            raise ValueError('Advanced side projection expects root-relative [batch, frames, 17, 2].')
        nonroot = self.last(self.activation(self.first(x[:, :, 1:])))
        nonroot = self.norm(_pad_reference_coordinate(nonroot))
        return torch.cat((nonroot.new_zeros(*nonroot.shape[:2], 1, 128), nonroot), dim=2)

    def profile_macs(self, inputs):
        x = _input_tensor(inputs)
        tokens = x.shape[0] * x.shape[1] * 16
        return tokens * (2 * 128 + 128 * 127 + 2 * 128)

class AdvancedCrossAttention(nn.Module):
    """Original single-head attention using side-LN coordinates and a root cache."""

    def __init__(self, query, value, norm, root_output):
        super().__init__()
        (self.query, self.value, self.norm) = (query, value, norm)
        self.register_buffer('root_output', root_output)
        self.requires_grad_(False)
        self.eval()

    def forward(self, x, y, attn_mask=None):
        if x.shape != y.shape or x.ndim != 4 or x.shape[-2:] != (17, 128):
            raise ValueError('Advanced attention expects matching [batch, frames, 17, 128] inputs.')
        query = self.query(x[:, :, 1:]).flatten(2).unsqueeze(1)
        side_coordinates = y[:, :, 1:, :127]
        key = side_coordinates.flatten(2).unsqueeze(1)
        value = self.value(side_coordinates).flatten(2).unsqueeze(1)
        scores = query @ key.transpose(-2, -1) / (17 * 128) ** 0.5
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        nonroot = (weights @ value).squeeze(1).reshape(x.shape[0], x.shape[1], 16, 127)
        nonroot = self.norm(_pad_reference_coordinate(nonroot))
        nan_guard = weights[:, 0, :, 0].unsqueeze(-1).unsqueeze(-1) * 0
        root = self.root_output.view(1, 1, 1, 128) + nan_guard
        return torch.cat((root, nonroot), dim=2)

    def profile_macs(self, inputs):
        x = _input_tensor(inputs)
        tokens = x.shape[0] * x.shape[1] * 16
        return tokens * (128 * 127 + 127 * 127 + 4 * 128)

def build_advanced_side(proj_side, cross_attn):
    """Return paired equivalent side/attention modules for the original Base path.

    ``cross_attn`` must already use the v3 equivalent Q/K and V/output fusion.
    The returned pair must be used together: the side output has its original
    LN affine transform absorbed into the returned attention weights.
    """
    from .posemamba_deploy import DeployCrossAttention
    _require_eval(proj_side)
    _require_eval(cross_attn)
    _require(isinstance(proj_side, nn.Sequential) and len(proj_side) == 4 and isinstance(proj_side[0], nn.Linear) and (tuple(proj_side[0].weight.shape) == (128, 2)) and isinstance(proj_side[1], nn.SiLU) and isinstance(proj_side[2], nn.Linear) and (tuple(proj_side[2].weight.shape) == (128, 128)) and (proj_side[2].bias is not None) and isinstance(proj_side[3], nn.LayerNorm) and (tuple(proj_side[3].normalized_shape) == (128,)) and proj_side[3].elementwise_affine, 'the original 2-to-128-to-128 side projection is required.')
    _require(isinstance(cross_attn, DeployCrossAttention), 'cross attention must already have its equivalent Q/K and V/output fusion.')
    _require(tuple(cross_attn.query.weight.shape) == (128, 128) and tuple(cross_attn.value.weight.shape) == (128, 128) and (cross_attn.query.bias is not None) and (cross_attn.value.bias is not None), 'expected the original fused 128-channel attention projections.')
    with torch.no_grad():
        side_norm = proj_side[3]
        side_weight = proj_side[2].weight[:127] - proj_side[2].weight[127:]
        side_bias = proj_side[2].bias[:127] - proj_side[2].bias[127]
        last = _linear_from(side_weight, side_bias)
        norm = nn.LayerNorm(128, eps=side_norm.eps, elementwise_affine=False)
        new_side = AdvancedSideProjection(copy.deepcopy(proj_side[0]), copy.deepcopy(proj_side[1]), last, norm)
        query_weight = cross_attn.query.weight * side_norm.weight.unsqueeze(1)
        query_bias = cross_attn.query.bias * side_norm.weight
        query = _linear_from(query_weight[:127] - query_weight[127:], query_bias[:127] - query_bias[127])
        value_weight = cross_attn.value.weight * side_norm.weight.unsqueeze(0)
        value_bias = cross_attn.value.weight @ side_norm.bias + cross_attn.value.bias
        rows = value_weight[:127] - value_weight[127:]
        value = _linear_from(rows[:, :127] - rows[:, 127:], value_bias[:127] - value_bias[127])
        zero = proj_side[0].weight.new_zeros(1, 1, 1, 2)
        root_output = cross_attn.norm(cross_attn.value(proj_side(zero))).reshape(128)
        new_attention = AdvancedCrossAttention(query, value, copy.deepcopy(cross_attn.norm), root_output.detach().clone())
    return (new_side, new_attention)
