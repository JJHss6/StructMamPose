"""Original-Base inference with active, online-composed graph projection weights.

The original training model is never changed. This export keeps the original
GCN weights and normalization affine parameters, forms the effective graph map
once per forward, and applies it to every frame/joint. Effective weights are
local tensors, never duplicate registered parameters. All normalization widths,
nonlinearities, scan directions and SSM dimensions retain their source values.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .deploy_graph_side import AdvancedGraphConditioning

def _require(condition, message):
    if not condition:
        raise ValueError('Online PoseMamba Base export: ' + message)

def _frozen_parameter(tensor):
    return nn.Parameter(tensor.detach().clone(), requires_grad=False)

class OnlineAffinePreparation(nn.Module):
    """Original second GCN Linear and LN affine, all active each forward."""

    def __init__(self, source_linear, source_norm):
        super().__init__()
        from .posemamba_deploy import _linear, _norm
        _linear(source_linear, 256, 256, 'online GCN second layer')
        _norm(source_norm, 256, 'online GCN normalization')
        self.weight = _frozen_parameter(source_linear.weight)
        self.bias = _frozen_parameter(source_linear.bias)
        self.gamma = _frozen_parameter(source_norm.weight)
        self.beta = _frozen_parameter(source_norm.bias)
        self.eval()

    def forward(self):
        weight = self.weight * self.gamma.unsqueeze(0)
        bias = self.weight @ self.beta + self.bias
        return (weight[:, :-1] - weight[:, -1:], bias)

    def profile_macs(self):
        return self.weight.numel() + self.weight.shape[0] * self.beta.numel()

class OnlineWeightComposer(nn.Module):
    """Keep the source 256-to-128 projection; form its effective map online.

    Output differences remove a common shift that the following LN128 cancels.
    The original 128 rows and biases are stored; all rows participate in each
    forward. The returned correction reuses this bias, not another Parameter.
    """

    def __init__(self, source_projection):
        super().__init__()
        from .posemamba_deploy import _linear
        _linear(source_projection, 256, 128, 'online graph projection')
        self.weight = _frozen_parameter(source_projection.weight)
        self.bias = _frozen_parameter(source_projection.bias)
        self.eval()

    def forward(self, folded_weight, folded_bias):
        weight = self.weight[:-1] - self.weight[-1:]
        correction = self.bias[:-1] - self.bias[-1]
        return (weight @ folded_weight, weight @ folded_bias + correction, correction)

    def profile_macs(self, inputs):
        (folded_weight, folded_bias) = inputs
        (rows, inner) = (self.weight.shape[0] - 1, self.weight.shape[1])
        _require(tuple(folded_weight.shape) == (inner, inner - 1) and tuple(folded_bias.shape) == (inner,), 'the online composer received an unsupported coefficient shape.')
        return rows * inner * (folded_weight.shape[1] + 1)

class OnlineFactorizedGraph(nn.Module):
    """Full-frame graph equivalent to the original dense Base graph in eval."""

    def __init__(self, source_op, advanced_graph):
        super().__init__()
        from .posemamba_deploy import _check_original_graph
        _require(not any((module.training for module in source_op.modules())), 'the source graph must be in evaluation mode.')
        _require(_check_original_graph(source_op) == 256, 'only the original GCN 256-to-256-to-256 is supported.')
        _require(type(advanced_graph) is AdvancedGraphConditioning and (not any((module.training for module in advanced_graph.modules()))), 'an evaluated advanced graph is required.')
        _require(torch.equal(advanced_graph.first[0].weight, source_op.gcn.gcn[0].weight) and torch.equal(advanced_graph.first[0].bias, source_op.gcn.gcn[0].bias) and (advanced_graph.first[2].eps == source_op.gcn.gcn[2].eps), 'the advanced graph must come from the supplied source weights.')
        self.first = copy.deepcopy(advanced_graph.first)
        self.preparation = OnlineAffinePreparation(source_op.gcn.gcn[3], source_op.gcn.gcn[2])
        self.composer = OnlineWeightComposer(source_op.proj_gcn[0])
        self.tail = copy.deepcopy(advanced_graph.tail)
        self.register_buffer('static_horizontal', advanced_graph.static_horizontal.detach().clone())
        self.register_buffer('static_vertical', advanced_graph.static_vertical.detach().clone())
        self.register_buffer('adj', advanced_graph.adj.detach().clone(), persistent=False)
        self.register_buffer('parents', advanced_graph.parents.detach().clone(), persistent=False)
        self.fold_affine = True
        self.requires_grad_(False)
        self.eval()
        self.validate_structure()

    @classmethod
    def from_op(cls, source_op, advanced_graph):
        return cls(source_op, advanced_graph)

    def validate_structure(self):
        """Reject unsupported graph variants before exporting or profiling."""
        from .posemamba_deploy import _check_topology, _linear, _norm
        _require(isinstance(self.first, nn.Sequential) and len(self.first) == 3, 'the graph input must remain Linear/SiLU/LayerNorm.')
        _linear(self.first[0], 256, 256, 'online graph first layer')
        _require(isinstance(self.first[1], nn.SiLU) and isinstance(self.first[2], nn.LayerNorm) and (self.first[2].normalized_shape == (256,)) and (not self.first[2].elementwise_affine), 'the graph input requires SiLU and non-affine LN256.')
        _require(type(self.preparation) is OnlineAffinePreparation and tuple(self.preparation.weight.shape) == (256, 256) and (tuple(self.preparation.bias.shape) == (256,)) and (tuple(self.preparation.gamma.shape) == (256,)) and (tuple(self.preparation.beta.shape) == (256,)), 'original second-GCN and normalization parameter shapes are required.')
        _require(type(self.composer) is OnlineWeightComposer and tuple(self.composer.weight.shape) == (128, 256) and (tuple(self.composer.bias.shape) == (128,)), 'the original graph projection parameter shapes are required.')
        _require(isinstance(self.tail, nn.Sequential) and len(self.tail) == 3, 'the graph tail must remain LayerNorm/SiLU/Linear.')
        _norm(self.tail[0], 128, 'online graph tail normalization')
        _require(isinstance(self.tail[1], nn.SiLU), 'the graph tail requires SiLU.')
        _linear(self.tail[2], 128, 16, 'online graph final projection', bias=False)
        _require(tuple(self.static_horizontal.shape) == (17, 16) and tuple(self.static_vertical.shape) == (17, 16), 'both fixed-topology output caches are required.')
        _check_topology(self.adj, self.parents)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (17, 256):
            raise ValueError('Online graph expects [batch, frames, 17, 256].')
        features = self.first(self.adj @ x)
        (folded_weight, folded_bias) = self.preparation()
        (weight, bias, correction) = self.composer(folded_weight, folded_bias)
        if getattr(self, 'strassen_effective_map', False):
            from .base_strassen import strassen_product
            q, self.last_strassen_details = strassen_product(features[..., :-1], weight=weight)
            q = q + bias
        else:
            q = F.linear(features[..., :-1], weight, bias)
        horizontal_q = q + q.index_select(-2, self.parents) - correction
        horizontal = self.tail(F.pad(horizontal_q, (0, 1))) + self.static_horizontal
        vertical = self.tail(F.pad(q, (0, 1))) + self.static_vertical
        horizontal = horizontal.flatten(1, 2).transpose(1, 2)
        vertical = vertical.transpose(1, 2).flatten(1, 2).transpose(1, 2)
        return torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)

    def profile_macs(self, inputs):
        """Count functional composition/application and all Linear/LN work.

        Coordinate additions/subtractions, adjacency aggregation and cache
        additions retain the project's inherited THOP omissions. Compositions
        run once per forward, independent of the number of clips in the batch.
        """
        x = inputs[0]
        tokens = x.shape[0] * x.shape[1] * x.shape[2]
        (first, tail) = (self.first[0], self.tail)
        hidden = first.out_features
        projected = self.composer.weight.shape[0] - 1
        inner = self.composer.weight.shape[1]
        token_ops = first.in_features * hidden + 2 * hidden + (hidden - 1) * projected + 2 * tail[2].in_features * (4 + tail[2].out_features)
        composition = projected * inner * hidden
        total = tokens * token_ops + self.preparation.profile_macs() + composition
        if getattr(self, 'strassen_effective_map', False):
            total += self.last_strassen_details['gemm_macs'] - tokens * (hidden - 1) * projected
        return total

def apply_online_factorization(converted, source):
    """Replace graphs on an independent advanced wrapper and return it.

    ``converted`` must have been produced from ``source``. All validation and
    new graph construction complete before modifying the inference wrapper.
    Source tensors, gradient flags and module modes are never changed.
    """
    from .posemamba_deploy import PoseMambaDeploy, _inspect_original_model
    info = _inspect_original_model(source)
    _require(info.get('architecture', 'original_b') == 'original_b', 'only original dense Base with GCN256/FFN256 is supported.')
    _require(isinstance(converted, PoseMambaDeploy) and converted.architecture == 'original_b' and (converted.construction_info == info) and converted.fold_affine and converted.fuse_attention and converted.advanced and (not getattr(converted, 'online_factorized', False)) and (not any((module.training for module in converted.modules()))), 'an unmodified original-Base advanced inference wrapper is required.')
    source_pointers = {parameter.data_ptr() for parameter in source.parameters()}
    _require(not source_pointers.intersection((parameter.data_ptr() for parameter in converted.parameters())), 'the inference wrapper must not share parameters with its training source.')
    source_blocks = list(source.STEblocks) + list(source.TTEblocks)
    inference_blocks = list(converted.model.STEblocks) + list(converted.model.TTEblocks)
    _require(len(source_blocks) == len(inference_blocks) == 20, 'the original ten spatial and ten temporal blocks are required.')
    replacements = [(block.op, OnlineFactorizedGraph.from_op(original.op, block.op.graph_conditioning)) for (original, block) in zip(source_blocks, inference_blocks)]
    for (op, graph) in replacements:
        op.graph_conditioning = graph
    converted.online_factorized = True
    return converted

def convert_to_online_deploy(source):
    """Copy an evaluated original Base into the frozen online-composed variant."""
    from .posemamba_deploy import _inspect_original_model, convert_to_deploy
    info = _inspect_original_model(source)
    _require(info.get('architecture', 'original_b') == 'original_b', 'only original dense Base with GCN256/FFN256 is supported.')
    devices = sorted({parameter.device.index for parameter in source.parameters() if parameter.is_cuda})
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        converted = convert_to_deploy(source, fold_affine=True, fuse_attention=True, advanced=True)
        return apply_online_factorization(converted, source)
