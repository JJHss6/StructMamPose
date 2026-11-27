"""Equivalent inference coordinates for dense original-width PoseMamba Base.

These transforms operate on an already exported copy. They do not change the
training model, SSM state size, or external channel dimensions. Zero-mean input
coordinates and output gauges require the surrounding LayerNorm operations;
the transformed Linear modules are not interchangeable with arbitrary Linear
layers outside those contexts.
"""
import torch
from torch import nn

def _linear_from_tensors(weight, bias=None):
    """Construct frozen inference weights without advancing the caller's RNG."""
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        linear = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None).to(weight)
        linear.weight.copy_(weight)
        if bias is not None:
            linear.bias.copy_(bias)
    linear.eval()
    linear.requires_grad_(False)
    return linear

class ZeroMeanLinear(nn.Module):
    """An affine map on sum-zero inputs, evaluated using its first N-1 entries.

    If sum(x)=0, W x = (W[:,:-1] - W[:,-1:]) x[:-1]. The logical input
    dimension stays N. Callers must supply a non-affine LayerNorm output.
    """

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features + 1
        self.out_features = linear.out_features
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_linear(cls, source):
        if not isinstance(source, nn.Linear) or source.in_features < 2:
            raise ValueError('ZeroMeanLinear requires a dense Linear with at least two inputs.')
        weight = source.weight.detach()
        bias = None if source.bias is None else source.bias.detach()
        return cls(_linear_from_tensors(weight[:, :-1] - weight[:, -1:], bias))

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f'ZeroMeanLinear expects {self.in_features} logical input channels.')
        return self.linear(x[..., :-1])

class GaugeLinear(nn.Module):
    """Return W x + b minus its final output channel, with that channel zero.

    This removes one common-shift degree of freedom only where downstream
    LayerNorm (possibly after residual addition) removes the same shift.
    """

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features + 1
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_linear(cls, source):
        if not isinstance(source, nn.Linear) or source.out_features < 2:
            raise ValueError('GaugeLinear requires a dense Linear with at least two outputs.')
        weight = source.weight.detach()
        bias = None if source.bias is None else source.bias.detach()[:-1] - source.bias.detach()[-1]
        return cls(_linear_from_tensors(weight[:-1] - weight[-1:], bias))

    def forward(self, x):
        if x.shape[-1] != self.in_features:
            raise ValueError(f'GaugeLinear expects {self.in_features} input channels.')
        output = self.linear(x)
        return torch.cat((output, torch.zeros_like(output[..., :1])), dim=-1)

def _nonaffine_norm(norm):
    return isinstance(norm, nn.LayerNorm) and tuple(norm.normalized_shape) == (128,) and (not norm.elementwise_affine)

def apply_block_coordinates(converted):
    """Rewrite an evaluated v3 inference copy in place; return that same object.

    Accept a PoseMambaDeploy wrapper or its internal PoseMamba. The previous
    export must already have folded FFN/head normalization affine parameters.
    A training model fails validation before any module is replaced.
    """
    backbone = getattr(converted, 'model', converted)
    if not hasattr(backbone, 'STEblocks') or not hasattr(backbone, 'TTEblocks') or len(backbone.STEblocks) != 10 or (len(backbone.TTEblocks) != 10) or backbone.training:
        raise ValueError('Block coordinates require an evaluated 20-block Base inference copy.')
    blocks = list(backbone.STEblocks) + list(backbone.TTEblocks)
    heads = (backbone.head_1, backbone.head_2)
    for block in blocks:
        if not _nonaffine_norm(block.norm2) or not isinstance(block.norm, nn.LayerNorm) or tuple(block.norm.normalized_shape) != (128,) or (not block.norm.elementwise_affine) or (block.norm.bias is None) or (not getattr(block.op, 'graph_equivalent', False)):
            raise ValueError('Expected the original-Base affine SSM norm and already folded FFN norm.')
        for (linear, shape) in ((block.op.in_proj, (128, 512)), (block.op.out_proj, (256, 128)), (block.mlp.fc1, (128, 256)), (block.mlp.fc2, (256, 128))):
            if not isinstance(linear, nn.Linear) or (linear.in_features, linear.out_features) != shape:
                raise ValueError('Coordinate conversion requires unchanged dense Base Linear shapes.')
    for head in heads:
        if not _nonaffine_norm(head[0]) or not isinstance(head[1], nn.Linear) or (head[1].in_features, head[1].out_features) != (128, 3):
            raise ValueError('Coordinate conversion requires already folded original Base heads.')
    for block in blocks:
        (norm, projection) = (block.norm, block.op.in_proj)
        with torch.no_grad():
            weight = projection.weight * norm.weight.unsqueeze(0)
            bias = projection.weight @ norm.bias
            if projection.bias is not None:
                bias = bias + projection.bias
            reduced = _linear_from_tensors(weight[:, :-1] - weight[:, -1:], bias)
        block.op.in_proj = ZeroMeanLinear(reduced)
        block.norm = nn.LayerNorm(norm.normalized_shape, eps=norm.eps, elementwise_affine=False).eval()
        block.op.out_proj = GaugeLinear.from_linear(block.op.out_proj)
        block.mlp.fc1 = ZeroMeanLinear.from_linear(block.mlp.fc1)
        block.mlp.fc2 = GaugeLinear.from_linear(block.mlp.fc2)
    for head in heads:
        head[1] = ZeroMeanLinear.from_linear(head[1])
    return converted

class FirstInputProjection(nn.Module):
    """Specialize the first STE normalization/projection to its raw 2-D input.

    The caller still builds and retains the original 128-D embedding/residual.
    ``embedded`` must be E(raw)+embedding_bias+spatial_position, before the
    first block and with evaluation-mode positional dropout. Its original
    128-channel variance is retained rather than approximated by a quadratic.
    """

    def __init__(self, linear, joint_numerator, offset, eps):
        super().__init__()
        self.linear = linear
        self.register_buffer('joint_numerator', joint_numerator.detach().clone())
        self.offset = nn.Parameter(offset.detach().clone(), requires_grad=False)
        self.eps = float(eps)
        self.num_joints = 17
        self.embedding_dim = 128
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_source(cls, original):
        """Read the untouched evaluated training model; never mutate its state."""
        if original.training:
            raise ValueError('FirstInputProjection requires the original model in evaluation mode.')
        embedding = original.Spatial_patch_to_embedding
        positions = original.Spatial_pos_embed
        block = original.STEblocks[0]
        (norm, projection) = (block.norm, block.op.in_proj)
        if not isinstance(embedding, nn.Linear) or (embedding.in_features, embedding.out_features) != (2, 128) or embedding.bias is None or (tuple(positions.shape) != (1, 17, 128)) or (not isinstance(norm, nn.LayerNorm)) or (tuple(norm.normalized_shape) != (128,)) or (not norm.elementwise_affine) or (norm.bias is None) or (not isinstance(projection, nn.Linear)) or ((projection.in_features, projection.out_features) != (128, 512)) or original.pos_drop.training:
            raise ValueError('FirstInputProjection requires the untouched original Base embedding/LN/in_proj.')
        with torch.no_grad():
            centered_embedding = embedding.weight - embedding.weight.mean(dim=0, keepdim=True)
            joint_offset = embedding.bias.unsqueeze(0) + positions[0]
            centered_offset = joint_offset - joint_offset.mean(dim=-1, keepdim=True)
            weighted_projection = projection.weight * norm.weight.unsqueeze(0)
            linear = _linear_from_tensors(weighted_projection @ centered_embedding)
            joint_numerator = centered_offset @ weighted_projection.T
            offset = projection.weight @ norm.bias
            if projection.bias is not None:
                offset = offset + projection.bias
        return cls(linear, joint_numerator, offset, norm.eps)

    def forward(self, raw, embedded):
        if raw.ndim != 4 or raw.shape[-2:] != (17, 2) or embedded.ndim != 4 or (embedded.shape[:-1] != raw.shape[:-1]) or (embedded.shape[-1] != 128):
            raise ValueError('Expected matching [B,T,17,2] raw and [B,T,17,128] embedded inputs.')
        variance = embedded.var(dim=-1, unbiased=False, keepdim=True)
        inverse_std = torch.rsqrt(variance + self.eps)
        return (self.linear(raw) + self.joint_numerator) * inverse_std + self.offset
