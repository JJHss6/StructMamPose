"""Trainable grouped PoseMamba-S with two full-frame graph context branches.

All 20 blocks, 243-frame processing, and original channel/state dimensions are
retained. Grouping changes channel connectivity; this is a new training
architecture, not an equivalent conversion of dense trained weights.
"""

import copy
import math

import torch
from torch import nn

from .PoseMamba import PoseMamba


ARCHITECTURE_NAME = 'posemamba_grouped_s_frame_context'
GROUPED_S_CONFIG = {
    'frame_context_dim': 32,
    'context_blocks': [4, 9],
    'graph_groups': 4,
    'ffn_groups': 2,
    'context_ffn_groups': 4,
}
_PARENTS = (0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16)


class GroupedLinear(nn.Module):
    """Apply independent linear maps to contiguous channel groups.

Only the used blocks are stored: weight has shape [groups, out/groups,
in/groups]. No dense masked matrix or unused off-diagonal parameters exist.
"""

    def __init__(self, in_features, out_features, groups, bias=True,
                 device=None, dtype=None):
        super().__init__()
        if any(type(value) is not int or value < 1
               for value in (in_features, out_features, groups)):
            raise ValueError('GroupedLinear dimensions and groups must be positive integers.')
        if in_features % groups or out_features % groups:
            raise ValueError('GroupedLinear input and output dimensions must be divisible by groups.')
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(
            groups, out_features // groups, in_features // groups,
            device=device, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Linear's standard initialization using each group's fan-in.
        # Applying kaiming_uniform_ to the whole 3-D weight would use the wrong
        # fan-in, because the leading dimension here is a group dimension.
        bound = 1 / math.sqrt(self.in_features // self.groups)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    def from_linear(cls, dense, groups):
        """Copy an already block-diagonal Linear without discarding weights.

        Arbitrary dense pretrained weights cannot be converted equivalently;
        any nonzero off-diagonal entry is rejected rather than silently pruned.
        New grouped training layers should use the regular constructor.
        """
        if not isinstance(dense, nn.Linear):
            raise TypeError('from_linear expects an nn.Linear module.')
        result = cls(dense.in_features, dense.out_features, groups,
                     bias=dense.bias is not None,
                     device=dense.weight.device, dtype=dense.weight.dtype)
        in_per_group = dense.in_features // groups
        out_per_group = dense.out_features // groups
        diagonal = torch.zeros_like(dense.weight)
        with torch.no_grad():
            for group in range(groups):
                rows = slice(group * out_per_group, (group + 1) * out_per_group)
                columns = slice(group * in_per_group, (group + 1) * in_per_group)
                diagonal[rows, columns] = dense.weight[rows, columns]
                result.weight[group].copy_(dense.weight[rows, columns])
            if not torch.equal(diagonal, dense.weight):
                raise ValueError('Dense Linear has nonzero off-diagonal weights; grouping would change its function.')
            if dense.bias is not None:
                result.bias.copy_(dense.bias)
        result.weight.requires_grad_(dense.weight.requires_grad)
        if result.bias is not None:
            result.bias.requires_grad_(dense.bias.requires_grad)
        result.train(dense.training)
        return result

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f'GroupedLinear expects a final dimension of {self.in_features}.')
        grouped = x.reshape(*x.shape[:-1], self.groups, self.in_features // self.groups)
        output = torch.einsum('...gi,goi->...go', grouped, self.weight)
        output = output.reshape(*x.shape[:-1], self.out_features)
        return output if self.bias is None else output + self.bias

    def extra_repr(self):
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'groups={self.groups}, bias={self.bias is not None}')


def _directions(horizontal, vertical):
    """[B,T,J,C] -> [B,4,C,T*J], preserving the original scan order."""
    horizontal = horizontal.flatten(1, 2).transpose(1, 2)
    vertical = vertical.transpose(1, 2).flatten(1, 2).transpose(1, 2)
    return torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)


def _scan_features(features, parents):
    # Autograd includes every parent contribution, including repeated parents.
    horizontal = features + features.index_select(-2, parents)
    return _directions(horizontal, features).transpose(-2, -1)


class FrameContextGraph(nn.Module):
    """Four dynamic/static graph paths, with joint-specific full-frame context.

    The context branch reads the same affine-normalized 17x128 GCN features,
    maps 2176 -> context_dim -> 1088, and produces distinct 64-D outputs for
    each joint. Parent addition is before the existing projection LN/SiLU.
    """

    def __init__(self, op, context_dim=32):
        super().__init__()
        if type(context_dim) is not int or context_dim < 1:
            raise ValueError('context_dim must be a positive integer.')
        if (getattr(op, 'graph_equivalent', False)
                or getattr(op, 'graph_pool_stride', None) is not None
                or not hasattr(op, 'gcn') or not hasattr(op, 'proj_gcn')
                or not hasattr(op, 'proj_adj')):
            raise ValueError('FrameContextGraph requires the original framewise S graph path.')
        graph, projection = op.gcn.gcn, op.proj_gcn
        if (len(graph) != 4 or len(projection) != 4
                or graph[0].in_features != 128 or graph[0].out_features != 128
                or not isinstance(graph[1], nn.SiLU)
                or not isinstance(graph[2], nn.LayerNorm)
                or tuple(graph[2].normalized_shape) != (128,)
                or not isinstance(graph[3], nn.Linear)
                or (graph[3].in_features, graph[3].out_features) != (128, 128)
                or not isinstance(projection[0], nn.Linear)
                or (projection[0].in_features, projection[0].out_features) != (128, 64)
                or not isinstance(projection[1], nn.LayerNorm)
                or tuple(projection[1].normalized_shape) != (64,)
                or not isinstance(projection[2], nn.SiLU)
                or not isinstance(projection[3], nn.Linear)
                or (projection[3].in_features, projection[3].out_features) != (64, 16)
                or (op.proj_adj.in_features, op.proj_adj.out_features) != (17, 128)
                or tuple(op.gcn.adj.shape) != (17, 17)):
            raise ValueError('FrameContextGraph requires GCN128, projection64 and state16.')
        self.first = copy.deepcopy(graph[:3])
        self.last = copy.deepcopy(graph[3])
        self.proj = copy.deepcopy(projection)
        self.proj_adj = copy.deepcopy(op.proj_adj)
        self.register_buffer('adj', op.gcn.adj.detach().clone(), persistent=False)
        self.register_buffer('parents', torch.tensor(
            _PARENTS, device=self.adj.device, dtype=torch.long), persistent=False)
        self.context = nn.Sequential(
            nn.Linear(17 * 128, context_dim), nn.GELU(), nn.Linear(context_dim, 17 * 64))
        self.context.to(op.proj_adj.weight)
        # Keep source train/eval modes; construction must not disable training.
        self.training = op.training
        self.context.train(op.training)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (17, 128):
            raise ValueError('FrameContextGraph expects [batch, frames, 17, 128].')
        batch, frames, joints, _ = x.shape
        features = self.first(self.adj @ x)
        graph = self.last(features)
        context = self.context(features.flatten(-2)).reshape(batch, frames, joints, 64)
        q = self.proj[0](_scan_features(graph, self.parents)) + _scan_features(context, self.parents)
        dynamic = self.proj[3](self.proj[2](self.proj[1](q))).transpose(-2, -1)
        # Keep the original four static training paths, rather than applying an
        # inference cache during training. All topology projection weights train.
        topology = self.proj_adj(self.adj.expand(batch, frames, joints, joints))
        static = self.proj(_scan_features(topology, self.parents)).transpose(-2, -1)
        return dynamic + static


def _new_grouped(linear, groups):
    result = GroupedLinear(linear.in_features, linear.out_features, groups,
                           bias=linear.bias is not None,
                           device=linear.weight.device, dtype=linear.weight.dtype)
    result.train(linear.training)
    return result


def build_grouped_s(frame_context_dim=32, context_blocks=(4, 9), graph_groups=4,
                    ffn_groups=2, context_ffn_groups=4, **kwargs):
    """Construct the fixed grouped S architecture for training from scratch.

    ``num_frame`` may be a positive smaller value for numerical tests. The
    channel widths, depth, grouping and context locations are fixed so that a
    configuration change cannot silently change the agreed architecture.
    """
    if (any(type(value) is not int for value in
            (frame_context_dim, graph_groups, ffn_groups, context_ffn_groups))
            or not isinstance(context_blocks, (tuple, list))
            or any(type(index) is not int for index in context_blocks)):
        raise ValueError('Grouped S group/context dimensions and block indices must be integers.')
    configuration = {
        'frame_context_dim': frame_context_dim,
        'context_blocks': list(context_blocks),
        'graph_groups': graph_groups,
        'ffn_groups': ffn_groups,
        'context_ffn_groups': context_ffn_groups,
    }
    if configuration != GROUPED_S_CONFIG:
        raise ValueError(f'The grouped S architecture requires {GROUPED_S_CONFIG}.')
    options = dict(kwargs)
    defaults = dict(num_frame=243, num_joints=17, in_chans=2,
                    embed_dim_ratio=64, depth=10, mlp_ratio=2.0,
                    lite_graph=False, graph_equivalent=False,
                    split_graph_by_joint_group=False, graph_pool_stride=1)
    for name, value in defaults.items():
        options.setdefault(name, value)
    if type(options['num_frame']) is not int or options['num_frame'] < 1:
        raise ValueError('Grouped S num_frame must be a positive integer.')
    fixed = dict(num_joints=17, in_chans=2, embed_dim_ratio=64, depth=10, mlp_ratio=2.0,
                 lite_graph=False, graph_equivalent=False, split_graph_by_joint_group=False)
    if any(options[name] != value for name, value in fixed.items()):
        raise ValueError('Grouped S requires 17 joints, input2, width64, depth10, mlp_ratio2 and no pooled graph path.')
    for name, width in (('graph_hidden_dim', 128), ('first_graph_hidden_dim', 128),
                        ('graph_output_dim', 128), ('graph_projection_dim', 64),
                        ('ffn_hidden_dim', 128)):
        if options.get(name) not in (None, width):
            raise ValueError(f'Grouped S requires {name}={width} (or None for the original default).')

    model = PoseMamba(**options)
    for module in model.modules():
        if (isinstance(module, nn.LayerNorm)
                and (not module.elementwise_affine or module.weight is None or module.bias is None)):
            raise ValueError('Grouped S training requires the original affine LayerNorm layers.')
    trunk_norms = [model.Spatial_norm, model.Temporal_norm]
    for block in list(model.STEblocks) + list(model.TTEblocks):
        trunk_norms.extend((block.norm, block.norm2))
    if any(not isinstance(module, nn.LayerNorm)
           or tuple(module.normalized_shape) != (64,) for module in trunk_norms):
        raise ValueError('Grouped S requires LayerNorm64 at the original trunk locations.')
    for stream in ('STEblocks', 'TTEblocks'):
        for index, block in enumerate(getattr(model, stream)):
            if block.op.graph_pool_stride is not None:
                raise ValueError('Grouped S must process all frames without temporal pooling.')
            selected = stream == 'STEblocks' and index in configuration['context_blocks']
            if index > 0:
                block.op.gcn.gcn[0] = _new_grouped(block.op.gcn.gcn[0], graph_groups)
                groups = context_ffn_groups if selected else ffn_groups
                block.mlp.fc1 = _new_grouped(block.mlp.fc1, groups)
                block.mlp.fc2 = _new_grouped(block.mlp.fc2, groups)
            if selected:
                op = block.op
                op.graph_conditioning = FrameContextGraph(op, context_dim=frame_context_dim)
                op.graph_equivalent = True
                del op.gcn, op.proj_gcn, op.proj_adj
    model.grouped_s_config = copy.deepcopy(configuration)
    model.architecture_name = ARCHITECTURE_NAME
    return model
