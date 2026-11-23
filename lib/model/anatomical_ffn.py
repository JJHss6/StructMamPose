"""Anatomical parameter sharing using eleven unmodified dense MLP modules.

Every joint retains its own feature vector and output. Anatomical pairs share
weights, not pooled features; each MLP remains Linear -> GELU -> Linear with
the original two dropout positions and complete channel connectivity.
"""

import copy

import torch
from torch import nn

from .mambablocks_my import Mlp


ANATOMICAL_JOINT_GROUPS = (
    (0,), (1, 4), (2, 5), (3, 6), (7,), (8,), (9,), (10,),
    (11, 14), (12, 15), (13, 16),
)


class AnatomicalFFN(nn.Module):
    """A dense C->H->Cout FFN with eleven independent anatomical weight sets."""

    def __init__(self, in_features, hidden_features, out_features=None, drop=0.0,
                 act_layer=nn.GELU, device=None, dtype=None):
        super().__init__()
        out_features = in_features if out_features is None else out_features
        self._set_metadata(in_features, hidden_features, out_features, device)
        self.ffns = nn.ModuleList([
            Mlp(in_features, hidden_features, out_features, act_layer=act_layer, drop=drop)
            .to(device=device, dtype=dtype)
            for _ in self.joint_groups
        ])

    def _set_metadata(self, in_features, hidden_features, out_features, device):
        if any(type(width) is not int or width < 1
               for width in (in_features, hidden_features, out_features)):
            raise ValueError('AnatomicalFFN dimensions must be positive integers.')
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_joints = 17
        self.num_groups = len(ANATOMICAL_JOINT_GROUPS)
        self.joint_groups = ANATOMICAL_JOINT_GROUPS
        self.group_sizes = tuple(len(joints) for joints in self.joint_groups)
        mapping = [None] * self.num_joints
        order = []
        for group_index, joints in enumerate(self.joint_groups):
            order.extend(joints)
            for joint in joints:
                mapping[joint] = group_index
        joint_order = torch.tensor(order, dtype=torch.long, device=device)
        self.register_buffer('joint_to_group', torch.tensor(mapping, dtype=torch.long, device=device),
                             persistent=False)
        self.register_buffer('joint_order', joint_order, persistent=False)
        self.register_buffer('restore_joint_order', joint_order.argsort(), persistent=False)

    @classmethod
    def from_mlp(cls, template):
        """Clone the original shared MLP without changing RNG or source tensors.

        Initial functions agree in eval mode or with zero dropout. Each clone
        subsequently learns independently, while its bilateral joints continue
        sharing that clone's parameters.
        """
        first, last = getattr(template, 'fc1', None), getattr(template, 'fc2', None)
        if (not isinstance(template, Mlp)
                or not isinstance(first, nn.Linear) or not isinstance(last, nn.Linear)
                or first.bias is None or last.bias is None
                or first.out_features != last.in_features
                or not isinstance(template.act, nn.GELU)
                or not isinstance(template.drop, nn.Dropout)):
            raise ValueError('AnatomicalFFN requires the original dense Linear/GELU/Dropout/Linear Mlp.')
        tensors = (first.weight, first.bias, last.weight, last.bias)
        if len({(tensor.device, tensor.dtype) for tensor in tensors}) != 1:
            raise ValueError('The source MLP parameters must share one dtype and device.')
        # Bypass random initialization: every registered child is a deep copy of
        # the existing Mlp, including its modes, activation and gradient flags.
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        result._set_metadata(first.in_features, first.out_features, last.out_features,
                             first.weight.device)
        result.ffns = nn.ModuleList([copy.deepcopy(template) for _ in result.joint_groups])
        result.ffns.training = template.training
        result.training = template.training
        return result

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2:] != (self.num_joints, self.in_features):
            raise ValueError(f'AnatomicalFFN expects [B,T,17,{self.in_features}].')
        ordered = x.index_select(2, self.joint_order)
        joint_features = ordered.split(self.group_sizes, dim=2)
        # nn.Linear acts only on the last channel dimension: even a two-joint
        # group produces two independent feature vectors, never an average.
        outputs = [ffn(features) for ffn, features in zip(self.ffns, joint_features)]
        return torch.cat(outputs, dim=2).index_select(2, self.restore_joint_order)

    def profile_macs(self, inputs):
        """Audit formula; default THOP can count the 22 child Linear calls directly."""
        x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
        if x.ndim != 4 or x.shape[-2:] != (self.num_joints, self.in_features):
            raise ValueError('Invalid AnatomicalFFN profiling input shape.')
        return x.shape[0] * x.shape[1] * self.num_joints * (
            self.in_features * self.hidden_features + self.hidden_features * self.out_features)

    def extra_repr(self):
        return (f'in_features={self.in_features}, hidden_features={self.hidden_features}, '
                f'out_features={self.out_features}, anatomical_groups={self.num_groups}')
