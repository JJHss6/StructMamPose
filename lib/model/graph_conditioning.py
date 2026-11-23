"""Framewise graph conditioning with equivalent projection reuse.

Every frame retains all 17 joint features. No temporal or joint pooling is
performed. Linear factors remain separately trainable and are composed on
every forward pass, so optimizer updates and gradients follow both factors.
"""

import torch
from torch import nn
from torch.nn import functional as F


class FramewiseGraphConditioning(nn.Module):
    def __init__(self, dim, hidden_dim, output_dim, projection_dim, state_dim, num_joints=17):
        super().__init__()
        if num_joints != 17:
            raise ValueError('Framewise graph conditioning uses the 17-joint skeleton.')
        if min(dim, hidden_dim, output_dim, projection_dim, state_dim) < 1:
            raise ValueError('Graph dimensions must be positive.')

        # Preserve the original graph: self weight 2, skeleton edge weight 1.
        adj = 2 * torch.eye(num_joints)
        edges = [(0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6),
                 (7, 8), (8, 9), (8, 11), (8, 14), (9, 10), (11, 12),
                 (12, 13), (14, 15), (15, 16)]
        for first, second in edges:
            adj[first, second] = adj[second, first] = 1
        self.register_buffer('adj', adj, persistent=False)
        self.register_buffer('parents', torch.tensor(
            [0, 0, 1, 2, 3, 0, 4, 5, 6, 8, 11, 12, 13, 8, 14, 15, 16]), persistent=False)
        self.gcn = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.proj_gcn = nn.Sequential(
            nn.Linear(output_dim, projection_dim), nn.LayerNorm(projection_dim),
            nn.SiLU(), nn.Linear(projection_dim, state_dim),
        )
        self.proj_adj = nn.Linear(num_joints, output_dim)

    def _projection_tail(self, preactivation):
        return self.proj_gcn[3](self.proj_gcn[2](self.proj_gcn[1](preactivation)))

    @staticmethod
    def _directions(horizontal, vertical):
        # [B,T,J,N] -> [B,4,N,T*J], matching CrossScan_plus_poselimbs.
        horizontal = horizontal.flatten(1, 2).transpose(1, 2)
        vertical = vertical.transpose(1, 2).flatten(1, 2).transpose(1, 2)
        return torch.stack((horizontal, vertical, horizontal.flip(-1), vertical.flip(-1)), dim=1)

    def forward(self, x):
        if x.ndim != 4 or x.shape[-2] != self.adj.shape[0]:
            raise ValueError('Expected graph input [batch, frames, 17, channels].')
        # Aggregate neighbors independently in every frame, retaining each joint.
        features = self.gcn[2](self.gcn[1](self.gcn[0](self.adj @ x)))

        # No nonlinearity separates these two factors. Recompute in the graph
        # each time; detach or persistent caches would invalidate training.
        graph_last, projection_first = self.gcn[3], self.proj_gcn[0]
        weight = projection_first.weight @ graph_last.weight
        bias = F.linear(graph_last.bias, projection_first.weight, projection_first.bias)
        preactivation = F.linear(features, weight, bias)

        # Parent addition is BEFORE LN/SiLU. Subtract the duplicated projection
        # bias to preserve P(G(h) + G(h_parent)), including parent gradients.
        horizontal = self._projection_tail(
            preactivation + preactivation.index_select(-2, self.parents) - projection_first.bias)
        vertical = self._projection_tail(preactivation)

        # The topology is fixed across frames/batch, but its learned projections
        # must be evaluated again on each forward pass during training.
        topology = self.proj_adj(self.adj)
        horizontal = horizontal + self.proj_gcn(topology + topology.index_select(0, self.parents))
        vertical = vertical + self.proj_gcn(topology)
        return self._directions(horizontal, vertical)
