import math
from functools import partial

from einops import repeat
import numpy as np
from timm.models.layers import DropPath
import torch
import torch.nn as nn

from .csms6s import CrossMerge_plus_poselimbs, CrossScan_plus_poselimbs, SelectiveScanCore


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(0.0)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class BiSTSSM(nn.Module):
    """Four-direction state-space operator with per-frame skeleton conditioning."""

    def __init__(self, d_model):
        super().__init__()
        d_inner = 2 * d_model
        d_state = 16
        dt_rank = math.ceil(d_model / 16)
        scan_directions = 4

        self.channel_first = False
        self.disable_z = False
        self.disable_z_act = False
        self.graph_pool_stride = None
        self.graph_equivalent = False
        self.forward_core = partial(
            self.forward_corev2,
            force_fp32=True,
            SelectiveScan=SelectiveScanCore,
            CrossScan=CrossScan_plus_poselimbs,
            CrossMerge=CrossMerge_plus_poselimbs,
        )

        self.out_norm = nn.LayerNorm(d_inner)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner)

        x_projs = [
            nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
            for _ in range(scan_directions)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([proj.weight for proj in x_projs]))

        self.out_act = nn.Identity()
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Identity()

        dt_projs = [self.dt_init(dt_rank, d_inner) for _ in range(scan_directions)]
        self.dt_projs_weight = nn.Parameter(torch.stack([proj.weight for proj in dt_projs]))
        self.dt_projs_bias = nn.Parameter(torch.stack([proj.bias for proj in dt_projs]))
        self.A_logs = self.A_log_init(d_state, d_inner, scan_directions)
        self.Ds = self.D_init(d_inner, scan_directions)

        self.gcn = GCN(d_inner, num_joints=17)
        self.proj_gcn = nn.Sequential(
            nn.Linear(d_inner, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_state),
        )
        self.proj_adj = nn.Linear(17, d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -std, std)

        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        # Inverse softplus keeps initialized time steps within the chosen range.
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies):
        a = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=d_inner).contiguous()
        a_log = nn.Parameter(repeat(torch.log(a), 'd n -> r d n', r=copies).flatten(0, 1))
        a_log._no_weight_decay = True
        return a_log

    @staticmethod
    def D_init(d_inner, copies):
        d = nn.Parameter(repeat(torch.ones(d_inner), 'n -> r n', r=copies).flatten(0, 1))
        d._no_weight_decay = True
        return d

    def forward_corev2(
        self, x, x_gcn, adj=None, *, force_fp32, SelectiveScan, CrossScan, CrossMerge,
    ):
        batch, _, height, width = x.shape
        directions, _, dt_rank = self.dt_projs_weight.shape
        state = self.A_logs.shape[1]
        length = height * width

        xs = CrossScan.apply(x)
        projected = torch.einsum('b k d l, k c d -> b k c l', xs, self.x_proj_weight)
        dts, bs, cs = torch.split(projected, [dt_rank, state, state], dim=2)
        dts = torch.einsum('b k r l, k d r -> b k d l', dts, self.dt_projs_weight)

        xs = xs.view(batch, -1, length)
        dts = dts.contiguous().view(batch, -1, length)
        a = -torch.exp(self.A_logs.to(torch.float))
        bs = bs.contiguous().view(batch, directions, state, length)
        cs = cs.contiguous().view(batch, directions, state, length)
        if self.graph_equivalent:
            # Converted inference graphs supply the bias in scan order.
            bs = bs + x_gcn
            cs = cs + x_gcn
        else:
            graph = self.proj_gcn(CrossScan.apply(x_gcn).permute(0, 1, 3, 2).contiguous())
            graph = graph.permute(0, 1, 3, 2).contiguous()
            adj = self.proj_adj(adj).permute(0, 3, 1, 2).contiguous()
            adj = self.proj_gcn(CrossScan.apply(adj).permute(0, 1, 3, 2).contiguous())
            adj = adj.permute(0, 1, 3, 2).contiguous()
            bs = bs + graph + adj
            cs = cs + graph + adj
        d = self.Ds.to(torch.float)
        delta_bias = self.dt_projs_bias.view(-1).to(torch.float)
        if force_fp32:
            xs, dts, bs, cs = (value.to(torch.float32) for value in (xs, dts, bs, cs))

        ys = SelectiveScan.apply(xs, dts, a, bs, cs, d, delta_bias, True, -1, -1, True)
        ys = ys.view(batch, directions, -1, height, width)
        y = CrossMerge.apply(ys).view(batch, -1, height, width)
        y = y.view(batch, -1, length).transpose(1, 2).contiguous().view(batch, height, width, -1)
        return self.out_norm(y).to(x.dtype)

    def forward(self, x):
        x, z = self.in_proj(x).chunk(2, dim=-1)
        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        frame_features = x.permute(0, 2, 3, 1).contiguous()
        if self.graph_equivalent:
            graph_bias = self.graph_conditioning(frame_features)
            y = self.forward_core(x, graph_bias)
        else:
            graph, adj = self.gcn(frame_features)
            y = self.forward_core(x, graph.permute(0, 3, 1, 2).contiguous(), adj)

        y = self.out_act(y) * z
        return self.dropout(self.out_proj(y))


class BiSTSSMBlock(nn.Module):
    def __init__(self, hidden_dim, drop_path, norm_layer, mlp_ratio, ffn_hidden_dim=None):
        super().__init__()
        self.norm = norm_layer(hidden_dim)
        self.op = BiSTSSM(hidden_dim)
        self.drop_path = DropPath(drop_path)
        self.norm2 = norm_layer(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio) if ffn_hidden_dim is None else ffn_hidden_dim
        if mlp_hidden_dim < 1:
            raise ValueError('FFN hidden dimension must be positive.')
        self.mlp = Mlp(hidden_dim, mlp_hidden_dim)

    def forward(self, x):
        x = x + self.drop_path(self.op(self.norm(x)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class GCN(nn.Module):
    def __init__(self, dim, num_joints):
        super().__init__()
        self.register_buffer('adj', self.construct_adj(num_joints), persistent=False)
        self.gcn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    @staticmethod
    def construct_adj(num_joints):
        adj = np.zeros((num_joints, num_joints), dtype=int)
        np.fill_diagonal(adj, 1)
        connections = [
            (0, 1), (0, 4), (0, 7), (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9),
            (8, 11), (8, 14), (9, 10), (11, 12), (12, 13), (14, 15), (15, 16),
        ]
        for start, end in connections:
            adj[start, end] = 1
            adj[end, start] = 1
        return torch.from_numpy(adj.astype(np.float32)) + torch.eye(num_joints)

    def forward(self, x):
        batch, frames = x.shape[:2]
        adj = self.adj.expand(batch, frames, -1, -1)
        return self.gcn(torch.matmul(adj, x)), adj
