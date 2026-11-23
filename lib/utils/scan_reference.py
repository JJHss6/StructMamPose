"""Differentiable PyTorch reference for PoseMamba's grouped selective scan.

This evaluates the actual recurrence and is intended for small CPU numerical
checks. It is not a fast replacement for the CUDA implementation.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch.nn import functional as F


def selective_scan_reference(u, delta, a, b, c, d=None, delta_bias=None,
                             delta_softplus=False, *_):
    """Evaluate a grouped scan while preserving PyTorch's autograd graph."""
    if u.ndim != 3 or b.ndim != 4 or c.shape != b.shape:
        raise ValueError('Expected u [B,D,L] and B/C [B,K,N,L].')
    batch, channels, length = u.shape
    groups, state_dim = b.shape[1:3]
    if (channels % groups or delta.shape != u.shape
            or b.shape[0] != batch or b.shape[-1] != length
            or a.shape != (channels, state_dim)):
        raise ValueError('Incompatible selective-scan tensor dimensions.')
    if delta_bias is not None:
        delta = delta + delta_bias.view(1, channels, 1)
    if delta_softplus:
        delta = F.softplus(delta)
    grouped_shape = (batch, groups, channels // groups, length)
    delta = delta.reshape(grouped_shape)
    grouped_u = u.reshape(grouped_shape)
    a = a.reshape(groups, channels // groups, state_dim)
    state = u.new_zeros(batch, groups, channels // groups, state_dim)
    output = []
    # Compute one step at a time to avoid allocating [B,D,N,L] intermediates.
    for step in range(length):
        delta_step = delta[..., step].unsqueeze(-1)
        update = delta_step * grouped_u[..., step].unsqueeze(-1)
        state = (torch.exp(delta_step * a) * state
                 + update * b[..., step].unsqueeze(2))
        output.append((state * c[..., step].unsqueeze(2)).sum(dim=-1))
    result = torch.stack(output, dim=-1).reshape(batch, channels, length)
    if d is not None:
        result = result + u * d.view(1, channels, 1)
    return result


@contextmanager
def reference_scan_context():
    """Temporarily replace only the CUDA scan with its true CPU recurrence."""
    from lib.model.csms6s import SelectiveScanCore

    with patch.object(SelectiveScanCore, 'apply', staticmethod(selective_scan_reference)):
        yield
