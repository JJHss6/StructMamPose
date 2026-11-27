"""Additional algebraic inference rewrites for the original dense Base model."""
import torch
from torch import nn
from .PoseMamba import PoseMamba

class AdvancedPoseMamba(PoseMamba):
    """Keep the original forward order, specializing only the first projection."""

    def __init__(self, converted, first_projection):
        nn.Module.__init__(self)
        for (name, parameter) in converted._parameters.items():
            self.register_parameter(name, parameter)
        for (name, buffer) in converted._buffers.items():
            self.register_buffer(name, buffer, persistent=name not in converted._non_persistent_buffers_set)
        for (name, module) in converted._modules.items():
            self.add_module(name, module)
        self.block_depth = converted.block_depth
        self.first_projection = first_projection
        self.STEblocks[0].norm = nn.Identity()
        self.STEblocks[0].op.in_proj = nn.Identity()
        self.eval()

    def STE_forward(self, raw):
        embedded = self.Spatial_patch_to_embedding(raw) + self.Spatial_pos_embed
        embedded = self.pos_drop(embedded)
        projected = self.first_projection(raw, embedded)
        block = self.STEblocks[0]
        x = embedded + block.drop_path(block.op(projected))
        x = x + block.drop_path(block.mlp(block.norm2(x)))
        return self.Spatial_norm(x)

def apply_advanced_equivalence(converted, source):
    """Transform an evaluated v3 inference copy; the source is read-only."""
    from .deploy_coordinates import FirstInputProjection, apply_block_coordinates
    from .deploy_graph_side import AdvancedGraphConditioning, build_advanced_side
    if any((module.training for module in source.modules())):
        raise ValueError('Advanced conversion requires an evaluated original Base.')
    devices = sorted({parameter.device.index for parameter in source.parameters() if parameter.is_cuda})
    with torch.no_grad(), torch.random.fork_rng(devices=devices):
        first_projection = FirstInputProjection.from_source(source)
        apply_block_coordinates(converted)
        for block in list(converted.STEblocks) + list(converted.TTEblocks):
            block.op.graph_conditioning = AdvancedGraphConditioning.from_graph(block.op.graph_conditioning)
        (converted.proj_side, converted.cross_attn) = build_advanced_side(converted.proj_side, converted.cross_attn)
        return AdvancedPoseMamba(converted, first_projection)
