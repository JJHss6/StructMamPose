from lib.model.PoseMamba import PoseMamba


class AverageMeter:
    """Track a weighted running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def partial_train_layers(model, partial_list):
    """Select the parameters to optimize by matching their names."""
    for name, parameter in model.named_parameters():
        parameter.requires_grad = any(layer in name for layer in partial_list)
    return model


def load_backbone(args):
    if args.backbone != 'PoseMamba':
        raise ValueError('The S/B release supports only the PoseMamba backbone.')

    retired_options = (
        'grouped_s', 'lite_graph', 'graph_equivalent', 'split_graph_by_joint_group',
        'anatomical_ffn_blocks', 'anatomical_ffn_hidden_dim', 'graph_output_dim',
        'graph_projection_dim',
    )
    for option in retired_options:
        if getattr(args, option, None):
            raise ValueError(f'{option} is unsupported by the S/B release.')
    for option in ('graph_hidden_dim', 'first_graph_hidden_dim'):
        if getattr(args, option, None) not in (None, 2 * args.dim_feat):
            raise ValueError(f'{option} must match the original graph width.')

    return PoseMamba(
        num_frame=args.maxlen,
        embed_dim_ratio=args.dim_feat,
        mlp_ratio=args.mlp_ratio,
        depth=args.depth,
        ffn_hidden_dim=getattr(args, 'ffn_hidden_dim', None),
    )
