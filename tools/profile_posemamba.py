import argparse
import sys
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

# Allow `python tools/profile_posemamba.py` without a PYTHONPATH override.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from lib.model.csms6s import SelectiveScanCore
from lib.model.posemamba_deploy import load_deploy_model
from lib.model.posemamba_online import convert_to_online_deploy
from lib.utils.learning import load_backbone
from lib.utils.model_profile import (
    deploy_cache_storage, profile_online_reported, profile_posemamba,
)
from lib.utils.model_logging import (
    reporting_note_for, uses_online_equivalent_scope,
)
from lib.utils.tools import get_config


def scan_shape_only(u, *_):
    # Default THOP does not count the CUDA scan. Preserve its output shape.
    return u


def main():
    parser = argparse.ArgumentParser(
        description='Profile PoseMamba and its current online equivalent S export.',
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--config', help='Training architecture YAML; defaults to the S YAML.')
    source.add_argument(
        '--deploy-checkpoint', type=Path,
        help='Exported checkpoint; use its saved architecture instead of a YAML.',
    )
    parser.add_argument(
        '--shape-only', action='store_true',
        help='Count on CPU using a shape-preserving stand-in for the CUDA scan.',
    )
    parser.add_argument(
        '--reported', action='store_true',
        help='Report the current online equivalent S scope with matched parameter and MAC exclusions.',
    )
    args = parser.parse_args()
    if not args.shape_only and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --shape-only to count on CPU.')

    device = torch.device('cpu' if args.shape_only else 'cuda')
    if args.shape_only:
        torch.set_num_threads(1)
    source_path = Path(
        args.deploy_checkpoint or args.config or 'pose3d/PoseMamba_train_h36m_S.yaml'
    ).expanduser()
    if not source_path.is_absolute() and not source_path.is_file():
        source_path = PROJECT_ROOT / source_path
    if not source_path.is_file():
        parser.error(f'Model source does not exist: {source_path}')
    config = None
    if args.deploy_checkpoint:
        model = load_deploy_model(source_path, map_location='cpu').to(device)
        if not getattr(model, 'online_factorized', False):
            parser.error('Only current online-factorized S deployment checkpoints are supported.')
        architecture = model.construction_info
        architecture_name = model.architecture
        frames = architecture['num_frame']
        joints = architecture['num_joints']
        channels = architecture['in_chans']
    else:
        config = get_config(str(source_path))
        model = load_backbone(config).to(device)
        architecture_name = getattr(model, 'architecture_name', f'PoseMamba_C{config.dim_feat}')
        frames, joints, channels = config.maxlen, config.num_joints, 2
    inputs = torch.zeros(1, frames, joints, channels, device=device)
    scan_context = (
        patch.object(SelectiveScanCore, 'apply', staticmethod(scan_shape_only))
        if args.shape_only else nullcontext()
    )
    with scan_context:
        if args.reported:
            try:
                uses_online_equivalent_scope(model, config)
                note = reporting_note_for(model, config)
                if not args.deploy_checkpoint:
                    model = convert_to_online_deploy(model.eval())
                report = profile_online_reported(model, inputs=(inputs,), verbose=False)
            except ValueError as error:
                parser.error(str(error))
            print(note)
            print(f'Params={report["thop_params"] / 1e6:.6f} M | MACs={report["macs"] / 1e9:.9f} G')
            return
        macs, params = profile_posemamba(model, inputs=(inputs,), verbose=False)

    full_params = sum(parameter.numel() for parameter in model.parameters())
    cache = deploy_cache_storage(model)
    if args.shape_only:
        print('Mode: CPU shape-only THOP')
    print(f'Architecture: {architecture_name}')
    print(f'Model source: {source_path.resolve()}')
    if args.deploy_checkpoint:
        print(f'Model: exported checkpoint (fold_affine={model.fold_affine}, fuse_attention={model.fuse_attention}, advanced={model.advanced}, online_factorized={model.online_factorized})')
        provenance = model.export_metadata.get('source', {}).get('kind', 'unspecified')
        print(f'Weight provenance: {provenance}')
    else:
        print('Weights: random initialization for architecture counting only; no accuracy evaluation.')
    print('Accounting: default THOP rules + exact equivalent-operation hooks.')
    print('Uncounted: selective scan, SSM einsum, graph adjacency/attention matmul, and tensor operations.')
    print(f'Input: {tuple(inputs.shape)}')
    print(f'THOP Params: {int(params):,} ({params / 1e6:.6f} M)')
    print(f'THOP MACs: {int(macs):,} ({macs / 1e9:.9f} G)')
    print(f'THOP MACs/frame: {macs / frames / 1e6:.6f} M')
    print(f'Full Params: {full_params:,} ({full_params / 1e6:.6f} M)')
    if args.deploy_checkpoint or cache['elements']:
        print(f'Derived inference caches: {cache["elements"]:,} scalars, {cache["bytes"]:,} bytes')
        print(f'Full Params + cache scalars: {full_params + cache["elements"]:,}')


if __name__ == '__main__':
    main()
