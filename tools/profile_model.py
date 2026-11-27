"""Profile either packaged model on an independent CPU deployment copy."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from lib.utils.learning import load_backbone
from lib.utils.model_logging import collect_model_complexity
from lib.utils.tools import get_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['S','B'],required=True)
    p.add_argument('--output',help='Optional JSON report; must be a new path.')
    args=p.parse_args()
    destination=Path(args.output) if args.output else None
    if destination and destination.exists():p.error('Output already exists.')
    torch.set_num_threads(1);torch.manual_seed(0)
    cfg=get_config(str(ROOT/f'pose3d/PoseMamba_train_h36m_{args.variant}.yaml'))
    model=load_backbone(cfg).cpu()
    report=collect_model_complexity(model,cfg)
    report['variant']=args.variant
    report['full_training_parameters']=sum(parameter.numel() for parameter in model.parameters())
    if destination:
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2)+'\n')
    measured=report['reported']
    print(f"{args.variant}: Params={measured['thop_params']/1e6:.6f} M | MACs={measured['macs']/1e9:.9f} G")
    print(f"Full training Params={report['full_training_parameters']/1e6:.6f} M")
    print('Reported deployment counts use a partial scope; they are not full-model training counts.')
    if 'additional_tensor_add_sub' in report:
        print(f"Additional tensor add/sub={report['additional_tensor_add_sub']/1e9:.9f} G")


if __name__=='__main__':main()
