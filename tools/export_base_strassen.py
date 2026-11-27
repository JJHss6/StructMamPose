"""Export original Base training weights into their equivalent Strassen inference form."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from lib.model.base_strassen import convert_to_base_strassen,save_strassen_model,load_strassen_model,execution_details
from lib.model.csms6s import SelectiveScanCore
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config
from lib.utils.model_profile import profile_online_reported
from lib.utils.scan_reference import reference_scan_context


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--random-init',action='store_true')
    parser.add_argument('--trusted-checkpoint',action='store_true')
    parser.add_argument('--output',required=True)
    parser.add_argument('--report')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--verify-frames',type=int,default=3)
    args=parser.parse_args()
    output=Path(args.output).resolve()
    protected=[Path(args.config).resolve()]+([Path(args.checkpoint).resolve()] if args.checkpoint else [])
    if output in protected or (args.report and Path(args.report).resolve() in protected+[output]):
        parser.error('Output/report must not overwrite inputs or each other.')
    if output.exists() or (args.report and Path(args.report).exists()):
        parser.error('Choose new output/report paths; existing results are not overwritten.')
    cfg=get_config(args.config)
    if not 1<=args.verify_frames<=cfg.maxlen:parser.error('Invalid verification frame count.')
    if args.trusted_checkpoint and not args.checkpoint:parser.error('--trusted-checkpoint needs --checkpoint.')
    torch.manual_seed(0);torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=load_backbone(cfg).float().eval()
    if args.checkpoint:
        data=torch.load(args.checkpoint,map_location='cpu',weights_only=not args.trusted_checkpoint)
        state=data.get('model_pos',data.get('state_dict',data))
        source.load_state_dict({k.removeprefix('module.'):v for k,v in state.items()},strict=True)
    source=source.to(args.device)
    converted=convert_to_base_strassen(source)
    x=torch.randn(1,args.verify_frames,17,2,device=args.device)
    context=reference_scan_context() if args.device=='cpu' else nullcontext()
    with context,torch.no_grad():
        reference=source(x);actual=converted(x)
        torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
    report={'random_initialization':bool(args.random_init),'verification_frames':args.verify_frames,
            'max_abs_error':(actual-reference).abs().max().item(),
            'mean_abs_error':(actual-reference).abs().mean().item(),
            'full_training_parameters':sum(p.numel() for p in source.parameters()),
            'full_dataset_accuracy_evaluation':'not_performed'}
    with torch.no_grad(),patch.object(SelectiveScanCore,'apply',staticmethod(lambda u,*_:u)):
        measured=profile_online_reported(converted,(torch.zeros(1,243,17,2,device=args.device),),verbose=False)
    report['profile']=measured
    details=execution_details(converted)
    report['additional_tensor_add_sub']=details['additional_tensor_add_sub']
    report['parameter_interpretation']='Stored deployment coefficients under partial accounting scope; no new independent learned parameters.'
    report['mac_interpretation']='Actual sub-GEMM MACs including padding; extra additions are reported separately.'
    save_strassen_model(converted,output,metadata=report)
    restored=load_strassen_model(output,map_location=args.device)
    for name,value in converted.state_dict().items():
        if not torch.equal(value,restored.state_dict()[name]):raise AssertionError('Reload state mismatch: '+name)
    context=reference_scan_context() if args.device=='cpu' else nullcontext()
    with context,torch.no_grad():
        reloaded=restored(x)
        if not torch.equal(actual,reloaded):raise AssertionError('Reload output differs.')
    report['reload_state_and_output_exact']=True
    report['status']='passed'
    if args.report:
        destination=Path(args.report);destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='profile'},indent=2))
    print(f"Params={measured['thop_params']/1e6:.6f} M | MACs={measured['macs']/1e9:.9f} G")

if __name__=='__main__':main()
