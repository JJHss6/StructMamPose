"""Base deployment accounting on independent CPU copies; original training unchanged."""
import copy
from unittest.mock import patch
import torch

ONLINE_REPORTING_NOTE = (
    "Following PoseMamba’s use of THOP, we report parameter and MAC counts "
    "excluding selected components functionally related to parameter generation."
)


def reporting_note_for(model=None, config=None):
    return ONLINE_REPORTING_NOTE


def uses_online_equivalent_scope(model=None, config=None):
    scope=getattr(config,'profile_scope',None)
    if scope not in (None,'base_strassen_partial'):
        raise ValueError(f'Unsupported Base scope: {scope!r}.')
    return True


def collect_model_complexity(model,config=None):
    from lib.model.base_strassen import convert_to_base_strassen,execution_details
    from lib.model.csms6s import SelectiveScanCore
    from .model_profile import profile_online_reported,deploy_cache_storage
    uses_online_equivalent_scope(model,config)
    source=model.module if isinstance(model,torch.nn.DataParallel) else model
    frames=int(getattr(config,'maxlen',source.Temporal_pos_embed.shape[1]))
    shape=(1,frames,17,2)
    previous_threads=torch.get_num_threads()
    try:
        with torch.random.fork_rng(devices=[]),torch.no_grad():
            torch.set_num_threads(1)
            candidate=copy.deepcopy(source).cpu().float().eval()
            deployment=convert_to_base_strassen(candidate)
            inputs=torch.zeros(shape,dtype=torch.float32)
            # Shape-only scan for accounting. Never used for training or equivalence checks.
            with patch.object(SelectiveScanCore,'apply',staticmethod(lambda u,*_:u)):
                measured=profile_online_reported(deployment,(inputs,),verbose=False)
            details=execution_details(deployment)
            return {'input_shape':list(shape),
                    'training':{'full_parameters':sum(p.numel() for p in source.parameters())},
                    'reported':measured,'deploy':measured,
                    'additional_tensor_add_sub':details['additional_tensor_add_sub'],
                    'deployment_cache':deploy_cache_storage(deployment),
                    'parameter_interpretation':'Stored deployment coefficient elements in the selected partial scope, not independent trainable parameters.',
                    'mac_interpretation':'Actual sub-GEMMs including padding; extra tensor additions/subtractions reported separately.'}
    finally:
        torch.set_num_threads(previous_threads)


def log_model_complexity(model,config,log):
    full=sum(p.numel() for p in model.parameters())
    if getattr(config,'backbone',None)!='PoseMamba':
        return {'training':{'full_parameters':full},'deploy':None}
    log.info(ONLINE_REPORTING_NOTE)
    try:
        report=collect_model_complexity(model,config)
    except Exception as error:
        message=f'{type(error).__name__}: {error}'
        log.warning(f'[PoseMamba/THOP] Base equivalent deployment: N/A | {message}')
        return {'training':{'full_parameters':full},'deploy':None,'profiling_error':message}
    measured=report['reported']
    log.info(f"Params={measured['thop_params']/1e6:.6f} M | MACs={measured['macs']/1e9:.9f} G")
    log.info(f"Deployment Params denote stored coefficients in the selected scope; full training Params={full/1e6:.6f} M. Additional tensor add/sub={report['additional_tensor_add_sub']/1e9:.9f} G.")
    return report
