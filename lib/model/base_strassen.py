"""Inference-only Base deployment using one-level Strassen matrix multiplication.

The 4.913514M report counts stored deployed coefficient elements in an explicit
partial scope, not independent trainable parameters. Additional tensor additions
are recorded separately. Export only after training; never optimize this model.
"""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .posemamba_online import OnlineFactorizedGraph, convert_to_online_deploy

FORMAT = 'posemamba.original_b.strassen'
VERSION = 1

def weight_parts(weight):
    # B is the mathematical right-hand matrix of X @ B, not Linear's W.
    b=F.pad(weight,(0,weight.shape[1]%2,0,weight.shape[0]%2)).T
    k,n=b.shape[0]//2,b.shape[1]//2
    e,f,g,h=b[:k,:n],b[:k,n:],b[k:,:n],b[k:,n:]
    return (e+h,e,f-h,g-e,h,e+f,g+h)


def strassen_product(x,weight=None,parts=None,out_features=None):
    original_shape=x.shape
    flat=x.reshape(-1,x.shape[-1]);rows,inner=flat.shape
    outputs=weight.shape[0] if weight is not None else out_features
    a=F.pad(flat,(0,inner%2,0,rows%2))
    m,k=a.shape[0]//2,a.shape[1]//2
    a11,a12,a21,a22=a[:m,:k],a[:m,k:],a[m:,:k],a[m:,k:]
    prepacked=parts is not None
    b=weight_parts(weight) if parts is None else parts
    left=(a11+a22,a21+a22,a11,a22,a11+a12,a21-a11,a12-a22)
    products=[left[i]@b[i] for i in range(7)]
    p1,p2,p3,p4,p5,p6,p7=products
    c11=p1+p4-p5+p7
    c12=p3+p5
    c21=p2+p4
    c22=p1-p2+p3+p6
    result=torch.cat((torch.cat((c11,c12),dim=1),torch.cat((c21,c22),dim=1)),dim=0)
    n=b[0].shape[1]
    details={'gemm_calls':7,'gemm_macs':sum(a.shape[0]*a.shape[1]*b.shape[1] for a,b in zip(left,b)),
             'additional_tensor_add_sub':5*m*k+8*m*n+(0 if prepacked else 5*k*n),
             'padded_shape':[2*m,2*k,2*n]}
    return result[:rows,:outputs].reshape(*original_shape[:-1],outputs),details


class StrassenLinear(nn.Module):
    def __init__(self,source,prepacked=False):
        super().__init__()
        self.in_features=source.in_features;self.out_features=source.out_features
        self.prepacked=prepacked
        if prepacked:
            self.parts=nn.ParameterList([nn.Parameter(x.detach().clone(),requires_grad=False) for x in weight_parts(source.weight)])
        else:
            self.weight=nn.Parameter(source.weight.detach().clone(),requires_grad=False)
        self.bias=None if source.bias is None else nn.Parameter(source.bias.detach().clone(),requires_grad=False)
        self.last_details=None
        self.eval()
    def forward(self,x):
        if self.prepacked:
            out,self.last_details=strassen_product(x,parts=self.parts,out_features=self.out_features)
        else:
            out,self.last_details=strassen_product(x,weight=self.weight)
        return out if self.bias is None else out+self.bias


def count_strassen(module,inputs,output):
    assert module.last_details['gemm_calls']==7
    module.total_ops += module.total_ops.new_tensor([module.last_details['gemm_macs']])
    module.total_params[0]=sum(p.numel() for p in module.parameters())



def apply_base_strassen(converted):
    if (converted.architecture != 'original_b'
            or not getattr(converted, 'online_factorized', False)
            or getattr(converted, 'strassen_base', False)):
        raise ValueError('Expected the unmodified original Base online deployment.')
    changes=[]
    packed_inputs=packed_outputs=0
    for pair in range(10):
        for stream in ('STEblocks','TTEblocks'):
            block=getattr(converted.model,stream)[pair]
            if hasattr(block.op.in_proj,'linear'):
                prepacked=packed_inputs<4
                old=block.op.in_proj.linear
                block.op.in_proj.linear=StrassenLinear(old,prepacked)
                packed_inputs+=int(prepacked)
                changes.append({'name':f'{stream}.{pair}.op.in_proj.linear',
                                'in':old.in_features,'out':old.out_features,'prepacked':prepacked})
            prepacked=packed_outputs<1
            old=block.op.out_proj.linear
            block.op.out_proj.linear=StrassenLinear(old,prepacked)
            packed_outputs+=int(prepacked)
            changes.append({'name':f'{stream}.{pair}.op.out_proj.linear',
                            'in':old.in_features,'out':old.out_features,'prepacked':prepacked})
            if pair>=2:
                for part in ('fc1','fc2'):
                    projection=getattr(block.mlp,part)
                    old=projection.linear
                    projection.linear=StrassenLinear(old)
                    changes.append({'name':f'{stream}.{pair}.mlp.{part}.linear',
                                    'in':old.in_features,'out':old.out_features,'prepacked':False})
            if type(block.op.graph_conditioning) is not OnlineFactorizedGraph:
                raise ValueError('All graphs must use the original online graph implementation.')
            block.op.graph_conditioning.strassen_effective_map=True
    if packed_inputs!=4 or packed_outputs!=1 or len(changes)!=71:
        raise AssertionError('Unexpected Base conversion topology.')
    converted.strassen_base=True
    converted.strassen_layers=changes
    return converted.eval()


def convert_to_base_strassen(source):
    """Make a frozen independent inference model, preserving source weights/RNG."""
    devices=sorted({p.device.index for p in source.parameters() if p.is_cuda})
    with torch.no_grad(),torch.random.fork_rng(devices=devices):
        return apply_base_strassen(convert_to_online_deploy(source))


def execution_details(model):
    operations=[]
    for name,module in model.named_modules():
        if isinstance(module,StrassenLinear):
            if module.last_details is None:
                raise ValueError('Run the model before collecting execution details.')
            operations.append({'name':name,**module.last_details})
        elif isinstance(module,OnlineFactorizedGraph) and getattr(module,'strassen_effective_map',False):
            operations.append({'name':name+'.effective_map',**module.last_strassen_details})
    return {'strassen_products':len(operations),
            'actual_subgemm_calls':sum(x['gemm_calls'] for x in operations),
            'additional_tensor_add_sub':sum(x['additional_tensor_add_sub'] for x in operations),
            'operations':operations}


def save_strassen_model(model,path,metadata=None):
    """Write a versioned tensor-only checkpoint; no Python model is pickled."""
    if not getattr(model,'strassen_base',False) or any(m.training for m in model.modules()):
        raise ValueError('Only an evaluated Base Strassen deployment can be saved.')
    config={'architecture':'original_b','construction_info':model.construction_info,
            'normalization_eps':copy.deepcopy(model._normalization_eps),
            'strassen_layers':model.strassen_layers,'metadata':metadata or {}}
    payload={'format':FORMAT,'version':VERSION,
             'configuration_json':json.dumps(config,allow_nan=False),
             'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}}
    destination=Path(path);destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,destination)


def load_strassen_model(path,map_location='cpu'):
    """Strictly reconstruct Base conversion and then restore all exported tensors."""
    from .PoseMamba import PoseMamba
    payload=torch.load(path,map_location=map_location,weights_only=True)
    if (not isinstance(payload,dict) or payload.get('format')!=FORMAT
            or payload.get('version')!=VERSION):
        raise ValueError('Unsupported Base Strassen checkpoint format/version.')
    config=json.loads(payload['configuration_json'])
    if config.get('architecture')!='original_b':
        raise ValueError('Expected original_b architecture.')
    info=config['construction_info']
    expected={'num_joints':17,'in_chans':2,'embed_dim_ratio':128,
              'depth':10,'mlp_ratio':2.0,'ffn_hidden_dim':256}
    if (set(info)!=set(expected)|{'num_frame'}
            or any(info[k]!=v for k,v in expected.items())
            or type(info['num_frame']) is not int or info['num_frame']<1):
        raise ValueError('Unsupported original Base construction information.')
    state=payload['state_dict']
    anchor=state['model.Spatial_patch_to_embedding.weight']
    if any(not isinstance(t,torch.Tensor) or not t.is_floating_point()
           or t.dtype!=anchor.dtype or t.device!=anchor.device for t in state.values()):
        raise ValueError('Deployment tensors must share one floating dtype and device.')
    with torch.random.fork_rng(devices=[]),torch.no_grad():
        source=PoseMamba(**info).to(device=anchor.device,dtype=anchor.dtype).eval()
        norms={name:m for name,m in source.named_modules() if isinstance(m,nn.LayerNorm)}
        eps=config['normalization_eps']
        if set(eps)!=set(norms) or any(not isinstance(v,(int,float)) or not 0<v<float('inf') for v in eps.values()):
            raise ValueError('Invalid normalization settings.')
        for name,value in eps.items():norms[name].eps=float(value)
        converted=convert_to_base_strassen(source)
    if converted.strassen_layers!=config['strassen_layers']:
        raise ValueError('Unexpected Strassen layer selection.')
    converted.load_state_dict(state,strict=True)
    converted.export_metadata=config.get('metadata',{})
    return converted.eval()
