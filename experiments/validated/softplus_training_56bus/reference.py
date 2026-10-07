"""Load the reference 56-bus policies used by the training evaluation."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from src.softplus_controller import voltage_branch

REF_ROOT = Path("D:/Code/Python/Flexible_Voltage_Control/check_points/policy_net/2025-02-18")

class Reference(nn.Module):
    def __init__(self, states):
        super().__init__()
        self.states=[]
        for i,state in enumerate(states):
            converted={}
            for key,value in state.items():
                name=f'a{i}_{key.replace(".","_")}'
                self.register_buffer(name,value.detach().cpu().clone())
                converted[key]=name
            self.states.append(converted)
    def forward(self,v,topology):
        outputs=[]
        topo=F.normalize(topology,dim=1)
        for i,keys in enumerate(self.states):
            s={k:getattr(self,n) for k,n in keys.items()}
            b=s['b'].clamp_min(0); b=b/b.sum()
            y=voltage_branch(v[:,i:i+1],s['q'].clamp(.01,1000),b)
            x=F.relu(F.linear(topo,s['topology_net.linear1.weight'],s['topology_net.linear1.bias']))
            x=F.relu(F.linear(x,s['topology_net.linear2.weight'],s['topology_net.linear2.bias']))
            gain=1.+F.elu(F.linear(x,s['topology_net.linear3.weight'],s['topology_net.linear3.bias']))
            outputs.append(.7*y*gain)
        return torch.cat(outputs,dim=1)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_reference(device):
    paths=[REF_ROOT/f'Step_500_Seed_4_a{i}.pth' for i in range(5)]
    states=[torch.load(p,map_location='cpu',weights_only=True) for p in paths]
    return Reference(states).to(device).eval(),[dict(path=str(p),sha256=sha(p)) for p in paths]
