"""Uncapped Softplus topology-conditioned controller and TD3 learner."""
from __future__ import annotations

# Shared 56-bus dependencies now live in the validated experiment.
from pathlib import Path as _MigrationPath
import sys as _migration_sys
_migration_sys.path.insert(0, str(_MigrationPath(__file__).resolve().parents[3]))

import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.validated.training_runtime.support import Replay
from src.softplus_controller import voltage_branch
from experiments.validated.softplus_training_123bus.learner import CentralizedLearner


class Policy(nn.Module):
    """Monotone voltage branch multiplied by an uncapped positive topology gain."""

    def __init__(self, hidden=1024, topology_hidden=256, gain_init_bias=.5):
        super().__init__()
        self.b_raw = nn.Parameter(torch.log(torch.rand(hidden).clamp_min(1e-6)))
        slopes = torch.rand(1, hidden) * .2 + .4
        self.q_raw = nn.Parameter(torch.logit((slopes - .01) / .99))
        self.topology_net = nn.Sequential(
            nn.Linear(113, topology_hidden),
            nn.ReLU(),
            nn.Linear(topology_hidden, topology_hidden),
            nn.ReLU(),
            nn.Linear(topology_hidden, 1),
        )
        for layer in self.topology_net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -.03, .03)
                nn.init.zeros_(layer.bias)
        nn.init.constant_(self.topology_net[-1].bias, float(gain_init_bias))

    def topology_gain(self, topology):
        # Constant scaling preserves line-admittance magnitude. There is no
        # gain cap: control effort is learned through the reward.
        return 1. + F.softplus(self.topology_net(topology / math.sqrt(113.)))

    def forward(self, voltage, topology):
        slopes = .01 + .99 * torch.sigmoid(self.q_raw)
        knots = torch.softmax(self.b_raw, dim=-1)
        return voltage_branch(voltage, slopes, knots) * self.topology_gain(topology)


class Controller(nn.Module):
    def __init__(self, hidden=1024, topology_hidden=256, gain_init_bias=.5):
        super().__init__()
        self.policies = nn.ModuleList([
            Policy(hidden, topology_hidden, gain_init_bias) for _ in range(14)
        ])

    def forward(self, voltage, topology):
        return torch.cat([
            policy(voltage[:, index:index + 1], topology)
            for index, policy in enumerate(self.policies)
        ], dim=1)


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
            outputs.append(y*gain)
        return torch.cat(outputs,dim=1)


class JointCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1=nn.Sequential(nn.Linear(155,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1))
        self.q2=copy.deepcopy(self.q1);nn.init.uniform_(self.q2[-1].weight,-.003,.003)
    def forward(self,v,x,last,q):
        z=torch.cat(((v-1)*20,x/math.sqrt(113.),last/5,(q-last)),dim=1)
        return self.q1(z),self.q2(z)

class Learner(CentralizedLearner):
    @staticmethod
    def action(previous,delta):return (previous-delta.clamp(-5.,5.)).clamp(-50.,50.)
    def __init__(self,device,seed=2601):
        torch.manual_seed(seed);self.device=device
        self.actor=Controller().to(device);self.target=copy.deepcopy(self.actor).requires_grad_(False)
        self.critics=nn.ModuleList([JointCritic() for _ in range(14)]).to(device)
        self.critic_target=copy.deepcopy(self.critics).requires_grad_(False)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=1e-4)
        self.critic_opt=torch.optim.Adam(self.critics.parameters(),lr=1e-3)
        self.actor_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.actor_opt,[10000,20000],gamma=.5)
        self.critic_scheduler=torch.optim.lr_scheduler.MultiStepLR(self.critic_opt,[20000,40000],gamma=.5)
        self.updates=0
