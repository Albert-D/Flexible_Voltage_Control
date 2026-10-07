"""Replay, actor initialization and feasibility helpers for controller training."""
import numpy as np
import torch

INITIAL_SLOPE_FACTOR = 0.05

class Replay:
    def __init__(self,capacity=100000,seed=912):
        self.rows=[]; self.capacity=capacity; self.pos=0; self.rng=np.random.default_rng(seed)
    def push(self,*row):
        item=tuple(np.asarray(z,dtype=np.float32).copy() for z in row)
        if len(self.rows)<self.capacity:self.rows.append(item)
        else:self.rows[self.pos]=item
        self.pos=(self.pos+1)%self.capacity
    def sample(self,n):
        ids=self.rng.choice(len(self.rows),n,replace=False)
        return [np.stack(z) for z in zip(*(self.rows[i] for i in ids))]

def safe_reset(env, scene):
    state = env.reset(scene)
    energized = np.asarray(env.voltage[list(env.nodes)], dtype=float)
    if state.min() < .80 or state.max() > 1.20:
        raise ValueError("controlled reset voltage outside 0.80..1.20")
    if energized.min() < .75 or energized.max() > 1.25:
        raise ValueError("energized reset voltage outside 0.75..1.25")
    return state

def initialize_weak_actor(actor, factor=INITIAL_SLOPE_FACTOR):
    """Change initial parameter values only; never multiply inference outputs."""
    if not 0 < factor <= 1:
        raise ValueError('Initialization factor must be in (0, 1]')
    with torch.no_grad():
        for policy in actor.policies:
            slopes = (.01 + .99 * torch.sigmoid(policy.q_raw.detach().cpu())) * factor
            unit = (slopes - .01) / .99
            if not bool(((unit > 0) & (unit < 1)).all()):
                raise ValueError('Requested initialization is outside existing slope bounds')
            policy.q_raw.copy_(torch.logit(unit))
    return actor
