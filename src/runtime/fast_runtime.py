"""Opt-in execution helpers; keep reward, sampling distribution and update counts.

CachedReplay stores one deterministic scalar per transition. After loading an old
checkpoint by assigning .rows, call rebuild_severity() before fast sampling.
ActorMirror synchronizes after EVERY actor update, not periodically/stale.
"""
import numpy as np
import torch
import copy
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


def configure_adam(learner, fused=True):
    for optimizer in (learner.actor_opt, learner.critic_opt):
        for group in optimizer.param_groups:
            group['fused'] = bool(fused)
            group['foreach'] = False if fused else None

class ActorMirror:
    def __init__(self, actor):
        if list(actor.buffers()):
            raise ValueError('CPU mirror requires explicit buffer synchronization for this model')
        self.actor = copy.deepcopy(actor).cpu().eval()
    def sync(self, source):
        with torch.no_grad():
            vector = torch.nn.utils.parameters_to_vector(source.parameters()).detach().cpu()
            torch.nn.utils.vector_to_parameters(vector, self.actor.parameters())

class CachedReplay(Replay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.severity = np.empty(self.capacity, dtype=np.float32)
    def push(self, *row):
        index = self.pos
        super().push(*row)
        self.severity[index] = np.maximum(np.abs(self.rows[index][0] - 1.) - .045, 0.).max()
    def rebuild_severity(self):
        if self.rows:
            voltage = np.stack([row[0] for row in self.rows])
            self.severity[:len(self.rows)] = np.maximum(np.abs(voltage - 1.) - .045, 0.).max(axis=1)

def sample_replay_fast(replay, count, hard_fraction=0., hard_quantile=.75):
    if hard_fraction <= 0:
        return replay.sample(count)
    if count > len(replay.rows):
        raise ValueError('replay sample exceeds available transitions')
    severity = replay.severity[:len(replay.rows)]
    threshold = float(np.quantile(severity, hard_quantile))
    hard_pool = np.flatnonzero((severity >= threshold) & (severity > 0))
    hard_count = min(int(round(count * hard_fraction)), len(hard_pool))
    hard_ids = replay.rng.choice(hard_pool, hard_count, replace=False)
    remaining = np.setdiff1d(np.arange(len(replay.rows)), hard_ids, assume_unique=False)
    uniform_ids = replay.rng.choice(remaining, count - hard_count, replace=False)
    indices = np.concatenate((hard_ids, uniform_ids))
    replay.rng.shuffle(indices)
    return [np.stack(values) for values in zip(*(replay.rows[i] for i in indices))]
