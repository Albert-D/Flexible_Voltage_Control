"""Opt-in execution helpers; keep reward, sampling distribution and update counts.

CachedReplay stores one deterministic scalar per transition. After loading an old
checkpoint by assigning .rows, call rebuild_severity() before fast sampling.
ActorMirror synchronizes after EVERY actor update, not periodically/stale.
"""
import numpy as np
import torch
import copy
from experiments.validated.training_runtime.support import Replay

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


def _grouped_mlp(networks, features):
    """Same independent Linear/ReLU layers, batched over agents, no shared weights."""
    from torch.nn import functional as F
    z = features.unsqueeze(0).expand(len(networks), -1, -1)
    for index in (0, 2, 4):
        weights = torch.stack([net[index].weight for net in networks])
        bias = torch.stack([net[index].bias for net in networks])
        z = torch.bmm(z, weights.transpose(1, 2)) + bias.unsqueeze(1)
        if index != 4:
            z = F.relu(z)
    return z

def grouped_actor(actor, voltage, topology):
    """Vectorized evaluation of the unchanged 123-bus Controller."""
    import math
    from torch.nn import functional as F
    policies = actor.policies
    slopes = .01 + .99 * torch.sigmoid(torch.stack([p.q_raw.squeeze(0) for p in policies]))
    b = torch.softmax(torch.stack([p.b_raw for p in policies]), dim=-1)
    knots = (torch.cumsum(b, dim=-1) - b).unsqueeze(1)
    weights = torch.cat((slopes[:, :1], slopes[:, 1:] - slopes[:, :-1]), dim=-1).unsqueeze(1)
    v = voltage.transpose(0, 1).unsqueeze(-1)
    branch = ((F.relu(v - 1.045 - knots) - F.relu(.955 - v - knots)) * weights).sum(dim=-1, keepdim=True)
    gain = 1. + F.softplus(_grouped_mlp([p.topology_net for p in policies], topology / math.sqrt(113.)))
    return (branch * gain).squeeze(-1).transpose(0, 1)

def grouped_critics(critics, voltage, topology, previous_q, q):
    import math
    features = torch.cat(((voltage - 1.) * 20., topology / math.sqrt(113.), previous_q / 5., q - previous_q), dim=1)
    return (_grouped_mlp([c.q1 for c in critics], features),
            _grouped_mlp([c.q2 for c in critics], features))

def grouped_update(learner, batch, batch_actor=True):
    """Identical TD3 equations; batch the independent actors/critics on the GPU.

    Uses the existing parameter objects, optimizers, schedulers, target networks,
    target noise, loss reductions, clipping, and every-third actor update.
    Restricted to the model123 architecture; not a replacement for 56-bus models.
    """
    v, x, last, q, reward, nv, nx, done = [torch.as_tensor(a, dtype=torch.float32, device=learner.device) for a in batch]
    with torch.no_grad():
        delta = grouped_actor(learner.target, nv, nx) if batch_actor else learner.target(nv, nx)
        delta = delta + (torch.randn_like(q) * .03).clamp(-.05, .05)
        next_q = learner.action(q, delta)
        q1, q2 = grouped_critics(learner.critic_target, nv, nx, q, next_q)
        target = reward.transpose(0, 1).unsqueeze(-1) + .99 * (1-done).unsqueeze(0) * torch.minimum(q1, q2)
    q1, q2 = grouped_critics(learner.critics, v, x, last, q)
    loss = ((q1-target).square().mean(dim=(1,2)) + (q2-target).square().mean(dim=(1,2))).sum()
    learner.critic_opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(learner.critics.parameters(),10.)
    learner.critic_opt.step();learner.critic_scheduler.step();learner.updates+=1
    actor_loss = None
    if learner.updates % 3 == 0:
        learner.critics.requires_grad_(False)
        actual = learner.action(last, grouped_actor(learner.actor, v, x) if batch_actor else learner.actor(v, x))
        policy_loss = -grouped_critics(learner.critics, v, x, last, actual)[0].mean()
        learner.actor_opt.zero_grad(set_to_none=True);policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.actor.parameters(),10.)
        learner.actor_opt.step();learner.actor_scheduler.step();learner.critics.requires_grad_(True)
        with torch.no_grad():
            for online,target_module in ((learner.actor,learner.target),(learner.critics,learner.critic_target)):
                for source,destination in zip(online.parameters(),target_module.parameters()):
                    destination.lerp_(source,.01)
        actor_loss = float(policy_loss.detach().cpu())
    return dict(critic_loss=float(loss.detach().cpu()),actor_loss=actor_loss)
