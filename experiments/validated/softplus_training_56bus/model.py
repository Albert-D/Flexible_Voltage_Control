"""Canonical Softplus actor with the established centralized TD3 learner."""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

from src.softplus_controller import SoftplusControllerSpec, SoftplusTopologyController


NUM_AGENTS = 5
TOPOLOGY_DIM = 55
ACTION_STEP_LIMIT = 2.5
ACTION_LIMIT = 25.0


def initialize_weak_actor(actor: SoftplusTopologyController, factor: float = 0.05) -> None:
    """Lower the stored slopes without adding an inference-time scale factor."""
    if not 0 < factor <= 1:
        raise ValueError("initial slope factor must lie in (0, 1]")
    spec = actor.spec
    with torch.no_grad():
        for policy in actor.policies:
            slopes = (
                spec.slope_min
                + (spec.slope_max - spec.slope_min) * torch.sigmoid(policy.q_raw)
            ) * factor
            unit = (slopes - spec.slope_min) / (spec.slope_max - spec.slope_min)
            if not bool(((unit > 0) & (unit < 1)).all()):
                raise ValueError("weak initialization falls outside slope bounds")
            policy.q_raw.copy_(torch.logit(unit))


class DeltaCentralCritic(nn.Module):
    def __init__(self, action_delta_scale: float = 1.0):
        super().__init__()
        if action_delta_scale <= 0:
            raise ValueError("action_delta_scale must be positive")
        self.action_delta_scale = float(action_delta_scale)
        width = NUM_AGENTS + TOPOLOGY_DIM + NUM_AGENTS + NUM_AGENTS
        self.q1 = nn.Sequential(
            nn.Linear(width, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = copy.deepcopy(self.q1)
        nn.init.uniform_(self.q2[-1].weight, -0.003, 0.003)

    def forward(self, voltage, topology, previous_q, current_q):
        action_delta = (current_q - previous_q) / self.action_delta_scale
        features = torch.cat(
            (
                (voltage - 1.0) * 20.0,
                topology / (TOPOLOGY_DIM ** 0.5),
                previous_q / 5.0,
                action_delta,
            ),
            dim=1,
        )
        return self.q1(features), self.q2(features)


class Learner:
    def __init__(
        self,
        device: torch.device,
        seed: int = 2601,
        actor_lr: float = 2.5e-5,
        critic_lr: float = 1e-3,
        initial_slope_factor: float = 0.05,
        hidden: int = 2048,
        topology_hidden: int = 256,
        action_delta_scale: float = 1.0,
        discount: float = 0.99,
        critic_init_seed: int | None = None,
        critic_output_init: str = "random",
    ):
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must lie in (0, 1]")
        if critic_output_init not in ("random", "zero"):
            raise ValueError("critic_output_init must be random or zero")
        self.critic_output_init = critic_output_init
        torch.manual_seed(seed)
        self.device = device
        self.discount = float(discount)
        self.spec = SoftplusControllerSpec(
            topology_dim=TOPOLOGY_DIM,
            num_agents=NUM_AGENTS,
            hidden=hidden,
            topology_hidden=topology_hidden,
        )
        self.actor = SoftplusTopologyController(self.spec).to(device)
        initialize_weak_actor(self.actor, initial_slope_factor)
        self.target = copy.deepcopy(self.actor).requires_grad_(False)
        if critic_init_seed is not None:
            cpu_rng_state = torch.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            torch.manual_seed(critic_init_seed)
        self.critics = nn.ModuleList(
            [DeltaCentralCritic(action_delta_scale) for _ in range(NUM_AGENTS)]
        ).to(device)
        if critic_output_init == "zero":
            # Zero value and action gradient without disabling actor optimization.
            # Apply the same initialization to every reward arm in a matched cohort.
            with torch.no_grad():
                for critic in self.critics:
                    for head in (critic.q1[-1], critic.q2[-1]):
                        head.weight.zero_()
                        head.bias.zero_()
        self.critic_target = copy.deepcopy(self.critics).requires_grad_(False)
        if critic_init_seed is not None:
            torch.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critics.parameters(), lr=critic_lr)
        self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.actor_opt, [10000, 20000], gamma=0.5
        )
        self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.critic_opt, [20000, 40000], gamma=0.5
        )
        self.updates = 0

    @staticmethod
    def action(previous, delta):
        return (
            previous - delta.clamp(-ACTION_STEP_LIMIT, ACTION_STEP_LIMIT)
        ).clamp(-ACTION_LIMIT, ACTION_LIMIT)

    def update(self, batch):
        voltage, topology, previous_q, q, reward, next_voltage, next_topology, done = [
            torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for value in batch
        ]
        with torch.no_grad():
            noisy_delta = self.target(next_voltage, next_topology)
            noisy_delta = noisy_delta + (torch.randn_like(q) * 0.03).clamp(-0.05, 0.05)
            next_q = self.action(q, noisy_delta)
            targets = []
            for index, critic in enumerate(self.critic_target):
                q1, q2 = critic(next_voltage, next_topology, q, next_q)
                targets.append(
                    reward[:, index:index + 1]
                    + self.discount * (1 - done) * torch.minimum(q1, q2)
                )

        critic_loss = 0.0
        for index, critic in enumerate(self.critics):
            q1, q2 = critic(voltage, topology, previous_q, q)
            critic_loss = (
                critic_loss
                + F.mse_loss(q1, targets[index])
                + F.mse_loss(q2, targets[index])
            )
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critics.parameters(), 10.0)
        self.critic_opt.step()
        self.critic_scheduler.step()
        self.updates += 1

        actor_loss = None
        if self.updates % 3 == 0:
            self.critics.requires_grad_(False)
            actual = self.action(previous_q, self.actor(voltage, topology))
            terms = [
                -critic(voltage, topology, previous_q, actual)[0].mean()
                for critic in self.critics
            ]
            policy_loss = torch.stack(terms).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()
            self.actor_scheduler.step()
            self.critics.requires_grad_(True)
            with torch.no_grad():
                for online, target in (
                    (self.actor, self.target),
                    (self.critics, self.critic_target),
                ):
                    for source, destination in zip(online.parameters(), target.parameters()):
                        destination.lerp_(source, 0.01)
            actor_loss = float(policy_loss.detach().cpu())
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": actor_loss,
        }

    def state(self):
        return {
            "model_config": self.actor.model_config(),
            "critic_output_init": self.critic_output_init,
            "actor": self.actor.state_dict(),
            "target": self.target.state_dict(),
            "critics": self.critics.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "actor_scheduler": self.actor_scheduler.state_dict(),
            "critic_scheduler": self.critic_scheduler.state_dict(),
            "updates": self.updates,
        }

    def load(self, state):
        if state.get("critic_output_init", "random") != self.critic_output_init:
            raise ValueError("checkpoint critic initialization does not match learner")
        if state.get("model_config") not in (None, self.actor.model_config()):
            raise ValueError("checkpoint model config does not match canonical actor")
        for key in (
            "actor", "target", "critics", "critic_target",
            "actor_opt", "critic_opt", "actor_scheduler", "critic_scheduler",
        ):
            getattr(self, key).load_state_dict(state[key])
        self.updates = int(state["updates"])


__all__ = [
    "ACTION_LIMIT", "ACTION_STEP_LIMIT", "Learner", "NUM_AGENTS",
    "TOPOLOGY_DIM", "initialize_weak_actor",
]
