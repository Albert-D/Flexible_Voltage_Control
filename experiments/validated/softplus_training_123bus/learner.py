"""Centralized TD3 updates and checkpoint state for the 123-bus learner."""
import torch
from torch import nn
from torch.nn import functional as F


class CentralizedLearner:
    def update(self, batch):
            voltage, topology, previous_q, q, reward, next_voltage, next_topology, done = [
                torch.as_tensor(value, dtype=torch.float32, device=self.device)
                for value in batch
            ]
            with torch.no_grad():
                noisy_delta = self.target(next_voltage, next_topology)
                noisy_delta = noisy_delta + (
                    torch.randn_like(q) * .03
                ).clamp(-.05, .05)
                next_q = self.action(q, noisy_delta)
                targets = []
                for index, critic in enumerate(self.critic_target):
                    q1, q2 = critic(next_voltage, next_topology, q, next_q)
                    targets.append(
                        reward[:, index:index + 1]
                        + .99 * (1 - done) * torch.minimum(q1, q2)
                    )

            critic_loss = 0.
            for index, critic in enumerate(self.critics):
                q1, q2 = critic(voltage, topology, previous_q, q)
                critic_loss = (
                    critic_loss
                    + F.mse_loss(q1, targets[index])
                    + F.mse_loss(q2, targets[index])
                )
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critics.parameters(), 10.)
            self.critic_opt.step()
            self.critic_scheduler.step()
            self.updates += 1

            actor_loss = None
            if self.updates % 3 == 0:
                self.critics.requires_grad_(False)
                actual = self.action(previous_q, self.actor(voltage, topology))
                policy_terms = [
                    -critic(voltage, topology, previous_q, actual)[0].mean()
                    for critic in self.critics
                ]
                policy_loss = torch.stack(policy_terms).mean()
                self.actor_opt.zero_grad(set_to_none=True)
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 10.)
                self.actor_opt.step()
                self.actor_scheduler.step()
                self.critics.requires_grad_(True)
                with torch.no_grad():
                    for online, target in (
                        (self.actor, self.target),
                        (self.critics, self.critic_target),
                    ):
                        for source, destination in zip(
                            online.parameters(), target.parameters()
                        ):
                            destination.lerp_(source, .01)
                actor_loss = float(policy_loss.detach().cpu())
            return {
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": actor_loss,
            }

    def state(self):
            return {
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
            for key in (
                "actor", "target", "critics", "critic_target",
                "actor_opt", "critic_opt", "actor_scheduler", "critic_scheduler",
            ):
                getattr(self, key).load_state_dict(state[key])
            self.updates = int(state["updates"])
