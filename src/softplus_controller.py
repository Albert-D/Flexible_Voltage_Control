"""Canonical uncapped Softplus topology-conditioned RLC-FT actor.

This module contains model structure only. Environments, rewards, critics,
optimizers and training schedules remain in their experiment packages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


SOFTPLUS_MODEL_VERSION = "softplus_topology_v1"


@dataclass(frozen=True)
class SoftplusControllerSpec:
    """Serializable architecture parameters shared across feeder sizes."""

    topology_dim: int
    num_agents: int
    hidden: int
    topology_hidden: int = 256
    gain_init_bias: float = 0.5
    slope_min: float = 0.01
    slope_max: float = 1.0
    initial_slope_low: float = 0.4
    initial_slope_high: float = 0.6

    def __post_init__(self) -> None:
        if self.topology_dim <= 0 or self.num_agents <= 0:
            raise ValueError("topology_dim and num_agents must be positive")
        if self.hidden <= 0 or self.topology_hidden <= 0:
            raise ValueError("hidden dimensions must be positive")
        if not self.slope_min < self.slope_max:
            raise ValueError("slope_min must be smaller than slope_max")
        if not self.slope_min < self.initial_slope_low:
            raise ValueError("initial_slope_low must exceed slope_min")
        if not self.initial_slope_low < self.initial_slope_high < self.slope_max:
            raise ValueError("initial slopes must lie inside the slope bounds")

    def as_config(self) -> dict[str, int | float | str]:
        return {"model_version": SOFTPLUS_MODEL_VERSION, **asdict(self)}


def voltage_branch(
    voltage: torch.Tensor,
    slopes: torch.Tensor,
    knots: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the symmetric monotone piecewise-linear voltage response."""

    weights = torch.cat(
        (slopes[..., :1], slopes[..., 1:] - slopes[..., :-1]), dim=-1
    )
    knot_locations = torch.cumsum(knots, dim=-1) - knots
    return (
        (
            F.relu(voltage - 1.045 - knot_locations)
            - F.relu(0.955 - voltage - knot_locations)
        )
        * weights
    ).sum(-1, keepdim=True)


class SoftplusTopologyPolicy(nn.Module):
    """One local voltage policy conditioned on the shared topology vector."""

    def __init__(self, spec: SoftplusControllerSpec):
        super().__init__()
        self.spec = spec
        self.b_raw = nn.Parameter(torch.log(torch.rand(spec.hidden).clamp_min(1e-6)))
        initial_slopes = (
            torch.rand(1, spec.hidden)
            * (spec.initial_slope_high - spec.initial_slope_low)
            + spec.initial_slope_low
        )
        slope_unit = (initial_slopes - spec.slope_min) / (
            spec.slope_max - spec.slope_min
        )
        self.q_raw = nn.Parameter(torch.logit(slope_unit))
        self.topology_net = nn.Sequential(
            nn.Linear(spec.topology_dim, spec.topology_hidden),
            nn.ReLU(),
            nn.Linear(spec.topology_hidden, spec.topology_hidden),
            nn.ReLU(),
            nn.Linear(spec.topology_hidden, 1),
        )
        for layer in self.topology_net:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.03, 0.03)
                nn.init.zeros_(layer.bias)
        nn.init.constant_(self.topology_net[-1].bias, spec.gain_init_bias)

    def topology_gain(self, topology: torch.Tensor) -> torch.Tensor:
        if topology.shape[-1] != self.spec.topology_dim:
            raise ValueError(
                f"expected topology_dim={self.spec.topology_dim}, "
                f"received {topology.shape[-1]}"
            )
        scaled = topology / math.sqrt(self.spec.topology_dim)
        return 1.0 + F.softplus(self.topology_net(scaled))

    def forward(
        self,
        voltage: torch.Tensor,
        topology: torch.Tensor,
    ) -> torch.Tensor:
        slopes = self.spec.slope_min + (
            self.spec.slope_max - self.spec.slope_min
        ) * torch.sigmoid(self.q_raw)
        knots = torch.softmax(self.b_raw, dim=-1)
        return voltage_branch(voltage, slopes, knots) * self.topology_gain(topology)


class SoftplusTopologyController(nn.Module):
    """Multi-agent controller with one policy per controlled voltage."""

    def __init__(self, spec: SoftplusControllerSpec):
        super().__init__()
        self.spec = spec
        self.policies = nn.ModuleList(
            [SoftplusTopologyPolicy(spec) for _ in range(spec.num_agents)]
        )

    def model_config(self) -> dict[str, int | float | str]:
        return self.spec.as_config()

    def forward(
        self,
        voltage: torch.Tensor,
        topology: torch.Tensor,
    ) -> torch.Tensor:
        if voltage.shape[-1] != self.spec.num_agents:
            raise ValueError(
                f"expected num_agents={self.spec.num_agents}, "
                f"received {voltage.shape[-1]} voltages"
            )
        return torch.cat(
            [
                policy(voltage[:, index : index + 1], topology)
                for index, policy in enumerate(self.policies)
            ],
            dim=1,
        )


__all__ = [
    "SOFTPLUS_MODEL_VERSION",
    "SoftplusControllerSpec",
    "SoftplusTopologyController",
    "SoftplusTopologyPolicy",
    "voltage_branch",
]
