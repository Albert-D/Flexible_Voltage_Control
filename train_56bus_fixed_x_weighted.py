"""Train a fixed-X 56-bus controller with calibrated action-cost weights.

This wrapper leaves the original environment and fixed-X training script
unchanged. It overrides the reward weights used by Environment.step_uncertain
before calling train_56bus_fixed_x.train(...), so we can discourage overly
aggressive fixed-X policies and select a controller with recovery time closer
to the full RLC-FT baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import Environment as env_module
from config import Config
from train_56bus_fixed_x import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train calibrated fixed-X 56-bus controller."
    )
    parser.add_argument("--episodes", type=int, default=Config.total_episodes)
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--train-seed", type=int, default=4)
    parser.add_argument("--topology-seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--local-action-cost", type=float, default=20.0)
    parser.add_argument("--global-action-cost", type=float, default=0.0)
    parser.add_argument("--local-voltage-cost", type=float, default=50.0)
    parser.add_argument("--global-voltage-cost", type=float, default=50.0)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env_module.cost_l_a_56bus = -abs(args.local_action_cost)
    env_module.cost_w_a_56bus = -abs(args.global_action_cost)
    env_module.cost_l_v_56bus = -abs(args.local_voltage_cost)
    env_module.cost_w_v_56bus = -abs(args.global_voltage_cost)

    if args.tag is None:
        args.tag = (
            "fixed_x_calibrated_"
            f"la{args.local_action_cost:g}_"
            f"ga{args.global_action_cost:g}_"
            f"seed{args.train_seed}"
        )

    output_dir = train(args)
    metadata_path = Path(output_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["reward_weight_override"] = {
        "cost_l_a_56bus": env_module.cost_l_a_56bus,
        "cost_w_a_56bus": env_module.cost_w_a_56bus,
        "cost_l_v_56bus": env_module.cost_l_v_56bus,
        "cost_w_v_56bus": env_module.cost_w_v_56bus,
    }
    metadata["calibration_note"] = (
        "This run increases the action penalty to avoid selecting an "
        "overly aggressive fixed-X controller. Checkpoints should be selected "
        "using recovery-time matching, not only objective minimization."
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
