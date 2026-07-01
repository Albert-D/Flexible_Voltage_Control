"""Evaluate topology-specific training under topology/admittance shifts.

This script keeps the original Environment.py unchanged. It creates a
load-retained evaluation scenario after env.reset_topo(...), so switching
topology changes affect the physical power-flow state. The goal is to compare
a controller trained on one fixed topology against the full topology-aware
RLC-FT policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import torch
from loguru import logger
from numpy import linalg as LA

from config import Config
from Environment import VoltageCtrl_Env, create_56bus
from NN_Module import FlexiblePolicyNet, TopologyNet


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Topology-specific versus topology/admittance-aware 56-bus evaluation."
    )
    parser.add_argument(
        "--specific-tag",
        type=str,
        default="topology_specific_seed0_scratch_seed4_20260617_0353",
    )
    parser.add_argument("--specific-steps", type=int, nargs="+", default=[300, 350])
    parser.add_argument("--specific-seed", type=int, default=4)
    parser.add_argument("--flexible-date", type=str, default="2025-02-18")
    parser.add_argument("--flexible-step", type=int, default=500)
    parser.add_argument("--flexible-seed", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--load-scale", type=float, default=0.7)
    parser.add_argument("--action-scale", type=float, default=0.7)
    parser.add_argument(
        "--admittance-scale",
        type=float,
        default=1.0,
        help="Extra multiplier for active-line admittances after reset_topo.",
    )
    return parser.parse_args()


def make_env() -> VoltageCtrl_Env:
    injection_bus = np.array([18, 21, 30, 45, 53]) - 1
    return VoltageCtrl_Env(create_56bus(), injection_bus)


def make_policy_set(env: VoltageCtrl_Env, checkpoint_paths: list[Path]):
    policies = []
    for path in checkpoint_paths:
        topology_net = TopologyNet(
            topology_dim=env.topology_dim,
            output_dim=1,
            hidden_dim=Config.topology_hidden_dim,
        )
        policy_net = FlexiblePolicyNet(
            env=env,
            topology_net=topology_net,
            obs_dim=Config.state_dim,
            action_dim=Config.action_dim,
            hidden_dim=Config.hidden_dim_56bus,
        ).to(device)
        policy_net.load_state_dict(torch.load(path, map_location=device))
        policy_net.eval()
        policies.append(policy_net)
    return policies


def full_policy_paths(args: argparse.Namespace, env: VoltageCtrl_Env) -> list[Path]:
    root = Path(Config.data_path)
    return [
        root
        / "check_points"
        / "policy_net"
        / args.flexible_date
        / f"Step_{args.flexible_step}_Seed_{args.flexible_seed}_a{i}.pth"
        for i in range(env.agentnum)
    ]


def specific_policy_paths(
    args: argparse.Namespace,
    env: VoltageCtrl_Env,
    specific_step: int,
) -> list[Path]:
    root = Path(Config.data_path)
    return [
        root
        / "check_points"
        / "topology_specific_56bus"
        / args.specific_tag
        / "policy_net"
        / f"Step_{specific_step}_Seed_{args.specific_seed}_a{i}.pth"
        for i in range(env.agentnum)
    ]


def load_fixed_switch(args: argparse.Namespace):
    metadata_path = (
        Path(Config.data_path)
        / "check_points"
        / "topology_specific_56bus"
        / args.specific_tag
        / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fixed_switch = {int(k): bool(v) for k, v in metadata["fixed_switch"].items()}
    return fixed_switch, metadata


def closed_line_mask(env: VoltageCtrl_Env) -> np.ndarray:
    mask = np.ones(len(env.network.line), dtype=bool)
    for _, sw in env.network.switch.iterrows():
        if sw.et == "l" and not bool(sw.closed):
            mask[int(sw.element)] = False
    return mask


def reset_shifted_env(
    env: VoltageCtrl_Env,
    seed: int,
    regime: str,
    fixed_switch: dict[int, bool],
    load_scale: float,
    admittance_scale: float,
):
    if regime == "training_topology":
        state, topology, scenario = env.reset_topo(seed=seed, manual_switch=fixed_switch)
    elif regime == "sampled_topologies":
        state, topology, scenario = env.reset_topo(seed=seed)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    topology_arr = np.asarray(topology, dtype=float).copy()
    if admittance_scale <= 0:
        raise ValueError("admittance_scale must be positive")

    if admittance_scale != 1.0:
        mask = closed_line_mask(env)
        x_current = env.network.line["x_ohm_per_km"].astype(float).to_numpy().copy()
        x_current[mask] = x_current[mask] / admittance_scale
        env.network.line.loc[:, "x_ohm_per_km"] = x_current
        topology_arr[mask] = topology_arr[mask] * admittance_scale

    env.network.load["p_mw"] = env.load0_p * load_scale
    env.network.load["q_mvar"] = env.load0_q * load_scale
    pp.runpp(env.network, algorithm="bfsw")
    env.state = env.network.res_bus.iloc[env.injection_bus].vm_pu.to_numpy()
    env.topology = topology_arr
    return env.state, topology_arr, scenario


def policy_action(policy_nets, state, topology_tensor, last_action, action_scale):
    raw_action = []
    with torch.no_grad():
        for i, policy_net in enumerate(policy_nets):
            state_i = torch.tensor(
                state[i].reshape(1,),
                device=device,
                dtype=torch.float32,
            ).unsqueeze(0)
            action_i = policy_net(state_i, topology_tensor)
            raw_action.append(action_i.detach().cpu().numpy()[0])
    return last_action - action_scale * np.asarray(raw_action)


def linear_action(env: VoltageCtrl_Env, state, last_action):
    state1 = np.asarray(state - env.vmax)
    state2 = np.asarray(env.vmin - state)
    d_v = (np.maximum(state1, 0) - np.maximum(state2, 0)).reshape((env.agentnum, 1))
    return last_action - 10 * d_v


def run_episode(
    env: VoltageCtrl_Env,
    method: str,
    policy_nets,
    seed: int,
    regime: str,
    fixed_switch: dict[int, bool],
    fixed_topology_input: np.ndarray,
    max_steps: int,
    load_scale: float,
    admittance_scale: float,
    action_scale: float,
):
    try:
        state, topology, scenario = reset_shifted_env(
            env,
            seed,
            regime,
            fixed_switch,
            load_scale,
            admittance_scale,
        )
    except Exception as exc:
        return {
            "regime": regime,
            "method": method,
            "seed": seed,
            "scenario": np.nan,
            "status": "initial_powerflow_fail",
            "recovery_time": max_steps,
            "control_cost": np.nan,
            "objective_cost": np.nan,
            "message": repr(exc),
        }

    if method == "Topology-specific fixed-X":
        policy_topology = fixed_topology_input
    else:
        policy_topology = topology
    topology_tensor = torch.tensor(policy_topology, dtype=torch.float32, device=device).unsqueeze(0)

    last_action = np.zeros((env.agentnum, 1))
    objective_cost = 0.0
    control_cost = 0.0
    status = "horizon_reached"
    message = ""

    for step in range(max_steps):
        if method == "Linear":
            action = linear_action(env, state, last_action)
        else:
            action = policy_action(policy_nets, state, topology_tensor, last_action, action_scale)
        last_action = np.copy(action)

        try:
            next_state, reward, done = env.step(action)
        except pp.powerflow.LoadflowNotConverged as exc:
            status = "loadflow_failed"
            message = repr(exc)
            break

        if np.min(next_state) < 0.70 or np.max(next_state) > 1.30:
            status = "guardrail"
            message = f"voltage guardrail: min={np.min(next_state):.3f}, max={np.max(next_state):.3f}"
            break

        objective_cost += -reward
        control_cost += LA.norm(action, 2)

        if done:
            status = "recovered"
            break

        state = next_state

    recovery_time = step if status == "recovered" else max_steps
    return {
        "regime": regime,
        "method": method,
        "seed": seed,
        "scenario": int(scenario),
        "status": status,
        "recovery_time": float(recovery_time),
        "control_cost": float(control_cost),
        "objective_cost": float(objective_cost),
        "message": message,
    }


def q25(x):
    return np.nanquantile(x, 0.25)


def q75(x):
    return np.nanquantile(x, 0.75)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["specific_step", "regime", "method"], as_index=False)
        .agg(
            recovery_median=("recovery_time", "median"),
            recovery_q25=("recovery_time", q25),
            recovery_q75=("recovery_time", q75),
            control_median=("control_cost", "median"),
            control_q25=("control_cost", q25),
            control_q75=("control_cost", q75),
            objective_median=("objective_cost", "median"),
            objective_q25=("objective_cost", q25),
            objective_q75=("objective_cost", q75),
            recovered_within_horizon=("status", lambda s: float(np.mean(s == "recovered"))),
            abnormal_stop_rate=("status", lambda s: float(np.mean(~s.isin(["recovered", "horizon_reached"])))),
            n=("seed", "count"),
        )
    )


def main():
    args = parse_args()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Using {} device", device)

    fixed_switch, metadata = load_fixed_switch(args)
    env = make_env()
    full_policies = make_policy_set(env, full_policy_paths(args, env))
    _, fixed_topology_input, _ = reset_shifted_env(
        env,
        metadata.get("actual_topology_seed", 0),
        "training_topology",
        fixed_switch,
        args.load_scale,
        args.admittance_scale,
    )
    fixed_topology_input = np.asarray(fixed_topology_input, dtype=float).copy()

    records = []
    for specific_step in args.specific_steps:
        env = make_env()
        specific_policies = make_policy_set(env, specific_policy_paths(args, env, specific_step))
        controllers = {
            "Full RLC-FT": full_policies,
            "Topology-specific current-X": specific_policies,
            "Topology-specific fixed-X": specific_policies,
            "Linear": None,
        }
        for regime in ["training_topology", "sampled_topologies"]:
            for method, policies in controllers.items():
                for seed in range(args.episodes):
                    row = run_episode(
                        env,
                        method,
                        policies,
                        seed,
                        regime,
                        fixed_switch,
                        fixed_topology_input,
                        args.steps,
                        args.load_scale,
                        args.admittance_scale,
                        args.action_scale,
                    )
                    row["specific_step"] = specific_step
                    row["load_scale"] = args.load_scale
                    row["admittance_scale"] = args.admittance_scale
                    records.append(row)
                logger.info(
                    "step={} regime={} method={} complete ({})",
                    specific_step,
                    regime,
                    method,
                    args.episodes,
                )

    df = pd.DataFrame(records)
    summary = summarize(df)
    output_dir = Path(Config.data_path) / "images" / "56bus" / "topology_admittance_shift"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"load{args.load_scale:g}_adm{args.admittance_scale:g}_{args.episodes}"
    raw_path = output_dir / f"topology_admittance_shift_raw_{suffix}.csv"
    summary_path = output_dir / f"topology_admittance_shift_summary_{suffix}.csv"
    metadata_path = output_dir / f"topology_admittance_shift_metadata_{suffix}.json"
    df.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(
        json.dumps({"args": vars(args), "training_metadata": metadata}, indent=2),
        encoding="utf-8",
    )

    logger.info("Saved raw results to {}", raw_path)
    logger.info("Saved summary to {}", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
