"""Evaluate a fixed-X controller against flexible RLC-FT.

Checkpoint selection uses only held-out operating points on the fixed training
network. The selected controller is then compared on (i) the same physical
network and (ii) sampled topology/admittance shifts. Both regimes use paired PV
operating points and retained base loads; only the physical network changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandapower as pp
import torch
from numpy import linalg as LA

from config import Config
from NN_Module import FlexiblePolicyNet, TopologyNet
from test_56bus_topology_admittance_shift import (
    closed_line_mask,
    full_policy_paths,
    make_env,
)
from train_56bus_fixed_x import reset_fixed_x


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and evaluate a truly fixed-X 56-bus controller."
    )
    parser.add_argument("--fixed-tag", type=str, required=True)
    parser.add_argument("--fixed-seed", type=int, default=4)
    parser.add_argument("--flexible-date", type=str, default="2025-02-18")
    parser.add_argument("--flexible-step", type=int, default=500)
    parser.add_argument("--flexible-seed", type=int, default=4)
    parser.add_argument("--validation-episodes", type=int, default=30)
    parser.add_argument("--validation-seed-start", type=int, default=1000)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--test-seed-start", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--admittance-scale", type=float, default=1.2)
    parser.add_argument("--action-scale", type=float, default=0.7)
    parser.add_argument(
        "--selected-step",
        type=int,
        default=None,
        help="Skip validation and evaluate this preselected fixed-X checkpoint.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["objective", "target-recovery"],
        default="objective",
        help=(
            "How to select the fixed-X checkpoint from validation episodes. "
            "'objective' keeps the original best-objective rule; "
            "'target-recovery' selects a checkpoint whose recovery time is "
            "closest to --target-recovery on the fixed training network."
        ),
    )
    parser.add_argument(
        "--target-recovery",
        type=float,
        default=9.0,
        help="Target median recovery time used when --selection-mode=target-recovery.",
    )
    parser.add_argument("--output-suffix", type=str, default=None)
    return parser.parse_args()


def load_metadata(args: argparse.Namespace) -> dict:
    path = (
        Path(Config.data_path)
        / "check_points"
        / "topology_specific_fixed_x_56bus"
        / args.fixed_tag
        / "metadata.json"
    )
    return json.loads(path.read_text(encoding="utf-8"))


def make_policy_set(env, checkpoint_paths: list[Path]):
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


def fixed_policy_paths(
    args: argparse.Namespace,
    env,
    step: int,
) -> list[Path]:
    root = (
        Path(Config.data_path)
        / "check_points"
        / "topology_specific_fixed_x_56bus"
        / args.fixed_tag
        / "policy_net"
    )
    return [
        root / f"Step_{step}_Seed_{args.fixed_seed}_a{i}.pth"
        for i in range(env.agentnum)
    ]


def available_fixed_steps(args: argparse.Namespace, env) -> list[int]:
    root = (
        Path(Config.data_path)
        / "check_points"
        / "topology_specific_fixed_x_56bus"
        / args.fixed_tag
        / "policy_net"
    )
    candidates = set()
    for path in root.glob(f"Step_*_Seed_{args.fixed_seed}_a0.pth"):
        candidates.add(int(path.name.split("_")[1]))
    return [
        step
        for step in sorted(candidates)
        if step > 0
        if all(path.exists() for path in fixed_policy_paths(args, env, step))
    ]


def set_paired_operating_point(env, seed: int, retain_loads: bool) -> int:
    rng = np.random.RandomState(seed)
    scenario = int(rng.choice([0, 1]))

    env.network.sgen.loc[:, "p_mw"] = 0.0
    env.network.sgen.loc[:, "q_mvar"] = 0.0
    if retain_loads:
        env.network.load.loc[:, "p_mw"] = env.load0_p
        env.network.load.loc[:, "q_mvar"] = env.load0_q
    else:
        env.network.load.loc[:, "p_mw"] = 0.0
        env.network.load.loc[:, "q_mvar"] = 0.0

    if scenario == 0:
        env.network.sgen.at[1, "p_mw"] = -0.5 * rng.uniform(2, 5)
        env.network.sgen.at[2, "p_mw"] = -0.4 * rng.uniform(10, 30)
        env.network.sgen.at[3, "p_mw"] = -0.3 * rng.uniform(2, 8)
        env.network.sgen.at[4, "p_mw"] = -0.3 * rng.uniform(2, 8)
        env.network.sgen.at[5, "p_mw"] = -0.4 * rng.uniform(2, 8)
    else:
        env.network.sgen.at[1, "p_mw"] = 0.5 * rng.uniform(2, 10)
        env.network.sgen.at[2, "p_mw"] = rng.uniform(5, 40)
        env.network.sgen.at[3, "p_mw"] = 0.2 * rng.uniform(2, 14)
        env.network.sgen.at[4, "p_mw"] = 0.4 * rng.uniform(2, 14)
        env.network.sgen.at[5, "p_mw"] = 0.4 * rng.uniform(2, 14)
    return scenario


def reset_reference_network(
    env,
    seed: int,
    metadata: dict,
    retain_loads: bool,
):
    fixed_switch = {int(k): bool(v) for k, v in metadata["fixed_switch"].items()}
    fixed_line_x = np.asarray(metadata["fixed_line_x_ohm_per_km"], dtype=float)
    fixed_x = np.asarray(metadata["fixed_topology_input"], dtype=float)

    # reset_fixed_x establishes the exact network; the operating point is then
    # reapplied so validation and test can choose whether to retain base loads.
    reset_fixed_x(env, seed, fixed_switch, fixed_line_x, fixed_x)
    scenario = set_paired_operating_point(env, seed, retain_loads)
    pp.runpp(env.network, algorithm="bfsw")
    env.state = env.network.res_bus.iloc[env.injection_bus].vm_pu.to_numpy()
    env.topology = fixed_x.copy()
    return env.state.copy(), fixed_x.copy(), scenario


def reset_shifted_network(
    env,
    seed: int,
    admittance_scale: float,
    max_tries: int = 50,
):
    """Sample a feasible changed network, then apply the paired operating point."""
    for offset in range(max_tries):
        network_seed = seed + 10_000 + offset
        try:
            _, topology, _ = env.reset_topo(seed=network_seed)
        except pp.powerflow.LoadflowNotConverged:
            continue

        topology = np.asarray(topology, dtype=float).copy()
        if admittance_scale != 1.0:
            mask = closed_line_mask(env)
            line_x = env.network.line["x_ohm_per_km"].astype(float).to_numpy().copy()
            line_x[mask] = line_x[mask] / admittance_scale
            env.network.line.loc[:, "x_ohm_per_km"] = line_x
            topology[mask] = topology[mask] * admittance_scale

        scenario = set_paired_operating_point(env, seed, retain_loads=True)
        try:
            pp.runpp(env.network, algorithm="bfsw")
        except pp.powerflow.LoadflowNotConverged:
            continue
        env.state = env.network.res_bus.iloc[env.injection_bus].vm_pu.to_numpy()
        env.topology = topology.copy()
        return env.state.copy(), topology, scenario
    raise pp.powerflow.LoadflowNotConverged(
        f"No feasible shifted network found for paired seed {seed}."
    )


def policy_action(policy_nets, state, topology, last_action, action_scale):
    topology_tensor = torch.tensor(
        topology,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    raw_action = []
    with torch.no_grad():
        for i, policy_net in enumerate(policy_nets):
            state_i = torch.tensor(
                state[i].reshape(1,),
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            raw_action.append(
                policy_net(state_i, topology_tensor).detach().cpu().numpy()[0]
            )
    return last_action - action_scale * np.asarray(raw_action)


def linear_action(env, state, last_action):
    high = np.asarray(state - env.vmax)
    low = np.asarray(env.vmin - state)
    delta_v = (np.maximum(high, 0) - np.maximum(low, 0)).reshape(
        (env.agentnum, 1)
    )
    return last_action - 10 * delta_v


def run_episode(
    env,
    method: str,
    policy_nets,
    seed: int,
    regime: str,
    metadata: dict,
    max_steps: int,
    admittance_scale: float,
    action_scale: float,
    retain_loads: bool = True,
):
    try:
        if regime == "training_network":
            state, current_x, scenario = reset_reference_network(
                env,
                seed,
                metadata,
                retain_loads=retain_loads,
            )
        elif regime == "flexible_shifts":
            state, current_x, scenario = reset_shifted_network(
                env,
                seed,
                admittance_scale,
            )
        else:
            raise ValueError(regime)
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

    if method == "Topology-specific RLC-FT":
        policy_x = np.asarray(metadata["fixed_topology_input"], dtype=float)
    else:
        policy_x = current_x

    last_action = np.zeros((env.agentnum, 1))
    objective_cost = 0.0
    control_cost = 0.0
    status = "horizon_reached"
    message = ""

    for step in range(max_steps):
        if method == "Linear":
            action = linear_action(env, state, last_action)
        else:
            action = policy_action(
                policy_nets,
                state,
                policy_x,
                last_action,
                action_scale,
            )
        last_action = np.copy(action)

        try:
            next_state, reward, done = env.step(action)
        except pp.powerflow.LoadflowNotConverged as exc:
            status = "loadflow_failed"
            message = repr(exc)
            break

        objective_cost += -reward
        control_cost += LA.norm(action, 2)

        if np.min(next_state) < 0.70 or np.max(next_state) > 1.30:
            status = "guardrail"
            message = (
                f"voltage guardrail: min={np.min(next_state):.3f}, "
                f"max={np.max(next_state):.3f}"
            )
            break
        if done:
            status = "recovered"
            break
        state = next_state

    return {
        "regime": regime,
        "method": method,
        "seed": seed,
        "scenario": int(scenario),
        "status": status,
        "recovery_time": float(step if status == "recovered" else max_steps),
        "control_cost": float(control_cost),
        "objective_cost": float(objective_cost),
        "message": message,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    def q25(values):
        return np.nanquantile(values, 0.25)

    def q75(values):
        return np.nanquantile(values, 0.75)

    return (
        df.groupby(["regime", "method"], as_index=False)
        .agg(
            recovery_median=("recovery_time", "median"),
            recovery_q25=("recovery_time", q25),
            recovery_q75=("recovery_time", q75),
            objective_median=("objective_cost", "median"),
            objective_q25=("objective_cost", q25),
            objective_q75=("objective_cost", q75),
            recovered_rate=("status", lambda x: float(np.mean(x == "recovered"))),
            n=("seed", "count"),
        )
    )


def paired_ratio_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["training_network", "flexible_shifts"]:
        sub = df[df["regime"] == regime]
        for metric in ["recovery_time", "objective_cost"]:
            pivot = sub.pivot(index="seed", columns="method", values=metric)
            for method in ["RLC-FT", "Topology-specific RLC-FT", "Linear"]:
                if method == "RLC-FT":
                    ratios = pd.Series(1.0, index=pivot.index)
                else:
                    ratios = (
                        pivot[method] / pivot["RLC-FT"]
                    ).replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "regime": regime,
                        "metric": metric,
                        "method": method,
                        "ratio_median": ratios.median(),
                        "ratio_q25": ratios.quantile(0.25),
                        "ratio_q75": ratios.quantile(0.75),
                        "n": len(ratios),
                    }
                )
    return pd.DataFrame(rows)


def select_checkpoint(args: argparse.Namespace, metadata: dict, output_dir: Path):
    env = make_env()
    steps = available_fixed_steps(args, env)
    if not steps:
        raise FileNotFoundError("No complete fixed-X checkpoints found.")

    rows = []
    seeds = range(
        args.validation_seed_start,
        args.validation_seed_start + args.validation_episodes,
    )
    for step in steps:
        policies = make_policy_set(env, fixed_policy_paths(args, env, step))
        for seed in seeds:
            row = run_episode(
                env=make_env(),
                method="Topology-specific RLC-FT",
                policy_nets=policies,
                seed=seed,
                regime="training_network",
                metadata=metadata,
                max_steps=args.steps,
                admittance_scale=args.admittance_scale,
                action_scale=args.action_scale,
                retain_loads=False,
            )
            row["checkpoint_step"] = step
            rows.append(row)
        print(f"validated fixed-X checkpoint {step}")

    validation = pd.DataFrame(rows)
    ranking = (
        validation.groupby("checkpoint_step", as_index=False)
        .agg(
            recovered_rate=("status", lambda x: float(np.mean(x == "recovered"))),
            objective_median=("objective_cost", "median"),
            recovery_median=("recovery_time", "median"),
        )
    )
    if args.selection_mode == "target-recovery":
        ranking["recovery_gap"] = (
            ranking["recovery_median"] - args.target_recovery
        ).abs()
        ranking = ranking.sort_values(
            ["recovered_rate", "recovery_gap", "objective_median"],
            ascending=[False, True, True],
        )
    else:
        ranking["recovery_gap"] = (
            ranking["recovery_median"] - args.target_recovery
        ).abs()
        ranking = ranking.sort_values(
            ["recovered_rate", "objective_median", "recovery_median"],
            ascending=[False, True, True],
        )
    ranking = ranking.reset_index(drop=True)
    validation.to_csv(output_dir / "fixed_x_checkpoint_validation_raw.csv", index=False)
    ranking.to_csv(output_dir / "fixed_x_checkpoint_ranking.csv", index=False)
    return int(ranking.iloc[0]["checkpoint_step"]), ranking


def make_plots(
    summary: pd.DataFrame,
    ratio_summary: pd.DataFrame,
    output_dir: Path,
):
    method_order = ["RLC-FT", "Topology-specific RLC-FT", "Linear"]
    method_labels = ["RLC-FT", "Topology-specific\nRLC-FT", "Linear"]
    regimes = ["training_network", "flexible_shifts"]
    regime_labels = {
        "training_network": "Training network",
        "flexible_shifts": "Flexible topology/admittance shifts",
    }
    colors = {
        "RLC-FT": "#2C7FB8",
        "Topology-specific RLC-FT": "#F28E2B",
        "Linear": "#8A8A8A",
    }

    def finish(ax):
        ax.grid(axis="y", alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.10,
        top=0.86,
        hspace=0.62,
        wspace=0.32,
    )
    for row, regime in enumerate(regimes):
        fig.text(
            0.515,
            0.905 if row == 0 else 0.462,
            regime_labels[regime],
            ha="center",
            fontsize=13,
            fontweight="semibold",
        )
        sub = summary[summary["regime"] == regime].set_index("method").loc[
            method_order
        ]
        for col, (metric, ylabel) in enumerate(
            [
                ("recovery", "Voltage recovery time (steps)"),
                ("objective", "Objective cost"),
            ]
        ):
            ax = axes[row, col]
            med = sub[f"{metric}_median"].to_numpy()
            q25 = sub[f"{metric}_q25"].to_numpy()
            q75 = sub[f"{metric}_q75"].to_numpy()
            x = np.arange(len(method_order))
            ax.bar(
                x,
                med,
                yerr=np.vstack([med - q25, q75 - med]),
                capsize=4,
                color=[colors[m] for m in method_order],
                edgecolor="black",
                linewidth=0.65,
            )
            ax.set_xticks(x, method_labels)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(q75) * 1.22)
            for xi, value, upper in zip(x, med, q75):
                ax.text(
                    xi,
                    upper + max(q75) * 0.035,
                    f"{value:.1f}",
                    ha="center",
                )
            finish(ax)
    fig.suptitle(
        "Topology-specific training versus flexible topology-aware control",
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output_dir / "fixed_x_median_iqr.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fixed_x_median_iqr.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.10,
        top=0.86,
        hspace=0.62,
        wspace=0.32,
    )
    for row, regime in enumerate(regimes):
        fig.text(
            0.515,
            0.905 if row == 0 else 0.462,
            regime_labels[regime],
            ha="center",
            fontsize=13,
            fontweight="semibold",
        )
        for col, (metric, ylabel) in enumerate(
            [
                ("recovery_time", "Recovery-time ratio to RLC-FT"),
                ("objective_cost", "Objective-cost ratio to RLC-FT"),
            ]
        ):
            ax = axes[row, col]
            sub = ratio_summary[
                (ratio_summary["regime"] == regime)
                & (ratio_summary["metric"] == metric)
            ].set_index("method").loc[method_order]
            med = sub["ratio_median"].to_numpy()
            q25 = sub["ratio_q25"].to_numpy()
            q75 = sub["ratio_q75"].to_numpy()
            x = np.arange(len(method_order))
            ax.bar(
                x,
                med,
                yerr=np.vstack([med - q25, q75 - med]),
                capsize=4,
                color=[colors[m] for m in method_order],
                edgecolor="black",
                linewidth=0.65,
            )
            ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
            ax.set_xticks(x, method_labels)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, max(q75) * 1.20)
            for xi, value, upper in zip(x, med, q75):
                ax.text(
                    xi,
                    upper + max(q75) * 0.035,
                    f"{value:.2f}",
                    ha="center",
                )
            finish(ax)
    fig.suptitle(
        "Paired normalized performance relative to RLC-FT",
        fontsize=15,
        y=0.975,
    )
    fig.savefig(output_dir / "fixed_x_paired_ratio.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fixed_x_paired_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    metadata = load_metadata(args)
    output_dir = (
        Path(Config.data_path)
        / "images"
        / "56bus"
        / "fixed_x_topology_shift"
        / args.fixed_tag
    )
    if args.output_suffix:
        output_dir = output_dir / args.output_suffix
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.selected_step is None:
        selected_step, ranking = select_checkpoint(args, metadata, output_dir)
        print("selected checkpoint:", selected_step)
        print(ranking.to_string(index=False))
    else:
        selected_step = args.selected_step
        if not all(
            path.exists()
            for path in fixed_policy_paths(args, make_env(), selected_step)
        ):
            raise FileNotFoundError(
                f"Incomplete fixed-X checkpoint at Step {selected_step}."
            )
        print("using preselected checkpoint:", selected_step)

    shape_env = make_env()
    fixed_policies = make_policy_set(
        shape_env,
        fixed_policy_paths(args, shape_env, selected_step),
    )
    flexible_args = SimpleNamespace(
        flexible_date=args.flexible_date,
        flexible_step=args.flexible_step,
        flexible_seed=args.flexible_seed,
    )
    flexible_policies = make_policy_set(
        shape_env,
        full_policy_paths(flexible_args, shape_env),
    )
    controllers = {
        "RLC-FT": flexible_policies,
        "Topology-specific RLC-FT": fixed_policies,
        "Linear": None,
    }

    rows = []
    seeds = range(
        args.test_seed_start,
        args.test_seed_start + args.test_episodes,
    )
    for regime in ["training_network", "flexible_shifts"]:
        for method, policies in controllers.items():
            for seed in seeds:
                rows.append(
                    run_episode(
                        env=make_env(),
                        method=method,
                        policy_nets=policies,
                        seed=seed,
                        regime=regime,
                        metadata=metadata,
                        max_steps=args.steps,
                        admittance_scale=args.admittance_scale,
                        action_scale=args.action_scale,
                        retain_loads=True,
                    )
                )
            print("finished:", regime, method)

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    ratio_summary = paired_ratio_summary(raw)
    raw["selected_fixed_step"] = selected_step
    raw.to_csv(output_dir / "fixed_x_comparison_raw.csv", index=False)
    summary.to_csv(output_dir / "fixed_x_comparison_summary.csv", index=False)
    ratio_summary.to_csv(
        output_dir / "fixed_x_comparison_paired_ratio.csv",
        index=False,
    )
    (output_dir / "evaluation_config.json").write_text(
        json.dumps({**vars(args), "selected_fixed_step": selected_step}, indent=2),
        encoding="utf-8",
    )
    make_plots(summary, ratio_summary, output_dir)
    print(summary.to_string(index=False))
    print(ratio_summary.to_string(index=False))


if __name__ == "__main__":
    main()
