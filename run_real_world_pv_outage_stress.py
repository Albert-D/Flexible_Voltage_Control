"""Evaluate PV outage/reconnection stress under empirical operating profiles.

This script leaves the training and environment modules unchanged. It keeps the
five monitored/controller buses fixed while toggling the bus-53 PV unit out of
service, which avoids changing the neural-controller input/output dimensions.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


VMIN = 0.95
VMAX = 1.05
HIGH_VOLTAGE_LIMIT = 1.10
SUSTAINED_LIMIT_MIN = 5.0
ENV_SEED = 10
FAILED_AGENT_INDEX = 4
FAILED_SGEN_INDEX = FAILED_AGENT_INDEX + 1
FAILED_BUS_NUMBER = 53
LINEAR_GAIN = 10.0
SAFE_SCALE = 10.0
RLCFT_SCALE = 0.7

METHODS = ("No control", "Linear", "Safe-DDPG", "RLC-FT")
CONTROL_METHODS = ("Linear", "Safe-DDPG", "RLC-FT")

METHOD_COLORS = {
    "No control": "#6B6B6B",
    "Linear": "#B76F67",
    "Safe-DDPG": "#6F8FAC",
    "RLC-FT": "#6D8F5E",
}
METHOD_FILLS = {
    "No control": "#9A9A9A",
    "Linear": "#D2877E",
    "Safe-DDPG": "#9FB4CD",
    "RLC-FT": "#98B486",
}
POWER_COLORS = {
    "Active load": "#6F8FAC",
    "Reactive load": "#C78F70",
    "Solar": "#6D8F5E",
}
BUS_COLORS = ("#6F8FAC", "#C78F70", "#6D8F5E", "#C97069", "#907AA4")
EVENT_COLOR = "#B2182B"
SAFE_FILL = "#E8F2E9"
GRID_COLOR = "#D9D9D9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=("one-day", "two-day"), required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"C:\Users\wdyao\OneDrive\Study\Code\Flexible_Voltage_Control"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"D:\Code\Python\Flexible_Voltage_Control"),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=600)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def load_runtime(repo_root: Path):
    sys.path.insert(0, str(repo_root))
    from Environment import VoltageCtrl_Env, create_56bus
    from NN_Module import FlexiblePolicyNet, SafePolicyNetwork, TopologyNet
    from config import Config

    return VoltageCtrl_Env, create_56bus, FlexiblePolicyNet, SafePolicyNetwork, TopologyNet, Config


def load_profiles(data_root: Path, scenario: str, max_steps: int | None):
    profile_path = data_root / "realworld_results.pkl.gz"
    with gzip.open(profile_path, "rb") as handle:
        payload = pickle.load(handle)
    source = payload["no_control"]
    one_day = {
        "p": np.asarray(source["p"]).reshape(-1),
        "q": np.asarray(source["q"]).reshape(-1),
        "pv_p": np.asarray(source["pv_p"]).reshape(-1),
    }
    repeats = 1 if scenario == "one-day" else 2
    profiles = {name: np.tile(values, repeats) for name, values in one_day.items()}
    if max_steps is not None:
        profiles = {name: values[:max_steps] for name, values in profiles.items()}
    return profiles, len(one_day["p"])


def event_steps(scenario: str, one_day_steps: int, pv_profile: np.ndarray):
    disconnect_step = int(round(9.0 * one_day_steps / 24.0))
    one_day_peak = int(np.argmax(pv_profile[:one_day_steps]))
    reconnect_step = one_day_peak if scenario == "one-day" else one_day_steps + one_day_peak
    return disconnect_step, reconnect_step


def build_environment(VoltageCtrl_Env, create_56bus):
    injection_bus = np.array([18, 21, 30, 45, 53]) - 1
    env = VoltageCtrl_Env(create_56bus(), injection_bus)
    env.reset(seed=ENV_SEED)
    env.pv_node_reset()
    state = env.reset0()
    return env, np.asarray(state, dtype=float), injection_bus


def load_policy_set(method, env, data_root, FlexiblePolicyNet, SafePolicyNetwork, TopologyNet, Config, device):
    if method not in ("Safe-DDPG", "RLC-FT"):
        return []

    policies = []
    for agent_index in range(5):
        if method == "RLC-FT":
            topology_net = TopologyNet(
                topology_dim=55,
                output_dim=1,
                hidden_dim=Config.topology_hidden_dim,
            )
            policy = FlexiblePolicyNet(
                env=env,
                topology_net=topology_net,
                obs_dim=1,
                action_dim=1,
                hidden_dim=Config.hidden_dim_56bus,
            ).to(device)
            checkpoint = (
                data_root
                / "check_points"
                / "policy_net"
                / "2025-02-18"
                / f"Step_500_Seed_4_a{agent_index}.pth"
            )
        else:
            policy = SafePolicyNetwork(
                env=env,
                obs_dim=1,
                action_dim=1,
                hidden_dim=100,
            ).to(device)
            checkpoint = (
                data_root.parent
                / "StableRL_VoltageCtrl-main"
                / "saved_models"
                / "stable_ddpg"
                / f"policy_net_checkpoint_a{agent_index}.pth"
            )
        policy.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        policy.eval()
        policies.append(policy)
    return policies


def controller_action(method, state, last_action, policies, topology_tensor, device):
    if method == "No control":
        return np.zeros(5, dtype=float)

    if method == "Linear":
        high_error = np.maximum(state - VMAX, 0.0)
        low_error = np.maximum(VMIN - state, 0.0)
        return last_action - LINEAR_GAIN * (high_error - low_error)

    increments = []
    for agent_index, policy in enumerate(policies):
        state_tensor = torch.tensor(
            [[state[agent_index]]],
            dtype=torch.float32,
            device=device,
        )
        if method == "RLC-FT":
            output = policy(state_tensor, topology_tensor)
            scale = RLCFT_SCALE
        else:
            output = policy(state_tensor)
            scale = SAFE_SCALE
        increments.append(float(output.detach().cpu().reshape(-1)[0]))
    return last_action - scale * np.asarray(increments)


def run_method(
    method,
    profiles,
    disconnect_step,
    reconnect_step,
    runtime,
    data_root,
    progress_every,
):
    VoltageCtrl_Env, create_56bus, FlexiblePolicyNet, SafePolicyNetwork, TopologyNet, Config = runtime
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, state, injection_bus = build_environment(VoltageCtrl_Env, create_56bus)
    policies = load_policy_set(
        method,
        env,
        data_root,
        FlexiblePolicyNet,
        SafePolicyNetwork,
        TopologyNet,
        Config,
        device,
    )
    topology_tensor = torch.tensor(
        np.asarray(env.topology_init),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    n_steps = len(profiles["p"])
    states = np.empty((n_steps + 1, 5), dtype=float)
    actions = np.empty((n_steps, 5), dtype=float)
    rewards = np.empty(n_steps, dtype=float)
    states[0] = state
    last_action = np.zeros(5, dtype=float)
    outage_active = False
    started = time.perf_counter()

    print(f"[{method}] start: {n_steps} steps on {device}", flush=True)
    for step in range(n_steps):
        if step == disconnect_step:
            env.network.sgen.at[FAILED_SGEN_INDEX, "in_service"] = False
            env.network.sgen.at[FAILED_SGEN_INDEX, "q_mvar"] = 0.0
            last_action[FAILED_AGENT_INDEX] = 0.0
            outage_active = True
            print(
                f"[{method}] bus-{FAILED_BUS_NUMBER} PV disconnected at step {step}",
                flush=True,
            )
        if step == reconnect_step:
            env.network.sgen.at[FAILED_SGEN_INDEX, "in_service"] = True
            env.network.sgen.at[FAILED_SGEN_INDEX, "q_mvar"] = 0.0
            last_action[FAILED_AGENT_INDEX] = 0.0
            outage_active = False
            print(
                f"[{method}] bus-{FAILED_BUS_NUMBER} PV reconnected at step {step}",
                flush=True,
            )

        action = controller_action(
            method,
            state,
            last_action,
            policies,
            topology_tensor,
            device,
        )
        if outage_active:
            action[FAILED_AGENT_INDEX] = 0.0

        next_state, topology, reward, _ = env.step_load(
            action.reshape(5, 1),
            profiles["p"][step],
            profiles["q"][step],
            profiles["pv_p"][step],
        )
        topology_tensor = torch.tensor(
            np.asarray(topology),
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        actions[step] = action
        rewards[step] = reward
        states[step + 1] = next_state
        last_action = action.copy()
        state = np.asarray(next_state, dtype=float)

        if progress_every and (step + 1) % progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                f"[{method}] {step + 1}/{n_steps} ({100 * (step + 1) / n_steps:.1f}%), "
                f"elapsed {elapsed / 60:.1f} min",
                flush=True,
            )

    elapsed = time.perf_counter() - started
    print(f"[{method}] complete in {elapsed / 60:.1f} min", flush=True)
    return {
        "method": method,
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "elapsed_sec": elapsed,
        "injection_bus": injection_bus,
    }


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def summarize_result(result, n_steps, minutes_per_step, reconnect_step):
    states = np.asarray(result["states"][:n_steps])
    actions = np.asarray(result["actions"])
    violation = np.maximum(states - VMAX, 0.0) + np.maximum(VMIN - states, 0.0)
    outside = (states < VMIN) | (states > VMAX)
    sustained_indicator = np.mean(outside, axis=1) > 0.10
    post_reconnect = slice(reconnect_step, n_steps)
    post_outside = sustained_indicator[post_reconnect]
    return {
        "method": result["method"],
        "peak_monitored_voltage_pu": float(np.max(states)),
        "minimum_monitored_voltage_pu": float(np.min(states)),
        "longest_sustained_violation_min": float(
            longest_true_run(sustained_indicator) * minutes_per_step
        ),
        "post_reconnect_longest_violation_min": float(
            longest_true_run(post_outside) * minutes_per_step
        ),
        "unsafe_duration_min": float(np.sum(np.any(outside, axis=1)) * minutes_per_step),
        "time_integrated_violation_pu_min": float(np.sum(violation) * minutes_per_step),
        "cumulative_reactive_action": float(np.sum(np.linalg.norm(actions, axis=1))),
        "objective_cost": float(-np.sum(result["rewards"])),
    }


def save_payload(output_dir, scenario, profiles, one_day_steps, disconnect_step, reconnect_step, results, metrics):
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": scenario,
        "settings": {
            "env_seed": ENV_SEED,
            "failed_bus_number": FAILED_BUS_NUMBER,
            "failed_sgen_index": FAILED_SGEN_INDEX,
            "selection_rule": "most-downstream PV among the largest installed PV units",
            "linear_gain": LINEAR_GAIN,
            "safe_scale": SAFE_SCALE,
            "rlcft_scale": RLCFT_SCALE,
            "one_day_steps": one_day_steps,
            "disconnect_step": disconnect_step,
            "reconnect_step": reconnect_step,
        },
        "profiles": profiles,
        "results": results,
        "metrics": metrics,
    }
    cache_path = output_dir / f"pv_outage_reconnection_{scenario}.pkl.gz"
    with gzip.open(cache_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    csv_path = output_dir / f"pv_outage_reconnection_{scenario}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    json_path = output_dir / f"pv_outage_reconnection_{scenario}_summary.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return cache_path, csv_path, json_path


def style_time_axis(ax, duration_hours):
    ax.set_xlim(0, duration_hours)
    ticks = (
        np.array([0.0, 12.0, 24.0])
        if duration_hours <= 24.1
        else np.linspace(0, duration_hours, 5)
    )
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(tick):02d}:00" if duration_hours <= 24 else f"{tick:.0f}" for tick in ticks])
    labels = ax.get_xticklabels()
    labels[0].set_ha("left")
    labels[-1].set_ha("right")
    ax.grid(True, color=GRID_COLOR, linewidth=0.55, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=0.7, pad=2)


def add_event_markers(ax, disconnect_hour, reconnect_hour, annotate=False):
    ax.axvspan(disconnect_hour, reconnect_hour, color="#F3E3E1", alpha=0.55, zorder=0)
    ax.axvline(disconnect_hour, color=EVENT_COLOR, linestyle=(0, (3, 2)), linewidth=0.9, zorder=5)
    ax.axvline(reconnect_hour, color=EVENT_COLOR, linestyle="-", linewidth=1.0, zorder=5)
    if annotate:
        ax.annotate(
            "PV outage",
            xy=(disconnect_hour, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color=EVENT_COLOR,
            fontsize=7.1,
            rotation=90,
        )
        ax.annotate(
            "PV reconnection",
            xy=(reconnect_hour, 0.03),
            xycoords=("data", "axes fraction"),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=EVENT_COLOR,
            fontsize=7.1,
            rotation=90,
        )


def panel_label(ax, label):
    ax.text(
        -0.18,
        1.16,
        label,
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def plot_figure(image_dir, scenario, payload):
    image_dir.mkdir(parents=True, exist_ok=True)
    settings = payload["settings"]
    profiles = payload["profiles"]
    result_map = {result["method"]: result for result in payload["results"]}
    metric_map = {metric["method"]: metric for metric in payload["metrics"]}
    n_steps = len(profiles["p"])
    duration_hours = 24.0 * n_steps / settings["one_day_steps"]
    hours = np.arange(n_steps) * duration_hours / n_steps
    disconnect_hour = settings["disconnect_step"] * duration_hours / n_steps
    reconnect_hour = settings["reconnect_step"] * duration_hours / n_steps
    bus_numbers = np.asarray(result_map["RLC-FT"]["injection_bus"]) + 1

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 9.2,
            "axes.titleweight": "normal",
            "legend.fontsize": 7.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 3, figsize=(7.09, 5.10))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.105, top=0.91, wspace=0.36, hspace=0.57)
    ax_a, ax_b, ax_c = axes[0]
    ax_d, ax_e, ax_f = axes[1]

    for label, key in (("Active load", "p"), ("Reactive load", "q"), ("Solar", "pv_p")):
        ax_a.plot(hours, profiles[key], color=POWER_COLORS[label], linewidth=1.15, label=label)
    style_time_axis(ax_a, duration_hours)
    add_event_markers(ax_a, disconnect_hour, reconnect_hour, annotate=True)
    ax_a.set_title("Empirical operating profiles", pad=5)
    ax_a.set_ylabel("Power (MW/MVar)")
    ax_a.set_ylim(bottom=0)
    ax_a.legend(frameon=False, loc="upper left", handlelength=1.4, borderaxespad=0.25, labelspacing=0.2)

    for target_ax, method in ((ax_b, "No control"), (ax_c, "RLC-FT")):
        states = np.asarray(result_map[method]["states"][:n_steps])
        for bus_index, (bus, color) in enumerate(zip(bus_numbers, BUS_COLORS)):
            target_ax.plot(hours, states[:, bus_index], color=color, linewidth=0.9, label=f"Bus {bus}")
        target_ax.axhspan(VMIN, VMAX, color=SAFE_FILL, alpha=0.75, zorder=0)
        target_ax.axhline(VMIN, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
        target_ax.axhline(VMAX, color="#555555", linestyle=(0, (3, 2)), linewidth=0.8)
        style_time_axis(target_ax, duration_hours)
        add_event_markers(target_ax, disconnect_hour, reconnect_hour)
        target_ax.set_ylim(0.90, max(1.14, float(np.max(states)) + 0.008))
        target_ax.set_yticks([0.90, 0.95, 1.00, 1.05, 1.10])
    ax_b.set_title("Voltages without control", pad=5)
    ax_b.set_ylabel("Bus voltage (p.u.)")
    ax_b.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.6, labelspacing=0.15, borderaxespad=0.25, fontsize=6.2)
    ax_c.set_title("Voltages with RLC-FT", pad=5)

    metric_specs = (
        (ax_d, "peak_monitored_voltage_pu", "Peak monitored voltage", "Voltage (p.u.)", HIGH_VOLTAGE_LIMIT, "1.10 p.u. limit", False),
        (ax_e, "longest_sustained_violation_min", "Longest sustained voltage violation", "Duration (min)", SUSTAINED_LIMIT_MIN, "5-min limit", True),
    )
    x_all = np.arange(len(METHODS))
    for ax, key, title, ylabel, limit, limit_label, log_scale in metric_specs:
        values = np.array([metric_map[method][key] for method in METHODS])
        if log_scale:
            plot_values = np.maximum(values, 0.05)
            base = 0.05
            ax.set_yscale("log")
        else:
            plot_values = values
            base = min(1.0, float(np.min(values)) - 0.01)
        ax.axhline(limit, color=EVENT_COLOR, linestyle=(0, (4, 2)), linewidth=1.0, zorder=1)
        label_x = 0.99 if key == "peak_monitored_voltage_pu" else 0.02
        label_ha = "right" if key == "peak_monitored_voltage_pu" else "left"
        ax.text(label_x, limit, limit_label, transform=ax.get_yaxis_transform(), ha=label_ha, va="bottom", color=EVENT_COLOR, fontsize=7.4)
        for index, method in enumerate(METHODS):
            ax.vlines(index, base, plot_values[index], color=METHOD_COLORS[method], linewidth=1.2, zorder=2)
            ax.scatter(index, plot_values[index], s=42, color=METHOD_FILLS[method], edgecolor=METHOD_COLORS[method], linewidth=0.7, zorder=3)
            label = f"{values[index]:.2f}" if key.endswith("_min") else f"{values[index]:.3f}"
            ax.annotate(label, (index, plot_values[index]), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom", fontsize=7.2)
        ax.set_xticks(x_all)
        ax.set_xticklabels(["No control", "Linear", "Safe-DDPG", "RLC-FT"], rotation=18, ha="right")
        ax.set_xlim(-0.35, 3.35)
        ax.set_title(title, pad=5)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.65, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=2.5, width=0.7, pad=2)
    ax_d.set_ylim(bottom=min(1.0, min(metric_map[m]["peak_monitored_voltage_pu"] for m in METHODS) - 0.01))
    ax_e.set_ylim(0.05, max(10.0, max(metric_map[m]["longest_sustained_violation_min"] for m in METHODS) * 1.7))

    objective = np.array([metric_map[method]["objective_cost"] for method in CONTROL_METHODS])
    normalized = objective / objective[0]
    x_control = np.arange(len(CONTROL_METHODS))
    lower = min(0.90, float(np.min(normalized)) - 0.02)
    ax_f.axhline(1.0, color="#777777", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
    for index, method in enumerate(CONTROL_METHODS):
        ax_f.vlines(index, lower, normalized[index], color=METHOD_COLORS[method], linewidth=1.2, zorder=2)
        ax_f.scatter(index, normalized[index], s=48, color=METHOD_FILLS[method], edgecolor=METHOD_COLORS[method], linewidth=0.75, zorder=3)
        offset = (4, 5) if method == "Linear" else (0, 5)
        ha = "left" if method == "Linear" else "center"
        ax_f.annotate(f"{normalized[index]:.3f}", (index, normalized[index]), xytext=offset, textcoords="offset points", ha=ha, va="bottom", fontsize=8.2)
    ax_f.set_xticks(x_control)
    ax_f.set_xticklabels(CONTROL_METHODS, rotation=12, ha="right")
    ax_f.set_xlim(-0.25, 2.25)
    ax_f.set_ylim(lower, max(1.014, float(np.max(normalized)) + 0.02))
    ax_f.set_title("Full-period objective cost", pad=5)
    ax_f.set_ylabel("Normalized objective cost")
    ax_f.set_axisbelow(True)
    ax_f.grid(axis="y", color=GRID_COLOR, linewidth=0.55, alpha=0.65, zorder=0)
    ax_f.spines[["top", "right"]].set_visible(False)
    ax_f.tick_params(length=2.5, width=0.7, pad=2)

    for ax, label in zip(axes.flat, "abcdef"):
        panel_label(ax, label)
    if duration_hours > 24:
        ax_a.set_xlabel("Elapsed time (h)")
        ax_b.set_xlabel("Elapsed time (h)")
        ax_c.set_xlabel("Elapsed time (h)")

    png_path = image_dir / f"pv_outage_reconnection_{scenario}_2x3.png"
    pdf_path = image_dir / f"pv_outage_reconnection_{scenario}_2x3.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main():
    args = parse_args()
    output_dir = args.data_root / "cache" / "real_world_pv_outage_stress"
    image_dir = args.data_root / "images" / "real_world_pv_outage_stress"
    cache_path = output_dir / f"pv_outage_reconnection_{args.scenario}.pkl.gz"
    if args.plot_only:
        with gzip.open(cache_path, "rb") as handle:
            payload = pickle.load(handle)
        png_path, pdf_path = plot_figure(image_dir, args.scenario, payload)
        print(f"PNG: {png_path}", flush=True)
        print(f"PDF: {pdf_path}", flush=True)
        return

    runtime = load_runtime(args.repo_root)
    profiles, one_day_steps = load_profiles(args.data_root, args.scenario, args.max_steps)
    disconnect_step, reconnect_step = event_steps(args.scenario, one_day_steps, profiles["pv_p"])
    if reconnect_step >= len(profiles["p"]):
        raise ValueError("The requested max-steps truncation excludes the reconnection event")

    print(
        f"Scenario={args.scenario}; n={len(profiles['p'])}; disconnect={disconnect_step}; "
        f"reconnect={reconnect_step}; PV peak={profiles['pv_p'][reconnect_step]:.3f}",
        flush=True,
    )
    print(f"CUDA available={torch.cuda.is_available()}", flush=True)

    results = []
    for method in METHODS:
        results.append(
            run_method(
                method,
                profiles,
                disconnect_step,
                reconnect_step,
                runtime,
                args.data_root,
                args.progress_every,
            )
        )

    n_steps = len(profiles["p"])
    duration_minutes = 24.0 * 60.0 * n_steps / one_day_steps
    minutes_per_step = duration_minutes / n_steps
    metrics = [
        summarize_result(result, n_steps, minutes_per_step, reconnect_step)
        for result in results
    ]
    cache_path, csv_path, json_path = save_payload(
        output_dir,
        args.scenario,
        profiles,
        one_day_steps,
        disconnect_step,
        reconnect_step,
        results,
        metrics,
    )
    payload = {
        "settings": {
            "one_day_steps": one_day_steps,
            "disconnect_step": disconnect_step,
            "reconnect_step": reconnect_step,
        },
        "profiles": profiles,
        "results": results,
        "metrics": metrics,
    }
    png_path, pdf_path = plot_figure(image_dir, args.scenario, payload)

    print("\nSummary", flush=True)
    for metric in metrics:
        print(json.dumps(metric), flush=True)
    print(f"Cache: {cache_path}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    print(f"PNG: {png_path}", flush=True)
    print(f"PDF: {pdf_path}", flush=True)


if __name__ == "__main__":
    main()
