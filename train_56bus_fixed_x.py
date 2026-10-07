"""Train a 56-bus RLC-FT controller on one truly fixed network matrix X.

The script reuses the existing environment, network architecture, TD3 agent,
and training hyperparameters. Switch states, physical line parameters, and the
topology/admittance input supplied to every actor and replay-buffer transition
remain fixed throughout training. Only the operating condition changes across
episodes through the existing low- and high-voltage PV injection scenarios.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandapower as pp
import torch
from loguru import logger

from config import Config
from Environment import VoltageCtrl_Env, create_56bus
from NN_Module import FlexiblePolicyNet, Q_Network, TopologyNet
from TD3 import TD3
from Utils import ReplayBuffer


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a topology-specific 56-bus controller with fixed X."
    )
    parser.add_argument("--episodes", type=int, default=Config.total_episodes)
    parser.add_argument("--steps", type=int, default=Config.total_steps)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--train-seed", type=int, default=4)
    parser.add_argument("--topology-seed", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def make_env() -> VoltageCtrl_Env:
    injection_bus = np.array([18, 21, 30, 45, 53]) - 1
    return VoltageCtrl_Env(create_56bus(), injection_bus)


def sample_reference_switches(
    env: VoltageCtrl_Env,
    topology_seed: int,
    max_tries: int = 100,
) -> tuple[dict[int, bool], int]:
    """Reuse the existing feasible-topology sampler, but not its random lines."""
    for seed in range(topology_seed, topology_seed + max_tries):
        try:
            env.reset_topo(seed=seed)
        except pp.powerflow.LoadflowNotConverged:
            continue
        fixed_switch = {
            int(idx): bool(env.network.switch.at[idx, "closed"])
            for idx in range(len(env.network.switch))
        }
        return fixed_switch, seed
    raise RuntimeError("Could not sample a feasible reference switch topology.")


def build_reference_network(
    env: VoltageCtrl_Env,
    fixed_switch: dict[int, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Return nominal physical line reactances and their fixed X encoding."""
    line_x = np.asarray(env.topology_init, dtype=float).copy()
    topology_input = np.reciprocal(line_x)

    for idx, closed in fixed_switch.items():
        env.network.switch.at[idx, "closed"] = closed
        if not closed and env.network.switch.at[idx, "et"] == "l":
            line_idx = int(env.network.switch.at[idx, "element"])
            topology_input[line_idx] = 0.0

    env.network.line.loc[:, "x_ohm_per_km"] = line_x
    return line_x, topology_input


def reset_fixed_x(
    env: VoltageCtrl_Env,
    seed: int,
    fixed_switch: dict[int, bool],
    fixed_line_x: np.ndarray,
    fixed_topology_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a new operating point while keeping the physical network fixed."""
    rng = np.random.RandomState(seed)
    scenario = int(rng.choice([0, 1]))

    env.network.line.loc[:, "x_ohm_per_km"] = fixed_line_x
    for idx, closed in fixed_switch.items():
        env.network.switch.at[idx, "closed"] = closed

    env.network.sgen.loc[:, "p_mw"] = 0.0
    env.network.sgen.loc[:, "q_mvar"] = 0.0
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

    pp.runpp(env.network, algorithm="bfsw")
    env.state = env.network.res_bus.iloc[env.injection_bus].vm_pu.to_numpy()
    env.topology = fixed_topology_input.copy()
    return env.state.copy(), fixed_topology_input.copy(), scenario


def create_agents(env: VoltageCtrl_Env, train_seed: int):
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    random.seed(train_seed)

    agents = []
    replay_buffers = []
    for _ in range(env.agentnum):
        value_net = Q_Network(
            obs_dim=1,
            action_dim=Config.action_dim,
            hidden_dim=256,
        ).to(device)
        target_value_net = Q_Network(
            obs_dim=1,
            action_dim=Config.action_dim,
            hidden_dim=256,
        ).to(device)

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
        target_policy_net = FlexiblePolicyNet(
            env=env,
            topology_net=topology_net,
            obs_dim=Config.state_dim,
            action_dim=Config.action_dim,
            hidden_dim=Config.hidden_dim_56bus,
        ).to(device)

        target_value_net.load_state_dict(value_net.state_dict())
        target_policy_net.load_state_dict(policy_net.state_dict())

        agents.append(
            TD3(
                policy_net=policy_net,
                value_net=value_net,
                target_policy_net=target_policy_net,
                target_value_net=target_value_net,
                value_lr=Config.value_learning_rate,
                policy_lr=Config.policy_learning_rate,
                max_action=Config.max_action,
            )
        )
        replay_buffers.append(ReplayBuffer(capacity=1_000_000))
    return agents, replay_buffers


def save_checkpoint(
    agents,
    output_dir: Path,
    episode: int,
    train_seed: int,
) -> None:
    for network_name in ("policy_net", "value_net"):
        (output_dir / network_name).mkdir(parents=True, exist_ok=True)

    for i, agent in enumerate(agents):
        torch.save(
            agent.policy_net.state_dict(),
            output_dir
            / "policy_net"
            / f"Step_{episode}_Seed_{train_seed}_a{i}.pth",
        )
        torch.save(
            agent.value_net.state_dict(),
            output_dir
            / "value_net"
            / f"Step_{episode}_Seed_{train_seed}_a{i}.pth",
        )


def train(args: argparse.Namespace) -> Path:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Using {} device", device)

    env = make_env()
    fixed_switch, actual_topology_seed = sample_reference_switches(
        env,
        args.topology_seed,
    )
    fixed_line_x, fixed_topology_input = build_reference_network(env, fixed_switch)

    # Confirm that the selected switch state is feasible with nominal line data.
    reset_fixed_x(
        env,
        seed=0,
        fixed_switch=fixed_switch,
        fixed_line_x=fixed_line_x,
        fixed_topology_input=fixed_topology_input,
    )

    agents, replay_buffers = create_agents(env, args.train_seed)
    tag = args.tag or (
        "fixed_x_56bus_"
        f"{datetime.now():%Y%m%d_%H%M}_"
        f"trainseed{args.train_seed}_toposeed{actual_topology_seed}"
    )
    output_dir = (
        Path(Config.data_path)
        / "check_points"
        / "topology_specific_fixed_x_56bus"
        / tag
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "experiment": "56bus_topology_specific_fixed_x",
        "train_seed": args.train_seed,
        "requested_topology_seed": args.topology_seed,
        "actual_topology_seed": actual_topology_seed,
        "fixed_switch": fixed_switch,
        "fixed_line_x_ohm_per_km": fixed_line_x.tolist(),
        "fixed_topology_input": fixed_topology_input.tolist(),
        "episodes": args.episodes,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": device,
        "config_data_path": Config.data_path,
        "note": (
            "Switch states, physical line parameters, and actor/replay-buffer "
            "topology input X are fixed throughout training. Only PV operating "
            "conditions vary across episodes."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    max_action = Config.max_action_56bus
    min_action = -Config.max_action_56bus
    reward_history = []
    episode_rows = []

    logger.info(
        "Training fixed-X controller with {}/{} closed switches; outputs: {}",
        sum(fixed_switch.values()),
        len(fixed_switch),
        output_dir,
    )

    for episode in range(args.episodes + 1):
        try:
            state, topology, scenario = reset_fixed_x(
                env,
                seed=episode,
                fixed_switch=fixed_switch,
                fixed_line_x=fixed_line_x,
                fixed_topology_input=fixed_topology_input,
            )
        except pp.powerflow.LoadflowNotConverged:
            logger.warning("Episode {} initial power flow did not converge.", episode)
            continue

        topology_tensor = torch.tensor(
            topology,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(0)
        episode_reward = 0.0
        last_action = np.zeros((env.agentnum, 1))
        status = "horizon"

        for step in range(args.steps):
            action = []
            for i, agent in enumerate(agents):
                state_i = torch.tensor(
                    state[i].reshape(1,),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                action_i = (
                    agent.policy_net(state_i, topology_tensor)
                    .detach()
                    .cpu()
                    .numpy()[0]
                )
                if episode < 30:
                    noise = np.random.normal(0, 0.5) / (episode + 1)
                else:
                    noise = np.random.normal(0, 0.05)
                action_i = np.clip(
                    action_i + np.clip(noise, -0.5, 0.5),
                    min_action,
                    max_action,
                )
                action.append(action_i)

            action = last_action - np.asarray(action)
            try:
                next_state, _, reward, reward_sep, done = env.step_uncertain(action)
            except pp.powerflow.LoadflowNotConverged:
                status = "loadflow_failed"
                break

            if np.min(next_state) < 0.75 or np.max(next_state) > 1.25:
                status = "guardrail"
                break

            for i, agent in enumerate(agents):
                replay_buffers[i].push(
                    state[i].reshape(1,),
                    fixed_topology_input.copy(),
                    action[i].reshape(1,),
                    last_action[i].reshape(1,),
                    Config.r_global_weight * reward
                    + Config.r_local_weight * reward_sep[i],
                    next_state[i].reshape(1,),
                    done,
                )
                if len(replay_buffers[i]) > args.batch_size:
                    agent.train(
                        replay_buffer=replay_buffers[i],
                        iterations=i,
                        batch_size=args.batch_size,
                        policy_noise=0.03,
                        noise_clip=0.05,
                        policy_freq=3,
                    )

            episode_reward += reward
            last_action = np.copy(action)
            if done:
                status = "recovered"
                break
            state = np.copy(next_state)

        reward_history.append(float(episode_reward))
        episode_rows.append(
            {
                "episode": episode,
                "scenario": scenario,
                "reward": float(episode_reward),
                "steps": int(step + 1),
                "status": status,
            }
        )

        if episode % args.progress_every == 0:
            logger.info(
                "Episode {:04d}: avg_reward(last 50)={:.3f}, status={}, steps={}",
                episode,
                float(np.mean(reward_history[-50:])),
                status,
                step + 1,
            )
        if episode % args.save_every == 0:
            save_checkpoint(agents, output_dir, episode, args.train_seed)

    save_checkpoint(agents, output_dir, args.episodes, args.train_seed)

    log_path = output_dir / "training_log.csv"
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write("episode,scenario,reward,steps,status\n")
        for row in episode_rows:
            stream.write(
                f"{row['episode']},{row['scenario']},{row['reward']},"
                f"{row['steps']},{row['status']}\n"
            )

    logger.info("Fixed-X training complete: {}", output_dir)
    return output_dir


if __name__ == "__main__":
    train(parse_args())
