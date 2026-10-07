"""Prepare, calibrate, smoke-test, and train the unified 56-bus controller."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import pandapower as pp
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandapower\..*")


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
SPEED = CODE_ROOT / "experiments/validated/training_runtime"
for path in (CODE_ROOT, SPEED, HERE):
    sys.path.insert(0, str(path))

from fast_environment import feeder_class
from fast_runtime import ActorMirror, CachedReplay, configure_adam, sample_replay_fast
from retraining_environment import BUS, Feeder, save_json

sys.path.insert(0, str(HERE))
from evaluation import (
    evaluate_pool,
    extract_actor_state,
    init_worker,
    paired_x_report,
    performance_gate,
    topology_gate,
    topology_probe,
)
from model import ACTION_LIMIT, ACTION_STEP_LIMIT, Learner, NUM_AGENTS
sys.path.insert(0, str(HERE))
import protocol as manifest_protocol

draw_graph = manifest_protocol.draw_graph
scene_for = manifest_protocol.scene_for
scenes_for_operating_point = manifest_protocol.scenes_for_operating_point


OUT = Path("D:/Code/Python/Flexible_Voltage_Control/experiments/unified_softplus_retraining")
PROTOCOL = OUT / "protocol.json"
CALIBRATION = OUT / "calibration.json"
FastFeeder = feeder_class(Feeder, BUS, "lightsim_recycle")
NEGATIVE_CONTROL = Path("D:/Code/Python/Flexible_Voltage_Control/experiments/topology_efficiency/seed2601_weak_lr_25e6_long/checkpoint_024243.pt")
POSITIVE_CONTROL = Path("D:/Code/Python/Flexible_Voltage_Control/experiments/controller_retraining_2026-09-11/topology_sensitive_controller_v2/runs/pilot_b1_seed2601_hard_state_replay/checkpoint_001920.pt")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append(path: Path, row) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def actor_from_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return extract_actor_state(payload)


def calibrate(workers: int = 4) -> dict:
    if not PROTOCOL.exists():
        raise FileNotFoundError("run prepare first")
    protocol = read(PROTOCOL)
    dev = protocol["groups"]["development_performance"]
    x_panel = protocol["groups"]["development_x"]
    confirmation = protocol["groups"]["confirmation"]
    learner = Learner(torch.device("cpu"), seed=2601)
    initial_state = {key: value.detach().cpu().clone() for key, value in learner.actor.state_dict().items()}
    controls = {
        "initial": initial_state,
        "negative_topology_control": actor_from_checkpoint(NEGATIVE_CONTROL),
        "positive_topology_control": actor_from_checkpoint(POSITIVE_CONTROL),
    }
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        reference_dev = evaluate_pool(pool, dev, mode="reference", workers=workers)
        reference_confirmation = evaluate_pool(pool, confirmation, mode="reference", workers=workers)
        initial_dev = evaluate_pool(pool, dev, initial_state, mode="policy", workers=workers)
        control_rows = {}
        for name, state in controls.items():
            correct = evaluate_pool(pool, x_panel, state, mode="policy", workers=workers)
            mismatched = evaluate_pool(pool, x_panel, state, mode="mismatched", workers=workers)
            probe = topology_probe(state, x_panel)
            paired = paired_x_report(correct["rows"], mismatched["rows"])
            control_rows[name] = {
                "probe": probe,
                "paired": paired,
                "topology_gate": topology_gate(probe, paired),
                "correct_metrics": correct["metrics"],
                "mismatched_metrics": mismatched["metrics"],
            }
    result = {
        "protocol_sha256": sha(PROTOCOL),
        "thresholds_frozen_before_new_training": {
            "normalized_action_difference": 0.02,
            "median_relative_gain_span": 0.05,
            "correct_objective_win_fraction": 0.55,
            "correct_objective_median_improvement_strictly_positive": True,
            "development_objective_multiplier": 2.0,
            "development_objective_rationale": "reference median recovery is 3 on the frozen panel while the preregistered candidate target is 4--6; final SI objective remains separately audited",
        },
        "reference_development": reference_dev,
        "reference_confirmation": reference_confirmation,
        "initial_development": initial_dev,
        "controls": control_rows,
        "sources": {
            "negative": {"path": str(NEGATIVE_CONTROL), "sha256": sha(NEGATIVE_CONTROL)},
            "positive": {"path": str(POSITIVE_CONTROL), "sha256": sha(POSITIVE_CONTROL)},
        },
        "workers": workers,
        "wall_seconds": time.perf_counter() - started,
    }
    save_json(CALIBRATION, result)
    torch.save({"actor": initial_state, "model_config": learner.actor.model_config()}, OUT / "initial_actor.pt")
    print(json.dumps({
        "calibration": str(CALIBRATION),
        "reference": reference_dev["metrics"],
        "initial": initial_dev["metrics"],
        "controls": {name: value["topology_gate"] for name, value in control_rows.items()},
        "seconds": result["wall_seconds"],
    }), flush=True)
    return result


def source_manifest():
    sources = [
        HERE / "README.md", HERE / "model.py", HERE / "protocol.py",
        HERE / "evaluation.py", HERE / "run.py",
        CODE_ROOT / "src/softplus_controller.py",
        CODE_ROOT / "retraining_environment.py",
        SPEED / "fast_environment.py", SPEED / "fast_runtime.py",
        HERE / "reference.py",
        SPEED / "support.py",
        CODE_ROOT / "data/SCE_56bus.mat",
    ]
    return {str(path): sha(path) for path in sources}


def audit_checkpoint(checkpoint_path: Path, name: str, workers: int = 4) -> dict:
    protocol = read(PROTOCOL)
    calibration = read(CALIBRATION)
    actor_state = actor_from_checkpoint(checkpoint_path)
    dev = protocol["groups"]["development_performance"]
    x_panel = protocol["groups"]["development_x"]
    started = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        performance = evaluate_pool(pool, dev, actor_state, mode="policy", workers=workers)
        correct = evaluate_pool(pool, x_panel, actor_state, mode="policy", workers=workers)
        mismatched = evaluate_pool(pool, x_panel, actor_state, mode="mismatched", workers=workers)
    probe = topology_probe(actor_state, x_panel)
    paired = paired_x_report(correct["rows"], mismatched["rows"])
    result = {
        "name": name,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha(checkpoint_path)},
        "protocol_sha256": sha(PROTOCOL),
        "performance": performance,
        "performance_gate": performance_gate(
            performance["metrics"], calibration["reference_development"]["metrics"]
        ),
        "topology_probe": probe,
        "topology_report": paired,
        "topology_gate": topology_gate(probe, paired),
        "wall_seconds": time.perf_counter() - started,
    }
    target = OUT / "checkpoint_audits" / f"{name}.json"
    save_json(target, result)
    print(json.dumps({
        "audit": str(target), "performance": performance["metrics"],
        "performance_gate": result["performance_gate"],
        "topology_probe": probe, "topology_report": paired,
        "topology_gate": result["topology_gate"], "seconds": result["wall_seconds"],
    }), flush=True)
    return result


def run_training(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the accepted runtime")
    if not PROTOCOL.exists() or not CALIBRATION.exists():
        raise FileNotFoundError("prepare and calibrate must complete first")
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("run-id must contain ASCII letters, digits, and underscores")
    protocol = read(PROTOCOL)
    calibration = read(CALIBRATION)
    protocol_hash = sha(PROTOCOL)
    if calibration["protocol_sha256"] != protocol_hash:
        raise ValueError("calibration does not match protocol")
    run = OUT / args.run_id
    if run.exists():
        raise FileExistsError(run)
    run.mkdir(parents=True)
    sources = source_manifest()
    config = {
        "command": "smoke" if args.smoke else "train",
        "run_id": args.run_id,
        "seed": args.seed,
        "critic_init_seed": args.critic_init_seed,
        "critic_output_init": args.critic_output_init,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "discount": args.discount,
        "initial_slope_factor": args.initial_slope_factor,
        "max_interactions": args.max_interactions,
        "eval_every": args.eval_every,
        "workers": args.workers,
        "topologies_per_operating_point": args.topologies_per_operating_point,
        "reward": {"band": args.reward_band, "deviation": args.reward_deviation,
                   "delta": args.reward_delta, "q": args.reward_q,
                   "step_clip": args.reward_step_clip,
                   "success": args.reward_success, "failure": args.reward_failure},
        "step_soft_threshold": args.step_soft_threshold,
        "disable_confirmation": args.disable_confirmation,
        "force_full_budget": args.force_full_budget,
        "initial_evaluation_cache": None if args.initial_evaluation_cache is None else {
            "path": str(args.initial_evaluation_cache),
            "sha256": sha(args.initial_evaluation_cache),
        },
        "initial_actor_checkpoint": None if args.initial_actor_checkpoint is None else {
            "path": str(args.initial_actor_checkpoint),
            "sha256": sha(args.initial_actor_checkpoint),
        },
        "runtime": {
            "powerflow": "exact lightsim_recycle",
            "cache_rebuild": ["reset", "topology", "admittance", "load"],
            "actor_rollout": "CPU mirror synchronized after every actor update",
            "learner": "CUDA fused Adam" if args.device == "cuda" else "CPU Adam fallback; CUDA unavailable",
            "replay": "CachedReplay, 50% hard at 75th severity quantile",
            "cpu_threads_per_process": 1,
        },
        "terminal_semantics": "terminal only on recovery or hard failure; bootstrap time-limit truncation",
        "acceptance": "two consecutive full development gates plus confirmation within 30000 interactions",
        "protocol_sha256": protocol_hash,
        "model_config": Learner(torch.device("cpu"), seed=args.seed).actor.model_config(),
        "sources": sources,
        "host": platform.node(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "device": args.device,
    }
    save_json(run / "config.json", config)
    shutil.copy2(PROTOCOL, run / "protocol.json")
    shutil.copy2(CALIBRATION, run / "calibration.json")
    for source in sources:
        source_path = Path(source)
        destination = run / "source" / source_path.relative_to(CODE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    started = time.perf_counter()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    learner = Learner(
        device,
        seed=args.seed,
        critic_init_seed=args.critic_init_seed,
        critic_output_init=args.critic_output_init,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        discount=args.discount,
        initial_slope_factor=args.initial_slope_factor,
    )
    if args.initial_actor_checkpoint is not None:
        source_payload = torch.load(args.initial_actor_checkpoint, map_location="cpu", weights_only=False)
        source_actor = extract_actor_state(source_payload)
        learner.actor.load_state_dict(source_actor, strict=True)
        learner.target.load_state_dict(source_actor, strict=True)
    cached_initial = None
    if args.initial_evaluation_cache is not None:
        cached_initial = read(args.initial_evaluation_cache)
        source_checkpoint = args.initial_evaluation_cache.with_name("checkpoint_000000.pt")
        source_payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        if source_payload["protocol_sha256"] != protocol_hash:
            raise ValueError("initial evaluation cache uses a different protocol")
        source_actor = extract_actor_state(source_payload)
        current_actor = learner.actor.state_dict()
        if source_actor.keys() != current_actor.keys() or any(
            not torch.equal(source_actor[key], current_actor[key].detach().cpu()) for key in current_actor
        ):
            raise ValueError("initial evaluation cache actor differs from this run's actor")
        if cached_initial["interactions"] != 0 or len(cached_initial["rows"]) != len(protocol["groups"]["development_performance"]):
            raise ValueError("initial evaluation cache does not match the expected panel")
    configure_adam(learner, fused=(device.type == "cuda"))
    mirror = ActorMirror(learner.actor)
    replay = CachedReplay(seed=args.seed + 3000)
    rng = np.random.default_rng(args.seed + 1000)
    noise_rng = np.random.default_rng(args.seed + 2000)
    env = FastFeeder()
    excluded = set(protocol["excluded_graphs"])
    used_graphs = set()
    used_x = set()
    records = []
    episodes = []
    solver_checks = []
    counts = {
        "interactions": 0, "successful_steps": 0, "failed_steps": 0,
        "graph_draws": 0, "reset_attempts": 0, "reset_failures": 0,
    }
    timing = {key: 0.0 for key in (
        "reset", "action", "powerflow", "reward_replay", "sample", "update",
        "actor_sync", "evaluation", "checkpoint", "solver_check",
    )}
    reference_dev = calibration["reference_development"]["metrics"]
    reference_confirmation = calibration["reference_confirmation"]["metrics"]
    x_panel = protocol["groups"]["development_x"]
    dev = protocol["groups"]["development_performance"]
    confirmation = protocol["groups"]["confirmation"]
    full_passes = 0
    x_collapse_passes = 0
    next_full_x = 5000
    performance_x_attempted = False
    last_x_gate = None
    pending_scenes = []
    operating_point_groups = 0

    def elapsed():
        return time.perf_counter() - started

    def checkpoint(label=None):
        tick = time.perf_counter()
        name = label or f'checkpoint_{counts["interactions"]:06d}.pt'
        path = run / name
        temp = path.with_suffix(".tmp")
        torch.save({
            "learner": learner.state(),
            "replay": replay.rows,
            "replay_pos": replay.pos,
            "replay_rng": replay.rng.bit_generator.state,
            "rng": rng.bit_generator.state,
            "noise_rng": noise_rng.bit_generator.state,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "counts": counts,
            "timing": timing,
            "used_graphs": sorted(used_graphs),
            "used_x": sorted(used_x),
            "records": records,
            "episodes": episodes,
            "solver_checks": solver_checks,
            "elapsed": elapsed(),
            "protocol_sha256": protocol_hash,
        }, temp)
        temp.replace(path)
        save_json(run / "latest_checkpoint.json", {"path": str(path), "sha256": sha(path)})
        timing["checkpoint"] += time.perf_counter() - tick
        return path

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
    ) as pool:
        def assess(force_x=False):
            nonlocal full_passes, x_collapse_passes, next_full_x
            nonlocal performance_x_attempted, last_x_gate
            tick = time.perf_counter()
            actor_state = {key: value.detach().cpu().clone() for key, value in mirror.actor.state_dict().items()}
            cached_initial_matches = cached_initial is not None or (
                args.seed == 2601
                and abs(args.initial_slope_factor - 0.05) < 1e-12
            )
            if counts["interactions"] == 0 and cached_initial_matches:
                if cached_initial is not None:
                    result = {"rows": cached_initial["rows"], "metrics": cached_initial["performance"], "wall_seconds": 0.0, "worker_seconds": 0.0}
                else:
                    evaluated = calibration["initial_development"]
                    result = {"rows": evaluated["rows"], "metrics": evaluated["metrics"], "wall_seconds": 0.0, "worker_seconds": 0.0}
            else:
                result = evaluate_pool(pool, dev, actor_state, mode="policy", workers=args.workers)
            perf = performance_gate(result["metrics"], reference_dev)
            probe = topology_probe(actor_state, x_panel)
            periodic_x = counts["interactions"] >= next_full_x
            first_performance_x = bool(perf["passed"] and not performance_x_attempted)
            confirm_previous_x = bool(
                perf["passed"] and last_x_gate is not None and last_x_gate["passed"]
            )
            run_full_x = bool(force_x or periodic_x or first_performance_x or confirm_previous_x)
            x_report = x_gate = None
            if run_full_x:
                if counts["interactions"] == 0 and cached_initial_matches:
                    if cached_initial is not None:
                        probe = cached_initial["topology_probe"]
                        x_report = cached_initial["topology_report"]
                        x_gate = cached_initial["topology_gate"]
                    else:
                        cached = calibration["controls"]["initial"]
                        probe = cached["probe"]
                        x_report = cached["paired"]
                        x_gate = cached["topology_gate"]
                else:
                    correct = evaluate_pool(pool, x_panel, actor_state, mode="policy", workers=args.workers)
                    mismatched = evaluate_pool(pool, x_panel, actor_state, mode="mismatched", workers=args.workers)
                    x_report = paired_x_report(correct["rows"], mismatched["rows"])
                    x_gate = topology_gate(probe, x_report)
                last_x_gate = x_gate
                if perf["passed"]:
                    performance_x_attempted = True
                while next_full_x <= counts["interactions"]:
                    next_full_x += 5000
            passed = bool(perf["passed"] and x_gate is not None and x_gate["passed"])
            if passed:
                full_passes += 1
            else:
                full_passes = 0
            if perf["passed"] and x_gate is not None and not x_gate["passed"]:
                x_collapse_passes += 1
            elif x_gate is not None and x_gate["passed"]:
                x_collapse_passes = 0
            timing["evaluation"] += time.perf_counter() - tick
            row = {
                "interactions": counts["interactions"],
                "num_graphs": len(used_graphs),
                "num_full_x": len(used_x),
                "updates": learner.updates,
                "performance": result["metrics"],
                "reference": reference_dev,
                "performance_gate": perf,
                "topology_probe": probe,
                "topology_report": x_report,
                "topology_gate": x_gate,
                "full_gate_passed": passed,
                "consecutive_full_passes": full_passes,
                "timing": timing.copy(),
                "elapsed": elapsed(),
            }
            records.append(row)
            save_json(run / "curve_data.json", records)
            save_json(run / f'evaluation_{counts["interactions"]:06d}.json', {**row, "rows": result["rows"]})
            checkpoint()
            print(json.dumps({
                "run": args.run_id,
                "interactions": counts["interactions"],
                "graphs": len(used_graphs),
                "performance": result["metrics"],
                "performance_pass": perf["passed"],
                "topology_probe": probe,
                "topology_pass": None if x_gate is None else x_gate["passed"],
                "consecutive": full_passes,
                "elapsed": elapsed(),
            }), flush=True)
            return row

        initial = assess(force_x=True)
        reason = "interaction_budget"
        next_eval = args.eval_every
        confirmation_result = None
        while counts["interactions"] < args.max_interactions:
            if (run / "STOP").exists():
                reason = "requested_stop"
                break
            tick = time.perf_counter()
            if not pending_scenes:
                topologies = []
                reserved = used_graphs | excluded
                for _ in range(args.topologies_per_operating_point):
                    topology = draw_graph(env, rng, reserved | {row["id"] for row in topologies}, counts)
                    topologies.append(topology)
                pending_scenes.extend(scenes_for_operating_point(
                    env,
                    topologies,
                    rng,
                    93000000 + operating_point_groups,
                    counts,
                ))
                operating_point_groups += 1
            scene = pending_scenes.pop(0)
            topology = {"id": scene["topology_id"], "switches": scene["switches"]}
            state = env.reset(scene).astype(np.float32)
            x_id = hashlib.sha256(np.asarray(scene["features"], dtype="<f4").tobytes()).hexdigest()[:24]
            if x_id in used_x:
                continue
            scene["x_id"] = x_id
            used_graphs.add(topology["id"])
            used_x.add(x_id)
            timing["reset"] += time.perf_counter() - tick
            features = env.features.copy()
            previous = np.zeros(NUM_AGENTS, dtype=np.float32)
            reward_sums = {key: 0.0 for key in ("band", "deviation", "delta", "q2", "step_clip", "success", "failure", "total")}
            losses = {}
            actual_steps = 0
            recovered = failed = False
            failure_reason = None
            episode_budget = min(60, args.max_interactions - counts["interactions"])
            for _ in range(episode_budget):
                tick = time.perf_counter()
                with torch.no_grad():
                    delta = mirror.actor(
                        torch.as_tensor(state).view(1, NUM_AGENTS),
                        torch.as_tensor(features).view(1, -1),
                    ).numpy().reshape(NUM_AGENTS)
                noise = noise_rng.normal(0.0, 0.25 if counts["interactions"] < 512 else 0.08, NUM_AGENTS)
                q = np.clip(
                    previous - np.clip(delta + noise, -ACTION_STEP_LIMIT, ACTION_STEP_LIMIT),
                    -ACTION_LIMIT,
                    ACTION_LIMIT,
                ).astype(np.float32)
                timing["action"] += time.perf_counter() - tick
                counts["interactions"] += 1
                actual_steps += 1
                tick = time.perf_counter()
                try:
                    next_state = env.step(q)
                    if not np.isfinite(next_state).all() or next_state.min() < 0.75 or next_state.max() > 1.25:
                        raise ValueError("hard voltage bound")
                    band = np.maximum(np.abs(next_state - 1.0) - 0.045, 0.0)
                    deviation = np.abs(next_state - 1.0)
                    delta_component = np.square(q - previous)
                    q_component = np.square(q)
                    step_clip_component = np.square(
                        np.maximum(np.abs(q - previous) - args.step_soft_threshold, 0.0)
                    )
                    local = (
                        args.reward_band * band
                        + args.reward_deviation * deviation
                        + args.reward_delta * delta_component
                        + args.reward_q * q_component
                        + args.reward_step_clip * step_clip_component
                    )
                    recovered = bool(np.all((next_state > 0.955) & (next_state < 1.045)))
                    reward = -(0.5 * local + 0.5 * local.mean())
                    if recovered:
                        reward += args.reward_success
                    counts["successful_steps"] += 1
                except (ValueError, pp.powerflow.LoadflowNotConverged) as exc:
                    counts["failed_steps"] += 1
                    next_state = state.copy()
                    band = deviation = delta_component = q_component = step_clip_component = np.zeros(NUM_AGENTS, dtype=np.float32)
                    reward = np.full(NUM_AGENTS, -args.reward_failure, dtype=np.float32)
                    failed = True
                    failure_reason = str(exc) or type(exc).__name__
                timing["powerflow"] += time.perf_counter() - tick
                reward_sums["band"] += float(band.sum())
                reward_sums["deviation"] += float(deviation.sum())
                reward_sums["delta"] += float(delta_component.sum())
                reward_sums["q2"] += float(q_component.sum())
                reward_sums["step_clip"] += float(step_clip_component.sum())
                reward_sums["success"] += float(args.reward_success * NUM_AGENTS if recovered else 0.0)
                reward_sums["failure"] += float(args.reward_failure * NUM_AGENTS if failed else 0.0)
                reward_sums["total"] += float(reward.sum())
                terminal = bool(recovered or failed)
                tick = time.perf_counter()
                replay.push(state, features, previous, q, reward, next_state, features, [float(terminal)])
                timing["reward_replay"] += time.perf_counter() - tick
                if counts["interactions"] >= 512 and len(replay.rows) >= 128:
                    tick = time.perf_counter()
                    batch = sample_replay_fast(replay, 128, 0.5, 0.75)
                    timing["sample"] += time.perf_counter() - tick
                    tick = time.perf_counter()
                    losses = learner.update(batch)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    timing["update"] += time.perf_counter() - tick
                    if any(value is not None and not np.isfinite(value) for value in losses.values()):
                        raise ValueError("nonfinite learner loss")
                    if learner.updates % 3 == 0:
                        tick = time.perf_counter()
                        mirror.sync(learner.actor)
                        timing["actor_sync"] += time.perf_counter() - tick
                state, previous = next_state, q
                if terminal:
                    break
            if len(used_graphs) in (1, 10, 30, 50) or len(used_graphs) % 200 == 0:
                tick = time.perf_counter()
                baseline = Feeder()
                baseline.reset(scene)
                baseline.step(previous)
                error = float(np.nanmax(np.abs(baseline.voltage - env.voltage)))
                solver_checks.append({
                    "interactions": counts["interactions"],
                    "graph": topology["id"],
                    "max_voltage_abs": error,
                    "passed": error <= 1e-6,
                })
                if error > 1e-6:
                    raise AssertionError(f"fast solver mismatch {error}")
                timing["solver_check"] += time.perf_counter() - tick
            episode = {
                "interactions": counts["interactions"],
                "num_graphs": len(used_graphs),
                "num_full_x": len(used_x),
                "operating_point_groups": operating_point_groups,
                "episode_steps": actual_steps,
                "recovered": recovered,
                "failed": failed,
                "failure_reason": failure_reason,
                "truncated": bool(actual_steps == episode_budget and not recovered and not failed),
                "noise_scale": 0.25 if counts["interactions"] <= 512 else 0.08,
                "reward_components": reward_sums,
                "return_per_agent_step": reward_sums["total"] / max(1, NUM_AGENTS * actual_steps),
                "losses": losses,
                "scene": scene,
                "counts": counts.copy(),
            }
            episodes.append(episode)
            append(run / "training.jsonl", episode)
            save_json(run / "progress.json", {
                "interactions": counts["interactions"],
                "graphs": len(used_graphs),
                "full_x": len(used_x),
                "updates": learner.updates,
                "timing": timing,
                "elapsed": elapsed(),
            })
            if counts["failed_steps"] >= 5 and counts["failed_steps"] / counts["interactions"] > 0.01:
                reason = "training_failure_stop"
                break
            if counts["interactions"] >= next_eval:
                row = assess(force_x=args.smoke)
                interval = 500 if row["performance"]["recovery_median"] < 30 else args.eval_every
                next_eval = counts["interactions"] + interval
                if full_passes >= 2 and not args.smoke and not args.disable_confirmation and (
                    confirmation_result is None or not confirmation_result["gate"]["passed"]
                ):
                    tick = time.perf_counter()
                    actor_state = {key: value.detach().cpu().clone() for key, value in mirror.actor.state_dict().items()}
                    candidate_confirmation = evaluate_pool(pool, confirmation, actor_state, mode="policy", workers=args.workers)
                    confirmation_gate = performance_gate(candidate_confirmation["metrics"], reference_confirmation)
                    timing["evaluation"] += time.perf_counter() - tick
                    confirmation_result = {
                        "interactions": counts["interactions"],
                        "metrics": candidate_confirmation["metrics"],
                        "reference": reference_confirmation,
                        "gate": confirmation_gate,
                        "rows": candidate_confirmation["rows"],
                    }
                    save_json(run / "confirmation.json", confirmation_result)
                    if confirmation_gate["passed"]:
                        checkpoint("controller_checkpoint.pt")
                        if not args.force_full_budget:
                            reason = "target_confirmed"
                            break
                    full_passes = 0
                if x_collapse_passes >= 2 and not args.smoke and not args.force_full_budget:
                    reason = "topology_utility_stop"
                    break
            if args.smoke and counts["interactions"] >= args.max_interactions:
                reason = "smoke_budget_complete"
                break
        if not records or records[-1]["interactions"] != counts["interactions"]:
            assess(force_x=args.smoke)

    final_checkpoint = checkpoint("final_checkpoint.pt")
    result = {
        "status": reason,
        "counts": counts,
        "num_graphs": len(used_graphs),
        "num_full_x": len(used_x),
        "operating_point_groups": operating_point_groups,
        "timing": timing,
        "elapsed": elapsed(),
        "records": len(records),
        "final_record": records[-1],
        "confirmation": confirmation_result,
        "solver_checks": solver_checks,
        "train_evaluation_graph_overlap": len(used_graphs & excluded),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else 0.0,
        "final_checkpoint": {"path": str(final_checkpoint), "sha256": sha(final_checkpoint)},
    }
    save_json(run / "result.json", result)
    print(json.dumps({"completed": args.run_id, "status": reason, "interactions": counts["interactions"], "result": str(run / 'result.json')}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--candidates", type=int, default=640)
    prepare_parser.add_argument("--workers", type=int, default=4)
    calibrate_parser = sub.add_parser("calibrate")
    calibrate_parser.add_argument("--workers", type=int, default=4)
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--checkpoint", type=Path, required=True)
    audit_parser.add_argument("--name", required=True)
    audit_parser.add_argument("--workers", type=int, default=4)
    for name, smoke in (("smoke", True), ("train", False)):
        child = sub.add_parser(name)
        child.set_defaults(smoke=smoke)
        child.add_argument("--run-id", required=True)
        child.add_argument("--seed", type=int, default=2601)
        child.add_argument("--critic-init-seed", type=int,
                           help="Change critic initialization while preserving the actor and training RNG streams")
        child.add_argument("--critic-output-init", choices=("random", "zero"), default="random",
                           help="Zero both critic output heads; use identical initialization in matched reward arms")
        child.add_argument("--actor-lr", type=float, default=2.5e-5)
        child.add_argument("--critic-lr", type=float, default=1e-3)
        child.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                           help="CPU is an explicit diagnostic fallback when CUDA is unavailable")
        child.add_argument("--discount", type=float, default=0.99,
                           help="TD discount; 1.0 makes terminal success reward independent of recovery time")
        child.add_argument("--initial-slope-factor", type=float, default=0.05)
        child.add_argument("--max-interactions", type=int, default=1000 if smoke else 30000)
        child.add_argument("--eval-every", type=int, default=1000)
        child.add_argument("--workers", type=int, default=4)
        child.add_argument("--topologies-per-operating-point", type=int, default=1)
        child.add_argument("--reward-q", type=float, default=0.0)
        child.add_argument("--reward-band", type=float, default=1.0)
        child.add_argument("--reward-deviation", type=float, default=0.001)
        child.add_argument("--reward-delta", type=float, default=0.2)
        child.add_argument("--reward-step-clip", type=float, default=0.0,
                           help="Penalty on squared applied step beyond the soft threshold")
        child.add_argument("--step-soft-threshold", type=float, default=2.0)
        child.add_argument("--disable-confirmation", action="store_true",
                           help="Keep the previously used confirmation panel out of reward exploration")
        child.add_argument("--reward-success", type=float, default=0.0,
                           help="Per-agent bonus only on the first successful recovery transition")
        child.add_argument("--reward-failure", type=float, default=100.0,
                           help="Per-agent penalty for a hard voltage or power-flow failure")
        child.add_argument("--force-full-budget", action="store_true",
                           help="Continue to the interaction budget after acceptance or X-gate stop; numerical safety stops remain active")
        child.add_argument("--initial-evaluation-cache", type=Path,
                           help="Reuse a verified matching initial-actor evaluation; cache computation is reported separately")
        child.add_argument("--initial-actor-checkpoint", type=Path,
                           help="Explicitly start from this frozen actor state, useful for matched CPU fallback")
    args = parser.parse_args()
    if args.command == "prepare":
        manifest_protocol.prepare(args.candidates, args.workers)
    elif args.command == "calibrate":
        calibrate(args.workers)
    elif args.command == "audit":
        audit_checkpoint(args.checkpoint, args.name, args.workers)
    else:
        if args.topologies_per_operating_point < 1:
            parser.error("--topologies-per-operating-point must be at least 1")
        if min(args.reward_q, args.reward_band, args.reward_deviation,
               args.reward_delta, args.reward_step_clip,
               args.reward_success, args.reward_failure) < 0:
            parser.error("reward coefficients must be nonnegative")
        if not 0.0 <= args.step_soft_threshold < ACTION_STEP_LIMIT:
            parser.error("--step-soft-threshold must lie in [0, 2.5)")
        if not 0.0 < args.discount <= 1.0:
            parser.error("--discount must lie in (0, 1]")
        run_training(args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
