"""Deterministic fixed-panel evaluation and internal topology-use diagnostics."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandapower as pp
import torch

warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandapower\..*")


CODE_ROOT = Path(__file__).resolve().parents[3]
SPEED = CODE_ROOT / "experiments/validated/training_runtime"
for path in (CODE_ROOT, SPEED):
    sys.path.insert(0, str(path))

from fast_environment import feeder_class
from reference import load_reference
from retraining_environment import BUS, Feeder
from src.softplus_controller import SoftplusControllerSpec, SoftplusTopologyController

from model import ACTION_LIMIT, ACTION_STEP_LIMIT, NUM_AGENTS, TOPOLOGY_DIM


FastFeeder = feeder_class(Feeder, BUS, "lightsim_recycle")
SPEC = SoftplusControllerSpec(
    topology_dim=TOPOLOGY_DIM,
    num_agents=NUM_AGENTS,
    hidden=2048,
    topology_hidden=256,
)
_env = _actor = _reference = None


def init_worker():
    global _env, _actor, _reference
    torch.set_num_threads(1)
    _env = FastFeeder()
    _actor = SoftplusTopologyController(SPEC).cpu().eval()
    _reference, _ = load_reference(torch.device("cpu"))
    _reference.eval()


def extract_actor_state(payload):
    if isinstance(payload, dict) and "learner" in payload:
        return payload["learner"]["actor"]
    if isinstance(payload, dict) and "actor" in payload:
        candidate = payload["actor"]
        if isinstance(candidate, dict):
            return candidate
    if isinstance(payload, dict) and payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise ValueError("cannot locate actor state_dict")


def rollout(model, scene, override_features=None, reference=False, horizon=200):
    env = _env if _env is not None else FastFeeder()
    env.counts = dict(reset=0, step=0, event=0, failed=0)
    voltage = env.reset(scene).astype(np.float32)
    q = np.zeros(NUM_AGENTS, dtype=np.float32)
    alpha_q = 1.0
    jv = jq = integrated_violation = 0.0
    max_step = max_q = 0.0
    step_hits = q_hits = 0
    failed = None
    recovered = False
    recovery = horizon
    trajectory = []
    features = np.asarray(
        env.features if override_features is None else override_features,
        dtype=np.float32,
    )
    for step in range(1, horizon + 1):
        jv += float(100.0 * np.square(voltage - 1.0).sum())
        jq += float(alpha_q * np.square(q).sum())
        integrated_violation += float(np.maximum(np.abs(voltage - 1.0) - 0.05, 0.0).sum())
        with torch.no_grad():
            delta = model(
                torch.as_tensor(voltage).view(1, NUM_AGENTS),
                torch.as_tensor(features).view(1, TOPOLOGY_DIM),
            ).numpy().reshape(NUM_AGENTS)
        clipped_delta = np.clip(delta, -ACTION_STEP_LIMIT, ACTION_STEP_LIMIT)
        new_q = np.clip(q - clipped_delta, -ACTION_LIMIT, ACTION_LIMIT).astype(np.float32)
        max_step = max(max_step, float(np.max(np.abs(new_q - q))))
        max_q = max(max_q, float(np.max(np.abs(new_q))))
        step_hits += int(np.any(np.abs(delta) >= ACTION_STEP_LIMIT - 1e-6))
        q_hits += int(np.any(np.abs(new_q) >= ACTION_LIMIT - 1e-6))
        try:
            next_voltage = env.step(new_q)
            if not np.isfinite(next_voltage).all() or next_voltage.min() < 0.75 or next_voltage.max() > 1.25:
                raise ValueError("hard voltage bound")
        except (ValueError, pp.powerflow.LoadflowNotConverged) as exc:
            failed = str(exc)
            remaining = horizon - step + 1
            jv += 1000.0 * remaining
            break
        if len(trajectory) < 12:
            trajectory.append({"step": step, "voltage": next_voltage.tolist(), "q": new_q.tolist()})
        voltage, q = next_voltage.astype(np.float32), new_q
        if bool(np.all((voltage >= 0.95) & (voltage <= 1.05))):
            recovered = True
            recovery = step
            break
    return {
        "seed": int(scene["seed"]),
        "topology_id": scene["topology_id"],
        "x_id": scene["x_id"],
        "reference": bool(reference),
        "success": bool(recovered and failed is None),
        "failure": failed,
        "recovery": int(recovery),
        "jv": float(jv),
        "jq": float(jq),
        "objective": float(jv + jq),
        "integrated_violation": float(integrated_violation),
        "max_action_increment": float(max_step),
        "max_abs_q": float(max_q),
        "action_step_hit_rate": float(step_hits / max(1, recovery if recovered else horizon)),
        "action_bound_hit_rate": float(q_hits / max(1, recovery if recovered else horizon)),
        "initial_severity": float(scene.get("initial_severity", 0.0)),
        "topology_influence": float(scene.get("topology_influence", 0.0)),
        "initial_safe": bool(scene.get("initial_safe", False)),
        "trajectory_head": trajectory,
        "counts": env.counts.copy(),
    }


def worker_job(payload):
    scenes, actor_state, mode = payload
    started = time.perf_counter()
    if mode != "reference":
        _actor.load_state_dict(actor_state, strict=True)
        model = _actor
    else:
        model = _reference
    rows = []
    for scene in scenes:
        override = scene.get("wrong_features") if mode == "mismatched" else None
        rows.append(rollout(model, scene, override, reference=mode == "reference"))
    return {"rows": rows, "worker_seconds": time.perf_counter() - started}


def evaluate_pool(pool, scenes, actor_state=None, mode="policy", workers=4):
    if mode not in ("policy", "reference", "mismatched"):
        raise ValueError(mode)
    started = time.perf_counter()
    size = max(1, (len(scenes) + workers - 1) // workers)
    jobs = [(scenes[index:index + size], actor_state, mode) for index in range(0, len(scenes), size)]
    results = list(pool.map(worker_job, jobs))
    rows = [row for result in results for row in result["rows"]]
    return {
        "rows": rows,
        "metrics": metrics(rows),
        "wall_seconds": time.perf_counter() - started,
        "worker_seconds": sum(result["worker_seconds"] for result in results),
    }


def metrics(rows):
    result = {
        "n": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "failures": sum(row["failure"] is not None for row in rows),
    }
    for key in (
        "recovery", "jv", "jq", "objective", "integrated_violation",
        "max_action_increment", "max_abs_q", "action_step_hit_rate",
        "action_bound_hit_rate",
    ):
        values = np.asarray([row[key] for row in rows], dtype=float)
        result[key + "_median"] = float(np.median(values))
        result[key + "_p90"] = float(np.quantile(values, 0.9))
    return result


def performance_gate(candidate, reference):
    checks = {
        "failures_zero": candidate["failures"] == 0,
        "success": candidate["success_rate"] >= 0.95,
        "recovery_median": 4.0 <= candidate["recovery_median"] <= 6.0,
        "recovery_p90": candidate["recovery_p90"] <= 1.25 * reference["recovery_p90"],
        # Reference-only calibration found median recovery 3 on this fixed panel,
        # while the preregistered candidate target remains 4--6. A 2x objective
        # guard prevents catastrophic cost without penalizing the intended slower
        # recovery. The sealed final audit still reports the exact SI objective.
        "objective": candidate["objective_median"] <= 2.00 * reference["objective_median"],
    }
    return {"passed": bool(all(checks.values())), "checks": checks}


def topology_probe(actor_state, scenes):
    actor = SoftplusTopologyController(SPEC).cpu().eval()
    actor.load_state_dict(actor_state, strict=True)
    features = torch.as_tensor(np.stack([row["features"] for row in scenes]), dtype=torch.float32)
    wrong = torch.as_tensor(np.stack([row["wrong_features"] for row in scenes]), dtype=torch.float32)
    voltage = torch.as_tensor(np.stack([row["initial_v"] for row in scenes]), dtype=torch.float32)
    with torch.no_grad():
        correct_delta = actor(voltage, features)
        wrong_delta = actor(voltage, wrong)
        gains = torch.stack(
            [policy.topology_gain(features)[:, 0] for policy in actor.policies],
            dim=1,
        ).numpy()
    action_difference = float(torch.mean(torch.abs(correct_delta - wrong_delta)).item() / ACTION_STEP_LIMIT)
    means = np.mean(np.abs(gains), axis=0)
    relative_span = np.ptp(gains, axis=0) / np.maximum(means, 1e-12)
    return {
        "normalized_action_difference": action_difference,
        "median_relative_gain_span": float(np.median(relative_span)),
        "per_agent_relative_gain_span": relative_span.tolist(),
        "correct_delta_abs_median": float(torch.median(torch.abs(correct_delta)).item()),
    }


def paired_x_report(correct_rows, mismatched_rows):
    assert [row["x_id"] for row in correct_rows] == [row["x_id"] for row in mismatched_rows]
    correct = np.asarray([row["objective"] for row in correct_rows], dtype=float)
    wrong = np.asarray([row["objective"] for row in mismatched_rows], dtype=float)
    relative = (wrong - correct) / np.maximum(correct, 1e-9)
    recovery_correct = np.asarray([row["recovery"] for row in correct_rows], dtype=float)
    recovery_wrong = np.asarray([row["recovery"] for row in mismatched_rows], dtype=float)
    return {
        "correct_success_rate": float(np.mean([row["success"] for row in correct_rows])),
        "mismatched_success_rate": float(np.mean([row["success"] for row in mismatched_rows])),
        "objective_relative_improvement_median": float(np.median(relative)),
        "objective_correct_win_fraction": float(np.mean(correct < wrong)),
        "objective_tie_fraction": float(np.mean(np.isclose(correct, wrong, rtol=1e-7, atol=1e-7))),
        "recovery_wrong_minus_correct_median": float(np.median(recovery_wrong - recovery_correct)),
        "recovery_changed_fraction": float(np.mean(recovery_wrong != recovery_correct)),
    }


def topology_gate(probe, paired):
    sensitivity = (
        probe["normalized_action_difference"] >= 0.02
        or probe["median_relative_gain_span"] >= 0.05
    )
    utility = (
        paired["correct_success_rate"] >= paired["mismatched_success_rate"]
        and paired["objective_relative_improvement_median"] > 0.0
        and paired["objective_correct_win_fraction"] > 0.55
    )
    return {
        "passed": bool(sensitivity and utility),
        "sensitivity_passed": bool(sensitivity),
        "utility_passed": bool(utility),
    }


__all__ = [
    "SPEC", "evaluate_pool", "extract_actor_state", "init_worker", "metrics",
    "paired_x_report", "performance_gate", "topology_gate", "topology_probe",
]
