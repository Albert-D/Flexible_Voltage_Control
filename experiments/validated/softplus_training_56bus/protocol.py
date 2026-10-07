"""Prepare disjoint, physically stratified 56-bus evaluation manifests."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandapower as pp

warnings.filterwarnings("ignore", category=FutureWarning, module=r"pandapower\..*")


CODE_ROOT = Path(__file__).resolve().parents[3]
SPEED = CODE_ROOT / "experiments/validated/training_runtime"
for path in (CODE_ROOT, SPEED):
    sys.path.insert(0, str(path))

from fast_environment import feeder_class
from retraining_environment import BUS, SWITCH_LINES, Feeder, sample_scene, save_json
from experiments.validated.training_runtime.support import safe_reset

def draw_graph(env,rng,excluded,counters):
    for _ in range(200000):
        counters['graph_draws']+=1
        key,nodes,mask=env.signature((rng.random(len(SWITCH_LINES))<.5).tolist())
        if key not in excluded and set(BUS).issubset(nodes):
            return dict(id=key,switches=[bool(mask[i]) for i in SWITCH_LINES])
    raise RuntimeError('Unique graph sampling exhausted')

def scene_for(env,topology,rng,seed,counters):
    mult=rng.uniform(.5,1.5,len(env.x0)).tolist()
    for attempt in range(80):
        counters['reset_attempts']+=1
        scene=sample_scene(env,topology,seed+attempt*200000)
        scene.update(multipliers=mult,events=[])
        try:
            state=safe_reset(env,scene)
            scene.update(initial_v=state.tolist(),initial_safe=bool(np.all((state>.9499)&(state<1.0501))),features=env.features.tolist())
            return scene,state
        except (ValueError,pp.powerflow.LoadflowNotConverged):
            counters['reset_failures']+=1
    raise RuntimeError('Operating-point feasibility sampling exhausted')


def scenes_for_operating_point(env, topologies, rng, seed, counters):
    """Build several unique X scenes with exactly shared load and PV injections."""
    multipliers = [rng.uniform(0.5, 1.5, len(env.x0)).tolist() for _ in topologies]
    for attempt in range(80):
        actual_seed = int(seed + attempt * 200000)
        rows = []
        valid = True
        for topology, line_multipliers in zip(topologies, multipliers):
            counters["reset_attempts"] += 1
            scene = sample_scene(env, topology, actual_seed)
            scene.update(multipliers=line_multipliers, events=[])
            try:
                state = env.reset(scene)
                energized = np.asarray(env.voltage[list(env.nodes)], dtype=float)
                if state.min() < 0.80 or state.max() > 1.20:
                    raise ValueError("controlled reset voltage outside 0.80..1.20")
                if energized.min() < 0.75 or energized.max() > 1.25:
                    raise ValueError("energized reset voltage outside 0.75..1.25")
            except (ValueError, pp.powerflow.LoadflowNotConverged):
                counters["reset_failures"] += 1
                valid = False
                break
            scene.update(
                initial_v=state.tolist(),
                initial_safe=bool(np.all((state > 0.9499) & (state < 1.0501))),
                features=env.features.tolist(),
            )
            rows.append(scene)
        if valid:
            injection = np.concatenate(
                (
                    np.asarray(rows[0]["load_p"], dtype="<f4"),
                    np.asarray(rows[0]["load_q"], dtype="<f4"),
                    np.asarray(rows[0]["pv_p"], dtype="<f4"),
                )
            )
            injection_id = hashlib.sha256(injection.tobytes()).hexdigest()[:24]
            for member, row in enumerate(rows):
                row["operating_point_id"] = injection_id
                row["operating_point_member"] = member
                row["operating_point_reuse"] = len(rows)
            return rows
    raise RuntimeError("Matched operating-point feasibility sampling exhausted")


OUT = Path("D:/Code/Python/Flexible_Voltage_Control/experiments/unified_softplus_retraining")
PROTOCOL = OUT / "protocol.json"
FastFeeder = feeder_class(Feeder, BUS, "lightsim_recycle")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def full_x_id(features) -> str:
    values = np.asarray(features, dtype="<f4")
    return hashlib.sha256(values.tobytes()).hexdigest()[:24]


def _score_scene(scene):
    env = FastFeeder()
    base = env.reset(scene).astype(np.float64)
    epsilon = 0.10
    sensitivity = []
    for index in range(len(BUS)):
        q = np.zeros(len(BUS), dtype=np.float32)
        q[index] = epsilon
        voltage = env.step(q).astype(np.float64)
        sensitivity.append((voltage - base) / epsilon)
    env.step(np.zeros(len(BUS), dtype=np.float32))
    band = np.maximum(np.abs(base - 1.0) - 0.05, 0.0)
    row = dict(scene)
    row.update(
        x_id=full_x_id(scene["features"]),
        initial_severity=float(band.max()),
        initial_integrated_violation=float(band.sum()),
        qv_sensitivity=np.stack(sensitivity, axis=1).reshape(-1).tolist(),
    )
    return row


def _assign_quantiles(rows, key: str, bins: int = 4) -> None:
    order = sorted(range(len(rows)), key=lambda index: (rows[index][key], rows[index]["x_id"]))
    for rank, index in enumerate(order):
        rows[index][key + "_bin"] = min(bins - 1, rank * bins // len(rows))


def _stratified_pick(rows, count: int):
    cells = {}
    for row in rows:
        key = (row["initial_severity_bin"], row["topology_influence_bin"])
        cells.setdefault(key, []).append(row)
    for values in cells.values():
        values.sort(key=lambda row: row["x_id"])
    chosen = []
    keys = sorted(cells)
    while len(chosen) < count and any(cells.values()):
        for key in keys:
            if cells[key] and len(chosen) < count:
                chosen.append(cells[key].pop(0))
    if len(chosen) != count:
        raise RuntimeError(f"only selected {len(chosen)} of {count} stratified rows")
    selected = {row["x_id"] for row in chosen}
    remaining = [row for row in rows if row["x_id"] not in selected]
    return chosen, remaining


def _strip_sensitivity(rows):
    return [
        {key: value for key, value in row.items() if key != "qv_sensitivity"}
        for row in rows
    ]


def prepare(candidate_count: int = 640, workers: int = 4) -> dict:
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    if candidate_count < 576:
        raise ValueError("candidate_count must be at least 576")
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026092001)
    env = FastFeeder()
    counters = dict(graph_draws=0, reset_attempts=0, reset_failures=0)
    used_graphs = set()
    scenes = []
    while len(scenes) < candidate_count:
        topology = draw_graph(env, rng, used_graphs, counters)
        scene, _ = scene_for(
            env, topology, rng, 92000000 + len(scenes), counters
        )
        scene["x_id"] = full_x_id(scene["features"])
        used_graphs.add(topology["id"])
        scenes.append(scene)
        if len(scenes) % 64 == 0:
            print(json.dumps({"prepared_scenes": len(scenes), "seconds": time.perf_counter() - started}), flush=True)

    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
    ) as pool:
        scored = list(pool.map(_score_scene, scenes, chunksize=4))
    matrix = np.stack([row["qv_sensitivity"] for row in scored])
    center = np.median(matrix, axis=0)
    scale = max(float(np.linalg.norm(center)), 1e-12)
    for row, values in zip(scored, matrix):
        row["topology_influence"] = float(np.linalg.norm(values - center) / scale)
    _assign_quantiles(scored, "initial_severity")
    _assign_quantiles(scored, "topology_influence")

    development, remaining = _stratified_pick(scored, 128)
    x_panel = sorted(
        remaining,
        key=lambda row: (-row["topology_influence"], row["x_id"]),
    )[:64]
    x_ids = {row["x_id"] for row in x_panel}
    remaining = [row for row in remaining if row["x_id"] not in x_ids]
    x_matrix = np.stack([row["qv_sensitivity"] for row in x_panel])
    for index, row in enumerate(x_panel):
        distance = np.linalg.norm(x_matrix - x_matrix[index], axis=1)
        distance[index] = -1.0
        other = int(np.argmax(distance))
        row["wrong_features"] = x_panel[other]["features"]
        row["wrong_x_id"] = x_panel[other]["x_id"]
        row["mismatch_sensitivity_distance"] = float(distance[other])

    confirmation, remaining = _stratified_pick(remaining, 256)
    freshness, remaining = _stratified_pick(remaining, 64)
    groups = {
        "development_performance": _strip_sensitivity(development),
        "development_x": _strip_sensitivity(x_panel),
        "confirmation": _strip_sensitivity(confirmation),
        "freshness": _strip_sensitivity(freshness),
    }
    graph_sets = [{row["topology_id"] for row in rows} for rows in groups.values()]
    x_sets = [{row["x_id"] for row in rows} for rows in groups.values()]
    assert all(not a & b for i, a in enumerate(graph_sets) for b in graph_sets[i + 1:])
    assert all(not a & b for i, a in enumerate(x_sets) for b in x_sets[i + 1:])
    excluded = sorted(set().union(*graph_sets))
    payload = {
        "schema": "unified_softplus_56_v1",
        "created": "2026-09-20",
        "groups": groups,
        "excluded_graphs": excluded,
        "distribution": {
            "switch_closed_probability": 0.5,
            "line_admittance": "independent uniform 0.5..1.5",
            "load": "retained; existing corrected sample_scene",
            "full_x": "effective energized-line mask times per-line admittance multiplier",
        },
        "selection": {
            "candidate_count": candidate_count,
            "development_performance": "4x4 severity/influence stratified, deterministic x_id order",
            "development_x": "64 highest physical Q-to-V influence among remaining candidates",
            "confirmation": "4x4 severity/influence stratified after development exclusion",
            "freshness": "4x4 severity/influence stratified after confirmation exclusion",
            "controller_performance_used": False,
        },
        "checks": {
            "unique_graphs": len(excluded),
            "unique_full_x": len(set().union(*x_sets)),
            "graph_overlap": 0,
            "full_x_overlap": 0,
            "sizes": {key: len(value) for key, value in groups.items()},
        },
        "preparation": {
            "workers": workers,
            "counters": counters,
            "powerflow_counts_main": env.counts,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    save_json(PROTOCOL, payload)
    save_json(OUT / "protocol_sha256.json", {"path": str(PROTOCOL), "sha256": sha(PROTOCOL)})
    print(json.dumps({"protocol": str(PROTOCOL), **payload["checks"], "wall_seconds": payload["preparation"]["wall_seconds"]}), flush=True)
    return payload


if __name__ == "__main__":
    prepare()
