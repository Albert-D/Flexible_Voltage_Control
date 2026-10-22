"""Resolve source paths stored in historical operational cache metadata."""
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_RELOCATED = {
    "Environment.py": "src/environments/Environment.py",
    "NN_Module.py": "src/models/NN_Module.py",
    "config.py": "configs/config.py",
    "run_real_world_pv_outage_stress.py": "experiments/validated/real_world_evaluation/run_real_world_pv_outage_stress.py",
}
for _name in ("joint_search.py", "maintenance.py", "plot_robustness_validation.py",
              "review.py", "robustness_validation.py", "run.py", "validation_subset30.py"):
    _RELOCATED["experiments/tmp/2026-09-16_operational_path_shift/" + _name] = (
        "experiments/validated/operational_robustness/simulation/" + _name)
for _name in ("review_operational_direction_screen.py", "run_operational_direction_screen.py",
              "run_operational_priority_pilot.py"):
    _RELOCATED["tests/operational_candidates/" + _name] = (
        "experiments/validated/operational_robustness/screening/" + _name)


def source_path(name):
    """Map old relative/absolute locations; retain the recorded content hash."""
    normalized = str(name).replace("\\", "/")
    for old, current in _RELOCATED.items():
        if normalized == old or normalized.endswith("/" + old):
            return REPO / current
    path = Path(name)
    return path if path.is_absolute() else REPO / path


def check_source(name, expected, digest):
    path = source_path(name)
    if not path.is_file() or digest(path) != expected:
        raise RuntimeError(
            f"Historical cached source differs or is unavailable: {name} -> {path}. "
            "Its original source hash is retained; use report for saved metrics, "
            "or restore the matching historical source before resuming simulations.")
