"""Invoke the hash-frozen original AC simulation protocol when a replay is needed."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
SOURCE = REPO / "experiments/tmp/2026-09-16_operational_path_shift/validation_subset30.py"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "run", "audit", "report"])
    args = parser.parse_args()
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Frozen original simulation entry is unavailable: {SOURCE}")
    subprocess.run([sys.executable, "-B", str(SOURCE), args.action], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
