"""Shared file I/O and process launcher for public experiment entries."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path, arguments):
    """Run the existing implementation with its own normal Python __main__."""
    script = PROJECT_ROOT / relative_path
    if not script.is_file():
        raise FileNotFoundError(script)
    return subprocess.call(
        [sys.executable, "-B", str(script), *arguments], cwd=PROJECT_ROOT
    )


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False) + "\n")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
