"""Main performance/operational evaluation entry."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from src.common import PROJECT_ROOT, run_script

NOTEBOOKS = {
    "56bus": "experiments/validated/performance_evaluation/test_56bus_performance.ipynb",
    "123bus": "experiments/validated/performance_evaluation/test_123_performance.ipynb",
}
OPERATION = "experiments/validated/operational_robustness/reproduce.py"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="Run the published performance notebooks or operational experiment.",
        epilog="Examples: evaluate.py 56bus --dry-run; evaluate.py operation report",
    )
    parser.add_argument("experiment", choices=(*NOTEBOOKS, "operation"))
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] == "operation":
        return run_script(OPERATION, argv[1:])
    if argv[0] not in NOTEBOOKS:
        parser.error("choose 56bus, 123bus, or operation")
    options = argparse.ArgumentParser(
        prog=f"evaluate.py {argv[0]}",
        description="Execute the original performance notebook, including its "
                    "fixed-topology comparison, and save the executed notebook to the result directory.",
    )
    options.add_argument("--output-dir", type=Path,
                         help="Executed-notebook directory; default is under Config.data_path")
    options.add_argument("--timeout", type=int, default=-1,
                         help="Maximum seconds per cell; -1 means no time limit")
    options.add_argument("--dry-run", action="store_true",
                         help="Show the notebook and command without running an evaluation")
    args = options.parse_args(argv[1:])
    from configs.config import Config
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or Path(Config.data_path) / "evaluations" / f"{argv[0]}_{stamp}"
    notebook = PROJECT_ROOT / NOTEBOOKS[argv[0]]
    command = [
        sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
        f"--ExecutePreprocessor.timeout={args.timeout}",
        "--output-dir", str(output), "--output", notebook.stem + "_executed",
        str(notebook),
    ]
    if args.dry_run:
        print(f"Notebook: {notebook}\nOutput: {output}")
        print(subprocess.list2cmdline(command))
        return 0
    if importlib.util.find_spec("nbconvert") is None:
        parser.error("Notebook execution requires nbconvert. Install repository requirements "
                     "with: python -m pip install -r requirements.txt")
    output.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
