"""Main training entry; implementations remain in their experiment modules."""
import argparse
import sys

from src.common import run_script

ENTRIES = {
    "56bus": "experiments/validated/softplus_training_56bus/run.py",
    "123bus": "experiments/validated/softplus_training_123bus/run.py",
    "performance": "experiments/validated/performance_training/train.py",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        description="Train RLC-FT controllers using the published implementations.",
        epilog="Examples: train.py 56bus train --help; train.py 123bus --help",
    )
    parser.add_argument("experiment", choices=ENTRIES)
    parser.add_argument("options", nargs=argparse.REMAINDER,
                        help="Options forwarded unchanged to the selected implementation")
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return 0
    if argv[0] not in ENTRIES:
        parser.error("choose 56bus, 123bus, or performance")
    experiment, options = argv[0], argv[1:]
    if experiment == "performance" and options:
        if options in (["-h"], ["--help"]):
            print("Original performance training: select ENV in "
                  "experiments/validated/performance_training/train.py and "
                  "hyperparameters in configs/config.py, then run:\n"
                  "  python train.py performance")
            return 0
        parser.error("performance settings are edited in its training file and configs/config.py")
    return run_script(ENTRIES[experiment], options)


if __name__ == "__main__":
    raise SystemExit(main())
