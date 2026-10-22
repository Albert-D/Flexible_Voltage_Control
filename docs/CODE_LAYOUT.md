# Code layout

For experiment commands and settings, see the [reproduction guide](REPRODUCING_RESULTS.md).

## Main entries

| Command | Purpose |
| --- | --- |
| python train.py 56bus ... | 56-bus Softplus learning experiment |
| python train.py 123bus ... | 123-bus Softplus learning experiment |
| python train.py performance | Original performance-controller training |
| python evaluate.py 56bus | Original 56-bus performance notebook |
| python evaluate.py 123bus | Original 123-bus performance notebook |
| python evaluate.py operation ... | Operational experiment |

The training entries forward all options unchanged to the existing scripts.
Use --help after the experiment/subcommand to see its options. Use --dry-run
with a performance evaluation to inspect its notebook and output location.
The evaluation notebooks can also be opened directly.

## Shared implementation

- src/common.py: shared file I/O and the main-entry process launcher.
- src/models/: RLC-FT and Safe-DDPG model components.
- src/softplus_controller.py: the single shared Softplus actor implementation.
- src/environments/: original power-flow and retraining environments.
- src/agents/: DDPG, TD3 and replay buffers.
- src/runtime/: accepted power-flow/replay acceleration and dashboard helpers;
  fast_runtime.py also contains shared replay, reset and actor initialization.
- configs/: system configurations and hyperparameters.
- data/: feeder/profile input files; data/reference/ retains the previously
  uploaded historical reference cache.

## Experiment-specific entries

Research evaluations stay under experiments/validated/:

| Directory | Contents |
| --- | --- |
| softplus_training_56bus/, softplus_training_123bus/ | Training efficiency |
| performance_training/ | Original performance-controller training |
| performance_evaluation/ | Main performance notebooks |
| operational_robustness/ | Reproduction entry and scenario-specific simulations; shared simulation helpers in simulation/support.py |
| real_world_evaluation/ | Daily-profile and PV stress evaluations |
| supplementary_evaluation/ | Policy, controller design, recovery and imperfect-information notebooks |
| si_robustness_figures/, reward_ablation_figures/ | Existing paper figure programs |

The 56-bus evaluation.py also contains its reference-controller loader. The
123-bus model123.py contains both its controller and learner. The adopted
123-bus protocol is identified by adopted_run.json.

The operational prepare/run/audit/report commands are implemented directly in
operational_robustness/reproduce.py and reuse simulation/support.py. Specialized
experiments remain callable inside their own directories. Root training entries
do not import or run those evaluations.

## Maintaining the structure

Put reusable implementation in src/ and keep scenario-specific code beside
its experiment. New explorations belong in experiments/tmp/YYYY-MM-DD_topic/.
Move accepted experiments to experiments/validated/ and preserve replaced
experiments under experiments/history/. Add a short README with the entry,
settings and output location for each new topic.

Checkpoints, logs and generated figures use the existing D-drive result
directory configured in configs/config.py. Do not add new experiment scripts,
notebooks or output files at the repository root. tests/ is for software checks.

## Checks

Run the software checks with:

    python -B -m unittest discover -s tests -p test_public_layout.py -v

The command-line notebook evaluator uses nbconvert from requirements.txt.
Notebook source is retained without routine execution output. Four Plotly
payloads used by the existing figure programs remain as reference inputs.
