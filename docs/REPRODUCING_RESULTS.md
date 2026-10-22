# Reproducing Paper Results

Run all commands from the repository root. Set the checkpoint and result
directory in [configs/config.py](../configs/config.py), and configure controller
checkpoint paths in each evaluation notebook's setup cells.

## Training Efficiency

The training-efficiency experiments use the following entries:

| System | Training entry |
| --- | --- |
| 56-bus | `experiments/validated/softplus_training_56bus/run.py` |
| 123-bus | `experiments/validated/softplus_training_123bus/run.py` |

For example, to train a 56-bus controller with the training-efficiency settings:

```bash
python train.py 56bus prepare
python train.py 56bus calibrate
python train.py 56bus train --run-id training_56bus_seed2601 --seed 2601 --initial-slope-factor 0.24 --reward-delta 0.35 --max-interactions 25000 --workers 2 --force-full-budget
```

Use seeds `2601`, `2602`, and `2603` with separate run IDs for the three 56-bus training runs. Both training entries provide their options through `--help`.

For the adopted 123-bus learning experiment, use the frozen protocol identified by `experiments/validated/softplus_training_123bus/adopted_run.json`:

```bash
python train.py 123bus --run-id delta010_seed2601 --seed 2601 --actor-lr 2.5e-5 --critic-lr 1e-3 --delta-weight 0.01 --noise 0.25 --warm-noise 0.75 --max-interactions 30000 --stop-at 30000 --eval-every 2000 --device cpu --workers 2
```

This is the October `delta010` configuration used by the current SI, with 30,000 interactions and no output rescaling. The entry checks the protocol hash and writes new runs to a separate reproduction directory. See the [123-bus training guide](../experiments/validated/softplus_training_123bus/README.md) and [manifest](../experiments/validated/softplus_training_123bus/adopted_run.json) for the frozen protocol, selected checkpoint, and `--protocol` / `--output-root` options.

The training directories contain the required model, environment and evaluation helpers. The 56-bus reference loader is included in `evaluation.py`; the 123-bus controller and learner share `model123.py`. Shared acceleration, replay and initialization utilities are in `src/runtime/fast_runtime.py`, and file I/O helpers are in `src/common.py`.

## Performance and Operational Results

| Experiment | Entry |
| --- | --- |
| 56-bus recovery time and transient costs, 5000 scenarios | `experiments/validated/performance_evaluation/test_56bus_performance.ipynb` |
| 123-bus recovery time and transient costs, 5000 scenarios | `experiments/validated/performance_evaluation/test_123_performance.ipynb` |
| Training efficiency | The 56-bus and 123-bus training entries listed above |
| 24-hour branch isolation and reconnection, 30 scenarios | `experiments/validated/operational_robustness/reproduce.py` |

For the performance comparisons, run python evaluate.py 56bus or python evaluate.py 123bus. Use --dry-run to inspect the command, or open the corresponding notebook directly and run its setup and controller-evaluation cells. The results are saved under `Config.data_path/cache/notebook_results/`.

For the 30-scenario operational experiment:

```bash
python evaluate.py operation prepare
python evaluate.py operation run
python evaluate.py operation report
```

The operational commands are implemented in `operational_robustness/reproduce.py`, with shared simulation helpers in `simulation/support.py`. Results are saved under `Config.data_path/experiments/operational_path_shift/20260916_robustness_validation/subset30/`.

## Supplementary Experiments

| Experiment | Entry |
| --- | --- |
| Recovery-time distributions | `experiments/validated/supplementary_evaluation/recovery/test_recovery_over_time_distribution.ipynb` |
| Topology-dependent policy behavior | `experiments/validated/supplementary_evaluation/policy/test_policy_adaptation.ipynb` |
| Voltage and control trajectories | `experiments/validated/supplementary_evaluation/policy/new_trajectory.ipynb`, `experiments/validated/supplementary_evaluation/policy/test_policy_output.ipynb` |
| Number of controllable buses | `experiments/validated/supplementary_evaluation/controller_design/subset.ipynb` |
| Stability slope bound | `experiments/validated/supplementary_evaluation/controller_design/test_alpha.ipynb` |
| Admittance stress | `experiments/validated/supplementary_evaluation/network_shifts/test_56bus_admittance_stress.ipynb` |
| Communication loss, delay, and topology-information error | `experiments/validated/supplementary_evaluation/imperfect_information/test_extra_topo_error.ipynb`, `experiments/validated/real_world_evaluation/test_real_world.ipynb` |
