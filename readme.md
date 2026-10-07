# RLC-FT: Provably Stable Multi-Agent Reinforcement Learning for Voltage Control

This repository contains the official implementation of the RLC-FT framework. RLC-FT is a multi-agent reinforcement learning controller designed for voltage regulation in distribution grids under flexible network topologies. The architecture guarantees voltage safety by construction, satisfying stability conditions without relying on external safety filters or post-hoc optimization steps.

## System Requirements

### Hardware Requirements
To reproduce the training and evaluation results, we recommend a machine with the following specifications:
* **RAM:** 16+ GB
* **CPU:** 4+ cores, 3.3+ GHz/core
* **GPU:** CUDA-enabled GPU is recommended for neural network training.

### Software Requirements
The codebase is tested on Windows operating systems using Python 3.11+. 

You can set up the environment using `uv` or `conda`. We provide configuration files for both package managers.

**Option 1: Using uv (Recommended)**
```bash
uv pip install -r requirements.txt
```

**Option 2: Using Conda**
```bash
conda env create -f "conda environment.yaml"
conda activate PowerSystem
```

## Repository Structure
* `data/`: Grid models and daily load/PV profiles.
* `Environment.py`: The 56-bus and 123-bus power-flow environments.
* `NN_Module.py`: RLC-FT and Safe-DDPG policy networks used by the performance evaluations.
* `src/softplus_controller.py`: Shared Softplus RLC-FT architecture.
* `Train.py`, `DDPG.py`, `TD3.py`, and `Utils.py`: Controller training and replay-buffer implementation.
* `experiments/`: Training-efficiency and operational experiment entries.
* `config.py`: Hyperparameters and the checkpoint/result directory, `Config.data_path`.

Run commands from the repository root. Set `Config.data_path` to your checkpoint and result directory, and set the controller checkpoint paths in the evaluation notebooks' setup cells.

## Training a New Model
For the controllers used in the performance evaluations, select `ENV = '56bus'` or `ENV = '123bus'` in `Train.py`, set the hyperparameters in `config.py`, and run:

```bash
python Train.py
```

The training-efficiency experiments use the following entries:

| System | Training entry |
| --- | --- |
| 56-bus | `experiments/validated/softplus_training_56bus/run.py` |
| 123-bus | `experiments/validated/softplus_training_123bus/run.py` |

For example, to train a 56-bus controller with the training-efficiency settings:

```bash
python experiments/validated/softplus_training_56bus/run.py prepare
python experiments/validated/softplus_training_56bus/run.py calibrate
python experiments/validated/softplus_training_56bus/run.py train --run-id training_56bus_seed2601 --seed 2601 --initial-slope-factor 0.24 --reward-delta 0.35 --max-interactions 25000 --workers 2 --force-full-budget
```

Use seeds `2601`, `2602`, and `2603` with separate run IDs for the three 56-bus training runs. Both training entries provide their options through `--help`.

The training directories contain the required model, environment and evaluation helpers. Shared execution utilities are in `experiments/validated/training_runtime/`.

## Reproducing Paper Results

| Experiment | Entry |
| --- | --- |
| 56-bus recovery time and transient costs, 5000 scenarios | `test_56bus_performance.ipynb` |
| 123-bus recovery time and transient costs, 5000 scenarios | `test_123_performance.ipynb` |
| Training efficiency | The 56-bus and 123-bus training entries listed above |
| 24-hour branch isolation and reconnection, 30 scenarios | `experiments/validated/operational_robustness/reproduce.py` |

For the performance comparisons, run the notebook setup and controller-evaluation cells. The results are saved under `Config.data_path/cache/notebook_results/`.

For the 30-scenario operational experiment:

```bash
python experiments/validated/operational_robustness/reproduce.py prepare
python experiments/validated/operational_robustness/reproduce.py run
python experiments/validated/operational_robustness/reproduce.py report
```

Results are saved under `Config.data_path/experiments/operational_path_shift/20260916_robustness_validation/subset30/`.

## Supplementary Experiments

| Experiment | Entry |
| --- | --- |
| Recovery-time distributions | `test_recovery_over_time_distribution.ipynb` |
| Topology-dependent policy behavior | `test_policy_adaptation.ipynb` |
| Voltage and control trajectories | `new_trajectory.ipynb`, `test_policy_output.ipynb` |
| Number of controllable buses | `subset.ipynb` |
| Stability slope bound | `test_alpha.ipynb` |
| Admittance stress | `test_56bus_admittance_stress.ipynb` |
| Communication loss, delay, and topology-information error | `perf.ipynb`, `test_extra_topo_error.ipynb`, `test_real_world.ipynb` |
