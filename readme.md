# One Multi-Agent Controller Suffices to Provably Stabilize Voltage over Flexible Grid Topologies

This repository contains the official implementation of RLC-FT accompanying the paper above. RLC-FT is a multi-agent reinforcement learning controller designed for voltage regulation in distribution grids under flexible network topologies. The architecture guarantees voltage safety by construction, satisfying stability conditions without relying on external safety filters or post-hoc optimization steps.

## System Requirements

### Hardware Requirements
To reproduce the training and evaluation results, we recommend a machine with the following specifications:
* **RAM:** 16+ GB
* **CPU:** 4+ cores, 3.3+ GHz/core
* **GPU:** CUDA-enabled GPU is recommended for neural network training.

### Software Requirements
The codebase is tested on Windows operating systems using Python 3.11+. 

Install the dependencies using:

```bash
python -m pip install -r requirements.txt
```

## Repository Structure

The root exposes two main entries, [train.py](train.py) and [evaluate.py](evaluate.py).
Shared code and individual experiments live in the following directories:

| Directory | Contents |
| --- | --- |
| src/ | Shared models, power-flow environments, learning algorithms and accelerated runtime |
| configs/ | System configurations and hyperparameters |
| experiments/validated/ | Training, performance evaluation, operational and supplementary experiments, grouped by topic |
| data/ | Grid models, daily profiles and historical reference data |
| tests/ | Software and entry-point checks |
| docs/ | [Detailed code map](docs/CODE_LAYOUT.md) and reference material |

Run commands from the repository root. Set Config.data_path in configs/config.py
to your checkpoint and result directory, and set controller checkpoint paths in
the evaluation notebooks' setup cells. Executed notebooks are saved under the
configured result directory.

## Training a New Model
For the controllers used in the performance evaluations, select `ENV = '56bus'` or `ENV = '123bus'` in `experiments/validated/performance_training/train.py`, set the hyperparameters in `configs/config.py`, and run:

```bash
python train.py performance
```

## Reproducing Paper Results

See the [reproduction guide](docs/REPRODUCING_RESULTS.md) for training-efficiency
settings, performance comparisons, operational tests and supplementary experiments.
