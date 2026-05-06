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
conda env create -f conda_environment.yaml
conda activate <your_env_name>
```

## Repository Structure
The codebase is modularized to separate the power system environment, neural network architectures, and experiment configurations.

* `data/`: Contains the original grid topology and load/generation profiles.

* `Environment.py`: Defines the power distribution network environment.
* `NN_Module.py` & `Hyper_Net_Module.py`: Contains the implementation of the RLC-FT policy networks.
* `Train.py`: The main training script.
* `DDPG.py` & `TD3.py`: Implementations of the foundational reinforcement learning algorithms used for optimization.
* `dashboard.py`: Auxiliary tools for interactive visualization of the grid states and control trajectories.
* `config.py`: Configuration and hyperparameter settings.

## Training a New Model
To train the RLC-FT controller from scratch, ensure your desired configuration is set in the respective `config` file, then execute the main training script:

```bash
python Train.py
```

## Reproducing Paper Results
We provide standalone scripts and Jupyter notebooks to reproduce the specific experimental scenarios discussed in the manuscript's Results section.

### 1. Inherent Stability and Topology Adaptation of the Learned Policy
To verify the controller's structural adaptation to dynamic network topology changes:

- **Evaluation:** Open and execute `test_policy_adaptation.ipynb`.

### 2. Exponential Stability Across Diverse Topologies via One Multi-agent Controller
To evaluate the comprehensive voltage regulation performance and trajectory stability on the 56-bus system across various topologies:

- **Evaluation:** Run `test_trajectory.py` for trajectory generation and explore `test_56bus_performance.ipynb` for performance metrics.

### 3. Scalability and Structural Generalization (IEEE 123-bus)
To test the framework's performance on the larger, high-dimensional test feeder:

- **Evaluation:** Open `test_123_performance.ipynb` and execute `test_123_trajectory.py`.

### 4. Robustness in Real-World Operational Scenarios
To evaluate the operational resilience using actual daily load profiles and dynamic PV integration across a 24-hour cycle on the SCE 56-bus system:

- **Evaluation:** Run `test_real_world_56bus.py` and `test_real_world.ipynb` to view the resulting voltage trajectories and power profiles.

### 5. Supplementary Evaluations
To reproduce the extended analyses and ablation studies detailed in the Supplementary Information:
* **Impact of Stability Parameter $\alpha$:** Open `test_alpha.ipynb` to evaluate the effect of the theoretical bound parameter $\alpha$ on system dynamics and conservatism.
* **Robustness to Communication Imperfections:** Execute `perf.ipynb` to analyze the system's tolerance to communication loss, broadcasting delays, and topology information errors.
