# Softplus controller training: 56-bus system

- Status: Validated three-seed training-efficiency experiment.
- Purpose: Train and evaluate the topology-dependent controller on the SCE 56-bus system.
- Entry: `run.py prepare`, `run.py calibrate` and `run.py train`, from the repository root.
- Acceptance: The adopted settings use seeds 2601, 2602 and 2603, each with 25,000 environment interactions.
- Outputs: `D:/Code/Python/Flexible_Voltage_Control/experiments/unified_softplus_retraining/`.
- Record: Each run stores its configuration, training records, learning curves and checkpoint metadata.
- Runtime: Newton-Raphson AC power flow with lightsim2grid/recycle, synchronized CPU actor inference, CUDA fused Adam and cached replay severity.

`model.py` supplies the learner and uses the shared actor in `src/softplus_controller.py`. `protocol.py` prepares evaluation scenarios; `evaluation.py` evaluates the controllers and loads the reference policies. File I/O helpers are shared through `src/common.py`.

Run from the repository root:

```bash
python experiments/validated/softplus_training_56bus/run.py --help
python experiments/validated/softplus_training_56bus/run.py prepare
python experiments/validated/softplus_training_56bus/run.py calibrate
python experiments/validated/softplus_training_56bus/run.py train --run-id training_56bus_seed2601 --seed 2601 --initial-slope-factor 0.24 --reward-delta 0.35 --max-interactions 25000 --workers 2 --force-full-budget
```

Shared exact power-flow and optimization helpers are in `src/runtime/`; the environment is in `src/environments/retraining_environment.py`. Each new run records the relocated source files and hashes. Existing frozen results and checkpoints are unchanged.
