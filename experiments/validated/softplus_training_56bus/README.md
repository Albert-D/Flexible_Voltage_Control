# Softplus controller training: 56-bus system

- Status: Validated three-seed training-efficiency experiment.
- Purpose: Train and evaluate the topology-dependent controller on the SCE 56-bus system.
- Entry: `run.py prepare`, `run.py calibrate` and `run.py train`, from the repository root.
- Acceptance: The adopted settings use seeds 2601, 2602 and 2603, each with 25,000 environment interactions.
- Outputs: `D:/Code/Python/Flexible_Voltage_Control/experiments/unified_softplus_retraining/`.
- Record: Each run stores its configuration, training records, learning curves and checkpoint metadata.
- Runtime: Newton-Raphson AC power flow with lightsim2grid/recycle, synchronized CPU actor inference, CUDA fused Adam and cached replay severity.

`model.py` supplies the learner and uses the shared actor in `src/softplus_controller.py`. `protocol.py` prepares evaluation scenarios; `evaluation.py` evaluates the controllers.

See the repository README for the adopted training command. `run.py --help` lists all available commands.
