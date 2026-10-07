# Softplus controller training: 123-bus system

- Status: Validated supplementary training-efficiency experiment.
- Purpose: Train and evaluate the topology-dependent controller on the 123-bus system.
- Entry: `run.py` trains and evaluates a controller; `prepare.py` prepares the evaluation protocol.
- Acceptance: The adopted run uses seed 2601 and 30,000 environment interactions; its selected checkpoint is at interaction 27,037.
- Outputs: `D:/Code/Python/Flexible_Voltage_Control/experiments/123bus_training_efficiency_2026-09-16/`.
- Record: Each run stores its configuration, training records, learning curves and checkpoint metadata.
- Runtime: Newton-Raphson AC power flow with lightsim2grid/recycle, synchronized CPU actor inference, CUDA fused Adam and cached replay severity.

Run from the repository root. With the prepared protocol and reference evaluations in the output directory:

```bash
python experiments/validated/softplus_training_123bus/run.py --run-id training_123bus_seed2601 --seed 2601 --actor-lr 1e-5 --critic-lr 1e-3 --delta-weight 0.001 --initial-factor 1 --noise 0.25 --warm-noise 0.75 --max-interactions 30000 --stop-at 30000 --wall-minutes 1440 --workers 2
```

`run.py --help` lists the training options. The environment, controller and evaluation helpers are maintained in this directory.
