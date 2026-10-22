# Softplus controller training: 123-bus system

- Status: Adopted supplementary training-efficiency configuration, delta010.
- Purpose: Reproduce the 30,000-interaction learning experiment used by the current SI.
- Entry: `run.py`; adopted provenance and required data hashes are in `adopted_run.json`.
- Acceptance: The frozen seed-2601 run ends at 30,000 interactions / 631 unique topologies. Its selected checkpoint has a 512-scenario acceptance median of 14 steps; the final development median is 15.
- Outputs: New runs default to `<Config.data_path>/experiments/123bus_recovery15_reproduction/`.
- Record: The unchanged adopted artifacts remain under `<Config.data_path>/experiments/123bus_recovery15_2026-10-02/`; `selection_lock.json` identifies `delta010/checkpoint_030000.pt` and `selected/controller.pt`.
- Runtime: Exact Newton-Raphson AC power flow with lightsim2grid/recycle; reset, topology, admittance and load changes rebuild the solver cache. One numerical thread, synchronized CPU actor mirror and CachedReplay with 50% hard sampling at the 75th severity quantile. CPU Adam is the adopted default; explicit CUDA runs use fused Adam. Timing separates reset, inference, power flow, replay, updates, synchronization, evaluation and checkpoint writes.

From the repository root, place the frozen protocol at the manifest's data-relative path (or supply `--protocol`), then run:

```bash
python experiments/validated/softplus_training_123bus/run.py --run-id delta010_seed2601 --seed 2601 --actor-lr 2.5e-5 --critic-lr 1e-3 --delta-weight 0.01 --noise 0.25 --warm-noise 0.75 --max-interactions 30000 --stop-at 30000 --eval-every 2000 --device cpu --workers 2
```

The entry verifies the frozen protocol SHA-256 before creating outputs. It uses its 64 development scenarios, 16 rotating scenarios per evaluation, excluded topology set and adopted scene seed stream. The actor is the shared `src/softplus_controller.py` implementation, with no output rescaling. Rewards, action limits, recovery/failure masks and TD3 updates follow the adopted October run. `--output-root` chooses a separate result directory. `--resume` accepts only this entry's matching source/configuration, preserving historical checkpoints and configurations.

`model123.py` contains the feeder-specific actor compatibility classes, critic and TD3 learner. `evaluation123.py` contains rollout and reference evaluation; `runtime.py` supplies the adopted training entry with its execution helpers. Shared runtime and file I/O are in `src/runtime/` and `src/common.py`.

The earlier 27,037-interaction September trajectory is a predecessor. Its separate preparation script is retained locally and is outside the public reproduction set. Its protocol and reported reference are not inputs to the current SI training curve. The current figure uses the frozen October run; retraining does not replace the adopted figure or checkpoint.
