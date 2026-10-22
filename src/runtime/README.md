# Shared training runtime

- Status: Validated shared runtime for the 56-bus and 123-bus training entries.
- Purpose: Reuse exact AC power-flow solves and accelerate actor inference and replay sampling.
- Entry: Imported by the controller training entries; no standalone training command.
- Acceptance: Solver equivalence, synchronized actor inference and replay/update equivalence checks.
- Outputs: Runtime measurements are recorded in each training run.
- Record: Training configuration and timing records.
- Runtime: lightsim2grid/recycle, CPU actor mirror, CUDA fused Adam and cached replay severity.

`fast_environment.py` provides the feeder adapter. `fast_runtime.py` provides the actor mirror, replay storage and sampling, optimizer setup, actor initialization and scene feasibility checks. File I/O helpers are shared through `src/common.py`.
