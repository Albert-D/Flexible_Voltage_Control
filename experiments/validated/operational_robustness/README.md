# Operational robustness under branch maintenance

- Status: Validated 30-scenario operational experiment.
- Purpose: Compare Linear, Safe-DDPG and RLC-FT controllers during 24-hour load-branch isolation and restoration on the SCE 56-bus system.
- Entry: `reproduce.py prepare|run|audit|report`, from the repository root.
- Acceptance: 30 scenarios, ten at each of three maintenance branches, with complete trajectories for all three controllers and archived replay checks.
- Outputs: `Config.data_path/experiments/operational_path_shift/20260916_robustness_validation/subset30/`.
- Record: Scenario manifest, metrics and replay audit in the output directory.
- Runtime: AC power-flow evaluation with a six-second control interval and fixed controller parameters.

`reproduce.py audit` checks saved trajectories; `reproduce.py report` generates the result summary. The simulation implementation is in `experiments/tmp/2026-09-16_operational_path_shift/`; `reproduce.py` provides the experiment entry.

The figure shows one example day in panels a-c and statistics across the 30 scenarios in panels d-f.
