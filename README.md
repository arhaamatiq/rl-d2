# Babysteps

**How fast can one GPU teach a humanoid to walk robustly?**

Babysteps turns "time-to-walk" into a benchmark with a frozen finish line and then
tries to set records against it. The robot is the 29-DoF G1 humanoid in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp). The baseline is mjlab's
stock velocity-tracking PPO recipe ([rsl_rl](https://github.com/leggedrobotics/rsl_rl)).
Everything is measured on a single RTX 4090, from a cold start, over 5 seeds.

{{HEADLINE_TABLE}}

## The bar

A checkpoint "walks robustly" when, evaluated offline with fixed seeds, it
(1) tracks 24 held-out velocity commands within the frozen linear and yaw error
thresholds, (2) completes 20 s episodes without falling under randomized friction,
center of mass and encoder bias, (3) recovers from a fixed-strength push from 8
directions, and (4) still walks when the same weights are replayed in a different
simulator (CPU MuJoCo instead of MuJoCo Warp). Thresholds were calibrated on the stock
recipe with separate seeds and frozen before any optimization: [`RULES.md`](RULES.md).

**Time-to-bar** is wall-clock from launch (caches wiped, so imports, kernel
compilation and graph capture are counted) to the first checkpoint that passes.
Checkpoints are written every 30 s by every recipe.

## Where the time went (stock, profiled)

{{PROFILE_SECTION}}

## What changed

{{ABLATION_TABLE}}

## Reproduce

```bash
docker build -t babysteps .
docker run --gpus all babysteps headline    # Track A record, 5 seeds, prints median/IQR
docker run --gpus all babysteps baseline    # stock T0, 5 seeds
```

Without Docker (Ubuntu 24.04, CUDA driver >= 12.8):

```bash
bash scripts/pod_setup.sh
bash scripts/reproduce.sh headline
```

Single pieces:

```bash
scripts/train.sh <recipe> <seed> runs/<name> <budget_min>     # cold-start timed run
.venv/bin/python -m babysteps.eval.time_to_bar --run runs/<name>
.venv/bin/python scripts/profile_report.py <trace.sqlite>      # NVTX phase breakdown
```

## Repository map

| path | what |
|---|---|
| `bar/` | frozen bar spec and the sim-to-sim model hash |
| `src/babysteps/eval/` | evaluator: GPU tracking/falls/push, CPU sim-to-sim, time-to-bar scan |
| `src/babysteps/train/` | timing harness, stock runner, Track A runner, recipes |
| `src/babysteps/systems/` | CUDA-graph env step, graph-capturable PPO update, NVTX, symmetry |
| `scripts/` | official runs, reproduction, profiling, calibration, video |
| `tests/` | parity tests (graphed env vs stock, graphed PPO vs rsl_rl), CPU unit tests |
| `LEDGER.md` | every reported number with command, commit, seeds and artifact |
| `DECISIONS.md` | design decisions in the order they were made |

## Related work

* **FastSAC / FastTD3** — Seo, Sferrazza, Chen, Shi, Duan, Abbeel, "Learning Sim-to-Real
  Humanoid Locomotion in 15 Minutes" (arXiv:2512.01996, Dec 2025): off-policy RL with
  massively parallel simulation trains G1 and T1 locomotion in about 15 minutes on one
  RTX 4090. FastSAC is available in the open-source
  [Holosoma](https://github.com/amazon-far/holosoma) framework (Apache-2.0), which also
  supports an MJWarp backend. Different task definition and success criterion from this
  benchmark, so the numbers are not directly comparable.
* **Agents' 100m** — AI agents iterating on training a G1 humanoid to sprint.
* Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021).
* Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies"
  (ICRA 2024).

Babysteps does not claim to be the first fast humanoid locomotion result. Its
contribution is a fixed, robustness-checked finish line with an offline judge, and an
accounting of where single-GPU PPO training time goes.

## Credits

See [`ATTRIBUTION.md`](ATTRIBUTION.md). Apache-2.0.
