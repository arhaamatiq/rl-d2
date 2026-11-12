# Rules

Babysteps measures **time-to-bar**: wall-clock time from launch to the first saved
checkpoint that passes a frozen evaluation, on a single RTX 4090.

## The bar (frozen: `bar/bar_v1.json`, sha256 `{{BAR_SHA}}`)

Task: the G1 humanoid (29 actuated joints) tracking commanded planar velocity and yaw
rate on flat ground, as defined by mjlab 1.6.0's `Mjlab-Velocity-Flat-Unitree-G1`
(robot model, physics, observations and action space are never modified).

A checkpoint passes when **all** of the following hold:

| check | where | threshold |
|---|---|---|
| mean linear-velocity tracking error | GPU, 24 held-out commands x 8 seeded robots, 20 s, after a 2 s settle | <= {{LIN}} m/s |
| mean yaw-rate tracking error | same episodes | <= {{YAW}} rad/s |
| fall-free 20 s episodes | same episodes (fall = torso tilt > 70 deg or pelvis < 0.5 m) | >= {{FF}} |
| push recovery | 8 commands x 8 directions x 2, horizontal kick of {{PUSH_DV}} m/s at t = 6 s | >= {{PUSH}} survive to 20 s |
| sim-to-sim | same policy in CPU MuJoCo (C engine), nominal model, 8 commands x 2 headings | >= {{S2S}} episodes fall-free with linear error <= {{S2S_LIN}} m/s |

Held-out commands: 8 structured (stand, forward 0.5 / 1.0, backward 0.5, strafe
+-0.5, turn +-0.5) and 16 drawn with a fixed seed from the stock training range before
any run was trained. The evaluation environment is mjlab's stock flat *play*
configuration (no observation noise, no random pushes) with startup domain
randomization (friction, center-of-mass offset, encoder bias) kept and seeded.

How the thresholds were set: see `DECISIONS.md` D6 (rule written before the
calibration runs finished) and the `calibration` block inside the bar file.

## Timing

* Clock starts in `scripts/train.sh` before the Python interpreter, after wiping the
  Warp kernel, Torch Inductor/Triton and CUDA JIT caches. Startup, kernel compilation,
  CUDA-graph capture and any `torch.compile` are inside the clock.
* Every run writes a portable checkpoint every 30 s of wall time. Time-to-bar is the
  timestamp of the first checkpoint that passes; earlier checkpoints are all evaluated.
* Also reported: time from the first learning iteration (startup and compile excluded).
* Evaluation runs offline, on saved checkpoints, with fixed seeds, by code that does not
  import any training code.

## What may change

Anything on the training side: algorithm hyperparameters, network, number of
environments, batching, reward shaping, curriculum, normalization, and systems work
(CUDA graphs, kernels, precision, overlap). Never the evaluation, the robot model, the
physics settings (timestep, decimation, solver) or the task definition used for
evaluation.

* **Track A**: the PPO family (on-policy actor-critic with clipped surrogate and GAE),
  same as the stock recipe.
* **Track B (open)**: any algorithm.

## Official records

* One RTX 4090, one run at a time, 5 seeds (0-4). Report the median and IQR of
  time-to-bar, the per-seed values, and robustness metrics at the passing checkpoint.
* A record is accepted after a independent re-run of the headline reproduces the
  median within 10%.
* Calibration seeds (100-102) and development seeds (200+) are never used for records.

## Submission format

A pull request adding `submissions/<name>/` with: the recipe (code or overrides), the
exact command, the commit, `run.json` + `checkpoints.jsonl` + `checkpoints/` for each
seed, and the evaluator output (`time_to_bar.json`). The maintainer re-runs the
evaluation on the submitted checkpoints and re-trains at least one seed.
