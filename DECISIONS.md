# Decisions

Every non-obvious choice, in the order it was made. Numbers come from `LEDGER.md`.

## D1. Stock reference = mjlab 1.6.0 G1 flat velocity PPO, exactly as shipped
`Mjlab-Velocity-Flat-Unitree-G1`, rsl_rl 5.4.2 PPO, 4096 envs (the README command),
24 steps/env, 5 epochs x 4 minibatches, adaptive KL LR. Only change: logger set to
TensorBoard instead of W&B (no account in the loop). mjlab's own periodic saves and
ONNX export stay on, because that is what a stock user pays.

## D2. One environment for every track
torch 2.9.0+cu128, warp-lang 1.14.0, mujoco / mujoco-warp 3.11.0, Python 3.13 — the
versions in mjlab 1.6.0's own lockfile. Stock and Track A run from the same venv, so a
speedup cannot come from a newer torch.

## D3. Time-to-bar measurement
* Clock starts in the shell wrapper (`scripts/train.sh`) before the interpreter, after
  compilation caches are wiped (Warp kernels, Inductor/Triton, CUDA JIT). Cold start is
  what a fresh machine sees, and it stops a recipe from hiding compile time in a cache.
* Checkpoints are written on a fixed 30 s wall-clock cadence for every track, so the
  measurement granularity does not depend on iteration speed.
* A checkpoint is a portable policy file (normalizer + MLP weights). The evaluator
  never imports training code, so a Track A network or a Track B actor is evaluated by
  the same code path.
* Two numbers per run: wall time from launch (official) and time from the first
  learning iteration (startup and compile excluded).

## D4. Pod choice
First pod (EPYC 7642 host) pulled from the Python package CDN at ~0.5 MB/s; a second pod
in another region pulled at ~74 MB/s. Kept the second, terminated the first. All
official timings run on one pod so host CPU differences cannot leak into the speedup.
The pod's persistent volume is a network FUSE mount; code and venvs live on the local
container disk (imports from the network mount would inflate startup time).

## D5. Sim-to-sim bundle
The CPU MuJoCo check runs the compiled model exported from mjlab (nominal, no domain
randomization). The action -> ctrl map is measured by perturbing each action through
mjlab's own action pipeline instead of being re-derived by hand. Observation parity
against mjlab: max abs error 7e-7 over 50 random-action steps. The 71 MB model binary
is not committed; its sha256 is pinned and the export is deterministic (same hash from
GPU and CPU export paths).
