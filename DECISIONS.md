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

## D6. Calibration rule (written before the calibration runs finished)
Seeds 100, 101, 102 calibrate; seeds 0-4 measure. The reference policy for each
calibration seed is the final checkpoint of a 120-minute stock run.
* `lin_err_max`, `yaw_err_max` = 1.25 x the worst seed's reference value.
* `fall_free_min` = 0.98 when all references are fall-free, else worst reference - 0.02.
* Push strength = largest of {0.5, 0.75, 1.0, 1.25} m/s at which every reference
  policy recovers from >= 95% of pushes; `push_success_min` = 0.90.
* `sim2sim_lin_err_max` = 1.5 x `lin_err_max` (no domain randomization in the CPU
  check, but a different solver); `sim2sim_success_min` = 0.875 (14 of 16 episodes),
  provided every reference scores >= 15/16.
The margins make the bar something stock reliably clears partway through training,
which is what makes T0 measurable.

## D7. Systems: one CUDA graph per rollout, one per PPO update
Baseline profile: GPU busy 29% of wall time, ~28k kernel launches and ~2,300
synchronizing calls per iteration, mjlab manager code 54% of host time. Physics alone
saturates near 580k samples/s; stock delivers ~120k. Rather than rewrite the task, the
whole env step runs mjlab's own code inside a CUDA graph:
* N+1 worlds; reset/resample/push indices from `torch.nonzero_static` padded with the
  extra world's index, so mjlab's `reset(env_ids)` code runs with static shapes.
* Warp kernels recorded into the PyTorch graph through Warp's external capture.
* Capture blockers fixed one at a time: Warp stream-switch event dependency,
  `torch.normal` std validation sync (reparameterized sampling instead), mjlab's
  `TorchArray.__setitem__` hopping to Warp's stream, pageable host copies from
  `_sample_se3_range` and from `t[idx] = <python scalar>`, `nonzero` in command
  resampling/heading, `singular.any()` in the raycast yaw alignment.
* PPO update rewritten with on-device adaptive LR and a capturable Adam (tensor LR).
First result at stock settings (4096 envs, same PPO): 0.38 s/iteration vs 0.82 s,
NVML utilization 100% vs 43%, per-iteration learning curve within the stock seed spread.

## D8. Graph capture order (driver crash workaround)
At >= 8k envs, capturing the PPO-update graph after the rollout graph made the
rollout graph's next launch segfault inside `cuGraphLaunch` (native backtrace via
gdb; driver 595.91, CUDA 13.2). Each graph alone was fine, memory was ample
(19.9 GiB free), and neither keeping the Warp graph object alive nor `gc.freeze()`
changed it. Capturing the update graph first and the rollout graph last is stable at
8k and 16k envs over 60 alternating iterations including a rollout re-capture, so the
trainer always uses that order. Root cause not isolated; recorded as an open question.

## D9. Bar v1 frozen (2026-11-12 04:00 UTC, sha256 3a1ee106cfdb...)
Reference policies (final checkpoints of seeds 100-102, ~4,200 iterations each):
lin error 0.0856 / 0.0742 / 0.0801 m/s, yaw error 0.0842 / 0.0882 / 0.0951 rad/s, all
fall-free, all 16/16 on sim-to-sim, push survival >= 0.99 at every candidate strength.
Rule output: lin <= 0.11, yaw <= 0.12, fall-free >= 0.98, push 1.25 m/s with >= 90%
survival, sim-to-sim lin <= 0.165 with >= 14/16.
Observations recorded, bar left unchanged (changing it now would be tuning after seeing
data): (a) every reference survived even the largest candidate push, so the push check is
the least discriminating of the five at convergence; it still bites early in training;
(b) on flat ground the two initial headings of the sim-to-sim check give identical CPU
trajectories, so its 16 episodes are effectively 8 distinct ones. Both are candidates
for a v2 bar, which would be versioned and never applied retroactively.
