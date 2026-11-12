# rl-d2

**How fast can one GPU teach a humanoid to walk robustly?**

rl-d2 turns "time-to-walk" into a benchmark with a frozen finish line, then sets
records against it. The robot is the 29-DoF G1 humanoid in
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp); the baseline is mjlab's stock
velocity-tracking PPO recipe ([rsl_rl](https://github.com/leggedrobotics/rsl_rl)), run
exactly as shipped. Every number below is on one RTX 4090 (Ryzen 9 7950X host), from a
cold start with compilation caches wiped, over 5 seeds, and comes from [`LEDGER.md`](LEDGER.md).

| | time-to-bar, median [IQR] | per seed (min) | iterations to bar | s / iteration | GPU util (NVML) |
|---|---|---|---|---|---|
| stock mjlab PPO (T0) | **13.26 min** [12.77, 15.78] | 18.05, 13.26, 12.77, 15.78, 12.77 | 1,432 | 0.528 | 73.1% |
| rl-d2 Track A (PPO) | **4.69 min** [4.69, 4.80] | 4.69, 5.20, 4.69, 4.68, 4.80 | 601 | 0.399 | 99.99% |
| speedup | **2.83x** (3.14x excluding startup/compile) | | 2.38x | 1.32x | |

All 10 runs pass every check. At the first passing checkpoint (medians), stock vs Track A:
linear error 0.098 vs 0.097 m/s, yaw error 0.113 vs 0.106 rad/s, fall-free 100% vs 100%,
push survival 100% vs 99.2% (1 of 128 pushes), sim-to-sim 16/16 vs 16/16.
The 3x target was not reached; this is the best result that passed the rules.

![stock vs Track A after 4.75 minutes of training](videos/minute_4p75_frame.png)
*Same wall-clock budget, same seed, same command schedule and push: stock (left, its
checkpoint at 4.4 min) falls after the push; Track A (right) keeps walking.
[video](videos/minute_4p75.mp4)*

## The bar

A checkpoint "walks robustly" when, evaluated offline with fixed seeds, it
1. tracks 24 held-out velocity commands (8 structured, 16 drawn before any training)
   within **0.11 m/s** linear and **0.12 rad/s** yaw-rate error,
2. completes **>= 98%** of 20 s episodes without falling under randomized friction,
   center-of-mass offset and encoder bias,
3. survives **>= 90%** of **1.25 m/s** horizontal pushes from 8 directions,
4. still walks when the same weights drive **CPU MuJoCo** instead of MuJoCo Warp
   (>= 14 of 16 episodes fall-free with linear error <= 0.165 m/s).

Thresholds were set by a rule written before calibration finished (1.25x the worst of
three stock reference policies, separate seeds), then frozen with a sha256 before any
optimization: [`RULES.md`](RULES.md), [`DECISIONS.md`](DECISIONS.md) D6 and D9.
The judge never imports training code: every recipe writes a portable checkpoint
(normalizer + MLP weights) every 30 s of wall time, and time-to-bar is the timestamp of
the first one that passes.

## Where the time went

Nsight Systems, NVTX ranges, 10 steady-state iterations, same host and procedure for both
([`scripts/profile_pair.sh`](scripts/profile_pair.sh)). Stock phases come from NVTX
(host) and kernel attribution (GPU); Track A runs as two CUDA graphs, so its phases
were timed as separately captured graphs ([`scripts/bench_components.py`](scripts/bench_components.py)).

| per iteration | stock host ms | stock GPU ms | Track A GPU ms |
|---|---|---|---|
| simulation (MuJoCo Warp step, forward, sensors) | 24.7 | 236.0 | 243.0 |
| env managers (rewards, obs, resets, commands, events) | **359.8** | 30.1 | 37.7 |
| policy inference | 14.9 | 4.1 | 3.4 |
| rollout storage | 11.4 | 1.7 | 1.7 |
| GAE | 2.2 | 0.4 | 0.4 |
| update forward + loss | 30.4 | 9.1 | 27.5 (8 epochs, symmetry, bf16) |
| backprop | 27.2 | 39.8 (autograd thread) | 42.7 |
| optimizer step + grad clip | 11.5 | 2.4 | 10.3 |
| host<->device: sync API calls / kernel launches | **2,278 / 27,898** | | **2 / 5 (+2 graph launches)** |
| kernel coverage (time any kernel runs / wall) | **57.6%** | | **99.9%** |

The stock loop is host-bound: mjlab's manager-based env issues ~1,100 small kernels and
~95 synchronizations per control step. Physics is already a CUDA graph and is the real floor.

## What changed (Track A)

Systems (no change to the MDP):
* **The whole env step in one CUDA graph, without rewriting the task.** The env runs N+1
  worlds; reset / resample / push indices come from `torch.nonzero_static(..., fill_value=N)`,
  so mjlab's own `reset(env_ids)` code runs with static shapes and padded entries hit a
  dummy world the learner never sees. MuJoCo Warp kernels are recorded into the PyTorch
  graph through Warp's external capture. A 24-step rollout (policy, physics, rewards,
  resets, storage) plus GAE is one graph launch.
* **PPO update as one graph:** rsl_rl's update with the same math, but the KL-adaptive
  learning rate lives on the device and feeds a capturable fused Adam; bf16 autocast.
* Stats read back through pinned memory and CUDA events one iteration late: **1 sync per
  iteration** instead of ~2,300.
* Checks: post-reset observations bitwise identical to stock over 203 matched resets;
  graphed update matches rsl_rl's to 1.6e-7 ([`tests/`](tests)).

Recipe (still PPO, same task, same evaluation): KL target 0.01 -> 0.02, left-right
symmetry augmentation of the actor loss (mirror map verified against physics),
8 epochs instead of 5, yaw-rate tracking reward weight 2 -> 4.

### Ablation: what each change bought

Cumulative, top to bottom; cold start, same host class (RTX 4090 + Ryzen 9 7950X). Rows 0
and 6 are the official 5-seed results; middle rows use seeds 0-2 (some ran two per
2-GPU host with separate compilation caches, see `DECISIONS.md` D12).

| # | change (cumulative) | seeds | time-to-bar, median [IQR] (min) | iterations to bar | s / iteration | startup (s) |
|---|---|---|---|---|---|---|
| 0 | stock mjlab PPO (T0) | 0-4 | **13.26** [12.77, 15.78] | 1432 | 0.527 | 38 |
| 1 | + rollout and PPO update as CUDA graphs (same hyperparameters) | 0-2 | **9.75** [9.49, 9.84] (1.36x) | 1426 | 0.375 | 42 |
| 2 | + KL target 0.01 -> 0.02 | 0-2 | **10.21** [9.73, 10.46] (0.96x) | 1525 | 0.375 | 39 |
| 3 | + left-right symmetry augmentation | 0-2 | **9.20** [8.45, 9.71] (1.11x) | 1273 | 0.399 | 39 |
| 4 | + 8 PPO epochs (was 5) | 0-2 | **7.74** [7.48, 8.50] (1.19x) | 930 | 0.454 | 41 |
| 5 | + bf16 autocast in the update | 0-2 | **7.20** [6.95, 7.23] (1.07x) | 980 | 0.399 | 38 |
| 6 | + yaw-rate tracking weight 2 -> 4 (= Track A) | 0-4 | **4.69** [4.69, 4.80] (1.53x) | 601 | 0.401 | 40 |

Row 1 is pure systems work: identical learning per iteration (1,426 vs 1,432 iterations),
29% less time per iteration. Rows 2-6 trade more work per iteration for far fewer
iterations; the yaw weight matters most because yaw-rate error is the check that gates
passing for every recipe (most of it is gait-induced pelvis yaw oscillation, measured at
50 Hz). KL 0.02 alone is neutral here but is part of the combination that won on dev seeds.

What did not help (dev seeds, details in `DECISIONS.md` D11): 8k envs, 12 steps/env,
tighter tracking kernels (they trade turning for linear precision), turn-in-place
curriculum, looser yaw posture, KL 0.03 (unstable), smaller initial action noise,
sparser training pushes, eager parallel Warp compilation (breaks mjlab's own capture).

## Independent verification

An independent re-run provisioned a new pod of
the same host class, set it up from the committed tree and ran `reproduce.sh headline`:
**5/5 seeds pass, median 5.15 min [4.66, 5.65], +9.8% vs the official 4.69 min**, inside
the 10% reproduction rule, but only just. The same seeds do not give the same run: MuJoCo
Warp is not bitwise deterministic, and time-to-bar moves in 30 s checkpoint steps, so one
seed shifting a checkpoint moves a 5-seed median by ~0.5 min. On the reproduction the
speedup is 2.57x. In 3 of the 5 fresh seeds the passing checkpoint sat exactly at the
sim-to-sim threshold (14/16). Re-judging the official checkpoints agreed on 7 of the 8
runs that were read back; the exception was a borderline checkpoint (track_a seed 1,
checkpoint 8) that failed officially and passed on re-judging. Details:
[`results/verify/REPORT.md`](results/verify/REPORT.md).

## Multi-GPU

The graphed trainer also runs data-parallel (`torchrun`, NCCL all-reduce of gradients and
of the adaptive-LR KL captured inside the update graph). On 2x RTX 4090 (PCIe, same host
class), Track A with 4096 envs per GPU:

| | time-to-bar, median [IQR] | iterations to bar | s / iteration | throughput scaling |
|---|---|---|---|---|
| 1 GPU (official, 5 seeds) | 4.69 min [4.69, 4.80] | 601 | 0.399 | 1x |
| 2 GPUs (3 seeds) | 4.22 min [3.97, 4.22] | 518 | 0.408 | **97.6%** efficiency |

Throughput scales almost perfectly; time-to-bar improves only 1.11x because PPO turns a
doubled batch into far fewer than half the iterations. Track A records stay single-GPU.

## Reproduce

```bash
docker build -t rl-d2 .
docker run --gpus all rl-d2 headline    # Track A: 5 seeds, cold start, judge, median/IQR
docker run --gpus all rl-d2 baseline    # stock T0: 5 seeds
```

Without Docker (Ubuntu 24.04, NVIDIA driver with CUDA >= 12.8):

```bash
bash scripts/pod_setup.sh          # locked env + rebuilds the sim-to-sim model, checks its hash
bash scripts/reproduce.sh headline
```

Pieces:

```bash
scripts/train.sh <recipe> <seed> runs/<name> <budget_min>    # cold-start timed run
.venv/bin/python -m rl_d2.eval.time_to_bar --run runs/<name> --procs 4
.venv/bin/python scripts/profile_report.py <trace.sqlite>     # NVTX phase breakdown
.venv/bin/python scripts/make_video.py --left runs/a --right runs/b --minute 5 --out v.mp4
```

Startup counts: the wrapper wipes the Warp, Inductor/Triton and CUDA JIT caches before
starting the clock. Cold startup is 38 s for stock and 40 s for Track A on the reference
host (mostly MuJoCo Warp kernel compilation); a slower host CPU adds tens of seconds to both.

## Repository map

| path | what |
|---|---|
| `bar/` | frozen bar spec, sim-to-sim model metadata and pinned hash |
| `src/rl_d2/eval/` | judge: GPU tracking/falls/push, CPU sim-to-sim, ordered time-to-bar scan |
| `src/rl_d2/train/` | timing harness, stock runner, graphed runner, recipes |
| `src/rl_d2/systems/` | graph-capturable env step, graphed PPO update, symmetry, NVTX |
| `scripts/` | official runs, reproduction, live early stop, profiling, calibration, video |
| `tests/` | parity tests vs stock mjlab and rsl_rl, mirror-map physics test, CPU unit tests |
| `LEDGER.md`, `DECISIONS.md` | every number, every decision |

## Related work

* **FastSAC / FastTD3**: Seo, Sferrazza, Chen, Shi, Duan, Abbeel, "Learning Sim-to-Real
  Humanoid Locomotion in 15 Minutes" (arXiv:2512.01996, Dec 2025). Off-policy RL with
  massively parallel simulation trains G1 and T1 locomotion in about 15 minutes on one
  RTX 4090. FastSAC is available in the open-source
  [Holosoma](https://github.com/amazon-far/holosoma) framework (Apache-2.0). Different task
  definition and success criterion, so the numbers are not directly comparable.
* **Agents' 100m**: AI agents iterating on training a G1 humanoid to sprint.
* Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL" (CoRL 2021).
* Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies"
  (ICRA 2024).

rl-d2 does not claim to be the first fast humanoid locomotion result. What it adds is
a fixed, robustness-checked finish line with an offline judge, and an accounting of where
single-GPU PPO time goes.

## Leaderboard

| track | entry | time-to-bar (median [IQR], 5 seeds) | host | verified |
|---|---|---|---|---|
| A (PPO) | rl-d2 Track A | 4.69 min [4.69, 4.80] | RTX 4090 + Ryzen 9 7950X | reproduced 5.15 min on a fresh pod (+9.8%) |
| A (PPO) | stock mjlab G1 PPO | 13.26 min [12.77, 15.78] | RTX 4090 + Ryzen 9 7950X | baseline |
| B (open) | open | | | |

Submissions: see [`RULES.md`](RULES.md) ("Submission format").

## Credits

See [`ATTRIBUTION.md`](ATTRIBUTION.md). Licensed under Apache-2.0.
