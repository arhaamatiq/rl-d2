# Independent verification report

The headline was re-run from scratch on a pod that had never run this project, set up
only from the committed tree.

## Host
- 1x RTX 4090 (driver 595.91.07), AMD Ryzen 9 7950X host (the reference host class), 16 vCPUs allocated.
- `scripts/pod_setup.sh` printed "setup ok"; the sim-to-sim model sha256 matched the pin (9294f780...).
- Command: `HEADLINE_BUDGET=8 bash scripts/reproduce.sh headline` (cold caches per seed, seeds 0-4 in sequence, nothing else running).

## Headline reproduction
| seed | first pass k | iteration | time-to-bar (min) | official (min) |
|---|---|---|---|---|
| 0 | 8 | 602 | 4.642 | 4.694 |
| 1 | 10 | 753 | 5.654 | 5.199 |
| 2 | 8 | 604 | 4.659 | 4.694 |
| 3 | 9 | 679 | 5.153 | 4.680 |
| 4 | 10 | 755 | 5.658 | 4.795 |

- Median **5.153 min** (IQR 4.66-5.65); 5/5 seeds passed.
- Versus ledger A1 (4.69): **+9.88%** (+9.78% vs the unrounded official median 4.694). Inside the 10% limit, but only by about 0.1 percentage point; the reproduction is 0.46 min slower than the official median.
- Seeds 1, 3 and 4 needed one or two more 30 s checkpoints than the official runs (k 10/9/10 vs 9/8/8); iterations to pass were 679-755 vs ~600. Seeds 0 and 2 matched closely. The gap comes from different training trajectories (MuJoCo Warp is not bitwise reproducible), not from per-iteration speed: startup was ~37 s (official ~39.8 s) and GPU utilization ~100%.
- Pass margins: `sim2sim_success` at the passing checkpoint was 0.875 (exactly the threshold) for seeds 0, 1 and 2 (official median 1.00).
- The 10% criterion is on the median only. With 5 seeds and 30 s checkpoint steps, one seed moving by a checkpoint shifts the median by ~0.5 min, so the result is fragile.
- Timings come from each run's own recorded timestamps.

Artifacts: `headline/` (repro.log, summary.json, per-seed time_to_bar.json and run.json).

## Re-judge of the official checkpoints (partial)
The 10 official run directories (checkpoints, checkpoints.jsonl, run.json) were copied to
the verification pod and re-evaluated, two runs at a time with `--procs 4`. The pod was
terminated (cause unknown) before the outputs were copied off, so `rejudge.json` has no
re-judged fields. 8 of the 10 results had already been read on the pod; they are recorded
here as observed values:

| run | official first-pass k / min | re-judged k / min |
|---|---|---|
| stock_s0 | 34 / 18.05 | 34 / 18.05 |
| stock_s1 | 25 / 13.26 | 25 / 13.26 |
| stock_s2 | 24 / 12.77 | 24 / 12.77 |
| stock_s3 | 30 / 15.78 | 30 / 15.78 |
| stock_s4 | 24 / 12.77 | 24 / 12.77 |
| track_a_s0 | 8 / 4.69 | 8 / 4.69 |
| track_a_s1 | 9 / 5.20 | **8 / 4.70** |
| track_a_s2 | 8 / 4.69 | 8 / 4.69 |
| track_a_s3, s4 | 8 / 4.68, 8 / 4.80 | not observed |

One disagreement: checkpoint 8 of track_a_s1 is borderline, and the judge (which runs
MuJoCo Warp) is not bitwise deterministic, so it failed in the official evaluation and
passed on re-judging. The official result is kept as measured; the medians are unaffected.

## Notes
- Marginal pass on the 10% rule (+9.88%), with sim-to-sim exactly at its threshold in 3 of 5 fresh seeds.
- The per-seed spread of the reproduction (4.64 to 5.66 min) is wider than the official IQR [4.69, 4.80] suggests.
