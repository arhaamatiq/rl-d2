# Ledger

Every number in the README and write-up comes from this table. Rendered from
`results/ledger.jsonl` by `bs-ledger render`; edit the jsonl, never this file.

| id | metric | value | unit | seeds | median_iqr | command | commit | gpu | date | artifact |
|---|---|---|---|---|---|---|---|---|---|---|
| V1 | sim2sim obs parity (CPU MuJoCo vs mjlab), max abs error | 7.2e-07 | obs units | 0 |  | python -m babysteps.eval.export_bundle --parity-only | 4b0cabf | RTX 4090 | 2026-11-12 | bar/sim2sim/g1_flat.json |
| V2 | graphed vs stock env: post-reset actor obs max abs diff (203 matched resets) | 0 | obs units | 1 |  | python tests/test_graph_env_parity.py | 9b3af13 | RTX 4090 | 2026-11-12 | tests/test_graph_env_parity.py |
| V3 | graphed vs stock env: obs max abs diff over first 30 steps (stock vs stock diverges after ~46) | 0.000285 | obs units | 1 |  | python tests/test_graph_env_parity.py | 9b3af13 | RTX 4090 | 2026-11-12 | tests/test_graph_env_parity.py |
| V4 | graphed PPO update vs rsl_rl update: max param diff after 5x4 minibatches | 1.64e-07 | param units | 0 |  | python tests/test_graphed_ppo_parity.py | 9b3af13 | RTX 4090 | 2026-11-12 | tests/test_graphed_ppo_parity.py |
| V5 | mirror map vs physics: max \|obs(mirror(s)) - mirror(obs(s))\| | <1e-6 | obs units | 0 |  | pytest tests/test_symmetry.py | 51e7b23 | CPU | 2026-11-12 | tests/test_symmetry.py |
| B1 | bar v1 sha256 (frozen 2026-11-12 04:00 UTC) | 3a1ee106cfdb |  | 100,101,102 |  | python scripts/calibrate_bar.py runs/calib/stock_s100 runs/calib/stock_s101 runs/calib/stock_s102 --freeze | 6fb9e96 | RTX 4090 | 2026-11-12 | bar/bar_v1.json |
| B2 | calibration seeds: first passing iteration under bar v1 (stock) | 1428 | iterations | 100,101,102 | 1428 [1423, 1569]; per seed 1428, 1418, 1709 | python -m babysteps.eval.time_to_bar --run runs/calib/stock_sXXX | 6fb9e96 | RTX 4090 | 2026-11-12 | runs/calib/*/time_to_bar.json |
| B3 | known-bad checkpoints fail / known-good pass (random init, standing-only, calibration finals) | pass |  | 100,101,102 |  | python -m babysteps.eval.time_to_bar --run runs/calib/stock_sXXX | 6fb9e96 | RTX 4090 | 2026-11-12 | runs/calib/*/eval.jsonl |
