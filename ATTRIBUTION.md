# Attribution

babysteps is built on the open-source projects below. All of them are used as
dependencies installed from PyPI (see `pyproject.toml` and `uv.lock`); no source is
vendored except where noted in the rsl_rl section.

| Project | License | Repository | License file | Use here |
| --- | --- | --- | --- | --- |
| mjlab | Apache-2.0 | https://github.com/mujocolab/mjlab | https://github.com/mujocolab/mjlab/blob/main/LICENSE | Task definitions, environments and the G1 flat-terrain velocity task used for training and for the frozen bar. |
| rsl_rl | BSD-3-Clause | https://github.com/leggedrobotics/rsl_rl | https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE | PPO implementation and actor/critic models used by the stock baseline. |
| MuJoCo | Apache-2.0 | https://github.com/google-deepmind/mujoco | https://github.com/google-deepmind/mujoco/blob/main/LICENSE | Physics engine; also the CPU simulator for the sim-to-sim check. |
| MuJoCo Warp | Apache-2.0 | https://github.com/google-deepmind/mujoco_warp | https://github.com/google-deepmind/mujoco_warp/blob/main/LICENSE | GPU-batched MuJoCo used for training and evaluation. |
| Warp | Apache-2.0 | https://github.com/NVIDIA/warp | https://github.com/NVIDIA/warp/blob/main/LICENSE.md | GPU kernel framework underlying MuJoCo Warp. |
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch | https://github.com/pytorch/pytorch/blob/main/LICENSE | Neural network training and inference. |

## G1 robot model

The G1 humanoid model comes from the asset zoo shipped inside mjlab
(https://github.com/mujocolab/mjlab/tree/main/src/mjlab/asset_zoo), which is derived
from MuJoCo Menagerie (https://github.com/google-deepmind/mujoco_menagerie). Menagerie
model files are distributed under BSD-3-Clause
(https://github.com/google-deepmind/mujoco_menagerie/blob/main/LICENSE; each model
directory there also carries its own license file). babysteps does not copy or modify
the model files; it loads them from the installed mjlab package.

## Code derived from rsl_rl

Two files reproduce the logic of rsl_rl 5.4.2 and are derived from it under
BSD-3-Clause (license linked above):

- `src/babysteps/train/stock_runner.py` follows `OnPolicyRunner.learn`.
- `src/babysteps/systems/graphed_ppo.py` follows `PPO.update`.

The first mirrors the stock training loop so the baseline measures the unmodified
algorithm; the second is a CUDA-graph version of the PPO update that is checked for
numerical parity against the original (`tests/test_graphed_ppo_parity.py`).

## Related work

Holosoma (https://github.com/amazon-far/holosoma) is a humanoid training framework
that serves as the reference for Track B. It is related work only: no code from it is
included or required. License: see repository LICENSE.
