#!/usr/bin/env bash
# Pod-local: wait for evaluations to finish, then 2-GPU Track A runs, then judge rank 0.
cd /root/rl-d2
while pgrep -f "[t]ime_to_bar" > /dev/null; do sleep 10; done
mkdir -p runs/ddp_official
for s in 0 1 2; do
  scripts/train_ddp.sh 2 track_a "$s" "runs/ddp_official/track_a_2gpu_s$s" 7 > "runs/ddp_official/track_a_2gpu_s$s.log" 2>&1
done
for s in 0 1 2; do
  MUJOCO_GL=egl .venv/bin/python -m rl_d2.eval.time_to_bar --run "runs/ddp_official/track_a_2gpu_s$s" --procs 8 \
    > "runs/ddp_official/track_a_2gpu_s$s.eval.log" 2>&1
done
