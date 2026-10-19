#!/usr/bin/env bash
# Task 3: Enhanced DQN on Atari Pong-v5
# Run from the project root: bash LAB5_M11415097_Code/run_task3.sh
#
# Enhancements enabled:
#   --use-double-dqn  : Double DQN target (Hasselt et al., 2016)
#   --use-per         : Prioritized Experience Replay (Schaul et al., 2016)
#   --n-step 3        : 3-step return (Sutton, 1988; used in Rainbow)
#   --use-dueling     : Dueling DQN (Wang et al., 2016)
#   --task3-mode      : Auto-save snapshots at 600k/1M/1.5M/2M/2.5M env steps
#
# First reach score 19 at approximately 2.26M env steps.
set -e

python LAB5_M11415097_Code/dqn.py \
    --env-name "ALE/Pong-v5" \
    --wandb-project "DLP-Lab5-Task3-Enhanced" \
    --wandb-run-name "task3-rainbow-lite-v3" \
    --save-dir "./results/task3" \
    --episodes 2500 \
    --batch-size 32 \
    --memory-size 100000 \
    --replay-start-size 10000 \
    --lr 1e-4 \
    --epsilon-decay 0.999995 \
    --epsilon-min 0.05 \
    --target-update-frequency 1000 \
    --max-episode-steps 27000 \
    --train-per-step 1 \
    --seed 42 \
    --use-double-dqn \
    --use-per \
    --n-step 3 \
    --use-dueling \
    --task3-mode

# To evaluate the submitted snapshots (each at the given milestone):
# python LAB5_M11415097_Code/test_model_task3.py \
#     --model-path LAB5_M11415097_task3_2500000.pt --episodes 20 --seed 0 --use-dueling
#
# Note: --use-dueling flag is required for the test script because the
# saved network state_dict has Dueling-style heads (value_head + advantage_head).
