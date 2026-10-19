#!/usr/bin/env bash
# Task 1: Vanilla DQN on CartPole-v1
# Run from the project root: bash LAB5_M11415097_Code/run_task1.sh
# These are the exact hyperparameters used to obtain the submitted task1.pt
# (20-seed average evaluation score: 500.00 / 500)
set -e

python LAB5_M11415097_Code/dqn.py \
    --env-name "CartPole-v1" \
    --wandb-project "DLP-Lab5-Task1-CartPole" \
    --wandb-run-name "task1-vanilla-v4" \
    --save-dir "./results/task1" \
    --episodes 1000 \
    --batch-size 128 \
    --lr 5e-4 \
    --memory-size 50000 \
    --replay-start-size 1000 \
    --target-update-frequency 1000 \
    --epsilon-decay 0.9999 \
    --epsilon-min 0.05 \
    --max-episode-steps 500 \
    --seed 42

# To evaluate the trained model:
# python LAB5_M11415097_Code/test_model_task1.py \
#     --model-path LAB5_M11415097_task1.pt --episodes 20 --seed 0
