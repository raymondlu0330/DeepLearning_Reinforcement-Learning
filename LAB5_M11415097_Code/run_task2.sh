#!/usr/bin/env bash
# Task 2: Vanilla DQN on Atari Pong-v5
# Run from the project root: bash LAB5_M11415097_Code/run_task2.sh
#
# This script runs in two stages:
#   Stage 1: Initial training from scratch (~17 hours on GTX 1080 Ti)
#   Stage 2: Low-LR fine-tuning continuation (~6 hours, +0.9 reward improvement)
#
# Submitted task2.pt is the best snapshot from Stage 2.
# (20-seed average evaluation score: 18.15 / 21)
set -e

# Stage 1: Initial training
python LAB5_M11415097_Code/dqn.py \
    --env-name "ALE/Pong-v5" \
    --wandb-project "DLP-Lab5-Task2-Pong" \
    --wandb-run-name "task2-vanilla-v1" \
    --save-dir "./results/task2" \
    --episodes 3000 \
    --batch-size 32 \
    --memory-size 100000 \
    --replay-start-size 10000 \
    --lr 1e-4 \
    --epsilon-decay 0.999995 \
    --epsilon-min 0.05 \
    --target-update-frequency 1000 \
    --max-episode-steps 27000 \
    --train-per-step 1 \
    --seed 42

# Stage 2: Fine-tuning from the best snapshot of stage 1 with much smaller lr
python LAB5_M11415097_Code/dqn.py \
    --env-name "ALE/Pong-v5" \
    --wandb-project "DLP-Lab5-Task2-Pong" \
    --wandb-run-name "task2-finetune-from-ep0980" \
    --save-dir "./results/task2_finetune" \
    --resume-from "./results/task2/best_ep0980_r20.00.pt" \
    --episodes 1500 \
    --batch-size 64 \
    --memory-size 100000 \
    --replay-start-size 1000 \
    --lr 1e-5 \
    --epsilon-start 0.05 \
    --epsilon-decay 1.0 \
    --epsilon-min 0.05 \
    --target-update-frequency 2000 \
    --max-episode-steps 27000 \
    --train-per-step 1 \
    --seed 7

# To evaluate the submitted model:
# python LAB5_M11415097_Code/test_model.py \
#     --model-path LAB5_M11415097_task2.pt --episodes 20 --seed 0
