# Spring 2026, 535518 Deep Learning - Lab 5 Task 3
# Pong evaluation script that supports both standard and Dueling DQN snapshots.
#
# 用法:
#   # 標準 DQN snapshot (Task 2 / Task 3 v1):
#   python test_model_task3.py --model-path LAB5_M11415097_task3_best.pt
#
#   # Dueling DQN snapshot (Task 3 v3 之後):
#   python test_model_task3.py --model-path LAB5_M11415097_task3_best.pt --use-dueling

import argparse
import os
import random

import gymnasium as gym
import ale_py
import numpy as np
import torch
import cv2
import imageio
from collections import deque

# 直接 import 主訓練程式裡的 DQN 與 AtariPreprocessor，確保架構與訓練時完全一致
# (避免另寫一份再 drift 的風險)
from dqn import DQN, AtariPreprocessor

gym.register_envs(ale_py)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor(frame_stack=4)
    num_actions = env.action_space.n

    # 用同一個 DQN class，依 --use-dueling 切換 architecture
    model = DQN(num_actions=num_actions, input_shape=(4, 84, 84), use_dueling=args.use_dueling).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0.0
        frames = []

        while not done:
            if args.save_video:
                frames.append(env.render())

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)
        print(f"Model: {os.path.basename(args.model_path)}, seed: {seed}, eval reward: {total_reward:.0f}")

        if args.save_video and len(frames) > 0:
            out = os.path.join(args.output_dir, f"task3_seed{seed:02d}_r{int(total_reward)}.mp4")
            with imageio.get_writer(out, fps=30) as writer:
                for f in frames:
                    writer.append_data(f)

    avg = float(np.mean(rewards))
    print(f"Average reward over {args.episodes} episodes: {avg:.2f}")
    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos_task3")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use-dueling", action="store_true",
                        help="Set this if the model was trained with --use-dueling")
    parser.add_argument("--save-video", action="store_true",
                        help="Save evaluation episodes as MP4 (default off to keep eval fast)")
    args = parser.parse_args()
    evaluate(args)
