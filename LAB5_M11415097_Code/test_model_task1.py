# Spring 2026, 535518 Deep Learning - Lab 5 Task 1
# Evaluation script for the CartPole-v1 vanilla DQN model.
# 對應的訓練腳本是 dqn.py (--env-name CartPole-v1)。
#
# 用法:
#   python test_model_task1.py --model-path LAB5_M11415097_task1.pt
#
# 預設會跑 seed 0 ~ 19 共 20 個 episode (符合助教評分協定)，並列出每個 seed
# 的 eval reward 與 20 個 seed 的 average reward。

import argparse
import os
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Task 1 用的 MLP Q-network 架構，必須與 dqn.py 中 input_shape=(4,) 那條分支完全一致。
    若改了 dqn.py 的 MLP 結構，這裡也要同步修改，否則 load_state_dict 會失敗。
    """

    def __init__(self, num_actions, input_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 依照 args.render 切換 gym render mode：
    # - "human"     : 開一個視窗即時顯示，可肉眼觀察 agent 行為 (CartPole 的彈跳桿子)
    # - "rgb_array" : 不開視窗，直接拿到 frame (numpy array)，方便寫成 mp4
    render_mode = "human" if args.render else "rgb_array"
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # 取得 state / action 維度
    input_dim = env.observation_space.shape[0]  # CartPole = 4
    num_actions = env.action_space.n            # CartPole = 2

    # 載入訓練好的 model weights，並切換到 eval mode (關掉 dropout / batchnorm 之類)
    model = DQN(num_actions=num_actions, input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 若需要存影片，準備輸出資料夾與 imageio writer
    if args.save_video:
        import imageio
        os.makedirs(args.video_dir, exist_ok=True)

    rewards = []  # 收集每個 seed 的 reward

    for ep in range(args.episodes):
        seed = args.seed + ep

        # 為每一輪重新設定 RNG seed，確保結果可重現
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 重置環境 (用該 seed 決定初始 state)
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)

        done = False
        total_reward = 0.0
        frames = []  # 累積一個 episode 的 frame，episode 結束後一次寫成 mp4

        while not done:
            # 若要錄影，每一步抓一張 RGB frame (要求 render_mode='rgb_array')
            if args.save_video:
                frames.append(env.render())

            # 直接用 greedy policy: a = argmax_a Q(s, a)
            state_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)
        # 沿用助教投影片 p21 的 print 格式，方便截圖貼進 report
        print(f"Model: {os.path.basename(args.model_path)}, seed: {seed}, eval reward: {total_reward:.0f}")

        # episode 結束才寫影片，避免每步寫檔造成 disk I/O 拖慢評估
        if args.save_video and len(frames) > 0:
            out_path = os.path.join(args.video_dir, f"task1_seed{seed:02d}_r{int(total_reward)}.mp4")
            with imageio.get_writer(out_path, fps=30) as writer:
                for f in frames:
                    writer.append_data(f)
            print(f"  -> video saved: {out_path}")

    avg = float(np.mean(rewards))
    print(f"Average reward over {args.episodes} episodes: {avg:.2f}")

    env.close()
    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the trained .pt model (Task 1 CartPole MLP)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes (default 20, per spec)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Starting seed; episodes use seeds [seed, seed+episodes)")
    parser.add_argument("--render", action="store_true",
                        help="Open a window to watch the agent play in real time (human mode)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save each episode as MP4 (requires render_mode='rgb_array', i.e. don't combine with --render)")
    parser.add_argument("--video-dir", type=str, default="./eval_videos_task1",
                        help="Directory to save recorded videos when --save-video is set")
    args = parser.parse_args()

    if args.render and args.save_video:
        raise SystemExit("--render and --save-video are mutually exclusive (gymnasium can only render to one target)")

    evaluate(args)
