# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DQN(nn.Module):
    def __init__(self, num_actions, input_shape=(4,), use_dueling=False):
        super(DQN, self).__init__()
        self.use_dueling = use_dueling

        if len(input_shape) == 1:
            # vector input (CartPole) -> MLP
            input_dim = input_shape[0]
            self.feature = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            feature_dim = 128
            self.is_image = False
        else:
            # image input (Atari) -> CNN (DeepMind Nature DQN)
            in_channels = input_shape[0]
            self.feature = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            )
            feature_dim = 512
            self.is_image = True

        if self.use_dueling:
            self.value_head = nn.Linear(feature_dim, 1)
            self.advantage_head = nn.Linear(feature_dim, num_actions)
        else:
            self.q_head = nn.Linear(feature_dim, num_actions)

    def forward(self, x):
        if self.is_image:
            x = x / 255.0
        feat = self.feature(x)
        if self.use_dueling:
            v = self.value_head(feat)
            a = self.advantage_head(feat)
            return v + (a - a.mean(dim=1, keepdim=True))
        return self.q_head(feat)


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """PER (Schaul et al., 2016). priority p_i = (|TD| + eps)^alpha, sampled by P(i) = p_i / sum p."""

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.eps = 1e-6

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error):
        max_priority = self.priorities[:len(self.buffer)].max() if len(self.buffer) > 0 else 1.0
        priority = max_priority if error is None else (abs(error) + self.eps) ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        n = len(self.buffer)
        prios = self.priorities[:n]
        probs = prios / prios.sum()
        indices = np.random.choice(n, batch_size, p=probs)
        batch = [self.buffer[i] for i in indices]
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(float(err)) + self.eps) ** self.alpha


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        seed = getattr(args, "seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        train_kwargs, eval_kwargs = {}, {}
        if "ALE/" in env_name:
            train_kwargs["repeat_action_probability"] = getattr(args, "repeat_action_prob", 0.25)
            eval_kwargs["repeat_action_probability"] = getattr(args, "eval_repeat_action_prob", 0.25)

        self.env = gym.make(env_name, render_mode="rgb_array", **train_kwargs)
        self.test_env = gym.make(env_name, render_mode="rgb_array", **eval_kwargs)
        self.env.reset(seed=seed)
        self.test_env.reset(seed=seed + 1)
        self.env.action_space.seed(seed)
        self.num_actions = self.env.action_space.n

        obs_shape = self.env.observation_space.shape
        self.is_atari = (len(obs_shape) > 1)

        if self.is_atari:
            self.preprocessor = AtariPreprocessor(frame_stack=4)
            input_shape = (4, 84, 84)
            self.best_reward = -21
        else:
            self.preprocessor = None
            input_shape = obs_shape
            self.best_reward = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)
        print(f"Env: {env_name} | obs_shape={obs_shape} | is_atari={self.is_atari} | num_actions={self.num_actions}")

        use_dueling = getattr(args, "use_dueling", False)
        self.q_net = DQN(self.num_actions, input_shape=input_shape, use_dueling=use_dueling).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_shape=input_shape, use_dueling=use_dueling).to(self.device)

        resume_path = getattr(args, "resume_from", None)
        if resume_path:
            print(f"Loading pretrained weights from: {resume_path}")
            state_dict = torch.load(resume_path, map_location=self.device)
            self.q_net.load_state_dict(state_dict)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        self.use_per = getattr(args, "use_per", False)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=args.memory_size,
                alpha=getattr(args, "per_alpha", 0.6),
                beta=getattr(args, "per_beta", 0.4),
            )
            self.per_beta_anneal_steps = getattr(args, "per_beta_anneal_steps", 1_000_000)
            self.per_beta_start = getattr(args, "per_beta", 0.4)
        else:
            self.memory = deque(maxlen=args.memory_size)

        self.use_double_dqn = getattr(args, "use_double_dqn", False)
        self.n_step = getattr(args, "n_step", 1)
        self.gamma_n = (args.discount_factor ** self.n_step)
        self.n_step_buffer = deque(maxlen=self.n_step)

        # task3: save snapshots at fixed env-step milestones for grading
        self.task3_mode = getattr(args, "task3_mode", False)
        self.task3_milestones = [600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000]
        self.task3_milestones_saved = set()
        self.task3_target_score = getattr(args, "task3_target_score", 19.0)
        self.task3_best_saved = False

        self.start_time = time.time()
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.env_count = 0
        self.train_count = 0
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def _elapsed_str(self):
        sec = int(time.time() - self.start_time)
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _push_to_memory(self, transition, error=None):
        if self.use_per:
            self.memory.add(transition, error)
        else:
            self.memory.append(transition)

    def _build_n_step_transition(self):
        R = 0.0
        gamma_k = 1.0
        for (_, _, r, _, d) in self.n_step_buffer:
            R += gamma_k * r
            gamma_k *= self.gamma
            if d:
                break
        s0, a0, _, _, _ = self.n_step_buffer[0]
        _, _, _, s_n, d_n = self.n_step_buffer[-1]
        return (s0, a0, R, s_n, d_n)

    def _to_state(self, obs, reset=False):
        if self.is_atari:
            return self.preprocessor.reset(obs) if reset else self.preprocessor.step(obs)
        return np.asarray(obs, dtype=np.float32)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self._to_state(obs, reset=True)
            done = False
            total_reward = 0
            step_count = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self._to_state(next_obs, reset=False)
                # store terminated only (not done) so truncation doesn't zero the bootstrap target
                self.n_step_buffer.append((state, action, reward, next_state, terminated))

                if len(self.n_step_buffer) == self.n_step:
                    self._push_to_memory(self._build_n_step_transition())

                if done:
                    # flush remaining partial transitions at episode end
                    while len(self.n_step_buffer) > 0:
                        self._push_to_memory(self._build_n_step_transition())
                        self.n_step_buffer.popleft()

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.task3_mode:
                    for milestone in self.task3_milestones:
                        if self.env_count >= milestone and milestone not in self.task3_milestones_saved:
                            ms_path = os.path.join(self.save_dir, f"task3_{milestone}.pt")
                            torch.save(self.q_net.state_dict(), ms_path)
                            self.task3_milestones_saved.add(milestone)
                            print(f"[Task3] Saved milestone @ {milestone} env steps -> {ms_path}")

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} Time: {self._elapsed_str()}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Elapsed Hours": (time.time() - self.start_time) / 3600.0,
                    })

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved checkpoint: {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    fixed_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), fixed_path)
                    history_path = os.path.join(self.save_dir, f"best_ep{ep:04d}_r{eval_reward:.2f}.pt")
                    torch.save(self.q_net.state_dict(), history_path)
                    print(f"New best -> {history_path} | reward={eval_reward:.2f}")

                if self.task3_mode and (not self.task3_best_saved) and (eval_reward >= self.task3_target_score):
                    best_path = os.path.join(self.save_dir, "task3_best.pt")
                    torch.save(self.q_net.state_dict(), best_path)
                    self.task3_best_saved = True
                    print(f"[Task3] First >= {self.task3_target_score} (got {eval_reward:.2f}) @ env_step={self.env_count} -> saved task3_best.pt")

                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count} Time: {self._elapsed_str()}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self._to_state(obs, reset=True)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self._to_state(next_obs, reset=False)

        return total_reward

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        if self.use_per:
            beta = min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * self.train_count / max(1, self.per_beta_anneal_steps))
            self.memory.beta = beta
            batch, per_indices, is_weights = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory, self.batch_size)
            per_indices, is_weights = None, None

        states, actions, rewards, next_states, dones = zip(*batch)

        states      = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                # action selected by main net, evaluated by target net
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_net(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma_n * next_q_values * (1.0 - dones)

        td_errors = target_q - q_values

        if self.use_per:
            is_weights_t = torch.from_numpy(is_weights).to(self.device)
            elementwise_loss = nn.functional.smooth_l1_loss(q_values, target_q, reduction='none')
            loss = (elementwise_loss * is_weights_t).mean()
        else:
            loss = nn.functional.smooth_l1_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.use_per and per_indices is not None:
            self.memory.update_priorities(per_indices, td_errors.detach().cpu().numpy())

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")
            wandb.log({
                "Train Step": self.train_count,
                "Loss": loss.item(),
                "Q_value_mean": q_values.mean().item(),
                "Q_value_std": q_values.std().item(),
                "Env Step Count": self.env_count,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-project", type=str, default="DLP-Lab5-Task1-CartPole")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=100)
    parser.add_argument("--replay-start-size", type=int, default=1000)
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-double-dqn", action="store_true")
    parser.add_argument("--use-per", action="store_true")
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--per-beta-anneal-steps", type=int, default=1_000_000)
    parser.add_argument("--task3-mode", action="store_true")
    parser.add_argument("--task3-target-score", type=float, default=19.0)
    parser.add_argument("--use-dueling", action="store_true")
    parser.add_argument("--repeat-action-prob", type=float, default=0.25)
    parser.add_argument("--eval-repeat-action-prob", type=float, default=0.25)
    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, name=args.wandb_run_name,
               config=vars(args), save_code=True)
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)
