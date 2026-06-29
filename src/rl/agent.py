"""
eco-logic/src/rl/agent.py
Q-Learning and Deep Q-Network (DQN) agents for thermal-aware workload placement.
"""

import numpy as np
import random
import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Agent (tabular, discretised state)
# ──────────────────────────────────────────────────────────────────────────────


class QLearningAgent:
    """
    Tabular Q-Learning agent.
    State is discretised into (avg_temp_bin, pue_bin) for fast convergence.
    Suitable for smaller action spaces or prototyping.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,  # learning rate
        gamma: float = 0.95,  # discount factor
        epsilon: float = 1.0,  # initial exploration rate
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        n_temp_bins: int = 10,
        n_pue_bins: int = 10,
    ):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.n_temp_bins = n_temp_bins
        self.n_pue_bins = n_pue_bins

        # Q-table: shape (temp_bins, pue_bins, n_actions)
        self.q_table = np.zeros((n_temp_bins, n_pue_bins, n_actions))

        self.step_count = 0
        self.episode_count = 0

    # ── State discretisation ──────────────────────────────────────────

    def _discretise(self, obs: np.ndarray) -> Tuple[int, int]:
        """Extract avg_temp and PUE from obs vector and discretise."""
        n_racks = (len(obs) - 2) // 2
        avg_temp = float(np.mean(obs[:n_racks]))
        pue = float(obs[-2])

        temp_bin = int(
            np.clip(
                (avg_temp - 28) / (90 - 28) * self.n_temp_bins, 0, self.n_temp_bins - 1
            )
        )
        pue_bin = int(
            np.clip((pue - 1.0) / (2.5 - 1.0) * self.n_pue_bins, 0, self.n_pue_bins - 1)
        )
        return temp_bin, pue_bin

    # ── Policy ────────────────────────────────────────────────────────

    def select_action(self, obs: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        t_bin, p_bin = self._discretise(obs)
        return int(np.argmax(self.q_table[t_bin, p_bin]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> float:
        t, p = self._discretise(obs)
        nt, np_ = self._discretise(next_obs)

        q_current = self.q_table[t, p, action]
        q_target = reward + (0 if done else self.gamma * np.max(self.q_table[nt, np_]))
        td_error = q_target - q_current

        self.q_table[t, p, action] += self.alpha * td_error
        self.step_count += 1
        return abs(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episode_count += 1

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = {
            "q_table": self.q_table.tolist(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
            "episode_count": self.episode_count,
        }
        Path(path).write_text(json.dumps(data))
        logger.info(f"Q-table saved → {path}")

    def load(self, path: str) -> None:
        data = json.loads(Path(path).read_text())
        self.q_table = np.array(data["q_table"])
        self.epsilon = data["epsilon"]
        self.step_count = data["step_count"]
        self.episode_count = data["episode_count"]
        logger.info(f"Q-table loaded ← {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: deque = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done) -> None:
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# ──────────────────────────────────────────────────────────────────────────────
# DQN Agent (PyTorch)
# ──────────────────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not found; DQNAgent unavailable. Install: pip install torch"
    )


if TORCH_AVAILABLE:

    class _QNetwork(nn.Module):
        def __init__(self, obs_size: int, n_actions: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_size, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, n_actions),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DQNAgent:
        """
        Deep Q-Network agent with experience replay and target network.
        Suitable for the full 64-rack observation space.
        """

        def __init__(
            self,
            obs_size: int,
            n_actions: int,
            lr: float = 1e-3,
            gamma: float = 0.99,
            epsilon: float = 1.0,
            epsilon_min: float = 0.05,
            epsilon_decay: float = 0.998,
            batch_size: int = 64,
            target_update_freq: int = 500,
            buffer_capacity: int = 50_000,
            device: Optional[str] = None,
        ):
            self.n_actions = n_actions
            self.gamma = gamma
            self.epsilon = epsilon
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.batch_size = batch_size
            self.target_update_freq = target_update_freq

            self.device = torch.device(
                device or ("cuda" if torch.cuda.is_available() else "cpu")
            )

            self.policy_net = _QNetwork(obs_size, n_actions).to(self.device)
            self.target_net = _QNetwork(obs_size, n_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.SmoothL1Loss()
            self.buffer = ReplayBuffer(buffer_capacity)

            self.step_count = 0
            self.episode_count = 0
            self.losses: list[float] = []

            logger.info(
                f"DQNAgent ready on {self.device} | obs={obs_size} actions={n_actions}"
            )

        # ── Policy ────────────────────────────────────────────────────

        def select_action(self, obs: np.ndarray) -> int:
            if random.random() < self.epsilon:
                return random.randrange(self.n_actions)
            with torch.no_grad():
                t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                return int(self.policy_net(t).argmax(dim=1).item())

        # ── Learning ──────────────────────────────────────────────────

        def push(self, obs, action, reward, next_obs, done) -> None:
            self.buffer.push(obs, action, reward, next_obs, done)

        def learn(self) -> Optional[float]:
            if len(self.buffer) < self.batch_size:
                return None

            obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

            obs_t = torch.FloatTensor(obs).to(self.device)
            actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
            rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
            next_obs_t = torch.FloatTensor(next_obs).to(self.device)
            dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

            # Current Q values
            q_vals = self.policy_net(obs_t).gather(1, actions_t)

            # Target Q values (Double DQN)
            with torch.no_grad():
                next_actions = self.policy_net(next_obs_t).argmax(1, keepdim=True)
                next_q = self.target_net(next_obs_t).gather(1, next_actions)
                target_q = rewards_t + self.gamma * next_q * (1 - dones_t)

            loss = self.loss_fn(q_vals, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
            self.optimizer.step()

            self.step_count += 1
            l = loss.item()
            self.losses.append(l)

            # Sync target network
            if self.step_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return l

        def decay_epsilon(self) -> None:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.episode_count += 1

        # ── Persistence ───────────────────────────────────────────────

        def save(self, path: str) -> None:
            torch.save(
                {
                    "policy_state_dict": self.policy_net.state_dict(),
                    "target_state_dict": self.target_net.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epsilon": self.epsilon,
                    "step_count": self.step_count,
                    "episode_count": self.episode_count,
                },
                path,
            )
            logger.info(f"DQN model saved → {path}")

        def load(self, path: str) -> None:
            ckpt = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(ckpt["policy_state_dict"])
            self.target_net.load_state_dict(ckpt["target_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.epsilon = ckpt["epsilon"]
            self.step_count = ckpt["step_count"]
            self.episode_count = ckpt["episode_count"]
            logger.info(f"DQN model loaded ← {path}")
