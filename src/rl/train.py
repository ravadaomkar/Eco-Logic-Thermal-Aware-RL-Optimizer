"""
eco-logic/src/rl/train.py
Training loop for the Thermal-Aware RL Optimizer.
Supports both Q-Learning (fast) and DQN (high performance) agents.
Logs episodes to MySQL and exports Prometheus metrics.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.api.prometheus import PrometheusExporter
from src.db.repository import EpisodeRepository
from src.rl.agent import QLearningAgent, ReplayBuffer
from src.rl.environment import DataCenterEnv
from src.rl.reward import reward_breakdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Episode runner
# ──────────────────────────────────────────────────────────────────────────────


def run_episode(
    env: DataCenterEnv,
    agent: QLearningAgent,
    train: bool = True,
) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    td_errors = []
    steps = 0

    while True:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            td_err = agent.update(obs, action, reward, next_obs, done)
            td_errors.append(td_err)

        total_reward += reward
        obs = next_obs
        steps += 1

        if done:
            break

    if train:
        agent.decay_epsilon()

    return {
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "final_pue": round(info["pue"], 4),
        "final_avg_temp": round(info["avg_temp"], 2),
        "final_max_temp": round(info["max_temp"], 2),
        "gpu_density": round(info["gpu_density"], 3),
        "mean_td_error": round(float(np.mean(td_errors)) if td_errors else 0, 4),
        "epsilon": round(agent.epsilon, 4),
        "terminated": terminated,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────────────────────


def train(
    n_episodes: int = 500,
    cooling_budget: float = 0.75,
    agent_type: str = "qlearning",  # "qlearning" | "dqn"
    save_every: int = 50,
    eval_every: int = 25,
    db_url: Optional[str] = None,
    push_prometheus: bool = False,
    seed: int = 42,
):
    logger.info(f"=== Eco-Logic RL Training ===")
    logger.info(
        f"Agent: {agent_type} | Episodes: {n_episodes} | Cooling budget: {cooling_budget}"
    )

    env = DataCenterEnv(cooling_budget=cooling_budget, max_steps=200, render_mode=None)
    env.reset(seed=seed)

    n_actions = env.action_space.n

    # Build agent
    if agent_type == "dqn":
        try:
            from src.rl.agent import DQNAgent

            obs_size = env.observation_space.shape[0]
            agent = DQNAgent(obs_size=obs_size, n_actions=n_actions)
            is_dqn = True
        except ImportError:
            logger.warning("PyTorch unavailable, falling back to Q-Learning")
            agent = QLearningAgent(n_actions=n_actions)
            is_dqn = False
    else:
        agent = QLearningAgent(n_actions=n_actions)
        is_dqn = False

    # Optional DB logger
    repo = EpisodeRepository(db_url) if db_url else None
    prom = PrometheusExporter() if push_prometheus else None

    history = []
    best_reward = float("-inf")

    for ep in range(1, n_episodes + 1):
        t0 = time.time()
        stats = run_episode(env, agent, train=True)
        elapsed = time.time() - t0

        stats["episode"] = ep
        stats["elapsed_s"] = round(elapsed, 3)
        history.append(stats)

        # Console log
        logger.info(
            f"Ep {ep:04d}/{n_episodes} | "
            f"R={stats['total_reward']:+.1f} | "
            f"PUE={stats['final_pue']:.3f} | "
            f"Temp={stats['final_avg_temp']:.1f}°C | "
            f"ε={stats['epsilon']:.3f} | "
            f"{elapsed*1000:.0f}ms"
        )

        # Persist episode to MySQL
        if repo:
            repo.log_episode(ep, stats)

        # Export Prometheus metrics
        if prom:
            prom.update(stats)

        # Checkpoint
        if ep % save_every == 0 or stats["total_reward"] > best_reward:
            best_reward = max(best_reward, stats["total_reward"])
            ckpt_path = CHECKPOINT_DIR / f"agent_ep{ep}.{'pt' if is_dqn else 'json'}"
            agent.save(str(ckpt_path))

        # Evaluation run (no exploration)
        if ep % eval_every == 0:
            orig_eps = agent.epsilon
            agent.epsilon = 0.0
            eval_stats = run_episode(env, agent, train=False)
            agent.epsilon = orig_eps
            logger.info(
                f"  [EVAL] R={eval_stats['total_reward']:+.1f} | "
                f"PUE={eval_stats['final_pue']:.3f} | "
                f"GPUDensity={eval_stats['gpu_density']:.2f}×"
            )

    # Final summary
    rewards = [h["total_reward"] for h in history]
    pues = [h["final_pue"] for h in history]
    logger.info("=== Training Complete ===")
    logger.info(f"Best reward   : {max(rewards):+.1f}")
    logger.info(f"Avg reward    : {np.mean(rewards):+.1f}")
    logger.info(f"Final PUE     : {pues[-1]:.3f}  (started {pues[0]:.3f})")
    logger.info(f"PUE reduction : {(pues[0]-pues[-1])/pues[0]*100:.1f}%")

    # Save final checkpoint
    final_path = CHECKPOINT_DIR / f"agent_final.{'pt' if is_dqn else 'json'}"
    agent.save(str(final_path))

    return history


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eco-Logic RL Trainer")
    parser.add_argument(
        "--episodes", type=int, default=200, help="Number of training episodes"
    )
    parser.add_argument(
        "--cooling", type=float, default=0.75, help="Cooling budget [0–1]"
    )
    parser.add_argument(
        "--agent", type=str, default="qlearning", choices=["qlearning", "dqn"]
    )
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--db", type=str, default=None, help="MySQL connection URL")
    parser.add_argument("--prometheus", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        n_episodes=args.episodes,
        cooling_budget=args.cooling,
        agent_type=args.agent,
        save_every=args.save_every,
        eval_every=args.eval_every,
        db_url=args.db,
        push_prometheus=args.prometheus,
        seed=args.seed,
    )
