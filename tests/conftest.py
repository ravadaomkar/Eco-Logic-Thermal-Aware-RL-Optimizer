"""
eco-logic/tests/conftest.py
Shared pytest fixtures and configuration.
"""

import pytest
import numpy as np
import sys, os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_obs(rng):
    """64-rack observation vector."""
    temps = rng.uniform(30, 80, 64).astype(np.float32)
    powers = rng.uniform(100, 900, 64).astype(np.float32)
    pue = np.array([1.45], dtype=np.float32)
    tod = np.array([0.5], dtype=np.float32)
    return np.concatenate([temps, powers, pue, tod])


@pytest.fixture
def sample_episode_stats():
    return {
        "episode": 1,
        "total_reward": 45.7,
        "mean_td_error": 0.032,
        "epsilon": 0.85,
        "steps": 120,
        "terminated": False,
        "final_pue": 1.42,
        "final_avg_temp": 63.1,
        "final_max_temp": 76.4,
        "gpu_density": 1.8,
        "elapsed_s": 0.412,
    }
