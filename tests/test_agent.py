"""
eco-logic/tests/test_agent.py
Unit tests for Q-Learning agent.
"""

import pytest
import numpy as np
import tempfile
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.agent import QLearningAgent, ReplayBuffer

N_RACKS   = 64
OBS_SIZE  = N_RACKS * 2 + 2
N_ACTIONS = N_RACKS * 5


def make_obs(avg_temp=55.0, pue=1.45):
    temps  = np.full(N_RACKS, avg_temp, dtype=np.float32)
    powers = np.full(N_RACKS, 400.0,   dtype=np.float32)
    return np.concatenate([temps, powers, [pue], [0.5]]).astype(np.float32)


@pytest.fixture
def agent():
    return QLearningAgent(n_actions=N_ACTIONS, epsilon=1.0,
                          epsilon_min=0.05, epsilon_decay=0.9)


class TestQLearningAgent:

    def test_action_in_valid_range(self, agent):
        obs = make_obs()
        for _ in range(50):
            assert 0 <= agent.select_action(obs) < N_ACTIONS

    def test_q_table_shape(self, agent):
        assert agent.q_table.shape == (agent.n_temp_bins, agent.n_pue_bins, N_ACTIONS)

    def test_update_changes_q_value(self, agent):
        obs = make_obs(60.0, 1.5); nobs = make_obs(58.0, 1.45); a = 10
        t, p = agent._discretise(obs)
        q_before = agent.q_table[t, p, a]
        agent.update(obs, a, 25.0, nobs, False)
        assert agent.q_table[t, p, a] != q_before

    def test_terminal_ignores_next_state(self, agent):
        obs = make_obs(88.0, 1.8); nobs = make_obs(30.0, 1.1); a = 0
        t, p = agent._discretise(obs)
        agent.q_table[t, p, :] = 0.0
        agent.update(obs, a, -50.0, nobs, True)
        assert agent.q_table[t, p, a] < 0

    def test_epsilon_decays(self, agent):
        eps = agent.epsilon; agent.decay_epsilon()
        assert agent.epsilon < eps

    def test_epsilon_floor(self, agent):
        for _ in range(1000): agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_min

    def test_exploitation_deterministic(self, agent):
        agent.epsilon = 0.0; obs = make_obs()
        t, p = agent._discretise(obs)
        agent.q_table[t, p, :] = 0.0; agent.q_table[t, p, 42] = 999.0
        for _ in range(10): assert agent.select_action(obs) == 42

    def test_save_load_roundtrip(self, agent):
        agent.update(make_obs(), 7, 15.0, make_obs(), False)
        q = agent.q_table.copy()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            a2 = QLearningAgent(n_actions=N_ACTIONS); a2.load(path)
            np.testing.assert_array_almost_equal(a2.q_table, q)
        finally:
            os.unlink(path)

    def test_step_count_increments(self, agent):
        obs = make_obs()
        for _ in range(5): agent.update(obs, 0, 1.0, obs, False)
        assert agent.step_count == 5


class TestReplayBuffer:

    def test_push_and_len(self):
        buf = ReplayBuffer(100); obs = np.zeros(OBS_SIZE, dtype=np.float32)
        for i in range(10): buf.push(obs, i % 5, float(i), obs, False)
        assert len(buf) == 10

    def test_capacity_eviction(self):
        buf = ReplayBuffer(5); obs = np.zeros(OBS_SIZE, dtype=np.float32)
        for i in range(20): buf.push(obs, 0, 0.0, obs, False)
        assert len(buf) == 5

    def test_sample_shapes(self):
        buf = ReplayBuffer(200); obs = np.random.rand(OBS_SIZE).astype(np.float32)
        for _ in range(100): buf.push(obs, 3, 1.5, obs, False)
        o, a, r, n, d = buf.sample(32)
        assert o.shape == (32, OBS_SIZE) and a.shape == (32,)

    def test_sample_raises_when_too_small(self):
        buf = ReplayBuffer(100)
        with pytest.raises(ValueError): buf.sample(10)
