"""eco-logic/tests/test_environment.py"""

import pytest, numpy as np, sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.environment import DataCenterEnv, N_RACKS, WL_LIST


@pytest.fixture
def env():
    e = DataCenterEnv(cooling_budget=0.75, max_steps=50)
    e.reset(seed=0)
    return e


class TestDataCenterEnv:
    def test_obs_shape(self, env):
        obs, _ = env.reset(seed=1)
        assert obs.shape == (N_RACKS * 2 + 2,)

    def test_obs_within_bounds(self, env):
        obs, _ = env.reset(seed=2)
        assert np.all(obs >= env.observation_space.low)
        assert np.all(obs <= env.observation_space.high)

    def test_action_space_size(self, env):
        assert env.action_space.n == N_RACKS * len(WL_LIST)

    def test_step_types(self, env):
        env.reset(seed=3)
        obs, r, term, trunc, info = env.step(0)
        assert isinstance(obs, np.ndarray) and isinstance(r, float)

    def test_step_count(self, env):
        env.reset(seed=4)
        for i in range(1, 6):
            env.step(env.action_space.sample())
            assert env.step_count == i

    def test_truncation_at_max_steps(self):
        e = DataCenterEnv(max_steps=5)
        e.reset(seed=5)
        done = False
        steps = 0
        while not done:
            _, _, term, trunc, _ = e.step(e.action_space.sample())
            done = term or trunc
            steps += 1
        assert steps <= 5

    def test_thermal_runaway_terminates(self):
        e = DataCenterEnv(cooling_budget=0.0, max_steps=1000)
        e.reset(seed=6)
        for r in e.racks:
            r.temp = 85.1
        _, _, term, _, _ = e.step(0)
        assert term

    def test_info_keys(self, env):
        env.reset(seed=7)
        _, _, _, _, info = env.step(0)
        for k in ("pue", "avg_temp", "max_temp", "gpu_density", "episode_return"):
            assert k in info

    def test_pue_valid(self, env):
        env.reset(seed=8)
        for _ in range(20):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            assert info["pue"] >= 1.0
            if term or trunc:
                break

    def test_rack_temps_bounded(self, env):
        env.reset(seed=9)
        for _ in range(10):
            env.step(env.action_space.sample())
        for r in env.racks:
            assert 28.0 <= r.temp <= 90.0
