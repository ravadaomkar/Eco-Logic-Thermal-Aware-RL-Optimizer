"""
eco-logic/src/rl/environment.py
Thermal-Aware Rack Environment (OpenAI Gym compatible)
Models a Dell PowerCool 8x8 rack grid with real thermal dynamics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

RACK_ROWS = 8
RACK_COLS = 8
N_RACKS = RACK_ROWS * RACK_COLS  # 64

# Workload type encoding
WL_TYPES = {
    "ML": {"power_draw": 850, "heat_factor": 1.4},
    "DB": {"power_draw": 320, "heat_factor": 0.8},
    "INFERENCE": {"power_draw": 520, "heat_factor": 1.1},
    "VIDEO": {"power_draw": 680, "heat_factor": 1.2},
    "IDLE": {"power_draw": 50, "heat_factor": 0.2},
}
WL_LIST = list(WL_TYPES.keys())
N_WL = len(WL_LIST)


@dataclass
class RackState:
    temp: float  # Celsius
    power: float  # Watts
    workload: str  # WL_TYPES key
    coolant_flow: float  # L/min


class DataCenterEnv(gym.Env):
    """
    State:
        - rack_temps     : (64,) float32   [28°C – 90°C]
        - rack_powers    : (64,) float32   [50W – 1200W]
        - pue            : (1,)  float32   [1.0 – 2.5]
        - time_of_day    : (1,)  float32   [0 – 1] normalised hour

    Action:
        Discrete(N_RACKS × N_WL) — assign workload type to a rack

    Reward:
        Composite of PUE improvement, thermal safety, and density bonus.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        cooling_budget: float = 0.75,
        max_steps: int = 500,
        target_pue: float = 1.20,
        critical_temp: float = 85.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.cooling_budget = cooling_budget  # 0–1
        self.max_steps = max_steps
        self.target_pue = target_pue
        self.critical_temp = critical_temp
        self.render_mode = render_mode

        # ── Observation space ──────────────────────────────────────────
        obs_size = N_RACKS + N_RACKS + 1 + 1  # temps + powers + pue + time
        self.observation_space = spaces.Box(
            low=np.float32([28.0] * N_RACKS + [50.0] * N_RACKS + [1.0, 0.0]),
            high=np.float32([90.0] * N_RACKS + [1200.0] * N_RACKS + [2.5, 1.0]),
            dtype=np.float32,
        )

        # ── Action space ───────────────────────────────────────────────
        self.action_space = spaces.Discrete(N_RACKS * N_WL)

        # ── Internal state ─────────────────────────────────────────────
        self.racks: list[RackState] = []
        self.pue: float = 1.61
        self.step_count: int = 0
        self.time_of_day: float = 0.0  # fraction of 24h
        self.episode_rewards: list[float] = []

    # ──────────────────────────────────────────────────────────────────
    # Gym API
    # ──────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random

        self.racks = [
            RackState(
                temp=float(rng.uniform(30, 65)),
                power=float(rng.uniform(100, 700)),
                workload=rng.choice(WL_LIST),
                coolant_flow=float(rng.uniform(40, 120)),
            )
            for _ in range(N_RACKS)
        ]
        self.pue = float(rng.uniform(1.50, 1.70))
        self.step_count = 0
        self.time_of_day = float(rng.uniform(0, 1))
        self.episode_rewards = []

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        rack_idx = action // N_WL
        wl_type = WL_LIST[action % N_WL]

        # Apply action: place workload on chosen rack
        self._place_workload(rack_idx, wl_type)

        # Simulate one thermal step for all racks
        self._thermal_step()

        # Recompute PUE
        self._update_pue()

        # Advance simulated time (30-min intervals)
        self.time_of_day = (self.time_of_day + 1 / 48) % 1.0
        self.step_count += 1

        # Compute reward
        reward = self._compute_reward()
        self.episode_rewards.append(reward)

        # Termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_steps

        info = {
            "pue": self.pue,
            "avg_temp": np.mean([r.temp for r in self.racks]),
            "max_temp": np.max([r.temp for r in self.racks]),
            "gpu_density": self._gpu_density(),
            "episode_return": sum(self.episode_rewards),
        }

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        avg_t = np.mean([r.temp for r in self.racks])
        print(
            f"Step {self.step_count:04d} | "
            f"PUE={self.pue:.3f} | "
            f"AvgTemp={avg_t:.1f}°C | "
            f"GPUDensity={self._gpu_density():.2f}x"
        )

    # ──────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        temps = np.array([r.temp for r in self.racks], dtype=np.float32)
        powers = np.array([r.power for r in self.racks], dtype=np.float32)
        return np.concatenate([temps, powers, [self.pue], [self.time_of_day]])

    def _place_workload(self, rack_idx: int, wl_type: str) -> None:
        wl = WL_TYPES[wl_type]
        rack = self.racks[rack_idx]
        rack.workload = wl_type
        rack.power = float(
            np.clip(
                rack.power * 0.7
                + wl["power_draw"] * 0.3
                + self.np_random.normal(0, 20),
                50,
                1200,
            )
        )

    def _thermal_step(self) -> None:
        """Simulate heat dissipation + workload heat generation for one step."""
        cooling_eff = self.cooling_budget
        for i, rack in enumerate(self.racks):
            wl = WL_TYPES[rack.workload]
            # Heat generated by workload
            heat_in = wl["heat_factor"] * (rack.power / 850) * 4.0
            # Cooling removes heat proportional to budget + coolant flow
            heat_out = cooling_eff * (rack.coolant_flow / 120) * 5.0
            noise = float(self.np_random.normal(0, 0.5))
            rack.temp = float(np.clip(rack.temp + heat_in - heat_out + noise, 28, 90))
            # Adjust coolant flow based on current temp (feedback control)
            if rack.temp > 70:
                rack.coolant_flow = min(150, rack.coolant_flow + 2.0)
            elif rack.temp < 45:
                rack.coolant_flow = max(20, rack.coolant_flow - 1.0)

    def _update_pue(self) -> None:
        """PUE = Total Facility Power / IT Equipment Power."""
        it_power = sum(r.power for r in self.racks)
        cooling_power = sum(
            r.coolant_flow * 0.05 for r in self.racks  # pump power estimate
        )
        total_power = it_power + cooling_power + 5000  # lighting/UPS overhead
        self.pue = float(np.clip(total_power / max(it_power, 1), 1.0, 2.5))

    def _compute_reward(self) -> float:
        temps = [r.temp for r in self.racks]
        avg_t = float(np.mean(temps))
        max_t = float(np.max(temps))
        avg_pw = float(np.mean([r.power for r in self.racks]))

        reward = 0.0

        # PUE reward: closer to 1.0 = better
        pue_bonus = (2.0 - self.pue) * 20.0
        reward += pue_bonus

        # Thermal safety
        reward += (85.0 - avg_t) * 0.5
        if avg_t < 70:
            reward += 15.0
        if max_t > self.critical_temp:
            reward -= 50.0  # hard penalty for thermal runaway

        # Efficiency milestone
        if self.pue <= self.target_pue + 0.05:
            reward += 30.0

        # Power cost penalty
        reward -= (avg_pw - 200.0) * 0.01

        # GPU density bonus
        reward += self._gpu_density() * 5.0

        return float(reward)

    def _is_terminated(self) -> bool:
        """Terminate if any rack hits critical temperature."""
        return any(r.temp >= self.critical_temp for r in self.racks)

    def _gpu_density(self) -> float:
        ml_racks = sum(1 for r in self.racks if r.workload == "ML")
        return max(1.0, ml_racks / 16.0 * 4.0)  # normalised to 4× target
