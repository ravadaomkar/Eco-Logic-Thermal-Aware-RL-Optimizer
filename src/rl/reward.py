"""
eco-logic/src/rl/reward.py
Composite reward function for thermal-aware workload placement.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class RewardWeights:
    pue_scale: float = 20.0
    temp_scale: float = 0.5
    power_penalty: float = 0.01
    safe_zone_bonus: float = 15.0
    efficiency_milestone: float = 30.0
    critical_penalty: float = 50.0
    density_bonus: float = 5.0


def compute_reward(
    avg_temp: float,
    max_temp: float,
    pue: float,
    avg_power: float,
    gpu_density: float,
    target_pue: float = 1.20,
    critical_temp: float = 85.0,
    weights: RewardWeights = None,
) -> float:
    """
    Compute the composite RL reward signal.

    Args:
        avg_temp     : Mean rack temperature across all 64 racks (°C)
        max_temp     : Maximum rack temperature (°C)
        pue          : Current Power Usage Effectiveness
        avg_power    : Mean rack power draw (W)
        gpu_density  : Current GPU rack density multiplier (1× – 4×)
        target_pue   : PUE target for efficiency milestone bonus
        critical_temp: Temperature above which a hard penalty applies
        weights      : RewardWeights dataclass (defaults used if None)

    Returns:
        Scalar reward (float)
    """
    if weights is None:
        weights = RewardWeights()

    reward = 0.0

    # 1. PUE improvement — primary objective
    #    PUE near 1.0 is ideal; 2.0 is very poor
    reward += (2.0 - pue) * weights.pue_scale

    # 2. Thermal safety
    reward += (critical_temp - avg_temp) * weights.temp_scale

    # 3. Safe zone bonus (below 70°C is operational comfort zone)
    if avg_temp < 70.0:
        reward += weights.safe_zone_bonus

    # 4. Critical temperature penalty (thermal runaway risk)
    if max_temp >= critical_temp:
        reward -= weights.critical_penalty

    # 5. Efficiency milestone (approaching target PUE)
    if pue <= target_pue + 0.05:
        reward += weights.efficiency_milestone

    # 6. Power cost penalty (energy pricing)
    reward -= max(0.0, avg_power - 200.0) * weights.power_penalty

    # 7. GPU density bonus (business value: higher density = more revenue)
    reward += gpu_density * weights.density_bonus

    return float(reward)


def reward_breakdown(
    avg_temp: float,
    max_temp: float,
    pue: float,
    avg_power: float,
    gpu_density: float,
    target_pue: float = 1.20,
    critical_temp: float = 85.0,
    weights: RewardWeights = None,
) -> dict:
    """Return a breakdown of each reward component for logging/debugging."""
    if weights is None:
        weights = RewardWeights()

    return {
        "pue_term":           round((2.0 - pue) * weights.pue_scale, 3),
        "temp_term":          round((critical_temp - avg_temp) * weights.temp_scale, 3),
        "safe_zone_bonus":    weights.safe_zone_bonus if avg_temp < 70 else 0.0,
        "critical_penalty":   -weights.critical_penalty if max_temp >= critical_temp else 0.0,
        "efficiency_bonus":   weights.efficiency_milestone if pue <= target_pue + 0.05 else 0.0,
        "power_penalty":      round(-max(0.0, avg_power - 200.0) * weights.power_penalty, 3),
        "density_bonus":      round(gpu_density * weights.density_bonus, 3),
        "total":              round(compute_reward(
            avg_temp, max_temp, pue, avg_power, gpu_density,
            target_pue, critical_temp, weights
        ), 3),
    }
