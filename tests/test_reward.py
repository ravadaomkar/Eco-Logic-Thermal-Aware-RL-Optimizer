"""eco-logic/tests/test_reward.py"""
import pytest, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rl.reward import compute_reward, reward_breakdown, RewardWeights

class TestReward:
    def test_low_pue_gives_higher_reward(self):
        r_good = compute_reward(50, 60, 1.2, 300, 2.0)
        r_bad  = compute_reward(50, 60, 1.8, 300, 2.0)
        assert r_good > r_bad

    def test_high_temp_penalised(self):
        r_cool = compute_reward(45, 55, 1.4, 300, 1.0)
        r_hot  = compute_reward(80, 90, 1.4, 300, 1.0)
        assert r_cool > r_hot

    def test_critical_temp_penalty(self):
        r_safe     = compute_reward(65, 75, 1.4, 300, 1.0, critical_temp=85.0)
        r_critical = compute_reward(65, 86, 1.4, 300, 1.0, critical_temp=85.0)
        assert r_safe > r_critical

    def test_safe_zone_bonus_applied(self):
        w = RewardWeights(safe_zone_bonus=15.0)
        r_safe = compute_reward(65, 70, 1.5, 300, 1.0, weights=w)
        r_over = compute_reward(72, 78, 1.5, 300, 1.0, weights=w)
        assert r_safe > r_over

    def test_efficiency_milestone_bonus(self):
        r_hit  = compute_reward(55, 65, 1.20, 300, 1.0, target_pue=1.20)
        r_miss = compute_reward(55, 65, 1.50, 300, 1.0, target_pue=1.20)
        assert r_hit > r_miss

    def test_higher_density_increases_reward(self):
        r_low  = compute_reward(55, 65, 1.4, 300, 1.0)
        r_high = compute_reward(55, 65, 1.4, 300, 4.0)
        assert r_high > r_low

    def test_breakdown_totals_match_compute(self):
        args = (60.0, 70.0, 1.35, 350.0, 2.0)
        total = compute_reward(*args)
        bd    = reward_breakdown(*args)
        assert abs(bd["total"] - round(total, 3)) < 0.001

    def test_breakdown_keys(self):
        bd = reward_breakdown(60, 70, 1.4, 300, 1.5)
        for k in ("pue_term","temp_term","safe_zone_bonus","critical_penalty",
                  "efficiency_bonus","power_penalty","density_bonus","total"):
            assert k in bd

    def test_power_penalty_increases_with_power(self):
        r_low  = compute_reward(55, 65, 1.4, 250, 1.0)
        r_high = compute_reward(55, 65, 1.4, 900, 1.0)
        assert r_low > r_high

    def test_return_type_is_float(self):
        assert isinstance(compute_reward(55, 65, 1.4, 300, 1.0), float)
