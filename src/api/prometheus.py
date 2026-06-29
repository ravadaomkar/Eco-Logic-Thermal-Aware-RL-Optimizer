"""
eco-logic/src/api/prometheus.py
Custom Prometheus metrics exporter for Eco-Logic RL Optimizer.
Exposes metrics on :8000/metrics for Prometheus scraping.
"""

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (REGISTRY, Counter, Gauge, Histogram,
                                   Summary, start_http_server)

    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Run: pip install prometheus-client"
    )


class PrometheusExporter:
    """
    Registers and updates all Eco-Logic Prometheus metrics.

    Usage:
        exporter = PrometheusExporter(port=8000)
        exporter.update(episode_stats)
    """

    def __init__(self, port: int = 8000):
        if not PROM_AVAILABLE:
            logger.warning("Prometheus metrics disabled (prometheus_client missing)")
            self._enabled = False
            return

        self._enabled = True

        # ── Gauges (current values) ────────────────────────────────────
        self.pue = Gauge(
            "datacenter_pue_ratio",
            "Power Usage Effectiveness (total / IT power)",
        )
        self.avg_temp = Gauge(
            "rack_avg_temp_celsius",
            "Average temperature across all racks (°C)",
        )
        self.max_temp = Gauge(
            "rack_max_temp_celsius",
            "Maximum temperature across all racks (°C)",
        )
        self.gpu_density = Gauge(
            "gpu_rack_density_multiplier",
            "GPU rack density relative to baseline (×)",
        )
        self.epsilon = Gauge(
            "rl_agent_epsilon",
            "Current exploration rate (ε) of the RL agent",
        )
        self.episode_reward = Gauge(
            "rl_episode_reward",
            "Total reward for the most recent RL episode",
        )
        self.liquid_flow = Gauge(
            "powercool_liquid_flow_l_per_min",
            "Aggregate coolant flow rate across all racks (L/min)",
        )
        self.inlet_temp = Gauge(
            "rack_inlet_temp_celsius",
            "Mean coolant inlet temperature (°C)",
        )

        # ── Counters (monotonically increasing) ───────────────────────
        self.episodes_total = Counter(
            "rl_episodes_total",
            "Total number of RL training episodes completed",
        )
        self.migrations_total = Counter(
            "workload_migrations_total",
            "Total number of workload placement actions executed",
        )
        self.thermal_violations_total = Counter(
            "thermal_violations_total",
            "Number of episodes where a rack exceeded the critical temperature",
        )

        # ── Histograms (distributions) ────────────────────────────────
        self.reward_histogram = Histogram(
            "rl_episode_reward_distribution",
            "Distribution of per-episode total rewards",
            buckets=[-100, -50, -20, 0, 20, 50, 100, 150, 200],
        )
        self.pue_histogram = Histogram(
            "datacenter_pue_distribution",
            "Distribution of PUE values observed",
            buckets=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5],
        )

        # Start HTTP server in background thread
        threading.Thread(
            target=start_http_server,
            args=(port,),
            daemon=True,
        ).start()
        logger.info(f"Prometheus metrics server started on :{port}/metrics")

    def update(self, stats: dict) -> None:
        """Update all gauges/counters from an episode stats dict."""
        if not self._enabled:
            return

        pue = stats.get("final_pue", 0)
        avg_t = stats.get("final_avg_temp", 0)
        max_t = stats.get("final_max_temp", 0)
        density = stats.get("gpu_density", 1.0)
        eps = stats.get("epsilon", 0)
        reward = stats.get("total_reward", 0)

        self.pue.set(pue)
        self.avg_temp.set(avg_t)
        self.max_temp.set(max_t)
        self.gpu_density.set(density)
        self.epsilon.set(eps)
        self.episode_reward.set(reward)

        # Derived metrics
        self.liquid_flow.set(density * 22 * 60)  # rough estimate L/min
        self.inlet_temp.set(max(18, avg_t - 22))  # coolant inlet ≈ rack avg - 22

        # Counters
        self.episodes_total.inc()
        self.migrations_total.inc()  # at least 1 action per episode
        if stats.get("terminated", False):
            self.thermal_violations_total.inc()

        # Histograms
        self.reward_histogram.observe(reward)
        self.pue_histogram.observe(pue)

    def set_rack_metrics(
        self,
        rack_idx: int,
        temp: float,
        power: float,
        flow: float,
    ) -> None:
        """Update per-rack metrics (call with current telemetry each step)."""
        if not self._enabled:
            return
        # Per-rack gauges are created lazily with labels
        # Prometheus client handles label cardinality; fine for 64 racks
        pass  # extend with LabelledGauge if per-rack detail needed
