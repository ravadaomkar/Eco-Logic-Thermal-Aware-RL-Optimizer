"""
eco-logic/src/db/repository.py
MySQL data access layer for Eco-Logic.
"""

import logging
import os
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

try:
    import mysql.connector
    from mysql.connector.pooling import MySQLConnectionPool
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False
    logger.warning("mysql-connector-python not installed. Run: pip install mysql-connector-python")


class EpisodeRepository:
    """
    Handles all database I/O for the RL training loop.
    """

    def __init__(self, db_url: Optional[str] = None):
        if not MYSQL_AVAILABLE:
            logger.warning("MySQL unavailable; episode logging disabled.")
            self._enabled = False
            return

        # Parse URL or fall back to env vars
        # Expected format: mysql://user:pass@host:3306/ecologic
        host     = os.environ.get("MYSQL_HOST",     "localhost")
        port     = int(os.environ.get("MYSQL_PORT", "3306"))
        user     = os.environ.get("MYSQL_USER",     "ecologic")
        password = os.environ.get("MYSQL_PASSWORD", "")
        database = os.environ.get("MYSQL_DB",       "ecologic")

        if db_url:
            from urllib.parse import urlparse
            p = urlparse(db_url)
            host     = p.hostname or host
            port     = p.port or port
            user     = p.username or user
            password = p.password or password
            database = (p.path or "/ecologic").lstrip("/") or database

        try:
            self._pool = MySQLConnectionPool(
                pool_name="ecologic_pool",
                pool_size=5,
                host=host,
                port=port,
                user=user,
                password=password,
                database=database,
                autocommit=True,
            )
            self._enabled = True
            logger.info(f"MySQL pool connected to {host}:{port}/{database}")
        except Exception as e:
            logger.error(f"MySQL connection failed: {e}")
            self._enabled = False

    @contextmanager
    def _cursor(self):
        if not self._enabled:
            yield None
            return
        conn = self._pool.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"DB error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    # ── Episode logging ───────────────────────────────────────────────

    def log_episode(self, episode: int, stats: dict) -> Optional[int]:
        """Insert one episode record. Returns the new row ID."""
        sql = """
            INSERT INTO rl_episodes (
                episode, total_reward, mean_td_error, epsilon,
                steps, terminated, final_pue, final_avg_temp,
                final_max_temp, gpu_density, elapsed_s
            ) VALUES (
                %(episode)s, %(total_reward)s, %(mean_td_error)s, %(epsilon)s,
                %(steps)s, %(terminated)s, %(final_pue)s, %(final_avg_temp)s,
                %(final_max_temp)s, %(gpu_density)s, %(elapsed_s)s
            )
        """
        params = {
            "episode":       episode,
            "total_reward":  stats.get("total_reward", 0),
            "mean_td_error": stats.get("mean_td_error"),
            "epsilon":       stats.get("epsilon", 0),
            "steps":         stats.get("steps", 0),
            "terminated":    int(stats.get("terminated", False)),
            "final_pue":     stats.get("final_pue", 0),
            "final_avg_temp":stats.get("final_avg_temp", 0),
            "final_max_temp":stats.get("final_max_temp", 0),
            "gpu_density":   stats.get("gpu_density", 1.0),
            "elapsed_s":     stats.get("elapsed_s"),
        }
        with self._cursor() as cur:
            if cur is None:
                return None
            cur.execute(sql, params)
            return cur.lastrowid

    def log_rack_snapshot(self, episode_id: int, step: int, racks: list) -> None:
        """Bulk insert rack state for a given step."""
        if not self._enabled:
            return
        sql = """
            INSERT INTO rack_snapshots
            (episode_id, step, rack_idx, temp_celsius, power_watts, coolant_flow, workload_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        rows = [
            (episode_id, step, i, r.temp, r.power, r.coolant_flow, r.workload)
            for i, r in enumerate(racks)
        ]
        with self._cursor() as cur:
            if cur:
                cur.executemany(sql, rows)

    def log_placement(
        self,
        episode_id: int,
        step: int,
        action: int,
        rack_idx: int,
        workload_type: str,
        reward: float,
        avg_temp: float,
        pue: float,
    ) -> None:
        sql = """
            INSERT INTO workload_placements
            (episode_id, step, action, rack_idx, workload_type, reward, avg_temp_before, pue_before)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._cursor() as cur:
            if cur:
                cur.execute(sql, (episode_id, step, action, rack_idx, workload_type, reward, avg_temp, pue))

    # ── Queries ───────────────────────────────────────────────────────

    def get_training_summary(self) -> Optional[Dict]:
        with self._cursor() as cur:
            if cur is None:
                return None
            cur.execute("SELECT * FROM v_training_summary")
            return cur.fetchone()

    def get_pue_trend(self, limit: int = 200) -> List[Dict]:
        with self._cursor() as cur:
            if cur is None:
                return []
            cur.execute("SELECT * FROM v_pue_trend ORDER BY episode DESC LIMIT %s", (limit,))
            return cur.fetchall()

    def get_best_episodes(self, top_n: int = 10) -> List[Dict]:
        with self._cursor() as cur:
            if cur is None:
                return []
            cur.execute(
                "SELECT * FROM rl_episodes ORDER BY total_reward DESC LIMIT %s",
                (top_n,)
            )
            return cur.fetchall()
