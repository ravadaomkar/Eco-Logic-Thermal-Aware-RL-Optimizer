-- eco-logic/src/db/schema.sql
-- MySQL 8.0 schema for the Eco-Logic Thermal-Aware RL Optimizer
-- Run: mysql -u root -p < src/db/schema.sql

CREATE DATABASE IF NOT EXISTS ecologic
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE ecologic;

-- ─────────────────────────────────────────────────────────────────────────────
-- rl_episodes: one row per training episode
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rl_episodes (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    episode         INT UNSIGNED     NOT NULL,
    created_at      DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- RL stats
    total_reward    FLOAT            NOT NULL,
    mean_td_error   FLOAT            DEFAULT NULL,
    epsilon         FLOAT            NOT NULL,
    steps           INT UNSIGNED     NOT NULL,
    terminated      TINYINT(1)       NOT NULL DEFAULT 0,

    -- Infrastructure metrics
    final_pue       DECIMAL(5,4)     NOT NULL,
    final_avg_temp  DECIMAL(5,2)     NOT NULL,
    final_max_temp  DECIMAL(5,2)     NOT NULL,
    gpu_density     DECIMAL(5,3)     NOT NULL,
    elapsed_s       DECIMAL(8,3)     DEFAULT NULL,

    INDEX idx_episode     (episode),
    INDEX idx_created_at  (created_at),
    INDEX idx_final_pue   (final_pue)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────────────────────
-- rack_snapshots: per-step rack telemetry (sampled every N steps)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rack_snapshots (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    episode_id      BIGINT UNSIGNED  NOT NULL,
    step            INT UNSIGNED     NOT NULL,
    captured_at     DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,

    rack_idx        TINYINT UNSIGNED NOT NULL,    -- 0-63
    temp_celsius    DECIMAL(5,2)     NOT NULL,
    power_watts     DECIMAL(7,2)     NOT NULL,
    coolant_flow    DECIMAL(6,2)     DEFAULT NULL,
    workload_type   VARCHAR(16)      DEFAULT NULL, -- ML, DB, INFERENCE, VIDEO, IDLE

    FOREIGN KEY (episode_id) REFERENCES rl_episodes(id) ON DELETE CASCADE,
    INDEX idx_episode_step (episode_id, step),
    INDEX idx_rack_idx     (rack_idx)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────────────────────
-- workload_placements: log every RL action (workload → rack assignment)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS workload_placements (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    episode_id      BIGINT UNSIGNED  NOT NULL,
    step            INT UNSIGNED     NOT NULL,
    placed_at       DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,

    action          INT UNSIGNED     NOT NULL,    -- raw action index
    rack_idx        TINYINT UNSIGNED NOT NULL,
    workload_type   VARCHAR(16)      NOT NULL,
    reward          FLOAT            NOT NULL,

    -- State snapshot at decision time
    avg_temp_before DECIMAL(5,2)     DEFAULT NULL,
    pue_before      DECIMAL(5,4)     DEFAULT NULL,

    FOREIGN KEY (episode_id) REFERENCES rl_episodes(id) ON DELETE CASCADE,
    INDEX idx_rack_type (rack_idx, workload_type)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────────────────────
-- dell_ome_jobs: track OpenManage Enterprise migration jobs
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dell_ome_jobs (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    submitted_at    DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at    DATETIME         DEFAULT NULL,

    ome_job_id      INT UNSIGNED     NOT NULL,
    source_rack_id  INT UNSIGNED     NOT NULL,
    target_rack_id  INT UNSIGNED     NOT NULL,
    workload_name   VARCHAR(128)     NOT NULL,
    status          VARCHAR(32)      NOT NULL DEFAULT 'Pending',  -- Pending, Running, Completed, Failed
    reason          TEXT             DEFAULT NULL,

    INDEX idx_status      (status),
    INDEX idx_ome_job_id  (ome_job_id)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────────────────────
-- twin_evaluations: digital twin accuracy tracking
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS twin_evaluations (
    id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    evaluated_at    DATETIME         NOT NULL DEFAULT CURRENT_TIMESTAMP,

    model_version   VARCHAR(64)      DEFAULT NULL,
    n_samples       INT UNSIGNED     NOT NULL,
    accuracy        DECIMAL(5,4)     NOT NULL,
    precision_val   DECIMAL(5,4)     DEFAULT NULL,
    recall_val      DECIMAL(5,4)     DEFAULT NULL,
    f1_score        DECIMAL(5,4)     DEFAULT NULL,
    pue_forecast_mae DECIMAL(6,4)    DEFAULT NULL,   -- mean absolute error

    INDEX idx_evaluated_at (evaluated_at)
) ENGINE=InnoDB;


-- ─────────────────────────────────────────────────────────────────────────────
-- Useful views
-- ─────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE VIEW v_training_summary AS
SELECT
    COUNT(*)                        AS total_episodes,
    MIN(final_pue)                  AS best_pue,
    AVG(final_pue)                  AS avg_pue,
    MAX(final_pue)                  AS worst_pue,
    AVG(final_avg_temp)             AS avg_temp_celsius,
    MAX(gpu_density)                AS max_gpu_density,
    SUM(terminated)                 AS thermal_violations,
    AVG(total_reward)               AS avg_reward,
    MAX(total_reward)               AS best_reward
FROM rl_episodes;


CREATE OR REPLACE VIEW v_pue_trend AS
SELECT
    episode,
    final_pue,
    AVG(final_pue) OVER (ORDER BY episode ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) AS pue_ma10,
    created_at
FROM rl_episodes
ORDER BY episode;
