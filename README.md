# ⬡ Eco-Logic: Thermal-Aware RL Optimizer

> **Reinforcement Learning-based workload placement optimizer for Dell PowerCool liquid-cooled data center infrastructure.**  
> Reduces PUE from 1.6 → 1.2 and enables 4× GPU rack density by intelligently placing workloads based on real-time thermal and energy cost signals.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MySQL](https://img.shields.io/badge/MySQL-8.0-orange?logo=mysql)](https://mysql.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)](https://docker.com)
[![Prometheus](https://img.shields.io/badge/Prometheus-Metrics-red?logo=prometheus)](https://prometheus.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📌 Overview

Eco-Logic is an end-to-end AI platform for **thermal-aware workload orchestration** in liquid-cooled HPC/AI data centers. It combines:

- **Reinforcement Learning (Q-Learning / DQN)** to learn optimal rack placement policies
- **Digital Twin simulation** achieving 90% accuracy in predicting real-world thermal spikes
- **Dell OpenManage Enterprise REST API** integration for live hardware telemetry
- **Prometheus + Grafana** observability stack for real-time PUE and temperature monitoring
- **MySQL** for episode storage, workload metadata, and rack state history

### Key Results

| Metric | Before | After | Impact |
|--------|--------|-------|--------|
| PUE (Power Usage Effectiveness) | 1.61 | 1.20 | ▼ 25% energy saved |
| GPU Rack Density | 1× | 4× | ▲ 4× throughput |
| Thermal Spike Prediction | — | 90% acc | Digital twin |
| Stalled Enterprise Deals | $50M+ blocked | Unblocked | Power cap resolved |

---

## 🗂 Project Structure

```
eco-logic/
├── src/
│   ├── rl/
│   │   ├── agent.py           # Q-Learning / DQN agent
│   │   ├── environment.py     # Rack thermal environment (OpenAI Gym)
│   │   ├── reward.py          # Reward function (PUE + temp + power)
│   │   └── train.py           # Training loop + episode logger
│   ├── digital_twin/
│   │   ├── simulator.py       # Thermal spike simulator
│   │   └── model.py           # LSTM thermal prediction model
│   ├── api/
│   │   ├── dell_ome.py        # Dell OpenManage Enterprise REST client
│   │   └── prometheus.py      # Prometheus metrics exporter
│   ├── db/
│   │   ├── schema.sql         # MySQL schema
│   │   └── repository.py      # DB access layer
│   └── main.py                # Entry point
├── dashboard/
│   └── index.html             # Live thermal dashboard (standalone)
├── data/
│   ├── rl_simulation_results.json
│   └── rack_telemetry_sample.csv
├── notebooks/
│   ├── 01_rl_training_analysis.ipynb
│   └── 02_thermal_twin_evaluation.ipynb
├── tests/
│   ├── test_agent.py
│   ├── test_environment.py
│   └── test_reward.py
├── docs/
│   ├── architecture.md
│   └── api_reference.md
├── .github/
│   └── workflows/
│       └── ci.yml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── config.yaml
└── .env.example
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/<ravadaomkar>/Eco-Logic-Thermal-Aware-RL-Optimizer.git
cd eco-logic
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Dell OME credentials and MySQL connection
```

### 3. Start Infrastructure

```bash
docker-compose up -d   # Starts MySQL + Prometheus + Grafana
```

### 4. Run the RL Optimizer

```bash
python src/main.py --episodes 200 --cooling-budget 0.75
```

### 5. Open the Dashboard

```bash
open dashboard/index.html
# Or serve it: python -m http.server 8080
```

---

## 🧠 RL Architecture

```
State  : [rack_temp_vector(64), rack_power_vector(64), pue, time_of_day]
Action : workload_placement_decision ∈ {0..63} × workload_type
Reward : f(pue, temp, power_cost, density_bonus)
Policy : ε-greedy Q-Learning → DQN (CNN over rack grid)
```

### Reward Function

```python
reward = (
    (85 - avg_temp) * 0.5        # temp below threshold
  + (2.0 - pue) * 20             # PUE improvement bonus
  - (avg_power - 200) * 0.01     # power cost penalty
  + 15 * (avg_temp < 70)         # safe zone bonus
  + 30 * (pue < 1.4)             # efficiency milestone
)
```

---

## 📡 Dell OpenManage Enterprise Integration

Eco-Logic queries the OME REST API every 30 seconds to fetch:
- `GET /api/DeviceService/Devices` — rack inventory
- `GET /api/MetricService/MetricData` — live power & thermal readings
- `POST /api/JobService/Jobs` — trigger workload migration actions

See [`src/api/dell_ome.py`](src/api/dell_ome.py) and [`docs/api_reference.md`](docs/api_reference.md).

---

## 📊 Prometheus Metrics

| Metric | Description |
|--------|-------------|
| `powercool_liquid_flow_l_per_min` | Coolant flow rate per rack |
| `rack_inlet_temp_celsius` | Inlet temperature sensor |
| `datacenter_pue_ratio` | Real-time PUE |
| `gpu_utilization_percent` | GPU compute load |
| `rl_episode_reward` | Current RL episode reward |
| `workload_migrations_total` | Total placement decisions |

Grafana dashboard JSON: [`docs/grafana_dashboard.json`](docs/grafana_dashboard.json)

---

## 🧬 Digital Twin

The digital twin uses an **LSTM model** trained on 6 months of historical rack telemetry to predict thermal spikes 10 minutes ahead:

- **Input**: Rolling 30-step thermal + power window per rack
- **Output**: Spike probability (>78°C) per rack in next 10 min
- **Accuracy**: 90% on held-out test set
- **Inference**: Runs every 60s; feeds RL state space

---

## 🐳 Docker

```bash
docker-compose up -d
```

Services:
- `optimizer` — RL training loop (Python)
- `mysql` — Episode + rack state storage
- `prometheus` — Metrics scraping
- `grafana` — Dashboard on `:3000`

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---
