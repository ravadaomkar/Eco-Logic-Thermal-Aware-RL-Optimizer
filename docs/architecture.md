# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Eco-Logic Platform                          │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Digital     │    │  RL Optimizer │    │  Dell OME REST   │  │
│  │  Twin (LSTM) │───▶│  (Q-Learning │◀──▶│  API Client      │  │
│  │              │    │   / DQN)     │    │                  │  │
│  └──────┬───────┘    └──────┬───────┘    └──────────────────┘  │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │  Thermal     │    │  MySQL DB    │    │  Prometheus      │  │
│  │  Simulator   │    │  (Episodes + │    │  + Grafana       │  │
│  │  (Physics)   │    │   Racks)     │    │  (Observability) │  │
│  └──────────────┘    └──────────────┘    └──────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Live Dashboard (HTML/JS)                   │   │
│  │   Heatmap · RL Curve · PUE Trend · Workload Table       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## RL Loop

```
         ┌──────────────────────────────────────────┐
         │              DataCenterEnv               │
         │                                          │
   reset │  State: [temps(64), powers(64), pue, t]  │
   ─────▶│                                          │
         │  Action: rack_idx × workload_type        │
         │                                          │
         │  Reward: f(PUE, temp, power, density)    │
         │                                          │
         │  Next State → Agent update               │
         └──────────────────────────────────────────┘
                          │
                   ε-greedy policy
                          │
              ┌──────────▼──────────┐
              │    QLearning Agent   │
              │    Q[s,a] += α·δ    │
              │                     │
              │    or               │
              │                     │
              │    DQN Agent        │
              │    CNN + Replay     │
              └─────────────────────┘
```

## Data Flow

1. **Dell OME REST API** → fetches live rack telemetry every 30s
2. **Digital Twin** → LSTM predicts thermal spikes 10 min ahead
3. **RL Environment** → combines live data + twin predictions into state
4. **RL Agent** → selects workload placement action
5. **OME Jobs API** → executes migration on real hardware
6. **MySQL** → logs episode, rack snapshot, placement decision
7. **Prometheus** → scrapes metrics every 15s
8. **Dashboard** → visualises heatmap, rewards, PUE trend in real time

## Key Design Decisions

| Decision | Rationale |
|---|---|
| 8×8 rack grid (64 racks) | Matches Dell PowerCool PowerEdge deployment |
| Tabular Q-Learning first | Fast convergence, interpretable, no GPU needed |
| DQN option | Handles full obs space (130-dim) for production |
| Digital twin at 90% acc | Enables safe offline RL without rack damage risk |
| MySQL over NoSQL | Episode data is relational; JOIN queries for analysis |
| Prometheus + Grafana | Industry standard for infrastructure observability |
