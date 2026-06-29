# API Reference

## Dell OpenManage Enterprise REST API

Base URL: `https://<OME_HOST>/api`

### Authentication
```
POST /SessionService/Sessions
Content-Type: application/json

{ "UserName": "admin", "Password": "...", "SessionType": "API" }

Response headers:
  X-Auth-Token: <token>
```

### Get Devices
```
GET /DeviceService/Devices?$filter=Type eq 1000
X-Auth-Token: <token>

Returns: { "value": [{ "Id": 1234, "DeviceName": "rack-01", ... }] }
```

### Get Metric Data
```
POST /MetricService/MetricData
X-Auth-Token: <token>

{
  "DeviceIds": [1234, 1235],
  "MetricNames": ["InletTemperature", "PowerConsumption", "CoolantFlowRate"],
  "GroupType": "AllDevices"
}
```

### Submit Workload Migration Job
```
POST /JobService/Jobs
X-Auth-Token: <token>

{
  "JobName": "EcoLogic-Migrate-llm-train",
  "Schedule": "startnow",
  "State": "Enabled",
  "JobType": { "Id": 5, "Name": "Update_Task" },
  "Params": [
    { "Key": "source_rack", "Value": "12" },
    { "Key": "target_rack", "Value": "28" },
    { "Key": "workload_name", "Value": "llm-train-001" }
  ],
  "Targets": [{ "Id": 1234, "Type": { "Id": 1000, "Name": "DEVICE" } }]
}
```

---

## Prometheus Metrics Reference

| Metric | Type | Description |
|---|---|---|
| `datacenter_pue_ratio` | Gauge | Real-time PUE |
| `rack_avg_temp_celsius` | Gauge | Mean rack temperature |
| `rack_max_temp_celsius` | Gauge | Maximum rack temperature |
| `gpu_rack_density_multiplier` | Gauge | GPU density ×1–4 |
| `rl_agent_epsilon` | Gauge | Exploration rate |
| `rl_episode_reward` | Gauge | Last episode reward |
| `powercool_liquid_flow_l_per_min` | Gauge | Coolant flow rate |
| `rack_inlet_temp_celsius` | Gauge | Coolant inlet temp |
| `rl_episodes_total` | Counter | Episodes completed |
| `workload_migrations_total` | Counter | Placement actions |
| `thermal_violations_total` | Counter | Critical temp events |
| `rl_episode_reward_distribution` | Histogram | Reward distribution |
| `datacenter_pue_distribution` | Histogram | PUE distribution |

Scrape endpoint: `http://localhost:8000/metrics`

---

## Python API

### DataCenterEnv
```python
from src.rl.environment import DataCenterEnv

env = DataCenterEnv(cooling_budget=0.75, max_steps=500)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

### QLearningAgent
```python
from src.rl.agent import QLearningAgent

agent = QLearningAgent(n_actions=env.action_space.n)
action = agent.select_action(obs)
td_error = agent.update(obs, action, reward, next_obs, done)
agent.decay_epsilon()
agent.save("checkpoints/agent.json")
agent.load("checkpoints/agent.json")
```

### compute_reward
```python
from src.rl.reward import compute_reward, reward_breakdown

r = compute_reward(avg_temp=62.0, max_temp=71.0, pue=1.35,
                   avg_power=380.0, gpu_density=2.5)

breakdown = reward_breakdown(62.0, 71.0, 1.35, 380.0, 2.5)
# {'pue_term': 13.0, 'temp_term': 11.5, 'safe_zone_bonus': 15.0, ...}
```

### DellOMEClient
```python
from src.api.dell_ome import DellOMEClient

client = DellOMEClient()
client.authenticate()
devices = client.get_devices(device_type="1000")
telemetry = client.get_rack_telemetry(device_ids=[1001, 1002, 1003])
job_id = client.trigger_workload_migration(
    source_rack_id=12, target_rack_id=28,
    workload_name="llm-train-001"
)
status = client.wait_for_job(job_id)
```

### ThermalPhysicsSimulator
```python
from src.digital_twin.simulator import ThermalPhysicsSimulator

sim = ThermalPhysicsSimulator()
result = sim.step(temp_c=65.0, power_w=800.0, coolant_flow_l_min=90.0)
# {'next_temp': 63.2, 'heat_removed_w': 5040.0, 'thermal_spike': False}

forecast = sim.predict_horizon(temp_c=65.0, power_w=800.0,
                               coolant_flow=90.0, horizon_steps=20)
```
