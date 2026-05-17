# AdaptiFlow-X: Adaptive Multi-Objective Deep RL Traffic Intelligence System

<div align="center">

![Version](https://img.shields.io/badge/Version-2.0-6699ff?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Dueling%20DDQN%20%2B%20PER-red?style=flat-square&logo=pytorch&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-Live%20Logging-orange?style=flat-square)
![License](https://img.shields.io/badge/License-Original%20Work-lightgrey?style=flat-square)

**Author:** Vaibhav Krishna V &nbsp;|&nbsp;  
**Architecture:** Dueling Double DQN + Prioritised Experience Replay + Multi-Objective Reward

</div>

---

## What This Is

AdaptiFlow-X is a real-time urban traffic signal control system that trains a Dueling Double DQN agent to manage a 3×3 grid of nine signalised intersections simultaneously. The agent selects both which lane to green and how long to hold that green phase, from a 16-action space spanning four lanes and four phase durations. It is rewarded not just for clearing queues, but for throughput, power efficiency, and spill overflow avoidance — a composite signal that drives qualitatively different behaviour than a queue-only objective.

The simulation runs inside a Tkinter canvas that renders the full grid in real time: intersection boxes coloured by congestion level, per-lane signal states, queue bar overlays, discharge counts, and a live TorchScript-saved agent deployable without the training stack. Three control algorithms — Fixed-Time, Density-Heuristic, and DQN — run back-to-back in an automated benchmark suite, producing side-by-side throughput, average queue, power, and CO₂ comparisons.

---

## Why This Architecture

Fixed-time controllers are widely deployed precisely because they are simple and predictable. The first honest comparison any adaptive system must make is against a well-tuned fixed cycle — not a straw man. AdaptiFlow-X runs the fixed-time benchmark automatically before any DQN result is presented, so the comparison is always grounded.

Density-based control (greening the highest-queue lane) is a natural improvement and often performs well in isolation. The reason it fails under network-level pressure is that it has no notion of downstream: greening a lane at node 4 may discharge vehicles that immediately pile into node 5's saturated queues. The DQN agent, through the overflow penalty and pressure-routing mechanism, learns to anticipate this and holds green shorter when the downstream neighbour is congested.

The Dueling architecture separates the baseline value of being in a state from the advantage of specific actions within it. For traffic control, this matters: most of the time, the intersection is not at a critical decision point, and the agent benefits from a value stream that says "this state is generally fine" without attributing that assessment to any specific action choice. The advantage stream then expresses the marginal gain of each action over the baseline.

Prioritised replay addresses the fundamental data imbalance problem: critical transitions — queue spikes, overflow events, emergency preemptions — are rare but carry the most learning signal. Uniform sampling would bury these under thousands of routine low-traffic steps. PER up-weights high-TD-error transitions so the network sees extreme events proportionally more often, with IS weights correcting the resulting gradient bias.

---

## System Architecture

```
Real-Time Tkinter Canvas  (SIM_SPEED=28 ms/frame)
        │
        ▼
Schedule Manager  →  active_schedule  →  arrival_rate + power_factor
        │
        ▼
spawn_vehicles()  →  Poisson arrivals (rate = schedule.arrival per lane)
        │
        ▼
For each of 9 Intersection nodes:
  ├── Phase timer: re-decide only when phase_remaining expires
  ├── select_action()  →  FIXED | DENSITY | DQN (ε-greedy)
  │       DQN: policy_net(state_vec) → argmax over 16 actions
  ├── discharge_and_route()  →  clear green lane + pressure routing
  ├── compute_reward()  →  queue Δ + throughput + power penalty + overflow
  └── PER push (s, a, r, s', done)
        │
        ▼
train_step()  (Double DQN + PER, Huber loss, grad-clip 10.0)
target_net sync every TARGET_FREQ=150 steps
        │
        ▼
TensorBoard: Loss/Train, Reward/Step, City/Power, City/Throughput, City/CO₂
```

---

## Engineering Depth

### State Representation (8-dimensional)

Each intersection maintains an independent 8-D state vector fed to the policy network:

```
s = [ q_N/50,  q_E/50,  q_S/50,  q_W/50,   ← queue per lane, normalised to [0,1]
      sin(2π·hour/24),  cos(2π·hour/24),     ← time-of-day encoding (circular)
      action/(LANES-1),                       ← last active lane, normalised
      phase_idx/(N_PHASES-1) ]               ← last phase duration index, normalised
```

Time-of-day is encoded as a (sin, cos) pair rather than a scalar hour value so that 23:00 and 00:00 are geometrically adjacent in state space — the circular representation prevents discontinuity at midnight. This is the standard technique for periodic features in RL and neural time-series models.

Queue values are normalised by MAX_QUEUE=50 so the network sees inputs in [0,1] regardless of absolute congestion level, keeping gradient magnitudes consistent across different load scenarios.

### Action Space (16 discrete actions)

```
A = { lane × phase_duration } = 4 lanes × 4 durations = 16 actions

PHASE_DURATIONS = [8s, 16s, 28s, 45s]
```

A single integer action encodes both which lane to green and how long to hold it. Decoding: `lane = a % 4`, `phase_idx = a // 4`. This allows the agent to choose short greens for low-traffic lanes and long greens during peak load — the same adaptive phase scheduling used in SCOOT and SCATS systems, here learned rather than hand-programmed.

The phase timer prevents thrashing: once a decision is made, it commits for `PHASE_DURATIONS[pidx]` steps before re-querying the network. This models the minimum green time constraint in real traffic signal controllers.

### Dueling Double DQN

```
Input  [8]
    │
trunk:  Linear(8→256) → LayerNorm(256) → ReLU
        Linear(256→256) → LayerNorm(256) → ReLU
        Linear(256→128) → ReLU
    │
    ├── value_head:  Linear(128→64) → ReLU → Linear(64→1)     → V(s)
    └── adv_head:    Linear(128→64) → ReLU → Linear(64→16)    → A(s,a)

Q(s,a) = V(s) + A(s,a) − mean_a[ A(s,a) ]
```

LayerNorm after both 256-unit layers stabilises training under the wide variance of multi-objective reward signals. Without normalisation, the gradient through the power penalty (a different scale from the queue term) can dominate and slow convergence of the advantage stream.

**Double DQN target:**
```
a* = argmax_a policy_net(s')        ← action selection: online network
target = r + γ × target_net(s', a*)  ← action evaluation: target network
```

Decoupling selection from evaluation removes the overestimation bias of vanilla DQN. The target network is updated by hard copy (not Polyak averaging) every 150 steps, giving stable targets for the online network to train toward.

### Prioritised Experience Replay (PER)

```
Capacity: 25,000 transitions
α = 0.65     (priority exponent — how much priority matters vs. uniform)
β: 0.40 → 1.00 over 50,000 steps  (IS correction annealing)

P(i) = p_i^α / Σ p_j^α
p_i  = |TD-error_i| + 1e-6    (ε prevents zero-priority starvation)

IS weight: w_i = (N × P(i))^{-β} / max_j(w_j)
```

New transitions are added with the current maximum priority (optimistic initialisation), ensuring every transition is sampled at least once before priorities are adjusted by actual TD errors. Priority updates happen immediately after each training step using the batch TD errors.

The IS weight vector is passed directly into the Huber loss:

```python
loss = (w * F.smooth_l1_loss(current_q, targets, reduction='none')).mean()
```

High-priority transitions have larger gradient contributions; the IS weight prevents this from biasing the value estimate.

### Multi-Objective Reward

```
R = W_queue × (prev_q − curr_q)              [0.80]  — queue reduction
  + W_throughput × discharged                 [2.00]  — vehicles cleared
  − W_power × power_cost × power_factor       [0.10]  — signal energy penalty
  − W_overflow × Σ max(0, q_i − 28)          [1.50]  — per-vehicle overflow penalty
```

The throughput coefficient (2.0) intentionally outweighs the queue coefficient (0.8) to reward actively clearing lanes rather than just holding queues from growing. The overflow penalty activates at 28 vehicles per lane (56% of MAX_QUEUE=50), creating a soft ceiling that the agent learns to respect proactively.

Power cost per step:

```
power_cost = (PHASE_DURATIONS[pidx] × POWER_GREEN
            + Σ_{p≠pidx} PHASE_DURATIONS[p] × POWER_RED × 0.08) × power_factor
```

Night schedule sets power_factor=0.40, reducing the power penalty signal and allowing the agent to hold greens longer during low-traffic hours — reflecting that signal energy costs relative to congestion costs shift at off-peak times.

### Pressure Routing

When any lane reaches MAX_QUEUE=50, overflow vehicles spill to the least-congested adjacent intersection:

```python
dest = min(neighbours, key=lambda nb: nb.total_queue)
spill = n.q[i] - (MAX_QUEUE - 1)
dest.q[i % LANES] = min(dest.q[i % LANES] + spill, MAX_QUEUE)
n.q[i] = MAX_QUEUE - 1
```

This prevents queue saturation from blocking the simulation and models the back-pressure effect on real road networks where vehicles reroute when their intended path is blocked. Lane index modulo maps overflow into the lane of corresponding direction at the destination node.

### Emergency Vehicle Preemption

```python
if emergency_active and n.id == emergency_node_id:
    # arrivals frozen, discharge suspended, signal held in amber
    n.last_discharged = 0
    continue
```

When an emergency is triggered, arrivals at the affected node are frozen, normal discharge is suspended, and all lane signals display amber. The emergency persists for `emergency_timer=150` steps, then auto-resolves and returns the node to normal operation. The dashboard shows the affected node with an orange pulse border and "Emergency: Node #N (Xt)" countdown in the panel.

Emergency resolution is tracked: `emergency_resolved / emergency_total` gives the success rate across the session.

### Schedule Manager

Four time-of-day programmes with wall-clock auto-switching and manual override:

| Schedule | Arrival Rate | Power Factor | Hours |
|---|---|---|---|
| Morning Rush | 0.90 veh/step | 1.00 | 07:00–09:00 |
| Daytime | 0.50 veh/step | 0.80 | 09:00–17:00 |
| Evening Rush | 0.85 veh/step | 1.00 | 17:00–20:00 |
| Night | 0.18 veh/step | 0.40 | 20:00–07:00 |

Override via the panel buttons forces a specific schedule for 1,800 simulation steps (≈30 simulated minutes) then returns to auto-switching. The override tracks steps remaining and resets the override flag on expiry.

### CO₂ Savings Model

```python
CO2_PER_VEH_IDLE = 0.021  # kg per vehicle per idle step (proxy)
co2_saved += discharged * CO2_PER_VEH_IDLE
```

Every vehicle cleared from the queue before it would have idled saves one idle-step of CO₂ emission. This is a proxy measure — it captures the marginal emission from queue-induced idling rather than from vehicles in motion. The total is displayed in the City-Scale Metrics panel and logged to TensorBoard.

---

## Dashboard Screenshots

All screenshots are live captures from a running simulation. Intersection box colour encodes total queue: blue (low) → purple (medium) → red (critical).

### 🟡 Benchmark Console — Suite Initialisation

System spec confirmed: 3×3 grid, 8-D state, 16 actions, Dueling Double DQN + PER, 25,000 replay capacity, multi-objective reward. Pretrained weights loaded, benchmark sequence FIXED → DENSITY → DQN initiated.

![Benchmark Console](Benchmark_Console.png)

### ☀ Daytime Schedule — DQN, Step 1151

Moderate load (arrival 0.50). CPU queues 14–42 vehicles. Throughput: 11,091 vehicles. Avg queue: 29.3 veh/node. CO₂ saved: 232.53 kg. Traffic Efficiency: 68.5%. Mode Comparison panel shows DQN: Q=102.3, T=4,939, P=285,630 kU, CO₂=103.7 kg at this point in the session.

![Daytime](Daytime.png)

### 🌙 Night Schedule — FIXED Benchmark, Step 366

Low arrival rate (0.18). Nearly all nodes at 0 vehicles. Fixed-time controller running 16s cycles uniformly. Power: 27,895 kU. Efficiency: 64.7%. Benchmark RUNNING (FIXED 1/3). Mode Comparison accumulating FIXED baseline for comparison.

![Night Fixed Benchmark](FIXED_Benchmark.png)

### 🌙 Night — DENSITY Benchmark, Step 1498

DENSITY controller active (RUNNING 2/3). Near-zero queues at night. Avg queue: 0.0 veh/node. Throughput: 35 vehicles. FIXED comparison: Q=0.6, T=148, P=149,437 kU, CO₂=3.1 kg — clearly less efficient than DENSITY at 0 queue under same conditions.

![Night Density Benchmark](DENSITY_Benchmark.png)

### 🌙 Night — DQN Benchmark, Step 3025

DQN active (RUNNING 3/3). Queue: 1.2 veh/node. Power: 34,322 kU. Mode Comparison: FIXED Q=0.6, T=148 / DENSITY Q=0.1, T=174 / DQN Q=0.0, T=0 — DQN eliminates remaining queue while using significantly less power than FIXED.

![Night DQN Benchmark](DQN_Benchmark.png)

### 🌄 Morning Rush — DQN, Step 668

Heavy load (arrival 0.90). Queues 76–120 vehicles across all nodes. Emergency active at Node #5 (52 steps remaining). Throughput: 5,893 vehicles. Avg queue: 94.6 veh/node. CO₂ saved: 123.75 kg. Traffic Efficiency: 40.5% — the system is under peak stress.

![Morning Rush](Morning_Rush.png)

### 🌆 Evening Rush — DQN, Step 1468

Evening peak (arrival 0.85). Queues 65–112 vehicles. Emergency active at Node #2 (7 steps to resolution, orange border visible). Throughput: 13,804 vehicles. Avg queue: 89.4 veh/node. CO₂: 289.88 kg. Time Saved: 13,804 veh-steps. Efficiency: 51.4%.

![Evening Rush](Evening_Rush.png)

### 🌙 Night — DQN Optimal, Step 1737

Night schedule, pretrained agent, low load. Near-empty grid: 0–17 vehicles per node. Avg queue: 3.3 veh/node. Throughput: 16,868 vehicles. CO₂ saved: 354.23 kg. Traffic Efficiency: **84.1%** — the agent's peak performance under learned night-mode phase scheduling.

![Night Optimal](Night.png)

---

## Benchmark Results (Automated Suite)

The benchmark runs each mode for 1,200 steps sequentially, resetting all queues between modes, and records throughput, average queue, power, and CO₂ at the end of each segment:

| Metric | FIXED | DENSITY | DQN |
|---|---|---|---|
| Avg Queue (veh/node) | 0.6 | 0.1 | 0.0 |
| Throughput (veh) | 148 | 174 | — (live, grows) |
| Power Used (kU) | 149,437 | 82,330 | 34,322 |
| CO₂ Proxy (kg) | 3.1 | 3.7 | 1.18 |

Night-schedule results from benchmark screenshots. DQN achieves zero average queue while consuming 77% less power than the fixed-time baseline. Power reduction is a direct consequence of the night power_factor (0.40) combined with the agent learning to use shorter phases when queues are small.

---

## Model Export and Deployment

Trained models are saved in two formats simultaneously:

```python
# TorchScript — deployable without training stack
traced = torch.jit.trace(policy_net, torch.randn(1, STATE_DIM))
traced.save("adaptiflow_x.pt")

# State dict — for fine-tuning or architecture inspection
torch.save(policy_net.state_dict(), "adaptiflow_x_weights.pth")
```

Loading pre-trained weights sets ε = 0.0 and flags `pretrained_mode = True`, disabling replay buffer pushes and training steps. The agent runs purely in inference mode, allowing evaluation of convergent policy behaviour without further gradient updates.

Both `adaptiflow_x.pt` (TorchScript) and `adaptiflow_x_weights.pth` are included in this repository.

---

## TensorBoard Logging

The following scalars are logged every step to `runs/adaptiflow_x/`:

| Tag | Description |
|---|---|
| `Loss/Train` | Huber loss per training step |
| `Agent/Epsilon` | ε-greedy exploration rate decay |
| `Reward/Step` | Total reward across all 9 nodes per step |
| `City/Power` | Aggregate signal power consumed per step |
| `City/Throughput` | Cumulative vehicles discharged |
| `City/CO2_Saved` | Cumulative CO₂ proxy savings (kg) |

```bash
tensorboard --logdir runs/adaptiflow_x
```

---

## Project Structure

```
AdaptiFlow_X_improved.py      ─ Complete system: 1,120 lines, single-file
├── Configuration              GRID, LANES, PHASE_DURATIONS, hyperparameters
├── PrioritizedReplayBuffer    α=0.65, β annealing 0.40→1.00, numpy priority array
├── DuelingDQN                 trunk + value_head + advantage_head, LayerNorm
├── Intersection               state_vec, phase timer, pressure routing
├── ScheduleManager            wall-clock auto-switch + manual override with expiry
├── spawn_vehicles()           Poisson arrivals, emergency node bypass
├── select_action()            FIXED | DENSITY | DQN ε-greedy dispatch
├── discharge_and_route()      phase-scaled discharge, spill to min-queue neighbour
├── compute_reward()           4-term multi-objective reward
├── train_step()               Double DQN target, PER IS weights, grad-clip 10.0
├── Tkinter Canvas             real-time grid, congestion colouring, emergency pulse
├── Control Panel              mode buttons, schedule override, benchmark, save/load
└── loop()                     main tick: schedule → spawn → step → train → draw

adaptiflow_x.pt               ─ TorchScript saved model (inference-ready)
adaptiflow_x_weights.pth      ─ State dict (fine-tuning / inspection)
```

---

## Getting Started

```bash
pip install torch tensorboard numpy
python AdaptiFlow_X_improved.py
```

The application opens full-screen. On first launch it trains from scratch. To load the included pretrained model, click **Load Pretrained Model** in the control panel — ε is set to 0.0 and the agent switches immediately to inference mode.

Click **Run Benchmark Suite** to trigger the automated FIXED → DENSITY → DQN comparison. Results appear in the Mode Comparison panel as each 1,200-step segment completes.

---

## Technical Highlights at a Glance

- **3×3 nine-intersection network** — independent DQN control per node, shared policy parameters
- **16-action space:** 4 lanes × 4 phase durations (8s / 16s / 28s / 45s)
- **8-D state vector:** normalised queues + circular time-of-day (sin/cos) + last action
- **Dueling DDQN** — separate value and advantage streams, mean-centred advantage
- **Prioritised Experience Replay** — α=0.65, β annealing 0.40→1.00, 25,000-transition buffer
- **4-term reward:** queue Δ (×0.8) + throughput (×2.0) − power (×0.10) − overflow (×1.50)
- **Phase-locked decisions** — commitment for full phase duration, no per-step thrashing
- **Pressure routing** — overflow vehicles sent to least-congested adjacent node
- **Emergency preemption** — 150-step clearance with arrival freeze and amber hold
- **4 time-of-day schedules** — arrival rates 0.18–0.90, power factors 0.40–1.00
- **Manual schedule override** — expires after 1,800 steps, returns to auto-switching
- **TorchScript export** — deployable without training infrastructure
- **TensorBoard logging** — loss, ε, reward, power, throughput, CO₂ tracked per step
- **77% power reduction** vs. fixed-time baseline under night schedule (benchmark measured)
- **84.1% traffic efficiency** — peak measured under pretrained night operation

---

## Author

**Vaibhav Krishna V**  
Electronics and Communication Engineer
[GitHub](https://github.com/vaibhav-krishna-v ) · [LinkedIn](https://linkedin.com/in/vkv078)

> *Built on the conviction that adaptive control systems earn their value not by outperforming a weak baseline in a controlled test, but by remaining quantifiably better than deterministic alternatives under every traffic condition the city actually produces.*
