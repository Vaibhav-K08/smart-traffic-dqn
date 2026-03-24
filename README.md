# 🚦 Multi-Intersection Deep RL Traffic Signal Controller

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A 3×3 grid of intersections controlled by a live training DQN agent, with TensorBoard logging, congestion color mapping, and a separate analytics pipeline that generates publication quality comparison plots.*

</div>

---

## What This Does

Nine intersections arranged in a 3×3 grid. Each one has four lanes with Poisson-distributed vehicle arrivals. Three control modes run in the same codebase: Fixed, Density, and DQN where you switch between them by changing one line. The DQN agent trains live during the simulation, updates a target network every 100 steps, and logs everything to TensorBoard. A second script reads those logs and produces bar charts and a radar plot comparing all three modes.

The trained policy network gets exported as a TorchScript traced model (`traffic_dqn.pt`) at the end of each run.

---

## How It Works

```
9 Intersections (3×3 grid), 4 lanes each
        ↓
  Poisson Arrivals (rate=0.6 per lane per step)
        ↓
  Control Mode (switch via MODE flag)
  ├── FIXED   → rotates green signal every second
  ├── DENSITY → always serves the most congested lane
  └── DQN     → policy network selects action, trains live
        ↓
  Discharge selected lane (2–6 vehicles per step)
        ↓
  Reward = previous total queue − current total queue
        ↓
  TensorBoard logs — loss, queue, throughput, epsilon,
                     episode reward, success rate
        ↓
  Tkinter canvas — live color coded intersection grid
  └── Export policy → traffic_dqn.pt (TorchScript)
```

---

## The DQN Agent

A fully connected policy network trained with experience replay and a separate target network.

```
Input:  lane queue counts (4 values, normalized by 10)
Hidden: Linear(4→128) → ReLU → Linear(128→128) → ReLU
Output: Q-values for all 4 lane actions
```

**Training setup:**

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.001) |
| Gamma | 0.95 |
| Replay memory | 8000 transitions |
| Batch size | 128 |
| Target network update | every 100 steps |
| Epsilon start | 1.0 |
| Epsilon min | 0.05 |
| Epsilon decay | 0.995 per step |
| Episode length | 500 steps |

Success is counted when average queue across all intersections drops below 15 vehicles at the end of an episode.

---

## Visualization

**Tkinter grid** — each intersection is drawn as a colored square. Color shifts from blue to red as congestion increases (0% to 100% of a normalized scale). Four signal dots per intersection show which lane is currently green. Total queue count is displayed at the center of each node.

**Metrics overlay** — live labels in the top left show Step Reward, Episode Reward, Average Reward (last 20 episodes), Success Rate, and current Epsilon.

---

## Analytics Pipeline

`elite_plots.py` reads the TensorBoard CSV exports and generates four plots:

| Plot | What It Shows |
|---|---|
| `elite_queue.png` | Average queue length per mode (bar chart) |
| `elite_throughput.png` | Vehicles per cycle per mode (bar chart) |
| `elite_radar.png` | Normalized radar across throughput, queue, success rate |
| `elite_summary.png` | Plain text summary card identifying best performing mode |

To generate the plots after a run:

```bash
# Export CSVs from TensorBoard first, organized by mode folder
python elite_plots.py
```

Expected folder structure:
```
FIXED/
  Average Queue.csv
  Throughput.csv
  Success Rate.csv
DENSITY/
  ...
DQN/
  ...
elite_plots.py
```

---

## Results

| Mode | Avg Queue | Throughput |
|---|---|---|
| Fixed | 26.7 vehicles | 8.07 veh/cycle |
| Density | 21.6 vehicles | 8.11 veh/cycle |
| DQN | **0.8 vehicles** | **8.20 veh/cycle** |

DQN eliminates congestion almost entirely while throughput stays equal. The agent learns to serve lanes efficiently rather than just rotating or picking the longest queue.

---

## Features

- 3×3 multi-intersection grid, all nodes simulated simultaneously
- Three control modes in one codebase, switchable with one variable
- Live training DQN with replay memory and target network
- TensorBoard logging — loss, queue, throughput, epsilon, rewards, success rate
- Tkinter canvas with congestion color mapping per intersection
- Separate analytics script generates publication-quality comparison plots
- Policy exported as TorchScript traced model at end of run

---

## Tech Stack

| Library | Purpose |
|---|---|
| PyTorch | DQN policy and target network, training loop |
| TensorBoard | Logging all training and simulation metrics |
| Tkinter | Live intersection grid visualization |
| NumPy | Queue state arrays and metric aggregation |
| Matplotlib + Pandas | Comparison plots in elite_plots.py |

---

## Running It

```bash
git clone https://github.com/YOUR_USERNAME/smart-traffic-dqn.git
cd smart-traffic-dqn
pip install -r requirements.txt

# Change MODE in traffic_control.py to FIXED | DENSITY | DQN
python traffic_control.py
```

To view TensorBoard logs:
```bash
tensorboard --logdir=runs/tokyo_rl
```

---

## Project Structure

```
smart-traffic-dqn/
├── traffic_control.py      # Main simulation — run this
├── elite_plots.py          # Analytics and comparison plots
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Industrial Applications

- Intelligent traffic signal control for smart cities
- Simulation platforms for transportation research
- Reinforcement learning in real time control systems
- Adaptive signal optimization for urban mobility infrastructure
- Transferable multi-mode control architecture for autonomous systems

## Societal Applications

- Reduces congestion at busy intersections
- Minimizes vehicle waiting time and travel delays
- Reduces fuel consumption and emissions
- Supports development of intelligent transportation systems
- Contributes to smarter urban mobility planning

---

## Author

**Vaibhav Krishna V**  
Electronics & Communication Engineer  
📧 vaibhavkv078@gmail.com

---

## License

MIT — see [LICENSE](LICENSE) for details.
