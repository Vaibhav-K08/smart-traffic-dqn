# 🚦 Smart Traffic Signal Control — DQN Reinforcement Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![RL](https://img.shields.io/badge/Reinforcement_Learning-DQN-orange?style=for-the-badge)
![Tkinter](https://img.shields.io/badge/Tkinter-GUI-blue?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*DQN reduced average queue length from 26.7 vehicles (fixed timing) to 0.8 — a 97% reduction.*

</div>

---

## 📌 Project Overview

A multi-mode intelligent traffic signal control system that implements and **directly compares** three control strategies at a simulated multi-lane intersection:

| Mode | Strategy | Avg Queue |
|---|---|---|
| Fixed Time | Equal green duration per lane | **26.7 vehicles** |
| Density Adaptive | Green time proportional to lane density | **21.6 vehicles** |
| DQN (RL) | Learned optimal policy from environment | **0.8 vehicles** ✅ |

The DQN agent achieves near-zero congestion **without suppressing throughput** — vehicles per cycle actually increased from 8.07 (Fixed) to 8.20 (DQN).

---

## 🏗️ System Architecture

```
Traffic Simulation (Multi-lane intersection)
        ↓
  State Extraction    ← Vehicle density per lane
        ↓
  ┌─────────────────────────────────┐
  │  Control Mode Selector          │
  │  ├── Fixed Time                 │
  │  ├── Density Adaptive           │
  │  └── DQN Agent (trained RL)     │
  └─────────────────────────────────┘
        ↓
  Signal Timing Action
        ↓
  Traffic Flow Update
        ↓
  Performance Logging + Visualization
```

---

## 📊 Key Results

### Congestion Reduction
| Strategy | Avg Queue Length | Improvement |
|---|---|---|
| Fixed Timing | 26.7 vehicles | Baseline |
| Density Adaptive | 21.6 vehicles | -19% |
| **DQN** | **0.8 vehicles** | **-97%** ✅ |

### Throughput (Vehicles per Cycle)
| Strategy | Vehicles/Cycle |
|---|---|
| Fixed Timing | 8.07 |
| Density Adaptive | 8.11 |
| **DQN** | **8.20** |

### DQN Training Stats
- Step Reward: 6.00
- Episode Reward: 3505.00
- Average Reward: -4101.00 → converging
- **Success Rate: 100%**
- Final Epsilon: 0.050

---

## ✅ Key Features

- 🔴 **3 control modes** in one unified simulation environment
- 🧠 **DQN agent** learns optimal signal policy through environment interaction
- 📊 **Tkinter GUI** with live traffic visualization and signal state display
- 📉 **Performance logger** records density, duration, and decisions per mode
- 🏙️ Directly applicable to real urban intersection control

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `NumPy` | Traffic density simulation & state computation |
| `Tkinter` | Real-time GUI visualization |
| `collections.deque` | Replay memory buffer for DQN |
| `random` | ε-greedy action sampling |

> Note: DQN is implemented from scratch using NumPy — no external RL library required.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/smart-traffic-dqn.git
cd smart-traffic-dqn
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the simulation
```bash
# Run all 3 modes and compare
python traffic_control.py

# Run specific mode: fixed | density | dqn
python traffic_control.py --mode dqn
```

---

## 📁 Project Structure

```
smart-traffic-dqn/
├── traffic_control.py      # Main simulation — run this
├── dqn_agent.py            # Deep Q-Network agent (from scratch)
├── traffic_env.py          # Intersection environment & simulation
├── fixed_controller.py     # Fixed-time signal control
├── density_controller.py   # Density-adaptive signal control
├── visualizer.py           # Tkinter real-time GUI
├── performance_logger.py   # Metrics recording & comparison
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🏙️ Applications

**Industrial:**
- Intelligent traffic signal control for smart cities
- Simulation platforms for transportation system research
- Reinforcement learning applications in real-time control systems
- Adaptive signal optimization for urban mobility infrastructure

**Societal:**
- Reduces traffic congestion at busy intersections
- Minimizes vehicle waiting time and travel delays
- Reduces fuel consumption and emissions
- Contributes to smarter urban mobility planning

---

## 👤 Author

**Vaibhav Krishna V**  
Electronics & Communication Engineering, NMIT Bengaluru  
USN: 1NT22EC182  
📧 vaibhavkv078@gmail.com

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
