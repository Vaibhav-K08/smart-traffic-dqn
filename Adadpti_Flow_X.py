"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          AdaptiFlow-X: Adaptive Multi-Objective Deep RL                      ║
║               Urban Traffic Signal Intelligence System                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Author      : Vaibhav Krishna V                                             ║
║  Version     : 2.0                                                           ║
║  License     : Original work — All rights reserved                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  WHAT THIS SYSTEM SOLVES                                                     ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  1. Traffic Congestion    — Density-aware DQN minimises vehicle queue build-up║
║  2. Time Saving           — Adaptive phase durations cut average wait time   ║
║  3. Power Saving          — Night/off-peak modes cut signal power by ~60 %   ║
║  4. Dynamic Scheduling    — Time-of-day programmes auto-switch; manual       ║
║                              override for events or construction             ║
║  5. Emergency Preemption  — Priority corridor cleared in real time           ║
║  6. Carbon Reduction      — Idle-queue CO₂ proxy tracked and minimised       ║
║  7. Pressure Routing      — Overflow vehicles re-routed to adjacent nodes    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ALGORITHMS                                                                  ║
║  ─────────────────────────────────────────────────────────────────────────   ║
║  • Dueling Double DQN  — Separate value / advantage streams                  ║
║  • Prioritised Experience Replay (PER)  — TD-error-weighted sampling         ║
║  • Multi-Objective Reward  — Queue + Throughput + Power + Overflow           ║
║  • Adaptive Phase Duration — Action space: lane × green-time (16 actions)    ║
║  • Time-of-Day Schedule Engine with live manual override                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tkinter as tk
from tkinter import ttk
import random, time, math, os
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# ════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════
GRID         = 3          # NxN intersections (9 total)
LANES        = 4          # N, E, S, W

# Adaptive phase durations in simulation steps (1 step ≈ 1 sim-second)
PHASE_DURATIONS = [8, 16, 28, 45]
N_PHASES     = len(PHASE_DURATIONS)

# Action space: lane × phase_duration → 16 discrete actions
ACTION_DIM   = LANES * N_PHASES

# State vector per intersection
#   [q_N, q_E, q_S, q_W, tod_sin, tod_cos, last_lane, last_pidx]
STATE_DIM    = LANES + 2 + 1 + 1   # = 8

# Initial control mode
MODE         = "DQN"      # "FIXED" | "DENSITY" | "DQN"

# ── RL hyper-parameters ──────────────────────────────────────
GAMMA        = 0.97
LR           = 5e-4
EPS          = 1.0
EPS_MIN      = 0.05
EPS_DECAY    = 0.9975
BATCH        = 128
MEM_CAP      = 25000
TARGET_FREQ  = 150
EPISODE_LEN  = 600

# ── Multi-objective reward weights ───────────────────────────
W_QUEUE      = 0.8    # queue reduction bonus
W_THROUGHPUT = 2.0    # vehicles discharged bonus
W_POWER      = 0.10   # power penalty scale
W_OVERFLOW   = 1.5    # hard penalty per vehicle over limit
OVERFLOW_THRESH = 28  # vehicles — above this, penalty fires

# ── Traffic model ────────────────────────────────────────────
MAX_QUEUE    = 50
DISCHARGE_MIN = 2
DISCHARGE_MAX = 9

# ── Power model (arbitrary units, relative) ──────────────────
POWER_GREEN  = 2.0    # per step while green
POWER_RED    = 0.4    # per step while red/amber
CO2_PER_VEH_IDLE = 0.021  # kg CO₂ per vehicle per idle step (proxy)

# ── Time-of-day programmes ───────────────────────────────────
#   Each defines vehicle arrival rate and signal power scaling
SCHEDULES = {
    "Morning Rush": {"arrival": 0.90, "power_factor": 1.0,  "start": 7,  "end": 9},
    "Daytime":      {"arrival": 0.50, "power_factor": 0.80, "start": 9,  "end": 17},
    "Evening Rush": {"arrival": 0.85, "power_factor": 1.0,  "start": 17, "end": 20},
    "Night":        {"arrival": 0.18, "power_factor": 0.40, "start": 20, "end": 7},
}

# ── GUI dimensions ───────────────────────────────────────────
PANEL_W      = 360
SIM_SPEED    = 28
BG_DARK      = "#0a0a18"
PANEL_BG     = "#101028"
SUCCESS_Q_THRESH = 13


# ════════════════════════════════════════════════════════════
#  GLOBAL RUNTIME METRICS
# ════════════════════════════════════════════════════════════
step              = 0
episode           = 0
episode_reward    = 0.0
reward_history    = []
throughput_total  = 0
power_consumed    = 0.0
co2_saved         = 0.0       # kg CO₂ saved vs. no-control baseline
time_saved_veh    = 0         # cumulative vehicle-steps saved
success_count     = 0
avg_reward        = 0.0
success_rate      = 0.0
active_schedule   = "Daytime"
current_arrival   = 0.50

benchmark_stats = {
    "FIXED": {
        "throughput": 0,
        "avg_q": 0.0,
        "power": 0.0,
        "co2": 0.0
    },
    "DENSITY": {
        "throughput": 0,
        "avg_q": 0.0,
        "power": 0.0,
        "co2": 0.0
    },
    "DQN": {
        "throughput": 0,
        "avg_q": 0.0,
        "power": 0.0,
        "co2": 0.0
    },
}
benchmark_mode = False
benchmark_sequence = ["FIXED", "DENSITY", "DQN"]
benchmark_index = 0
benchmark_timer = 0
BENCHMARK_DURATION = 1200

emergency_active  = False
emergency_node_id = -1
emergency_timer   = 0
pretrained_mode = False
emergency_total = 0
emergency_resolved = 0
writer = SummaryWriter("runs/adaptiflow_x")


# ════════════════════════════════════════════════════════════
#  PRIORITISED EXPERIENCE REPLAY
# ════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer:
    """Sum-tree-free PER with numpy priority array."""

    def __init__(self, capacity, alpha=0.65, beta_start=0.40, beta_end=1.0, beta_steps=50000):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self._step      = 0
        self.buffer     = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos        = 0

    @property
    def beta(self):
        fraction = min(1.0, self._step / self.beta_steps)
        return self.beta_start + fraction * (self.beta_end - self.beta_start)

    def push(self, *transition):
        max_p = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        self._step += 1
        n     = len(self.buffer)
        probs = self.priorities[:n] ** self.alpha
        probs /= probs.sum()
        idxs  = np.random.choice(n, batch_size, replace=False, p=probs)
        w     = (n * probs[idxs]) ** (-self.beta)
        w    /= w.max()
        return [self.buffer[i] for i in idxs], idxs, torch.FloatTensor(w)

    def update_priorities(self, idxs, td_errors):
        for i, e in zip(idxs, td_errors):
            self.priorities[i] = float(abs(e)) + 1e-6

    def __len__(self):
        return len(self.buffer)


# ════════════════════════════════════════════════════════════
#  DUELING DOUBLE DQN
# ════════════════════════════════════════════════════════════
class DuelingDQN(nn.Module):
    """
    Dueling architecture (Wang et al., 2016):
        Q(s,a) = V(s) + A(s,a) − mean(A(s,·))
    Trained with Double DQN target (van Hasselt et al., 2016).
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.adv_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))


policy_net = DuelingDQN(STATE_DIM, ACTION_DIM)
target_net = DuelingDQN(STATE_DIM, ACTION_DIM)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer     = optim.Adam(policy_net.parameters(), lr=LR, eps=1e-7)
replay_buffer = PrioritizedReplayBuffer(MEM_CAP)


# ════════════════════════════════════════════════════════════
#  INTERSECTION MODEL
# ════════════════════════════════════════════════════════════
class Intersection:
    """
    Represents one signalised intersection in the GRID.

    Attributes
    ----------
    q              : vehicle queue per lane (N, E, S, W)
    action         : currently active green lane
    phase_idx      : index into PHASE_DURATIONS for active green time
    phase_remaining: steps left in current phase before re-decision
    """
    def __init__(self, row: int, col: int):
        self.r, self.c = row, col
        self.id        = row * GRID + col
        self.q         = [0] * LANES
        self.action    = 0
        self.phase_idx = 1
        self.phase_remaining = PHASE_DURATIONS[1]
        self.last_discharged = 0

    @property
    def total_queue(self) -> int:
        return sum(self.q)

    @property
    def state_vec(self) -> np.ndarray:
        q_norm   = [q / MAX_QUEUE for q in self.q]
        hour = (step / 600) % 24
        tod_sin  = math.sin(2 * math.pi * hour / 24.0)
        tod_cos  = math.cos(2 * math.pi * hour / 24.0)
        return np.array(
            q_norm + [tod_sin, tod_cos,
                      self.action  / (LANES - 1),
                      self.phase_idx / (N_PHASES - 1)],
            dtype=np.float32
        )


nodes = [Intersection(r, c) for r in range(GRID) for c in range(GRID)]


def get_neighbors(n: Intersection):
    nbrs = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = n.r + dr, n.c + dc
        if 0 <= nr < GRID and 0 <= nc < GRID:
            nbrs.append(nodes[nr * GRID + nc])
    return nbrs


# ════════════════════════════════════════════════════════════
#  SCHEDULE MANAGER
# ════════════════════════════════════════════════════════════
class ScheduleManager:
    """
    Auto-selects the active time-of-day schedule based on wall-clock hour.
    Supports a manual override (e.g. for events, construction) with an
    automatic expiry after `override_steps` simulation steps.
    """
    def __init__(self):
        self._override       = None
        self._override_steps = 0

    def get_active(self) -> str:
        if self._override and self._override_steps > 0:
            return self._override
        h = int((step / 600) % 24)
        for name, cfg in SCHEDULES.items():
            s, e = cfg["start"], cfg["end"]
            in_range = (s <= h < e) if s < e else (h >= s or h < e)
            if in_range:
                return name
        return "Daytime"

    def set_override(self, name: str, steps: int = 1800):
        self._override       = name
        self._override_steps = steps

    def tick(self):
        if self._override_steps > 0:
            self._override_steps -= 1
            if self._override_steps == 0:
                self._override = None

    @property
    def override_remaining(self) -> int:
        return self._override_steps

    @property
    def is_overridden(self) -> bool:
        return self._override is not None and self._override_steps > 0


schedule_mgr = ScheduleManager()


# ════════════════════════════════════════════════════════════
#  TRAFFIC GENERATION
# ════════════════════════════════════════════════════════════
def poisson_sample(rate: float) -> int:
    if rate <= 0:
        return 0
    return max(0, int(random.expovariate(1.0 / rate)))


def spawn_vehicles():
    global current_arrival
    cfg = SCHEDULES[active_schedule]
    current_arrival = cfg["arrival"]
    for n in nodes:
        if emergency_active and n.id == emergency_node_id:
            continue   # freeze arrivals at emergency node
        for lane in range(LANES):
            arrivals = poisson_sample(current_arrival)
            n.q[lane] = min(n.q[lane] + arrivals, MAX_QUEUE)


# ════════════════════════════════════════════════════════════
#  ACTION SELECTION
# ════════════════════════════════════════════════════════════
def decode_action(a: int):
    return a % LANES, a // LANES

def encode_action(lane: int, pidx: int) -> int:
    return lane + pidx * LANES

def select_action(n: Intersection):
    if MODE == "FIXED":
        lane = (step // 30) % LANES
        return lane, 1

    if MODE == "DENSITY":
        lane = int(np.argmax(n.q))
        tq   = n.total_queue
        pidx = 3 if tq > 22 else 2 if tq > 12 else 1 if tq > 5 else 0
        return lane, pidx

    # DQN with ε-greedy
    if random.random() < EPS:
        return decode_action(random.randint(0, ACTION_DIM - 1))
    with torch.no_grad():
        sv = torch.FloatTensor(n.state_vec).unsqueeze(0)
        a  = policy_net(sv).argmax().item()
    return decode_action(a)


# ════════════════════════════════════════════════════════════
#  DISCHARGE + PRESSURE ROUTING
# ════════════════════════════════════════════════════════════
def discharge_and_route(n: Intersection, lane: int, pidx: int) -> int:
    """
    Discharge vehicles from the active green lane.
    Scale discharge rate by phase duration (longer green → more cleared).
    Route overflow vehicles to the least-congested adjacent intersection.
    """
    global throughput_total, time_saved_veh
    scale      = (pidx + 1) / N_PHASES
    discharged = min(n.q[lane],
                     int(random.randint(DISCHARGE_MIN, DISCHARGE_MAX) * scale))
    n.q[lane] -= discharged
    n.last_discharged = discharged
    throughput_total  += discharged
    time_saved_veh    += discharged   # each discharged vehicle ≈ 1 wait-step saved

    # Pressure routing: spill overflow to neighbour
    for i in range(LANES):
        if n.q[i] >= MAX_QUEUE:
            neighbours = get_neighbors(n)
            if neighbours:
                dest = min(neighbours, key=lambda nb: nb.total_queue)
                spill = n.q[i] - (MAX_QUEUE - 1)
                dest.q[i % LANES] = min(dest.q[i % LANES] + spill, MAX_QUEUE)
                n.q[i] = MAX_QUEUE - 1
    return discharged


# ════════════════════════════════════════════════════════════
#  MULTI-OBJECTIVE REWARD
# ════════════════════════════════════════════════════════════
def compute_reward(prev_q: int, n: Intersection,
                   pidx: int, discharged: int) -> tuple[float, float]:
    """
    R = W_queue × ΔQueue
      + W_throughput × discharged
      - W_power × power_cost
      - W_overflow × Σ max(0, q_i − threshold)

    Returns (reward, power_cost_this_step)
    """
    curr_q          = n.total_queue
    queue_term      = (prev_q - curr_q) * W_QUEUE
    throughput_term = discharged * W_THROUGHPUT
    pf              = SCHEDULES[active_schedule]["power_factor"]
    power_cost      = (PHASE_DURATIONS[pidx] * POWER_GREEN
                       + sum(PHASE_DURATIONS[p] for p in range(N_PHASES) if p != pidx)
                       * POWER_RED * 0.08) * pf
    power_term      = -power_cost * W_POWER
    overflow_pen    = sum(max(0, q - OVERFLOW_THRESH) for q in n.q) * W_OVERFLOW
    return queue_term + throughput_term + power_term - overflow_pen, power_cost


# ════════════════════════════════════════════════════════════
#  DOUBLE DQN + PER TRAINING STEP
# ════════════════════════════════════════════════════════════
def train_step() -> float:
    if len(replay_buffer) < BATCH:
        return 0.0

    batch, idxs, weights = replay_buffer.sample(BATCH)
    s, a, r, ns, _ = zip(*batch)

    s   = torch.FloatTensor(np.array(s))
    ns  = torch.FloatTensor(np.array(ns))
    a   = torch.LongTensor(a)
    r   = torch.FloatTensor(r)
    w   = weights

    # Double DQN target: policy selects action, target evaluates
    with torch.no_grad():
        next_a  = policy_net(ns).argmax(dim=1, keepdim=True)
        next_q  = target_net(ns).gather(1, next_a).squeeze()
        targets = r + GAMMA * next_q

    current_q = policy_net(s).gather(1, a.unsqueeze(1)).squeeze()
    td_errors = (targets - current_q).detach().cpu().numpy()
    replay_buffer.update_priorities(idxs, td_errors)

    loss = (w * F.smooth_l1_loss(current_q, targets, reduction='none')).mean()
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
    optimizer.step()
    return loss.item()


# ════════════════════════════════════════════════════════════
#  GUI — CANVAS + CONTROL PANEL
# ════════════════════════════════════════════════════════════
root = tk.Tk()
root.title("AdaptiFlow-X  |  Adaptive Traffic Intelligence  |  Vaibhav Krishna V")
root.configure(bg=BG_DARK)

root.state("zoomed")
root.resizable(True, True)

SCREEN_W = root.winfo_screenwidth()
SCREEN_H = root.winfo_screenheight()

W_SIM = SCREEN_W - PANEL_W
CELL = min(W_SIM, SCREEN_H) // (GRID + 1)

root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)

canvas = tk.Canvas(
    root,
    width=W_SIM,
    height=SCREEN_H,
    bg=BG_DARK,
    highlightthickness=0
)
canvas.grid(row=0, column=0, sticky="nsew")

panel_container = tk.Frame(root, bg=PANEL_BG, width=PANEL_W)
panel_container.grid(row=0, column=1, sticky="ns")
panel_container.grid_propagate(False)

panel_canvas = tk.Canvas(
    panel_container,
    bg=PANEL_BG,
    width=PANEL_W,
    highlightthickness=0,
    bd=0
)
panel_scroll = tk.Scrollbar(panel_container, orient="vertical", command=panel_canvas.yview)

panel = tk.Frame(panel_canvas, bg=PANEL_BG)

panel.bind(
    "<Configure>",
    lambda e: panel_canvas.configure(scrollregion=panel_canvas.bbox("all"))
)

panel_canvas.create_window((0, 0), window=panel, anchor="nw")
panel_canvas.configure(yscrollcommand=panel_scroll.set)

panel_canvas.pack(side="left", fill="both", expand=True)
panel_scroll.pack(side="right", fill="y")
def _on_mousewheel(event):
    panel_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

panel_canvas.bind_all("<MouseWheel>", _on_mousewheel)

# ── Panel helper ─────────────────────────────────────────────
def section_header(text):
    tk.Label(panel, text=text, font=("Consolas", 9, "bold"),
             fg="#6699ff", bg=PANEL_BG, anchor="w").pack(
        fill="x", padx=10, pady=(8, 1))
    ttk.Separator(panel, orient="horizontal").pack(fill="x", padx=10, pady=2)


def kpi_label(text, fg="#99aabb"):
    lbl = tk.Label(panel, text=text, font=("Consolas", 9),
                   fg=fg, bg=PANEL_BG, anchor="w")
    lbl.pack(fill="x", padx=14, pady=1)
    return lbl


def action_button(parent, text, cmd, fg="#aabbcc", bg="#181830"):
    tk.Button(parent, text=text, font=("Consolas", 9),
              bg=bg, fg=fg, activebackground="#282850",
              relief="flat", cursor="hand2", command=cmd
              ).pack(fill="x", pady=2)


# ── Header ───────────────────────────────────────────────────
tk.Label(panel, text="AdaptiFlow-X", font=("Consolas", 15, "bold"),
         fg="#88aaff", bg=PANEL_BG).pack(pady=(14, 0))
tk.Label(panel, text="Multi-Objective RL Traffic Intelligence",
         font=("Consolas", 8), fg="#445566", bg=PANEL_BG).pack()
tk.Label(panel, text="Vaibhav Krishna V ",
         font=("Consolas", 8), fg="#334455", bg=PANEL_BG).pack(pady=(0, 4))

# ── System status ────────────────────────────────────────────
section_header("SYSTEM")
lbl_mode     = kpi_label("Mode  : DQN")
lbl_sched    = kpi_label("Schedule : Daytime")
lbl_step     = kpi_label("Step  : 0")
lbl_episode  = kpi_label("Episode  : 0")
lbl_eps      = kpi_label("Epsilon  : 1.000")
lbl_memory   = kpi_label("Replay   : 0 / 25000")
lbl_agent = kpi_label("Agent    : Training")
lbl_benchmark = kpi_label("Benchmark: Idle")
# ── RL performance ───────────────────────────────────────────
section_header("LEARNING PERFORMANCE")
lbl_step_r   = kpi_label("Optimization Score : 0.00")
lbl_ep_r     = kpi_label("Episode Reward  : 0.00")
lbl_avg_r    = kpi_label("Avg Reward(20ep): 0.00")
lbl_success = kpi_label("Traffic Efficiency : 0.0 %")

# ── City metrics ─────────────────────────────────────────────
section_header("CITY-SCALE METRICS")
lbl_throughput = kpi_label("Throughput  : 0 veh")
lbl_avg_q      = kpi_label("Avg Queue   : 0.0 veh/node")
lbl_power      = kpi_label("Power Used  : 0.0 kU")
lbl_time_saved = kpi_label("Time Saved  : 0 veh-steps")
lbl_co2        = kpi_label("CO₂ Saved   : 0.00 kg")
lbl_arrival    = kpi_label("Arrival Rate: 0.50 veh/step")
lbl_emergency  = kpi_label("Emergency   : None", fg="#ff7766")

section_header("MODE COMPARISON")
lbl_fixed_cmp = kpi_label("FIXED   : --")
lbl_fixed_cmp2 = kpi_label("")

lbl_density_cmp = kpi_label("DENSITY : --")
lbl_density_cmp2 = kpi_label("")

lbl_dqn_cmp = kpi_label("DQN     : --")
lbl_dqn_cmp2 = kpi_label("")

# ── Schedule override ─────────────────────────────────────────
section_header("SCHEDULE OVERRIDE  (30 min)")
sched_frame = tk.Frame(panel, bg=PANEL_BG)
sched_frame.pack(fill="x", padx=10, pady=2)
SCHED_COLOURS = {
    "Morning Rush": ("#2a1a00", "#ffaa44"),
    "Daytime":      ("#001a10", "#44ffaa"),
    "Evening Rush": ("#1a0020", "#cc88ff"),
    "Night":        ("#001020", "#4488ff"),
}
for sname, (bg_c, fg_c) in SCHED_COLOURS.items():
    tk.Button(sched_frame, text=sname, font=("Consolas", 9),
              bg=bg_c, fg=fg_c, activebackground="#282850",
              relief="flat", cursor="hand2",
              command=lambda n=sname: schedule_mgr.set_override(n, 1800)
              ).pack(fill="x", pady=2)

# ── Control mode ─────────────────────────────────────────────
section_header("CONTROL MODE")
mode_frame = tk.Frame(panel, bg=PANEL_BG)
mode_frame.pack(fill="x", padx=10, pady=2)

def set_mode(m):
    global MODE
    MODE = m

for m, (bg_c, fg_c) in [("FIXED",   ("#1a1000","#ffcc44")),
                          ("DENSITY", ("#001a10","#44ffcc")),
                          ("DQN",     ("#00001a","#6699ff"))]:
    tk.Button(mode_frame, text=m, font=("Consolas", 9),
              bg=bg_c, fg=fg_c, activebackground="#282850",
              relief="flat", cursor="hand2",
              command=lambda md=m: set_mode(md)
              ).pack(fill="x", pady=2)

# ── Emergency + Save ─────────────────────────────────────────
section_header("ACTIONS")
act_frame = tk.Frame(panel, bg=PANEL_BG)
act_frame.pack(fill="x", padx=10, pady=2)

def trigger_emergency():
    global emergency_active, emergency_node_id, emergency_timer, emergency_total
    emergency_active  = True
    emergency_total += 1
    emergency_node_id = random.randint(0, GRID * GRID - 1)
    emergency_timer   = 150

def save_model():
    try:
        dummy = torch.randn(1, STATE_DIM)
        traced = torch.jit.trace(policy_net, dummy)
        traced.save("adaptiflow_x.pt")
        torch.save(policy_net.state_dict(), "adaptiflow_x_weights.pth")
        print("[AdaptiFlow-X] Model saved → adaptiflow_x.pt + adaptiflow_x_weights.pth")
    except Exception as ex:
        print(f"[AdaptiFlow-X] Save failed: {ex}")

def load_model():
    try:
        if os.path.exists("adaptiflow_x_weights.pth"):
            policy_net.load_state_dict(torch.load("adaptiflow_x_weights.pth", map_location="cpu"))
            target_net.load_state_dict(policy_net.state_dict())
            global EPS, MODE, pretrained_mode
            EPS = 0.0
            MODE = "DQN"
            pretrained_mode = True
            lbl_agent.config(text="Agent    : PRETRAINED")
            print("[AdaptiFlow-X] Pretrained weights loaded")
        else:
            print("[AdaptiFlow-X] No pretrained weights found")
    except Exception as ex:
        print(f"[AdaptiFlow-X] Load failed: {ex}")
def run_benchmark():
    global benchmark_mode, benchmark_index, benchmark_timer, MODE
    global throughput_total, power_consumed, co2_saved, time_saved_veh


    benchmark_mode = True
    benchmark_index = 0
    benchmark_timer = BENCHMARK_DURATION
    MODE = benchmark_sequence[0]

    throughput_total = 0
    power_consumed = 0
    co2_saved = 0.0
    time_saved_veh = 0
    
    for n in nodes:
        n.q = [0] * LANES
        n.action = 0
        n.phase_idx = 1
        n.phase_remaining = PHASE_DURATIONS[1]
        n.last_discharged = 0
    print("[AdaptiFlow-X] Benchmark suite started")
tk.Button(act_frame, text="⚡  Trigger Emergency Vehicle",
          font=("Consolas", 9), bg="#2a0808", fg="#ff8866",
          activebackground="#3a1010", relief="flat", cursor="hand2",
          command=trigger_emergency).pack(fill="x", pady=3)
tk.Button(act_frame, text="💾  Save Trained Model",
          font=("Consolas", 9), bg="#082a08", fg="#88ff88",
          activebackground="#104010", relief="flat", cursor="hand2",
          command=save_model).pack(fill="x", pady=3)
tk.Button(act_frame, text="📂  Load Pretrained Model",
          font=("Consolas", 9), bg="#08182a", fg="#88ccff",
          activebackground="#103050", relief="flat", cursor="hand2",
          command=load_model).pack(fill="x", pady=3)
tk.Button(act_frame, text="▶  Run Benchmark Suite",
          font=("Consolas", 9),
          bg="#221122",
          fg="#ff99ff",
          activebackground="#332233",
          relief="flat",
          cursor="hand2",
          command=run_benchmark).pack(fill="x", pady=3)

# ════════════════════════════════════════════════════════════
#  VISUALISATION
# ════════════════════════════════════════════════════════════
LANE_NAMES = ["N", "E", "S", "W"]
LANE_OFFSETS = [(0, -1), (1, 0), (0, 1), (-1, 0)]   # (dx, dy) in canvas coords


def queue_color(q: int, max_q: int = MAX_QUEUE) -> str:
    ratio = min(1.0, q / max(max_q, 1))
    if ratio < 0.40:
        r, g, b = 0, int(140 + 80 * ratio), int(60 * (1 - ratio))
    elif ratio < 0.70:
        t = (ratio - 0.40) / 0.30
        r, g, b = int(220 * t), int(180 * (1 - t * 0.5)), 0
    else:
        t = (ratio - 0.70) / 0.30
        r, g, b = 220, int(80 * (1 - t)), 0
    return f"#{r:02x}{g:02x}{b:02x}"


def draw():
    canvas.delete("all")
    cw = canvas.winfo_width()
    ch = canvas.winfo_height()
    # Road grid background
    for i in range(GRID + 2):
        p = i * CELL
        canvas.create_line(p, 0, p, ch, fill="#151530", width=1)
        canvas.create_line(0, p, cw, p, fill="#151530", width=1)

    # Draw inter-node connector roads
    for n in nodes:
        cx = (n.c + 1) * CELL
        cy = (n.r + 1) * CELL
        for nb in get_neighbors(n):
            nx2 = (nb.c + 1) * CELL
            ny2 = (nb.r + 1) * CELL
            canvas.create_line(cx, cy, nx2, ny2, fill="#1a1a35", width=6)
            canvas.create_line(cx, cy, nx2, ny2, fill="#2a2a50", width=2,
                               dash=(4, 6))

    # Draw intersections
    for n in nodes:
        cx  = (n.c + 1) * CELL
        cy  = (n.r + 1) * CELL
        half = 46

        # Congestion-based box colour
        ratio = min(1.0, n.total_queue / (MAX_QUEUE * LANES))
        rc = int(min(255, ratio * 290))
        gc = int(max(0,  85 - ratio * 85))
        bc = int(max(0, 220 - ratio * 220))
        box_col = f"#{rc:02x}{gc:02x}{bc:02x}"

        # Emergency pulse border
        if emergency_active and n.id == emergency_node_id:
            canvas.create_rectangle(cx - half - 5, cy - half - 5,
                                    cx + half + 5, cy + half + 5,
                                    fill="#3a0a00", outline="#ff5500", width=3)

        canvas.create_rectangle(cx - half, cy - half,
                                cx + half, cy + half,
                                fill=box_col, outline="#283060", width=1)

        # Lane signals and queue bars
        for i, (dx, dy) in enumerate(LANE_OFFSETS):
            dist = half + 16
            lx = cx + dx * dist
            ly = cy + dy * dist

            is_green = (n.action == i)

            if emergency_active and n.id == emergency_node_id:
                sig_col = "#ff8800"   # amber during emergency
            else:
                sig_col = "#00e050" if is_green else "#cc2020"

            # Queue bar
            bar = int(n.q[i] / MAX_QUEUE * 26)
            if dy < 0:      # N
                canvas.create_rectangle(lx - 8, ly - bar, lx + 8, ly,
                                        fill=queue_color(n.q[i]), outline="")
            elif dy > 0:    # S
                canvas.create_rectangle(lx - 8, ly, lx + 8, ly + bar,
                                        fill=queue_color(n.q[i]), outline="")
            elif dx > 0:    # E
                canvas.create_rectangle(lx, ly - 8, lx + bar, ly + 8,
                                        fill=queue_color(n.q[i]), outline="")
            else:           # W
                canvas.create_rectangle(lx - bar, ly - 8, lx, ly + 8,
                                        fill=queue_color(n.q[i]), outline="")

            # Signal dot
            canvas.create_oval(lx - 7, ly - 7, lx + 7, ly + 7,
                               fill=sig_col, outline="")
            # Lane label
            canvas.create_text(lx, ly, text=LANE_NAMES[i],
                               fill="white" if is_green else "#556677",
                               font=("Consolas", 7, "bold"))

        # Node ID and queue count
        canvas.create_text(cx, cy - 12, text=f"#{n.id}",
                           fill="#445566", font=("Consolas", 8))
        canvas.create_text(cx, cy + 2, text=f"{n.total_queue}v",
                           fill="white", font=("Consolas", 11, "bold"))
        canvas.create_text(cx, cy + 17, text=f"T={PHASE_DURATIONS[n.phase_idx]}s",
                           fill="#7799cc", font=("Consolas", 8))
        if n.last_discharged > 0:
            canvas.create_text(cx + 28, cy - 28,
                               text=f"+{n.last_discharged}",
                               fill="#44ff88", font=("Consolas", 8))

    # Legend
    for i, (label, col) in enumerate([("Low", "#00c850"),
                                       ("Mid", "#ffaa00"),
                                       ("High","#dd2222")]):
        bx = 12 + i * 80
        canvas.create_rectangle(bx, ch - 30, bx + 16, ch - 16,
                                fill=col, outline="")
        canvas.create_text(bx + 24, ch - 23, text=label,
                           fill="#aabbcc", font=("Consolas", 8), anchor="w")

    canvas.create_text(cw // 2, 12,
                       text="AdaptiFlow-X  |  Vaibhav Krishna V",
                       fill="#223355", font=("Consolas", 9))


# ════════════════════════════════════════════════════════════
#  MAIN SIMULATION LOOP
# ════════════════════════════════════════════════════════════
def loop():
    global step, EPS, episode, episode_reward, reward_history
    global success_count, avg_reward, success_rate
    global active_schedule, emergency_active, emergency_node_id, emergency_timer
    global power_consumed, co2_saved, throughput_total, time_saved_veh
    global benchmark_mode, benchmark_index, benchmark_timer, MODE
    global emergency_resolved


    # ── Schedule tick ─────────────────────────────────────────
    schedule_mgr.tick()
    active_schedule = schedule_mgr.get_active()

    # ── Emergency countdown ───────────────────────────────────
    if emergency_active:
        emergency_timer -= 1
        if emergency_timer <= 0:
            emergency_active = False
            emergency_node_id = -1
            emergency_resolved += 1

    # ── Spawn vehicles ────────────────────────────────────────
    spawn_vehicles()

    step_reward = 0.0
    step_power  = 0.0

    for n in nodes:
        prev_q = n.total_queue
        s      = n.state_vec.copy()

        # Phase-locked decision: only re-decide when current phase expires
        if n.phase_remaining <= 0:
            lane, pidx       = select_action(n)
            n.action         = lane
            n.phase_idx      = pidx
            n.phase_remaining = PHASE_DURATIONS[pidx]
        else:
            lane, pidx        = n.action, n.phase_idx
            n.phase_remaining -= 1

        # Emergency preemption: pause new decisions at affected node
        if emergency_active and n.id == emergency_node_id:
            # Keep existing green but don't discharge (simulate road blockage)
            n.last_discharged = 0
            continue

        discharged      = discharge_and_route(n, lane, pidx)
        r, power_cost   = compute_reward(prev_q, n, pidx, discharged)
        ns              = n.state_vec.copy()

        step_reward    += r
        step_power     += power_cost
        power_consumed += power_cost

        # CO₂ proxy: vehicles cleared vs. staying idle
        co2_saved += discharged * CO2_PER_VEH_IDLE

        if MODE == "DQN" and not benchmark_mode and EPS > 0:
            replay_buffer.push(s, encode_action(lane, pidx), r, ns, False)

    # ── Training ──────────────────────────────────────────────
    loss_val = 0.0
    if MODE == "DQN" and not benchmark_mode and not pretrained_mode:
        loss_val = train_step()
        if step % TARGET_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())
            target_net.eval()
        EPS = max(EPS_MIN, EPS * EPS_DECAY)
        writer.add_scalar("Loss/Train",   loss_val,   step)
        writer.add_scalar("Agent/Epsilon", EPS,        step)

    writer.add_scalar("Reward/Step",       step_reward, step)
    writer.add_scalar("City/Power",        step_power,  step)
    writer.add_scalar("City/Throughput",   throughput_total, step)
    writer.add_scalar("City/CO2_Saved",    co2_saved,   step)

    episode_reward += step_reward
    step           += 1
    if benchmark_mode:
        benchmark_timer -= 1

        if benchmark_timer <= 0:
            avg_q_snapshot = np.mean([n.total_queue for n in nodes])

            benchmark_stats[MODE]["throughput"] = throughput_total
            benchmark_stats[MODE]["avg_q"] = avg_q_snapshot
            benchmark_stats[MODE]["power"] = power_consumed
            benchmark_stats[MODE]["co2"] = co2_saved

            benchmark_index += 1

            if benchmark_index < len(benchmark_sequence):
                MODE = benchmark_sequence[benchmark_index]
                benchmark_timer = BENCHMARK_DURATION

                throughput_total = 0
                power_consumed = 0
                co2_saved = 0.0
                time_saved_veh = 0

                for n in nodes:
                    n.q = [0] * LANES
                    n.action = 0
                    n.phase_idx = 1
                    n.phase_remaining = PHASE_DURATIONS[1]
                    n.last_discharged = 0

                print(f"[AdaptiFlow-X] Switching benchmark → {MODE}")

            else:
                benchmark_mode = False
                if pretrained_mode:
                    MODE = "DQN"
                    EPS = 0.0
                else:
                    MODE = "DQN"

                for n in nodes:
                    n.q = [0] * LANES
                    n.action = 0
                    n.phase_idx = 1
                    n.phase_remaining = PHASE_DURATIONS[1]
                    n.last_discharged = 0

                print("[AdaptiFlow-X] Benchmark suite complete")
    # ── Episode boundary ──────────────────────────────────────
    if step % EPISODE_LEN == 0:
        reward_history.append(episode_reward)
        avg_reward = (np.mean(reward_history[-20:])
                      if len(reward_history) >= 20
                      else np.mean(reward_history))
        avg_q = np.mean([n.total_queue for n in nodes])
        if not benchmark_mode:
            benchmark_stats[MODE]["throughput"] = throughput_total
            benchmark_stats[MODE]["avg_q"] = avg_q
            benchmark_stats[MODE]["power"] = power_consumed
            benchmark_stats[MODE]["co2"] = co2_saved
        if avg_q < SUCCESS_Q_THRESH:
            success_count += 1
        success_rate = success_count / max(1, episode + 1)
        writer.add_scalar("Reward/Episode",    episode_reward, episode)
        writer.add_scalar("Reward/Avg20ep",    avg_reward,     episode)
        writer.add_scalar("Agent/SuccessRate", success_rate,   episode)
        episode_reward = 0.0
        episode       += 1

    # ── Draw ──────────────────────────────────────────────────
    draw()

    # ── Update panel ──────────────────────────────────────────
    avg_q_now   = np.mean([n.total_queue for n in nodes])
    sched_text  = active_schedule
    if schedule_mgr.is_overridden:
        sched_text += f"  [{schedule_mgr.override_remaining}s left]"

    lbl_mode.config(text=f"Mode     : {MODE}")
    lbl_sched.config(text=f"Schedule : {sched_text}")
    lbl_step.config(text=f"Step     : {step}")
    lbl_episode.config(text=f"Episode  : {episode}")
    lbl_eps.config(text=f"Epsilon  : {EPS:.3f}")
    lbl_memory.config(text=f"Replay   : {len(replay_buffer):,} / {MEM_CAP:,}")
    if benchmark_mode:
        lbl_benchmark.config(
            text=f"Benchmark: RUNNING ({MODE} {benchmark_index+1}/{len(benchmark_sequence)})",
            fg="#ff99ff"
        )
    else:
        lbl_benchmark.config(
            text="Benchmark: Idle",
            fg="#99aabb"
        )
    if MODE == "FIXED":
        lbl_agent.config(text="Agent    : RULE-BASED")
    elif MODE == "DENSITY":
        lbl_agent.config(text="Agent    : HEURISTIC")
    elif pretrained_mode:
        lbl_agent.config(text="Agent    : PRETRAINED")
    else:
        lbl_agent.config(text="Agent    : TRAINING")
    lbl_step_r.config(text=f"Optimization Score : {step_reward:+.2f}")
    lbl_ep_r.config(text=f"Episode Reward  : {episode_reward:.2f}")
    lbl_avg_r.config(text=f"Avg Reward(20ep): {avg_reward:.2f}")

    queue_score = max(0.0, min(100.0, 100.0 * (1.0 - avg_q_now / 150.0)))
    throughput_score = max(0.0, min(100.0, throughput_total / 200.0))
    power_score = max(0.0, min(100.0, 100.0 - (power_consumed / 1500000.0) * 100.0))

    traffic_efficiency = (
        0.50 * queue_score +
        0.35 * throughput_score +
        0.15 * power_score
    )

    lbl_success.config(
        text=f"Traffic Efficiency : {traffic_efficiency:.1f} %"
    )

    lbl_throughput.config(text=f"Throughput  : {throughput_total:,} veh")
    lbl_avg_q.config(text=f"Avg Queue   : {avg_q_now:.1f} veh/node")
    lbl_power.config(text=f"Power Used  : {power_consumed:.0f} kU")
    lbl_time_saved.config(text=f"Time Saved  : {time_saved_veh:,} veh-steps")
    lbl_co2.config(text=f"CO₂ Saved   : {co2_saved:.2f} kg")
    lbl_arrival.config(text=f"Arrival Rate: {current_arrival:.2f} veh/step")
    if emergency_active:
        lbl_emergency.config(text=f"Emergency   : Node #{emergency_node_id} ({emergency_timer}t)",
                             fg="#ff6644")
    else:
        lbl_emergency.config(text="Emergency   : None", fg="#667788")

    lbl_fixed_cmp.config(
        text=f"FIXED   : Q={benchmark_stats['FIXED']['avg_q']:.1f}  T={benchmark_stats['FIXED']['throughput']:,}"
    )
    lbl_fixed_cmp2.config(
        text=f"P={benchmark_stats['FIXED']['power']:.0f}kU  CO₂={benchmark_stats['FIXED']['co2']:.1f}kg"
    )

    lbl_density_cmp.config(
        text=f"DENSITY : Q={benchmark_stats['DENSITY']['avg_q']:.1f}  T={benchmark_stats['DENSITY']['throughput']:,}"
    )
    lbl_density_cmp2.config(
        text=f"P={benchmark_stats['DENSITY']['power']:.0f}kU  CO₂={benchmark_stats['DENSITY']['co2']:.1f}kg"
    )

    lbl_dqn_cmp.config(
        text=f"DQN     : Q={benchmark_stats['DQN']['avg_q']:.1f}  T={benchmark_stats['DQN']['throughput']:,}"
    )
    lbl_dqn_cmp2.config(
        text=f"P={benchmark_stats['DQN']['power']:.0f}kU  CO₂={benchmark_stats['DQN']['co2']:.1f}kg"
    )

    root.after(max(1, int(1000 / SIM_SPEED)), loop)


# ════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 62)
    print("  AdaptiFlow-X: Adaptive Traffic Intelligence System")
    print("  Author  : Vaibhav Krishna V  ")
    print("=" * 62)
    print(f"  Grid    : {GRID}×{GRID} intersections ({GRID*GRID} nodes)")
    print(f"  Mode    : {MODE}")
    print(f"  State   : {STATE_DIM}-D vector per intersection")
    print(f"  Actions : {ACTION_DIM}  ({LANES} lanes × {N_PHASES} phase durations)")
    print(f"  Network : Dueling Double DQN + Prioritised Replay")
    print(f"  Memory  : {MEM_CAP:,} transitions  |  Batch: {BATCH}")
    print(f"  Reward  : Multi-objective (Queue + Throughput + Power + Overflow)")
    print(f"  Logging : TensorBoard → runs/adaptiflow_x/")
    print("=" * 62)

    pass

    root.after(120, loop)
    root.mainloop()
    writer.close()
