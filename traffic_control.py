"""
Smart Traffic Signal Control System
Using Adaptive and Reinforcement Learning Methods
Author: Vaibhav Krishna V | NMIT Bengaluru | 1NT22EC182
"""

import numpy as np
import random
import argparse
from collections import deque

NUM_LANES        = 4
MAX_VEHICLES     = 30
SIMULATION_STEPS = 200
ARRIVAL_RATE     = 0.3
GAMMA            = 0.95
EPSILON_START    = 1.0
EPSILON_MIN      = 0.05
EPSILON_DECAY    = 0.995
MEMORY_SIZE      = 2000
BATCH_SIZE       = 32

class TrafficIntersection:
    def __init__(self):
        self.lanes = np.zeros(NUM_LANES, dtype=int)
        self.step_count = 0
        self.total_wait = 0
        self.throughput = 0

    def reset(self):
        self.lanes = np.random.randint(0, 10, NUM_LANES)
        self.step_count = 0
        self.total_wait = 0
        self.throughput = 0
        return self.lanes / MAX_VEHICLES

    def step(self, action):
        discharged = min(self.lanes[action], np.random.randint(3, 8))
        self.lanes[action] = max(0, self.lanes[action] - discharged)
        self.throughput += discharged
        for i in range(NUM_LANES):
            if np.random.random() < ARRIVAL_RATE:
                self.lanes[i] = min(MAX_VEHICLES, self.lanes[i] + np.random.randint(1, 4))
        reward = discharged * 2.0 - np.sum(self.lanes) * 0.1
        self.total_wait += np.sum(self.lanes)
        self.step_count += 1
        done = self.step_count >= SIMULATION_STEPS
        return self.lanes / MAX_VEHICLES, reward, done

    def get_metrics(self):
        return {
            "avg_queue":  round(self.total_wait / max(self.step_count, 1) / NUM_LANES, 1),
            "throughput": round(self.throughput / max(self.step_count, 1), 2)
        }

class DQNAgent:
    def __init__(self):
        self.memory  = deque(maxlen=MEMORY_SIZE)
        self.epsilon = EPSILON_START
        hidden = 24
        self.W1 = np.random.randn(NUM_LANES, hidden) * 0.1
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, NUM_LANES) * 0.1
        self.b2 = np.zeros((1, NUM_LANES))

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def act(self, state):
        if np.random.random() < self.epsilon:
            return random.randrange(NUM_LANES)
        return int(np.argmax(self.forward(state.reshape(1, -1))))

    def remember(self, s, a, r, ns, d):
        self.memory.append((s, a, r, ns, d))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        for s, a, r, ns, d in random.sample(self.memory, BATCH_SIZE):
            target = r if d else r + GAMMA * np.max(self.forward(ns.reshape(1, -1)))
            pred   = self.forward(s.reshape(1, -1))
            pred[0][a] = target
            grad   = (pred - self.forward(s.reshape(1, -1))) * 0.001
            h      = np.maximum(0, s.reshape(1, -1) @ self.W1 + self.b1)
            self.W2 += h.T @ grad
            self.b2 += grad
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

def fixed_time_control(step):
    return (step // 20) % NUM_LANES

def density_adaptive_control(lanes):
    return int(np.argmax(lanes))

def run_simulation(mode="all"):
    env   = TrafficIntersection()
    agent = DQNAgent()
    results = {}
    modes_to_run = ["fixed", "density", "dqn"] if mode == "all" else [mode]

    for ctrl_mode in modes_to_run:
        print(f"\n{'='*50}\n  MODE: {ctrl_mode.upper()}\n{'='*50}")

        if ctrl_mode == "dqn":
            print("  Training DQN agent (50 episodes)...")
            for ep in range(50):
                state = env.reset()
                for _ in range(SIMULATION_STEPS):
                    action = agent.act(state)
                    ns, rew, done = env.step(action)
                    agent.remember(state, action, rew, ns, done)
                    agent.replay()
                    state = ns
                    if done:
                        break
                if (ep + 1) % 10 == 0:
                    print(f"  Episode {ep+1}/50 | epsilon={agent.epsilon:.3f}")
            agent.epsilon = EPSILON_MIN

        state = env.reset()
        for step in range(SIMULATION_STEPS):
            if ctrl_mode   == "fixed":   action = fixed_time_control(step)
            elif ctrl_mode == "density": action = density_adaptive_control(env.lanes)
            else:                        action = agent.act(state)
            state, _, done = env.step(action)
            if done:
                break

        m = env.get_metrics()
        results[ctrl_mode] = m
        print(f"  Avg Queue: {m['avg_queue']} | Throughput: {m['throughput']} veh/cycle")

    if len(results) > 1:
        print(f"\n{'='*50}\n  FINAL COMPARISON\n{'='*50}")
        for k, v in results.items():
            print(f"  {k.upper():<12} Queue={v['avg_queue']:5.1f}  Throughput={v['throughput']:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all", choices=["fixed","density","dqn","all"])
    args = parser.parse_args()
    print("Smart Traffic Signal Control | Vaibhav Krishna V | NMIT")
    run_simulation(mode=args.mode)
