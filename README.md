# Reinforcement Learning in GridWorld

This repository contains implementations of three classic reinforcement learning (RL) algorithms — **Monte Carlo**, **SARSA**, and **Q-learning** — applied to a common GridWorld environment.

The goal is to compare the performance of these RL methods under various hyperparameter settings in a stochastic grid environment with walls, doors, and a defined goal state.

## Files

- `monte-carlo.py` — Implements a Monte Carlo learning agent.
- `sarsa.py` — Implements a SARSA learning agent.
- `qlearning.py` — Implements a Q-learning agent.

Each script:
- Defines the `GridWorld` environment with stochastic transitions.
- Runs training episodes using its respective RL algorithm.
- Evaluates the resulting policy and prints it as a grid.

## Environment

The `GridWorld` environment is a 10x10 grid with:
- A goal state at the bottom-right corner (9,9).
- Walls and doors that restrict movement between rooms.
- Rewards: +1 for reaching the goal, -1 otherwise.
- Stochastic transitions controlled by parameters `p1` (intended move), `p2` (stay in place), and `p3` (adjacent moves).

## How to Run

Each script can be run independently:

```bash
python monte-carlo.py
python sarsa.py
python qlearning.py
