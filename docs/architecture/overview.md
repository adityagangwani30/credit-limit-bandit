---
title: System Architecture
category: architecture
file_reference: none
---
# System Architecture

This document describes the high-level components and interactions that make up the Credit Limit Bandit project. The system is composed of five primary modules: Simulator, Context Builder, Bandit Policies, Reward Engine (with delayed buffer), and Evaluation & Dashboard.

ASCII component map
``text
[Synthetic Users] --> [Simulator] --> [Simulation Results CSV]
                                 |
                                 v
                         [Context Builder] --> [Bandit Policies] --> [Action Log]
                                 |                                |
                                 v                                v
                          [Reward Engine (Buffer)]  <---  [Simulator Outcomes]
                                 |
                                 v
                         [Evaluation & Dashboard]
``

Responsibilities table
| Component | File | Inputs | Outputs | Key classes |
| --- | --- | --- | --- | --- |
| Simulator | src/simulator.py | seed, n_users, policy actions | simulated transactions | Simulator |
| Context Builder | src/context.py | user profile, history | context vectors (ndarray) | ContextBuilder |
| Bandit Policies | src/bandits/*.py | context vectors | selected actions | ContextualBandit, ThompsonBandit |
| Reward Engine | src/reward.py | outcomes, defaults | normalized rewards, buffer | RewardBuffer, RewardEngine |
| Evaluation & Dashboard | src/evaluate.py, dashboard/app.py | action log, rewards | metrics, visualizations | Evaluator |

Component descriptions
- Simulator: Generates month-level synthetic users and outcomes using parameterized risk and spending models. It enables reproducible experiments without sensitive data.
- Context Builder: Extracts and normalizes features (10 fields) into fixed-size vectors to allow contextual decision-making per user and month.
- Bandit Policies: Implements contextual policies (`Thompson`, `UCB`, `EpsilonGreedy`) that consume context vectors and produce actions; each policy exposes `select_action` and `update` APIs.
- Reward Engine: Computes net rewards, applies normalization, and buffers outcomes for a 3-month delay before exposing them to policy updates.
- Evaluation & Dashboard: Computes metrics (revenue, regret, default rate) and provides an interactive Streamlit app for exploration and presentation.

What happens in one simulation month (walkthrough)
1. Simulator emits monthly user activity and synthetic outcomes (spend, outstanding, default flags).
2. Context Builder reads user profile + recent history and produces a context vector for each user.
3. Bandit Policies receive contexts and return an action (limit change) per user.
4. Actions are logged to an Action Log (CSV row per user/month/action).
5. Simulator applies actions to the next month’s financial behavior model and produces outcomes after internal processing.
6. Reward Engine receives simulator outcomes and stores them in a RewardBuffer to enforce a 3-month delay.
7. Evaluator reads released rewards (from buffer when due) and computes monthly metrics.
8. Dashboard updates visualizations and stores artifacts for offline analysis.

Technology choices
- NumPy over PyTorch: the bandit algorithms use lightweight probability updates and sampling; NumPy is faster to iterate, simpler to serialize, and reduces engineering overhead for a simulation-focused repo.
- Streamlit over Flask/FastAPI: Streamlit provides fast interactive exploration without building a separate frontend; it suits demos and interviews where visual iteration is critical.

Related docs
- [docs/architecture/data-flow.md](docs/architecture/data-flow.md)
- [docs/components/simulator.md](docs/components/simulator.md)
