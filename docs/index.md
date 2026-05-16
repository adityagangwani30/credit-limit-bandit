---
title: Documentation Home
category: guide
file_reference: none
---
# Project Documentation — Credit Limit Bandit

Credit Limit Bandit is a research and demonstration repository that applies contextual multi-armed bandit methods to personalized credit limit decisions. It includes a synthetic user simulator, a context builder, three bandit policies, delayed reward handling, evaluation tooling, and an interactive Streamlit dashboard for visualization.

## What this project demonstrates
- Reinforcement learning concepts applied to a fintech decisioning problem
- Contextual personalization with limited, delayed feedback
- Practical handling of delayed rewards and production constraints
- Comparative evaluation of algorithms under economic shocks
- Reproducible simulation and interpretability for policy selection

## Navigation table
| File | What it covers | Read if you want to... |
| --- | --- | --- |
| [docs/architecture/overview.md](docs/architecture/overview.md) | System architecture and responsibilities | Understand how modules interact and where to extend |
| [docs/architecture/data-flow.md](docs/architecture/data-flow.md) | Data flow and types | Know exact shapes and buffers for debugging |
| [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md) | Thompson Sampling deep dive | Implement, tune, or explain Thompson Sampling |
| [docs/algorithms/ucb.md](docs/algorithms/ucb.md) | UCB algorithm details | Compare deterministic exploration strategies |
| [docs/components/simulator.md](docs/components/simulator.md) | Synthetic user model | Regenerate data and validate assumptions |
| [docs/guides/getting-started.md](docs/guides/getting-started.md) | Setup and first run instructions | Run the repo locally and reproduce results |

## Suggested reading order
- Interviewer / recruiter: architecture → results → index
- ML engineer evaluating code: guides/getting-started → components/context-builder → algorithms/thompson-sampling
- Student learning bandits: concepts/multi-armed-bandit → concepts/exploration-exploitation → algorithms/comparison

## Quick stats
- 10,000 users · 4 actions · 3 algorithms · 12-month horizon · 3-month reward delay

Related docs
- [docs/guides/getting-started.md](docs/guides/getting-started.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
