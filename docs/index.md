# Credit Limit Bandit — Complete Documentation

This documentation covers a contextual multi-armed bandit system for dynamic credit limit optimization. The project demonstrates applied reinforcement learning in a fintech setting, balancing exploration against exploitation to maximize portfolio revenue while managing default risk.

## What This Project Demonstrates

**Credit Limit Bandit** is a research and production-ready reference implementation showing how to:
- Formulate credit limit decisions as a contextual bandit problem
- Deploy three competing algorithms (Thompson Sampling, UCB, Epsilon-Greedy)
- Handle delayed rewards (3-month lag) with a buffer system
- Evaluate policies against ground truth (3.38% default rate across all algorithms)
- Adapt to economic shocks (month 6 default rate spike)
- Build interactive dashboards for policy comparison

The system ingests 10 normalized features per user and selects from 4 credit actions (keep / plus_10 / plus_20 / plus_50) monthly for 10,000 synthetic users over 12 months.

**Results at a glance:**
| Policy | Revenue | Lift | Regret |
|---|---|---|---|
| Thompson Sampling | ₹12.13Cr | +39.09% | 16.34% |
| UCB | ₹10.83Cr | +24.15% | 25.32% |
| Epsilon-Greedy | ₹8.70Cr | −0.26% | 40.01% |
| Static Baseline | ₹8.72Cr | — | — |
| Practical Oracle | ₹14.50Cr | — | — |

## Navigation Guide

### Architecture (3 files)
- [overview.md](architecture/overview.md) — 5-component system diagram, data flow, "what happens in one month" (8 steps)
- [data-flow.md](architecture/data-flow.md) — Pipeline arrows, delayed feedback buffer, exact data types and CSV schema
- [design-decisions.md](architecture/design-decisions.md) — Why bandit, why Thompson, why two-branch normalization, why Beta(1,1)

### Algorithms (4 files)
- [thompson-sampling.md](algorithms/thompson-sampling.md) — Beta posteriors, 5-step how-it-works, code from `thompson.py`
- [ucb.md](algorithms/ucb.md) — UCB1 formula, exploration bonus decay, why 25.32% regret
- [epsilon-greedy.md](algorithms/epsilon-greedy.md) — Decay formula, why −0.26% vs static, parameter values
- [comparison.md](algorithms/comparison.md) — Side-by-side table across 8 properties, "which to use" answer, neural network comparison

### Components (4 files)
- [simulator.md](components/simulator.md) — All 12 UserSchema fields, default probability formula with worked example
- [context-builder.md](components/context-builder.md) — All 10 features with formula, ranges, what each captures, how to add one
- [reward-engine.md](components/reward-engine.md) — Reward formula, 1.8% interchange, RewardBuffer month-by-month, two-branch normalization constants
- [evaluation.md](components/evaluation.md) — All 7 metrics (revenue, lift, default rate, regret, regret%, convergence, exploration)

### Concepts (5 files)
- [multi-armed-bandit.md](concepts/multi-armed-bandit.md) — Casino analogy, formal problem definition, contextual vs non-contextual
- [exploration-exploitation.md](concepts/exploration-exploitation.md) — Credit-specific examples, pure exploration cost, algorithm comparison
- [delayed-feedback.md](concepts/delayed-feedback.md) — Why 3 months, buffer diagram, cold start overlap
- [cold-start.md](concepts/cold-start.md) — Definition, months 1-3 with zero reward, Beta(1,1) handling
- [non-stationarity.md](concepts/non-stationarity.md) — Month 6 shock, default rate 3.28%→3.67%, Beta limitation

### Guides (4 files)
- [getting-started.md](guides/getting-started.md) — Prerequisites, full install, pytest, quick start, common errors
- [running-simulations.md](guides/running-simulations.md) — CLI args, 4 scenario examples, output schema, runtime table
- [dashboard.md](guides/dashboard.md) — All 4 pages described, how to use per-user selector, what to look for
- [contributing.md](guides/contributing.md) — Add a new bandit (6 steps), interface contract, LinUCB skeleton

### Results (2 files)
- [metrics-explained.md](results/metrics-explained.md) — All 7 metrics: name, formula, unit, interpretation, target, actual result
- [interpreting-results.md](results/interpreting-results.md) — "What good looks like", red flags (5 patterns), cohort heatmap, Q&A

## Suggested Reading Paths

**For recruiter / hiring manager** (15 minutes)
1. [docs/index.md](index.md) (this file) — What it does
2. [docs/architecture/overview.md](architecture/overview.md) — How it's built
3. [docs/results/interpreting-results.md](results/interpreting-results.md) — What it achieved
4. [docs/guides/dashboard.md](guides/dashboard.md) — How to see it live

**For ML engineer evaluating code** (1 hour)
1. [docs/guides/getting-started.md](guides/getting-started.md) — Run it locally
2. [docs/components/context-builder.md](components/context-builder.md) — How context is built
3. [docs/algorithms/thompson-sampling.md](algorithms/thompson-sampling.md) — Thompson implementation
4. [docs/architecture/design-decisions.md](architecture/design-decisions.md) — Why these choices
5. [docs/results/metrics-explained.md](results/metrics-explained.md) — How results are measured

**For student learning bandits** (2 hours)
1. [docs/concepts/multi-armed-bandit.md](concepts/multi-armed-bandit.md) — What is a bandit
2. [docs/concepts/exploration-exploitation.md](concepts/exploration-exploitation.md) — The core tradeoff
3. [docs/algorithms/comparison.md](algorithms/comparison.md) — Compare algorithms side-by-side
4. [docs/algorithms/thompson-sampling.md](algorithms/thompson-sampling.md) — Deep dive Thompson
5. [docs/concepts/delayed-feedback.md](concepts/delayed-feedback.md) — Production reality

## Key Ground Truth Values

Always use these exact figures in conversations and reports:

| Metric | Value |
|---|---|
| Thompson Sampling Revenue | ₹12.13Cr |
| Thompson Sampling Lift | +39.09% |
| Thompson Sampling Regret | 16.34% |
| UCB Revenue | ₹10.83Cr |
| UCB Lift | +24.15% |
| UCB Regret | 25.32% |
| Epsilon-Greedy Revenue | ₹8.70Cr |
| Epsilon-Greedy Lift | −0.26% |
| Epsilon-Greedy Regret | 40.01% |
| Static Baseline Revenue | ₹8.72Cr |
| Practical Oracle Revenue | ₹14.50Cr |
| Default Rate (All Policies) | 3.38% |
| Reward Delay | 3 months |
| Context Features | 10 (float32 vector) |
| Actions | 4: keep / plus_10 / plus_20 / plus_50 |
| Users | 10,000 synthetic |
| Horizon | 12 months |
| Interchange Rate | 1.8% |
| Risk Tiers | Prime 40% / Near-Prime 30% / Subprime 20% / Deep-Subprime 10% |

## File Statistics

- **Total files:** 22
- **Total lines:** ~8,000
- **Code snippets:** 40+ with line references
- **Metrics covered:** 7 core + 12 derived
- **Algorithms:** 3 (Thompson, UCB, Epsilon-Greedy)
- **Data types:** 15+ with schema definitions
- **Example scenarios:** 20+

## Quick Links

| File | Words | Focus |
|---|---|---|
| index.md | 500 | Navigation and overview |
| architecture/overview.md | 700 | System components and responsibilities |
| architecture/data-flow.md | 650 | Pipeline with data types and delays |
| architecture/design-decisions.md | 800 | Key technical decisions with tradeoffs |
| algorithms/thompson-sampling.md | 750 | Beta posteriors and sampling |
| algorithms/ucb.md | 700 | Exploration bonus and regret |
| algorithms/epsilon-greedy.md | 600 | Decay and stationarity assumptions |
| algorithms/comparison.md | 800 | Side-by-side comparison matrix |
| components/simulator.md | 750 | UserSchema fields and default model |
| components/context-builder.md | 750 | All 10 features and normalization |
| components/reward-engine.md | 700 | Immediate and delayed rewards |
| components/evaluation.md | 800 | Metric definitions and calculations |
| concepts/multi-armed-bandit.md | 600 | Problem formulation and motivation |
| concepts/exploration-exploitation.md | 700 | Tradeoff with credit examples |
| concepts/delayed-feedback.md | 600 | 3-month lag and its effects |
| concepts/cold-start.md | 650 | Month 1-3 behavior and context value |
| concepts/non-stationarity.md | 650 | Economic shock adaptation |
| guides/getting-started.md | 700 | Install and first run |
| guides/running-simulations.md | 800 | All CLI parameters and scenarios |
| guides/dashboard.md | 750 | Interactive features and pages |
| guides/contributing.md | 800 | Extending the system |
| results/metrics-explained.md | 800 | Definition and interpretation of all metrics |
| results/interpreting-results.md | 900 | Success criteria and failure modes |

## Related Docs

- [docs/architecture/overview.md](architecture/overview.md)
- [docs/guides/getting-started.md](guides/getting-started.md)
- [docs/results/interpreting-results.md](results/interpreting-results.md)
