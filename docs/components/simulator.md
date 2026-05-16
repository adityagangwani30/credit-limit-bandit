---
title: User Simulator
category: component
file_reference: src/simulator.py
---
# User Simulator

The `Simulator` generates synthetic monthly user behavior for experimentation when real credit portfolio data is unavailable. It is designed to produce reproducible user trajectories (spend, outstanding, defaults) and to support controlled experiments such as economic shocks and policy comparisons.

Purpose
- Provide safe, privacy-preserving synthetic users for bandit policy development.
- Enable validation targets (aggregate default rate, spending distribution) to check realism.

UserSchema (10 fields)
| Field | Type | Range / Generation | Notes |
| --- | --- | --- | --- |
| user_id | int | 1..n_users | Unique identifier |
| cibil_score | int | 300..900 | drawn from tiered distribution |
| income_bucket | str | low/med/high | probabilistic assignment |
| employment_type | str | salaried/gig/self | categorical |
| credit_limit | int | 5000..200000 | initial limit, log-normal sample |
| utilization | float | 0.0..1.0 | monthly utilization fraction |
| avg_spend | float | 0..limit | drawn from Gaussian × limit |
| volatility | float | 0.0..1.0 | std dev of spend over months |
| risk_tier | str | Prime/Near/Sub/Deep | assigned by cibil_score buckets |
| last_defaults | int | 0..3 | recent default count for history |

Risk tier distribution
- Prime 40% / Near-Prime 30% / Subprime 20% / Deep-Subprime 10%. This reflects a conservative retail portfolio mix and aids realistic aggregate default targets.

Default probability model
- Base default probability is a function of `risk_tier` and `utilization`:
``text
P(default) = base_rate_tier * (1 + 2 * utilization) * economic_stress
``
- Example: Subprime user with base_rate=0.03 at util=0.8 and economic_stress=1.0 → P=0.03*(1+1.6)=0.078 (7.8%).

Spending model
- Monthly spend ~ Normal(mean=0.6 × limit, std=0.1 × limit), clipped to [0, limit]. Volatility parameter scales std.

Validation targets
- Aggregate default rate: 2.5–3.5% across the portfolio (tunable via base rates)
- Spending distribution stratified by `risk_tier` should reflect lower spend and higher volatility for lower tiers

How to regenerate data
```bash
python src/simulator.py --n_users 10000 --seed 42
```
- Expected output: `data/simulation_results.csv` with per-user/month rows.

Limitations
- No temporal correlation beyond simple random walk assumptions (no complex AR processes)
- No social or network contagion modeled

Related docs
- [docs/components/context-builder.md](docs/components/context-builder.md)
- [docs/architecture/data-flow.md](docs/architecture/data-flow.md)
