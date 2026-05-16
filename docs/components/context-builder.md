---
title: Context Builder & Feature Extraction
category: component
file_reference: src/context.py
---
# Context Builder & Feature Extraction

The Context Builder converts raw user attributes and short-term history into fixed-size numeric vectors consumed by contextual bandits. Good features enable personalization and reduce regret by differentiating users with similar aggregate statistics.

Why context matters
- The same limit change has different marginal value for different users; context allows per-user expected reward estimation and targeted exploration.

Feature table (10 features)
| Feature | Formula / Source | Raw range | Normalized range | What it captures |
| --- | --- | --- | --- | --- |
| cibil_score | raw int | 300–900 | 0–1 (min-max) | creditworthiness signal |
| credit_limit | raw int | 5k–200k | log + min-max | account scale (log-scale) |
| utilization | balance / limit | 0–1 | 0–1 | short-term stress |
| avg_spend_3m | mean last 3 spends | 0–limit | 0–1 | spending propensity |
| volatility_3m | std last 3 spends | 0–limit | 0–1 | unpredictability signal |
| income_bucket | categorical | low/med/high | one-hot | affordability proxy |
| employment_type | categorical | salaried/gig/self | one-hot | stability proxy |
| tenure_months | int | 0–240 | 0–1 | relationship age |
| last_defaults | int | 0–3 | clipped 0–1 | recent risk signal |
| risk_tier | categorical | Prime/... | ordinal 0..1 | coarse risk grouping |

Normalization strategy
- Min-max for bounded features to preserve interpretability.
- Log-transform for `credit_limit` to compress skew.
- Standardization avoided to keep feature ranges consistent across users and months.

Spending volatility
- Computed as standard deviation over last 3 months normalized by limit. High volatility signals unpredictability and increases conservative actions.

Cold start vector
- For new users with no history: use population means for history-derived features, keep categorical fields from onboarding data. Example vector: cibil=0.8, credit_limit=log-median, utilization=0.2, avg_spend_3m=population_mean, volatility=0.15.

Feature importance intuition
- Most influential: `cibil_score`, `utilization`, and `avg_spend_3m` — they directly correlate with default and revenue.

Adding new features
1. Add the column to `UserSchema` in `src/simulator.py` and ensure generation.
2. Update `src/context.py` to compute and normalize the new feature.
3. Update downstream tests in `tests/` and add a brief note in `docs/components/context-builder.md`.

Related docs
- [docs/components/simulator.md](docs/components/simulator.md)
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
