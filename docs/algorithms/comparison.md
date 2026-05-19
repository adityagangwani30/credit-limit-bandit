# Algorithm Comparison

Thompson Sampling delivers ₹12.13Cr revenue (+39.09% lift, 16.34% regret) vs UCB (₹10.83Cr, 25.32% regret) and Epsilon-Greedy (₹8.70Cr, −0.26% regret). This document explains why Thompson wins and when alternatives apply.

## Algorithm Performance Table

| Algorithm | Revenue | Lift | Regret | Convergence | Exploration |
|-----------|---------|------|--------|-------------|-------------|
| Thompson | ₹12.13Cr | +39.09% | 16.34% | Month 5 | Adaptive |
| UCB | ₹10.83Cr | +24.15% | 25.32% | Month 6 | √(ln(n)/count) |
| Epsilon | ₹8.70Cr | −0.26% | 40.01% | Month 7 | Decay 0.995^t |
| Baseline | ₹8.72Cr | 0% | 39.80% | N/A | None |
| Oracle | ₹14.50Cr | +66.39% | 0% | Instant | N/A |

Thompson outperforms because Beta posteriors adapt exploration to actual uncertainty. By Month 5, Thompson concentrates on high-value actions (plus_20 for Prime users, keep for Deep-Subprime). Epsilon-Greedy's fixed decay (ε = 0.36 at Month 5) wastes 36% on random actions despite having converged 1-2 months earlier.

## Why Not Deep Learning?

1. **Delayed feedback (90 days):** Neural networks assume <10 step feedback; credit defaults arrive 3 months late
2. **Sample scarcity:** 10K users × 12 months = 120K decisions, only 1 outcome per user; NN needs 100K+ samples
3. **Exploration cost:** Testing high-limits on risky users causes real defaults (₹1-3M loss each)
4. **Interpretability:** RBI requires explainable credit decisions; Thompson's posterior is interpretable
5. **Non-stationarity:** Thompson adapts to Month 6 economic shock in 1 month; NN retraining takes weeks

## Production Recommendation

Use Thompson Sampling (₹12.13Cr). Deploy UCB if deterministic decisions required (compliance).

## Related Docs

- [thompson-sampling.md](thompson-sampling.md)
- [ucb.md](ucb.md)
- [epsilon-greedy.md](epsilon-greedy.md)
