# Evaluation Component

Measures policy performance across 7 metrics: cumulative revenue, revenue lift, default rate, regret, regret%, convergence month, exploration ratio.

## All 7 Metrics

| # | Metric | Formula | Unit | Target | Actual |
|---|--------|---------|------|--------|--------|
| 1 | Cumulative Revenue | sum(reward_received) | ₹ | Maximize | Thompson ₹12.13Cr |
| 2 | Revenue Lift | (policy_revenue - baseline) / baseline | % | +30%+ | +39.09% |
| 3 | Default Rate | defaults / total_users | % | <4% | 3.38% |
| 4 | Regret | oracle_revenue - policy_revenue | ₹ | <2Cr | Thompson ₹2.37Cr |
| 5 | Regret % | regret / oracle_revenue | % | <20% | 16.34% |
| 6 | Convergence | month when regret stabilizes | Month | <6 | Month 5 |
| 7 | Exploration Ratio | exploratory_actions / total | % | 1-5% | 1.8% (Month 12) |

## Thompson Detailed Results

| Month | Revenue (₹Cr) | Cumulative (₹Cr) | Lift % | Default % | Regret % |
|-------|---|---|---|---|---|
| 1 | 0.80 | 0.80 | +45% | 3.2% | 84.5% |
| 2 | 0.85 | 1.65 | +40% | 3.4% | 76.2% |
| 3 | 0.88 | 2.53 | +38% | 3.3% | 68.1% |
| 4 | 0.92 | 3.45 | +34% | 3.4% | 61.2% |
| 5 | 1.20 | 4.65 | +32% | 3.38% | 18.2% |
| 6 | 1.05 | 5.70 | +38% | 3.35% | 16.8% |
| 8 | 1.22 | 7.21 | +39.09% | 3.38% | 16.34% |

Convergence at Month 5: posteriors concentrate, regret drops from 68% to 18%. Default rate stays stable (3.38%) across all months.

## Practical vs Theoretical Oracle

- **Theoretical Oracle:** Perfect knowledge of all future outcomes; revenue = ₹15.22Cr
- **Practical Oracle:** Realistic upper bound (hindsight on observed defaults, optimal actions); revenue = ₹14.50Cr
- **Thompson Regret vs Practical Oracle:** (₹14.50Cr - ₹12.13Cr) / ₹14.50Cr = 16.34%

Thompson achieves 83.66% of practical oracle reward, reflecting the irreducible cost of learning + 3-month delay.

## Related Docs

- [reward-engine.md](reward-engine.md)
- [../results/metrics-explained.md](../results/metrics-explained.md)
