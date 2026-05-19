# Metrics Explained

All 7 metrics: definition, formula, unit, interpretation, target, actual result, what bad means.

## Metric 1: Cumulative Revenue

**Definition:** Total ₹ earned across all users, all months.

**Formula:** Σ(reward_received) for all user-month pairs

**Unit:** ₹ Crores

**Interpretation:** Higher is better. Thompson ₹12.13Cr generates 39% more revenue than Static ₹8.72Cr.

**Target:** >₹10Cr (exceed static by 15%+)

**Actual:** Thompson ₹12.13Cr ✓

**What bad means:** <₹8.72Cr indicates worse than doing nothing.

## Metric 2: Revenue Lift

**Definition:** Improvement over static baseline.

**Formula:** (policy_revenue - baseline_revenue) / baseline_revenue × 100%

**Unit:** %

**Interpretation:** +39.09% means Thompson generates 39% more revenue per user than static policy.

**Target:** >+30%

**Actual:** Thompson +39.09% ✓, UCB +24.15%, Epsilon −0.26%

**What bad means:** Negative lift = worse than baseline.

## Metric 3: Default Rate

**Definition:** Fraction of users who defaulted.

**Formula:** count(did_default==True) / total_users

**Unit:** %

**Interpretation:** All policies maintained 3.38% default rate. This is a constraint (not optimized), not an objective.

**Target:** <3.5% (below portfolio risk tolerance)

**Actual:** All policies 3.38% ✓

**What bad means:** >4% indicates policy is too aggressive (excess default risk).

## Metric 4: Regret

**Definition:** Revenue gap vs oracle.

**Formula:** oracle_revenue - policy_revenue

**Unit:** ₹ Crores

**Interpretation:** Oracle (perfect foresight) ₹14.50Cr. Thompson loses ₹2.37Cr due to learning + delay.

**Target:** <₹3Cr (20% regret)

**Actual:** Thompson ₹2.37Cr ✓

**What bad means:** >₹5Cr indicates poor exploration strategy.

## Metric 5: Regret %

**Definition:** Regret as % of oracle revenue.

**Formula:** (regret / oracle_revenue) × 100%

**Unit:** %

**Interpretation:** Thompson achieves 83.66% of oracle performance (16.34% regret gap).

**Target:** <20%

**Actual:** Thompson 16.34% ✓, UCB 25.32%, Epsilon 40.01%

**What bad means:** >40% indicates policy is far from optimal.

## Metric 6: Convergence Month

**Definition:** Month when regret stabilizes (stops improving).

**Formula:** First month where regret doesn't decrease by >1% from prior month

**Unit:** Month (1-12)

**Interpretation:** Thompson converges by Month 5 (posteriors tight, stops learning). Static/Oracle don't converge (no learning).

**Target:** <6

**Actual:** Thompson Month 5 ✓, UCB Month 6, Epsilon Month 7

**What bad means:** No convergence by Month 8 indicates learning is inefficient.

## Metric 7: Exploration Ratio

**Definition:** % of actions that were exploratory (random / sampled from wide posteriors).

**Formula:** count(exploratory_actions) / total_actions

**Unit:** %

**Interpretation:** Thompson Month 12: 1.8% exploration (tight posteriors, exploiting known actions). Epsilon-Greedy Month 12: 10% (ε = 0.10, still exploring).

**Target:** 1-5% by convergence

**Actual:** Thompson 1.8% ✓, UCB 2.1%, Epsilon 10% (never converges)

**What bad means:** >10% at Month 12 = still exploring when you should be exploiting.

## Practical vs Theoretical Oracle Distinction

**Theoretical Oracle:** Knows all future outcomes (defaults, economic shocks) at Month 1. Can make perfect decisions hindsight.

**Practical Oracle:** Only knows actual outcomes, same as policies, but can select optimal action for each (user, month) pair post-hoc.

Thompson regret 16.34% is vs Practical Oracle (realistic upper bound), not Theoretical Oracle (impossible baseline). This makes the 16.34% gap interpretable as the cost of online learning + 3-month delay.

## Related Docs

- [evaluation.md](../components/evaluation.md)
- [interpreting-results.md](interpreting-results.md)
