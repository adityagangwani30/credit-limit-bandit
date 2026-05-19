# UCB1 Algorithm

Upper Confidence Bound algorithm selects action with highest upper bound: UCB = mean + √(ln(n)/count). Achieves ₹10.83Cr revenue (+24.15% lift, 25.32% regret).

## UCB1 Formula

For action a, after trying it n_a times:

UCB_a = μ_a + c × √(ln(n) / n_a)

- μ_a: average reward for action a
- n_a: count times action a selected
- n: total selections across all actions
- c: exploration constant (typically 1.0-2.0)

## How It Works

1. **Initialization:** All actions have count 0, mean 0.5 (neutral prior)
2. **Compute UCB:** For each action, compute mean + bonus √(ln(total)/count)
3. **Select action with highest UCB**
4. **Update mean:** μ_a ← (μ_a × n_a + reward) / (n_a + 1)
5. **Increment count:** n_a += 1
6. **Repeat**

The exploration bonus √(ln(n)/n_a) decays over time: less-tried actions get high bonus, forcing exploration. As n grows large (ln(n) grows slowly), bonus shrinks, shifting toward exploitation.

## Why UCB Underperforms Thompson (25.32% vs 16.34% Regret)

Thompson's posterior sampling adapts exploration to uncertainty. UCB's √(ln(n)/n_a) is a fixed function of counts, not problem difficulty.

**Example:** After 100 trials each:
- plus_10: 95 successes → mean 0.95, bonus √(6.91/100) = 0.26, UCB = 1.21
- plus_50: 68 successes → mean 0.68, bonus 0.26, UCB = 0.94

Plus_10 wins. But what if plus_50's true mean is 0.90 (we just got unlucky)? Thompson's loose Beta(68, 32) would still sample it ~40% of the time to explore. UCB's count-based bonus has decayed too much—bonus is same for both, so mean dominates.

This mismatch causes UCB to over-exploit early-lucky actions.

## Performance

| Metric | UCB | Thompson | Gap |
|--------|-----|----------|-----|
| Revenue | ₹10.83Cr | ₹12.13Cr | −₹1.30Cr |
| Regret | 25.32% | 16.34% | +8.98% |
| Convergence | Month 6 | Month 5 | +1 month |

UCB is solid but not optimal. Use when deterministic decisions required (compliance mandates non-random recommendations).

## Related Docs

- [thompson-sampling.md](thompson-sampling.md)
- [epsilon-greedy.md](epsilon-greedy.md)
