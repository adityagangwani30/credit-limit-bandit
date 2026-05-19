# Thompson Sampling Algorithm

Thompson Sampling maintains Beta(α,β) posteriors for each action. At decision time, sample θ from posteriors and select action with highest sample. Achieves ₹12.13Cr revenue (+39.09% lift, 16.34% regret).

## Beta Posterior Model

Each action tracks two parameters:
- **α:** Count of successes (no default, interchange earned)
- **β:** Count of failures (default, no reward)

**Initial:** Beta(1,1) uniform prior (maximum entropy, express complete uncertainty)

**After first feedback (Month 4):** Actions diverge. Example:
- plus_10: Beta(95, 5) — 95 successes, 5 failures, mean 0.95
- keep: Beta(88, 12) — 88 successes, 12 failures, mean 0.88
- plus_50: Beta(68, 32) — 68 successes, 32 failures, mean 0.68

## How It Works

1. **Sampling:** For each action, sample θ ~ Beta(α, β)
2. **Selection:** Choose action with highest θ
3. **Execute:** Apply limit increase, record action
4. **Wait:** 3-month observation window
5. **Update:** When outcome arrives, increment α (success) or β (failure)

Example Month 5 decision:
- Plus_10 sample: θ1 ~ Beta(95, 5) → draw 0.93
- Keep sample: θ2 ~ Beta(88, 12) → draw 0.87
- Plus_20 sample: θ3 ~ Beta(72, 28) → draw 0.71
- Select plus_10 (highest sample 0.93)

## Why Thompson Outperforms

Thompson's 39.09% lift vs Epsilon's −0.26% comes from:

1. **Adaptive exploration:** Actions with high-uncertainty posteriors (loose Beta distributions) get sampled more. Thompson auto-focuses exploration where uncertainty remains.

2. **Faster convergence:** By Month 5, tight posteriors → exploitation. Epsilon-Greedy's ε = 0.26 at Month 5 wastes 26% on random exploration despite having converged.

3. **Delayed feedback handling:** Months 1-3 Beta(1,1) priors ensure balanced exploration. Month 4 feedback immediately tightens posteriors. Epsilon-Greedy's decay doesn't adapt to when information arrives.

## Performance

| Metric | Thompson | UCB | Epsilon | Baseline |
|--------|----------|-----|---------|----------|
| Revenue | ₹12.13Cr | ₹10.83Cr | ₹8.70Cr | ₹8.72Cr |
| Lift | +39.09% | +24.15% | −0.26% | 0% |
| Regret | 16.34% | 25.32% | 40.01% | 39.80% |
| Convergence | Month 5 | Month 6 | Month 7 | N/A |

## Related Docs

- [ucb.md](ucb.md)
- [epsilon-greedy.md](epsilon-greedy.md)
- [comparison.md](comparison.md)
