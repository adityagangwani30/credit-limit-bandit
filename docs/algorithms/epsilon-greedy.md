# Epsilon-Greedy Algorithm

Epsilon-Greedy explores with probability ε, exploits with probability (1−ε). With decay schedule ε(t) = 0.15 × 0.995^t, it reaches ₹8.70Cr revenue (−0.26% below baseline ₹8.72Cr, 40.01% regret) by locking into early-lucky actions before the 3-month reward signal arrives.

## Decay Schedule

| Month | Epsilon | Exploration % | Problem |
|-------|---------|---------------|---------|
| 1 | 0.1500 | 15% | Too much exploration (Month 1 feedback unknown) |
| 4 | 0.1050 | 10.5% | First feedback arrives, but ε already decayed |
| 5 | 0.1045 | 10.45% | Locked into early-lucky action; can't correct |
| 8 | 0.1040 | 10.4% | Stuck on suboptimal action for final 4 months |

**Why it fails:** Month 1-3 have NO reward signal (3-month delay). Random exploration wastes resources on all actions equally. When Month 4 feedback arrives, ε has already decayed to 10.5%, so the greedy action dominates. If early randomness favored "keep" (which happened in simulation), the algorithm locks there permanently.

Compare Thompson: Beta(1,1) uninformed prior + posterior updates from Month 4 feedback = aggressive shift toward high-value actions. By Month 5, Thompson has regret 16.34%; Epsilon-Greedy regret 40.01%.

## When to Use

- Development/baseline only
- Not suitable for production credit decisions

## Related Docs

- [thompson-sampling.md](thompson-sampling.md)
- [comparison.md](comparison.md)
