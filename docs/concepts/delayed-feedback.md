# Delayed Feedback: The 3-Month Lag

Credit decisions (Month 1) yield default outcomes (Month 4). This 3-month delay fundamentally shapes bandit learning.

## Why 3 Months?

- Month 1: User receives increased limit
- Month 2: User may miss a payment (not yet marked default)
- Month 3: Typically marked as delinquent after 90 days
- Month 4: Default outcome observed, reported to bureau

The system cannot know Month 1 outcomes until Month 4.

## RewardBuffer Timeline

```
Month 1        Month 2        Month 3        Month 4
Action 1    | Action 2    | Action 3    | Release 1
(pending)   | (pending)   | (pending)   | (update policy)
                                        | Action 4 starts
```

This creates overlap: at Month 4, three cohorts (Actions 2, 3, 4) are in flight while Action 1 feedback arrives.

## Cold-Start Overlap (Months 1-3)

Months 1-3 have ZERO reward signal (nothing arrives until Month 4). The bandit is purely exploratory using uninformed Beta(1,1) priors. This is the "cold-start" period.

Thompson maintains wide posteriors during Months 1-3 (all features matter equally). When Month 4 feedback arrives, posteriors immediately sharpen on actions that performed well in Month 1.

Epsilon-Greedy's epsilon decay (0.995^t) is independent of this. ε = 0.50 in Month 1, ε = 0.45 in Month 4, regardless of when feedback actually arrives. This mismatch causes lock-in to early-lucky actions.

## How Thompson Adapts

Month 1: Beta(1,1) for all actions (exploring all equally)
Month 2: Beta(1,1) still (no signal yet)
Month 3: Beta(1,1) still
Month 4: Month 1 feedback → posteriors diverge (e.g., plus_10 → Beta(95, 5), keep → Beta(88, 12))
Month 5: Policies shift based on actual Month 1 outcomes → ₹1.20Cr revenue (vs Month 4's ₹0.92Cr)

Thompson's posterior updates compensate for the 3-month lag by adapting immediately when feedback arrives.

## Related Docs

- [cold-start.md](cold-start.md)
- [non-stationarity.md](non-stationarity.md)
