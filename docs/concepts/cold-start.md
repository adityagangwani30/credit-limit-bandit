# Cold-Start Problem

Months 1-3 have zero reward signal (3-month delay means first feedback arrives Month 4). The bandit must act without observing outcomes.

## What Happens Months 1-3

- All actions have Beta(1,1) posteriors (uniform, uninformed)
- Sampling from Beta(1,1) produces roughly equal action selection
- Users receive random limit increases (50% keep, 16% plus_10, 17% plus_20, 17% plus_50)
- No posterior update occurs (no reward data)
- The bandit explores blindly, which is expensive

For Deep-Subprime users, testing plus_50 on 167 users (17% of 1,000) causes ~25 defaults each costing ₹2M = ₹50M cold-start cost.

## Why Context Features Help

Even with zero historical outcomes, context features (CIBIL, income, age) encode prior knowledge. Thompson uses context to assign different action eligibility:

- Prime user (CIBIL 800+): all 4 actions eligible
- Subprime user (CIBIL 600): keep + plus_10 only
- Deep-Subprime (CIBIL 300-579): keep only

Safety-score gating applies before action selection, reducing exploration damage. Deep-Subprime never tests plus_50 (eligibility gates it out).

## Beta(1,1) Prior Justification

Beta(1,1) is the maximum-entropy distribution over [0,1], expressing "complete uncertainty." It's Bayesian default for unknown Bernoulli outcomes. As evidence accumulates (successes α, failures β), posteriors concentrate. By Month 5, Beta distributions are tight (e.g., Beta(88, 12) has mean 0.88, std ~0.035), and Thompson exploits them.

## Feature Importance for Cold-Start

Ranked by impact on action eligibility:

1. **CIBIL score** (Feature 7): determines risk tier → eligibility set
2. **Utilization** (Feature 0): high utilization narrows eligible actions
3. **Payment streak** (Feature 1): poor streak gates out aggressive actions
4. **Income** (Feature 2): low income limits plus_50 eligibility
5. **Delinquency count** (Feature 3): high count restricts actions

Months 1-3 rely on these features to avoid catastrophic exploration (e.g., plus_50 for CIBIL 300).

## Related Docs

- [delayed-feedback.md](delayed-feedback.md)
- [non-stationarity.md](non-stationarity.md)
