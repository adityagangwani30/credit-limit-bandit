# Reward Engine Component

Computes monthly rewards from spending and defaults. Positive: 1.8% interchange fee on spend (max ₹700,000). Negative: full outstanding if default (min −₹27,000,000).

## Reward Formula

net_reward = (amount_spent × 0.018) - (outstanding × did_default)

## Worked Example 1: Prime user, no default

- amount_spent = ₹50,000
- outstanding_amount = ₹8,000 (16% of spend)
- did_default = False
- reward = (₹50,000 × 0.018) - (₹8,000 × 0) = **₹900**

This reward updates Thompson Beta: α += 1 (success). Repeated 95 times with similar outcomes + 5 defaults gives Beta(96, 6).

## Worked Example 2: Subprime user, defaults

- amount_spent = ₹25,000
- outstanding_amount = ₹25,000 (100% of spend, didn't pay anything back)
- did_default = True
- reward = (₹25,000 × 0.018) - (₹25,000 × 1) = ₹450 - ₹25,000 = **−₹24,550**

This updates Thompson Beta: β += 1 (failure). Subprime cohort shows higher failure rate (25-30% default), leading to tight posteriors favoring "keep" action over "plus_10"/"plus_20".

## Normalization Constants

Two-branch linear normalization for reward before Beta update:

| Constant | Value | Purpose |
|----------|-------|---------|
| POSITIVE_MAX | ₹700,000 | Cap on positive rewards (annual interchange limit) |
| NEGATIVE_MIN | −₹27,000,000 | Cap on negative rewards (largest bulk default loss) |

Normalization: if reward > 0, map to (0.5, 1.0]; if < 0, map to [0.0, 0.5). Maps ₹700K → 1.0, ₹0 → 0.5, −₹27M → 0.0.

## RewardBuffer: 3-Month Delay Mechanics

Buffer holds actions for 3 months before releasing for policy updates.

| Month | Action Taken | Stored State | Released from Buffer |
|-------|--------------|--------------|----------------------|
| 1 | 10,000 actions | Pending | None (month < 4) |
| 2 | 10,000 actions | Pending | None |
| 3 | 10,000 actions | Pending | None |
| 4 | 10,000 actions | Pending | Month 1 outcomes (10,000) → update |
| 5 | 10,000 actions | Pending | Month 2 outcomes (10,000) → update |

The buffer matches (user_id, action_month) to (user_id, outcome_month) and releases at month = action_month + 3.

## Related Docs

- [simulator.md](simulator.md)
- [context-builder.md](context-builder.md)
- [evaluation.md](evaluation.md)
