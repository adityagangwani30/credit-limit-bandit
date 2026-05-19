# Data Flow and Schemas

Traces exact data shapes from user input through context, action selection, reward calculation, to evaluation metrics.

## Pipeline

User CSV → ContextBuilder → Bandit → Action → RewardEngine → RewardBuffer → Evaluation

## UserSchema (13 fields)

| Field | Type | Range |
|-------|------|-------|
| user_id | str | USER_00001 – USER_10000 |
| age_bucket | str | 18-25, 26-35, 36-50, 51+ |
| income_bucket | str | low, mid, high, very_high |
| cibil_score | int | 300–900 |
| employment_type | str | salaried, self_employed, gig, student |
| payment_streak | int | 0–36 months |
| utilization_ratio | float | 0.02–0.98 |
| spending_category | str | essentials, lifestyle, mixed |
| account_age_months | int | 1–120 |
| delinquency_count | int | 0–5 |
| transaction_frequency | int | 1–40/month |
| risk_tier | str | Prime, Near-Prime, Subprime, Deep-Subprime |
| initial_credit_limit | float | ₹10K–₹500K |

## Context Vector (10 float32)

| # | Feature | Formula | Range |
|---|---------|---------|-------|
| 0 | credit_utilization | outstanding/limit | [0,1] |
| 1 | payment_streak_norm | streak/36 | [0,1] |
| 2 | income_percentile | {low:0.25, mid:0.5, high:0.75, very_high:1.0} | [0.25,1.0] |
| 3 | delinquency_score | 1-(count/5) | [0,1] |
| 4 | account_age_norm | months/120 | [0,1] |
| 5 | spending_volatility | std/mean | [0,1] |
| 6 | transaction_freq_norm | freq/40 | [0,1] |
| 7 | cibil_norm | (score-300)/600 | [0,1] |
| 8 | months_of_history_norm | month/12 | [0,1] |
| 9 | current_limit_norm | log10(limit)/log10(500K) | [0,1] |

**Cold start (Month 1):** All features populated from UserSchema. Context rich even with no outcome history.

## Reward Calculation

net_reward = (amount_spent × 0.018) - (outstanding × did_default)

**Example 1:** Prime user ₹50K spend, no default
- Interchange: ₹50K × 0.018 = ₹900
- Default loss: ₹0 (did_default=False)
- Reward = ₹900 → Thompson β+= 1 (success)

**Example 2:** Subprime user ₹25K spend, defaults
- Interchange: ₹25K × 0.018 = ₹450
- Default loss: ₹25K × 1 = ₹25K
- Reward = ₹450 - ₹25K = −₹24,550 → Thompson β += 1 (failure)

## Normalization Constants

| Constant | Value | Purpose |
|----------|-------|---------|
| POSITIVE_MAX | ₹700,000 | Cap positive rewards |
| NEGATIVE_MIN | −₹27,000,000 | Cap negative rewards |

Maps ₹700K → 1.0, ₹0 → 0.5, −₹27M → 0.0 (two-branch linear).

## RewardBuffer: Month-by-Month

| Month | Stored | Pending | Released |
|-------|--------|---------|----------|
| 1 | 10K actions | Pending | None |
| 2 | 10K actions | Pending (20K total) | None |
| 3 | 10K actions | Pending (30K total) | None |
| 4 | 10K actions | Pending (40K total) | Month 1 outcomes (10K) |
| 5 | 10K actions | Pending (40K total) | Month 2 outcomes (10K) |

Buffer releases (user_id, action_month) ← (user_id, outcome_month) when month = action_month + 3.

## Related Docs

- [overview.md](overview.md)
- [design-decisions.md](design-decisions.md)
