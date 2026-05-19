# Simulator Component

Generates 10,000 synthetic users across 4 risk tiers: Prime (40%), Near-Prime (30%), Subprime (20%), Deep-Subprime (10%). All 12 UserSchema fields with realistic ranges and correlation.

## All 12 UserSchema Fields

| Field | Type | Range | Example |
|-------|------|-------|---------|
| user_id | str | USER_00001 to USER_10000 | USER_04321 |
| age_bucket | str | 18-25, 26-35, 36-50, 51+ | 36-50 |
| income_bucket | str | low, mid, high, very_high | high |
| cibil_score | int | 300-900 (+ income drift) | 742 |
| employment_type | str | salaried, self_employed, gig, student | salaried |
| payment_streak | int | 0-36 months | 18 |
| utilization_ratio | float | 0.02-0.98 (Beta distribution) | 0.65 |
| spending_category | str | essentials, lifestyle, mixed | mixed |
| account_age_months | int | 1-120 (1-10 years) | 48 |
| delinquency_count | int | 0-5 (previous defaults) | 1 |
| transaction_frequency | int | 1-40 monthly | 22 |
| risk_tier | str | Prime, Near-Prime, Subprime, Deep-Subprime | Near-Prime |
| initial_credit_limit | float | ₹10K-₹500K (tier-dependent) | ₹145,230 |

## Default Probability Formula

For month M, user U, utilization U%, default probability:

P(default) = base_rate × utilization_multiplier × stress_multiplier × delinquency_penalty × payment_streak_discount

**Worked Example: Subprime user at 80% utilization**
- base_rate = 0.06 (6% for Subprime)
- utilization_multiplier = 0.81 + 0.20×(0.80) + 0.08×(0.65) = 1.21
- stress_multiplier = 1.0 (normal month; 2.0 if economic shock)
- delinquency_penalty = 1.0 + 0.02×1 = 1.02 (1 prior default)
- payment_streak_discount = 1.0 - 0.002×8 = 0.98 (8-month payment streak)
- P(default) = 0.06 × 1.21 × 1.0 × 1.02 × 0.98 = **0.0722** (7.22%)

This 7.22% default probability drives reward outcomes. With ₹30K spent: reward = (₹30K × 0.018) - (₹30K × 0.0722) = ₹540 - ₹2,166 = **−₹1,626** (expected loss if action taken).

## Regenerate Users

```python
from src.simulator import generate_users
users_df = generate_users(n=10000, seed=42)
users_df.to_csv("data/synthetic_users.csv", index=False)
```

## Related Docs

- [context-builder.md](context-builder.md)
- [reward-engine.md](../components/reward-engine.md)
