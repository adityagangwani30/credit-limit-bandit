# Context Builder Component

Builds a 10-dimensional normalized context vector for each user monthly. All features normalized to [0,1] range using two-branch linear normalization (NOT sigmoid).

## All 10 Features

| # | Feature | Formula | Raw Range | Norm Range | Example |
|---|---------|---------|-----------|-----------|---------|
| 0 | credit_utilization | outstanding/limit | 0-∞ | [0,1] | 0.45 |
| 1 | payment_streak_norm | payment_streak/36 | 0-36 | [0,1] | 0.50 |
| 2 | income_percentile | {low:0.25, mid:0.50, high:0.75, very_high:1.0} | {0.25,0.50,0.75,1.0} | [0,1] | 0.75 |
| 3 | delinquency_score | 1-(count/5) clipped | 0-5 | [0,1] | 0.80 |
| 4 | account_age_norm | months/120 | 0-120 | [0,1] | 0.40 |
| 5 | spending_volatility | std/mean (3 months) | 0-∞ | [0,1] | 0.12 |
| 6 | transaction_freq_norm | freq/40 | 0-40 | [0,1] | 0.55 |
| 7 | cibil_norm | (score-300)/600 | 300-900 | [0,1] | 0.74 |
| 8 | months_of_history_norm | month/12 | 1-12 | [0,1] | 0.50 |
| 9 | current_limit_norm | log10(limit)/log10(500K) | 10K-500K | [0,1] | 0.62 |

**Cold start (Month 1):** All features default to normalized values from UserSchema (utilization_ratio, payment_streak, income, etc.). Context is rich even with no transaction history.

**Example Near-Prime user, Month 6:**
```python
[0.45, 0.50, 0.75, 0.80, 0.40, 0.12, 0.55, 0.74, 0.50, 0.62]  # float32
```

## Why Two-Branch Linear (Not Sigmoid)?

Sigmoid squashes: sigmoid(−3) ≈ 0.05, sigmoid(3) ≈ 0.95. Extreme credit scores (300 vs 850) get compressed. Linear preserves: (300−300)/600 = 0.0, (850−300)/600 = 0.92. Thompson Sampling needs this distinction to learn different posteriors for 300 vs 850 scores.

## Add a New Feature

1. Compute raw value for user U at month M
2. Choose min/max thresholds from observed data
3. Apply: normalized = (raw - min) / (max - min) clipped to [0,1]
4. Append to context vector (becomes 11-dimensional)
5. Update ContextBuilder.build_context() method
6. Retrain policies (previous posteriors become obsolete)

## Related Docs

- [simulator.md](simulator.md)
- [reward-engine.md](reward-engine.md)
