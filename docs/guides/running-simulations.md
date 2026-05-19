# Running Simulations

CLI arguments for `simulate_run.py`. All parameters with types/defaults. Example commands. Output schema. Expected runtimes.

## CLI Arguments

```
python src/simulate_run.py [OPTIONS]

Options:
  --policy POLICY               Algorithm: thompson_sampling, ucb, epsilon_greedy, static
  --n_users INT                 Number of users to simulate (default: 1000, max 10000)
  --n_months INT                Simulation horizon in months (default: 12, range 1-12)
  --seed INT                    Random seed (default: 42)
  --output_path PATH            Results CSV location (default: data/simulation_results.csv)
  --economic_stress FLOAT       Stress multiplier 1.0 (normal), 2.0 (shock) (default: 1.0)
  --help                        Show this help
```

## Example Commands

### Quick Dev Run (5 seconds)
```bash
python src/simulate_run.py --policy thompson_sampling --n_users 100 --n_months 1
# Output: 100 users × 1 month = 100 rows, 5 seconds
```

### Full Run (Production Baseline)
```bash
python src/simulate_run.py --policy thompson_sampling --n_users 10000 --n_months 12 --seed 42
# Output: 10,000 users × 12 months = 120,000 rows, ~45 seconds
```

### Stress Test (Economic Shock)
```bash
python src/simulate_run.py --policy thompson_sampling --n_users 5000 --n_months 12 --economic_stress 2.0
# Default rates double at month 6; tests adaptation
```

### Compare All Policies
```bash
for policy in thompson_sampling ucb epsilon_greedy static; do
  python src/simulate_run.py --policy $policy --n_users 1000 --n_months 12
done
# Generates 4 CSV outputs for comparison
```

## Output Schema

CSV columns:
- user_id: USER_00001 format
- month: 1-12
- action_taken: keep / plus_10 / plus_20 / plus_50
- policy: thompson_sampling / ucb / epsilon_greedy / static
- reward_received: ₹ amount (−27M to +700K)
- did_default: True/False
- outstanding_amount: ₹ amount user owed
- risk_tier, income_bucket, account_age_bucket: cohort fields

## Runtime Table

| Users | Months | Runtime | Rows | Notes |
|-------|--------|---------|------|-------|
| 100 | 1 | <1s | 100 | Dev |
| 1,000 | 12 | ~5s | 12K | Dev baseline |
| 5,000 | 12 | ~20s | 60K | Cohort testing |
| 10,000 | 12 | ~45s | 120K | Production |

## Related Docs

- [getting-started.md](getting-started.md)
- [dashboard.md](dashboard.md)
