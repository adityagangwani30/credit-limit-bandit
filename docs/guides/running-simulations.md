---
title: Running Simulations
category: guide
file_reference: src/simulate_run.py
---
# Running Simulations

CLI arguments for `src/simulate_run.py`:
- `--policy` [thompson|ucb|epsilon_greedy|static|oracle|all] (default: all)
- `--n_months` [int] (default: 12)
- `--n_users` [int] (default: 10000)
- `--seed` [int] (default: 42)
- `--economic_shock` [bool] (default: True)
- `--output` [path] (default: data/simulation_results.csv)

Example commands
- Quick dev run:
```bash
python src/simulate_run.py --policy thompson --n_months 3 --n_users 1000
```
- Full production run:
```bash
python src/simulate_run.py --policy all --n_months 12
```
- Stress test with shock:
```bash
python src/simulate_run.py --policy all --economic_shock True
```

Output schema (simulation_results.csv)
| Column | Type | Description |
| --- | --- | --- |
| user_id | int | unique user id |
| month | int | simulation month (1..n_months) |
| action | int | action code selected |
| spend | float | observed spend in month |
| outstanding | float | outstanding balance end of month |
| default | bool | whether user defaulted |
| reward | float | net reward computed (may be buffered) |

Expected runtimes
- `n_users=1000, n_months=3`: < 30s
- `n_users=10000, n_months=12`: ~3 minutes (single-threaded CPU)

Tips
- For iteration, run with `--n_users 1000` and check behavior before scaling up.
- Use `--seed` for reproducibility across runs.

Related docs
- [docs/components/simulator.md](docs/components/simulator.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
