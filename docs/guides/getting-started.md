# Getting Started

Prerequisites & installation for Mac/Linux/Windows. Verify with pytest. Run first simulation in 2 commands.

## Prerequisites

- Python 3.10+
- pip (or conda)
- ~2GB disk (for synthetic_users.csv, notebooks)
- 10 minutes setup

## Install (All Platforms)

```bash
# Clone repo
git clone <repo_url>
cd credit-limit-bandit

# Create venv
python -m venv venv

# Activate
# Mac/Linux:
source venv/bin/activate
# Windows:
venv\Scripts\Activate

# Install
pip install -r requirements.txt
pip install -e .
```

## Verify Install

```bash
pytest tests/ -v
# Should pass all 22 tests
```

## Quick Start: 2 Commands

```bash
# 1. Generate synthetic users
python -c "from src.simulator import generate_users; df = generate_users(1000); df.to_csv('data/synthetic_users.csv', index=False)"

# 2. Run 12-month simulation
python src/simulate_run.py --policy thompson_sampling --n_months 12 --n_users 1000
# Output: simulation_results.csv
```

## Common Errors & Fixes

### ModuleNotFoundError: No module named 'src'

**Cause:** PYTHONPATH doesn't include project root.

**Fix:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Mac/Linux
set PYTHONPATH=%PYTHONPATH%;%CD%          # Windows
```

### FileNotFoundError: data/synthetic_users.csv not found

**Cause:** Users CSV doesn't exist yet.

**Fix:** Run generate_users (Quick Start step 1).

### Port already in use (Streamlit)

**Cause:** Port 8501 occupied.

**Fix:**
```bash
streamlit run dashboard/app.py --server.port 8502
```

## Related Docs

- [running-simulations.md](running-simulations.md)
- [dashboard.md](dashboard.md)
