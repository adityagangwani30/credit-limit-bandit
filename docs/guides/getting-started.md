---
title: Getting Started
category: guide
file_reference: none
---
# Getting Started

Prerequisites
- Python 3.11+ · git · 4GB RAM

Quick setup (copy-paste)
```bash
git clone https://github.com/your-org/credit-limit-bandit.git
cd credit-limit-bandit
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Generate synthetic data
```bash
python src/simulator.py --n_users 10000 --seed 42
```
- Output: `data/simulation_results.csv` (per-user/month rows). Validate aggregate default rate in `data/`.

Run a full simulation
```bash
python src/simulate_run.py --policy all --n_months 12
```
- Expected runtime: ~3 minutes for `n_users=10000` on a modern laptop; CPU-only.

Launch the dashboard
```bash
streamlit run dashboard/app.py
```
- First load shows portfolio overview. Use sliders on Live Simulation page to run quick scenarios.

Run tests
```bash
pytest tests/ -v
```

Common errors & fixes
- `ModuleNotFoundError: src` → run `pip install -e .` to register package in editable mode.
- `FileNotFoundError: data/simulation_results.csv` → run `python src/simulator.py` or `python src/simulate_run.py`.
- Streamlit port conflict → `streamlit run dashboard/app.py --server.port 8502`

Related docs
- [docs/guides/running-simulations.md](docs/guides/running-simulations.md)
- [docs/guides/dashboard.md](docs/guides/dashboard.md)
