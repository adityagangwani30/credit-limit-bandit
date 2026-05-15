# Credit-Limit Bandit

[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-red?logo=streamlit)](YOUR_STREAMLIT_URL)

A contextual multi-armed bandit system for optimizing credit-card limit decisions. The project simulates a credit-card portfolio, builds monthly user context vectors, learns limit-increase policies, applies delayed reward feedback, and exposes the results through notebooks and a Streamlit dashboard.

## Architecture

```text
+-------------+     +----------------+     +---------------+     +---------------+     +-------------+
|  Simulator  | --> | ContextBuilder | --> | Bandit Policy | --> | Reward Engine | --> | Evaluation  |
+-------------+     +----------------+     +---------------+     +---------------+     +-------------+
       |                    |                      |                      |                    |
       | synthetic users    | normalized features  | action selection     | delayed rewards    | metrics/dashboard
       v                    v                      v                      v                    v
```

## How It Works

Every month, the simulator creates or updates a customer's financial behavior: spend, utilization, and default outcomes. The `ContextBuilder` turns that information into a compact numeric summary. A bandit policy then decides whether to keep the limit unchanged or increase it.

Thompson Sampling works like a cautious experimenter. For each user-action pair, it keeps a belief about how good that action is. When it needs to choose, it samples from those beliefs and picks the action with the strongest sampled outcome. Good rewards strengthen the belief that an action works well. Bad rewards strengthen the belief that it does not. Over time, this naturally balances exploration and exploitation without hard-coded rules.

## Project Components

- `src/simulator.py`: Generates synthetic users and monthly outcomes.
- `src/context.py`: Builds the 10-feature context vector used by policies.
- `src/bandits/`: Thompson Sampling, UCB, and Epsilon-Greedy implementations.
- `src/reward.py`: Computes interchange revenue, default penalties, and delayed feedback.
- `src/simulate_run.py`: Runs end-to-end portfolio simulations.
- `src/evaluate.py`: Regret, cohort, shock, and policy-comparison metrics.
- `dashboard/app.py`: Four-page Streamlit dashboard for portfolio analysis.

## Getting Started

```bash
git clone https://github.com/your-username/credit-limit-bandit.git
cd credit-limit-bandit
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows PowerShell
python -m pip install -r requirements.txt
```

Generate the synthetic users:

```bash
python -m src.simulator
```

Run a short simulation during development:

```bash
python -m src.simulate_run --policy thompson
```

Run the dashboard locally:

```bash
python -m streamlit run dashboard/app.py
```

Run the test suite:

```bash
python -m pytest tests/ -v --tb=short
```

## Results

Populate the table below from `data/simulation_results.csv` after running the full 12-month simulation.

| Policy | Total Revenue (INR) | Default Rate | Regret vs Oracle | Revenue Lift vs Static |
|---|---:|---:|---:|---:|
| Thompson Sampling | TBD | TBD | TBD | TBD |
| UCB | TBD | TBD | TBD | TBD |
| Epsilon-Greedy | TBD | TBD | TBD | TBD |
| Static Baseline | TBD | TBD | TBD | 0.00% |
| Oracle | TBD | TBD | 0.00% | TBD |

## Resume Bullet Points

- Built a contextual multi-armed bandit system for dynamic credit-limit optimization across a synthetic credit-card portfolio.
- Simulated 10,000-cardholder monthly behavior with risk-aware features including CIBIL score, utilization, delinquency, tenure, and transaction frequency.
- Implemented Thompson Sampling, UCB, and Epsilon-Greedy policies with delayed reward feedback based on interchange revenue and default losses.
- Developed a Streamlit analytics dashboard with portfolio KPIs, per-user drilldowns, regret analysis, live policy simulation, and policy comparison views.
- Added production hardening with integration tests, CI, Docker packaging, and deployment guidance for Streamlit Community Cloud.

## Streamlit Community Cloud Deployment

1. Push the repository to a public GitHub repository.
2. Generate `data/synthetic_users.csv` and `data/simulation_results.csv` locally, then commit both files to the repo.
3. Go to `https://share.streamlit.io` and sign in with GitHub.
4. Create a new app and select your public repository.
5. Set the main file path to `dashboard/app.py`.
6. Set the Python version to `3.11`.
7. Deploy the app and wait for the build to finish.
8. Replace `YOUR_STREAMLIT_URL` in this README with the live app URL.

## Docker

Build and run the container locally:

```bash
docker build -t credit-limit-bandit .
docker run -p 8501:8501 credit-limit-bandit
```

## CI

GitHub Actions runs:

- `pytest tests/ -v --tb=short`
- a 3-month Thompson Sampling smoke simulation on 100 users

The workflow file lives at `.github/workflows/ci.yml`.

## Final Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Dashboard runs locally (`streamlit run dashboard/app.py`)
- [ ] Thompson Sampling shows >30% revenue lift vs static in `simulation_results.csv`
- [ ] Default rate <4%
- [ ] GitHub repo is public with clean commit history
- [ ] Streamlit Cloud URL is live and in README

## License

MIT
