# 💳 Credit Limit Bandit

**Contextual Multi-Armed Bandit for Dynamic Credit Limit Optimization**

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![CI](https://github.com/adityagangwani30/credit-limit-bandit/actions/workflows/ci.yml/badge.svg) ![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit) ![Code style](https://img.shields.io/badge/code%20style-black-000000) ![Domain](https://img.shields.io/badge/domain-Fintech%20%2F%20RL-blueviolet) ![Status](https://img.shields.io/badge/status-production--ready-success)

<div align="center">
  <h3>Dynamic credit limit optimization using Reinforcement Learning</h3>
  <p>
    Thompson Sampling · UCB · Epsilon-Greedy · 10,000 users · 12-month
    simulation · 3-month delayed feedback · Streamlit dashboard
  </p>
</div>

## 📋 Table of Contents

- [Live Demo](#-live-demo)
- [Problem Statement](#problem-statement)
- [How It Works](#how-it-works)
- [Results](#results)
- [Production Challenges](#production-challenges--how-we-handled-them)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Running Tests](#running-tests)
- [Dashboard Guide](#dashboard-guide)
- [Tech Stack](#tech-stack)
- [Bandit Algorithms](#bandit-algorithms-implemented)
- [Key ML Concepts](#key-ml-concepts)
- [Known Limitations](#known-limitations--future-work)
- [Project Highlights](#project-highlights-for-resume)

## 🚀 Live Demo

🚀 [Launch Streamlit App](https://credit-limit-banditgit.streamlit.app/)

Explore real-time credit limit recommendations across 10,000 simulated users
with interactive policy comparisons, regret analysis, and performance
dashboards.

> **For recruiters:** The 2-minute walkthrough below covers the live
> simulation controls, policy comparison curves, and the per-user deep dive.
> [Add Loom/YouTube link once recorded]

**What you can do in the live app:**

| Page | What it shows |
|---|---|
| 💼 Portfolio Overview | Revenue curves for all 5 policies, default rate by tier, action distribution over time |
| 👤 Per-User Deep Dive | Any user's 12-month credit limit trajectory, monthly reward, and bandit decision history |
| 📊 Policy Comparison | Side-by-side regret curves, revenue lift vs static, full metrics table |
| ⚡ Live Simulation | Re-run Thompson Sampling with custom economic stress, fraud spike, and risk distribution |

## ❓ Problem Statement

Every major Indian fintech—Cred, Slice, Jupiter, HDFC, ICICI, and others—
manually reviews credit limits every 6 months. This creates a fundamental
business challenge: **set limits too low and you lose interchange revenue;
set them too high and defaults spike.**

Today's approach is reactive and static. Credit decisions are reviewed on a
fixed calendar, heuristics are applied inconsistently, and there is no
systematic way to learn from outcomes. The result is money left on the table
and unnecessary risk exposure.

**Credit Limit Bandit automates this decision using Reinforcement Learning.**
It continuously learns from user behavior—spending patterns, repayment
history, utilization trends—and recommends personalized limit increases in
real time. By framing credit limit optimization as a contextual multi-armed
bandit problem, we balance exploration (testing higher limits to find the
sweet spot) against exploitation (sticking with what we know works).

### Why Multi-Armed Bandit - not a classifier or rule engine?

| Approach | Why it falls short |
|---|---|
| Rule-based (if CIBIL > 750, increase limit) | Static thresholds can't adapt to changing user behaviour or macroeconomic conditions |
| Supervised classifier (predict default probability) | Requires labeled training data; doesn't optimize for revenue - only for default prediction |
| A/B test | Requires a fixed control group; can't personalize per user; slow to adapt |
| **Contextual Bandit (this project)** | **Learns per-user, adapts continuously, optimizes directly for the revenue-minus-default objective** |

The bandit framing is natural here because:

- Each user is a decision point visited monthly
- Actions have delayed, noisy rewards (3-month lag)
- The optimal action differs by user context (CIBIL, income, utilization)
- We want to maximize cumulative portfolio revenue, not just predict a
  label

## ⚙️ How It Works

```
User Context (CIBIL, Utilization, Tenure...)
            ↓
    [Bandit Policy: Thompson Sampling]
            ↓
    Credit Limit Action (Increase / Hold)
            ↓
    Reward Signal (Revenue + Default Loss)
            ↓
    Policy Update (Bayesian Belief Refinement)
```

Every month, we observe a user's financial snapshot and feed it to a bandit
policy. The policy samples from its learned beliefs about which limit would
generate the best outcome, then makes a decision. After a 3-month lag
(reflecting real-world delayed feedback), we observe whether the user defaulted
and calculate the true reward. That reward updates the policy's belief,
making it smarter over time.

### The 4 credit limit actions

| Action | Effect | When the bandit chooses it |
|---|---|---|
| `keep` | No change to limit | User has high utilization or recent default signals |
| `plus_10` | Limit × 1.10 | Moderate confidence - user shows steady repayment |
| `plus_20` | Limit × 1.20 | High confidence - strong CIBIL, low utilization |
| `plus_50` | Limit × 1.50 | Very high confidence - Prime user, long history |

### The 10 context features

| Feature | What it captures | Range |
|---|---|---|
| `credit_utilization` | Current spend / limit | 0-1 |
| `payment_streak_norm` | Consecutive on-time payments | 0-1 |
| `income_percentile` | Income bucket rank | 0.25-1.0 |
| `delinquency_score` | Inverse of delinquency count | 0-1 |
| `account_age_norm` | Account tenure | 0-1 |
| `spending_volatility` | Std / mean of last 3 months spend | 0-1 |
| `transaction_freq_norm` | Monthly transaction frequency | 0-1 |
| `cibil_norm` | Normalized CIBIL score (300-900) | 0-1 |
| `months_of_history_norm` | How much history the bandit has | 0-1 |
| `current_limit_norm` | log10(limit) / log10(500000) | 0-1 |

### The reward function

```text
net_reward = (amount_spent * 0.018) - (outstanding_amount * did_default)

Example - Prime user, INR 50,000 spend, no default:
  reward = 50,000 * 0.018 = INR 900

Example - Subprime user, INR 30,000 outstanding, defaults:
  reward = (30,000 * 0.018) - 30,000 = INR -29,460
```

The 1.8% rate reflects standard interchange fees in Indian fintech. Rewards
are delayed 3 months - action at month T updates the bandit at month T+3.

## 📈 Results

All results are from a single reproducible run:
`python src/simulate_run.py --policy all --n_months 12 --seed 42`

The simulation generates 10,000 synthetic users, runs all 5 policies in
parallel for 12 months, applies an economic shock at month 6 (default rates
double), and evaluates each policy against both a practical oracle and a
theoretical oracle.

### Policy Comparison (12-month simulation, 10,000 users)

| Policy | Revenue (INR) | Lift vs Static | Default Rate | Regret vs Practical Oracle | Convergence | Exploration |
|---|---:|---:|---:|---:|---:|---:|
| Thompson Sampling | ₹12.13Cr | +39.09% | 3.38% | 16.34% | Month 9 | 37.61% |
| UCB | ₹10.83Cr | +24.15% | 3.38% | 25.32% | Month 10 | 43.95% |
| Epsilon-Greedy | ₹8.70Cr | −0.26% | 3.38% | 40.01% | Month 4 | 6.44% |
| Static Baseline | ₹8.72Cr | 0% | 3.38% | 39.85% | N/A | 0% |
| Practical Oracle | ₹14.50Cr | +66.25% | 3.38% | 0% | N/A | N/A |
| Theoretical Oracle (Reference) | ₹29.40Cr | +237.05% | 3.38% | N/A | N/A | N/A |

> Regret measured vs **practical oracle** (best fixed action per user for the
> full 12 months - a realistic upper bound). Theoretical per-month hindsight
> oracle = ₹29.40Cr (shown for completeness but not a meaningful online
> policy target).

> Thompson Sampling achieves **16.34% regret vs the practical oracle** in
> month 12. Since the practical oracle has perfect 12-month hindsight per
> user, a sub-20% gap for an online policy operating under 3-month delayed
> feedback and cold-start constraints represents strong performance. Regret
> is still declining at month 12, suggesting further improvement with a
> longer horizon.

> Epsilon-Greedy underperforms static (−0.26%) because random exploration
> increases limits for high-risk users without learned discrimination,
> generating defaults that outweigh the interchange revenue gained.

### Target Check

| Target | Latest Result | Status |
|---|---:|:---:|
| Revenue lift >30% | +39.09% | ✓ |
| Default rate <4% | 3.38% | ✓ |
| Exploration 10–25% | 37.61% | ✗ |
| Convergence M3–5 | Month 9 | ✗ |

### Cohort Analysis - Which users benefit most?

| Risk Tier | Revenue (Thompson) | Lift vs Static | Default Rate |
|---|---:|---:|---:|
| Prime | ₹13.88Cr | +30.0% | 0.39% |
| Near-Prime | ₹74.4L | +45.9% | 1.85% |
| Subprime | ₹−1.34Cr | −2.0% | 5.87% |
| Deep-Subprime | ₹−1.15Cr | +0.0% | 15.32% |

> **Key insight:** Prime users show the highest absolute revenue (₹13.88Cr,
> +30.0% lift) because their pristine credit history (0.39% default rate)
> lets the bandit confidently increase limits with minimal downside risk.
> Near-Prime users show the strongest proportional lift (+45.9%), the sweet
> spot where sufficient repayment history meets unused credit capacity.
> Subprime and Deep-Subprime tiers generate net losses - default penalties
> outweigh interchange revenue, and the bandit correctly learns conservatism
> for these users within the first few months.

## 🛠️ Production Challenges & How We Handled Them

| Challenge | Simulated As | Result |
|---|---|---|
| **Delayed reward feedback** | Reward for month T arrives at T+3 | Bandit stays uncertain (wide Beta) during lag; adapts correctly once signal arrives |
| **Cold start - new users** | First 3 months: no reward history | Months 4-6 show higher reward than months 7-12 (−49.1%) because the month-6 economic shock suppresses revenue in months 7-9. Excluding the shock window, the underlying trend is positive. |
| **Non-stationarity** | Economic shock doubles default rates at month 6 | Default rate remained within 0.4% of pre-shock baseline (3.28% → 3.67% at month 12). Thompson's conservative prior (Beta(1,1)) limited aggressive increases to risky users. |
| **Reward sparsity** | Defaults are 3.38% of events | Sigmoid normalization prevents rare large penalties from dominating Beta updates |

## 🏗️ System Architecture

```
User Simulator -> Context Builder -> Bandit Policy
     |                 |                |
     v                 v                v
  Input data      10-feature vector   Action choice
                 |
                 v
            Reward Engine -> Evaluation Module
```

**Component Breakdown:**

- **User Simulator**: Generates synthetic 10,000-user portfolio with dynamic
  spend, utilization, and default behavior.
- **Context Builder**: Extracts and normalizes 10 financial features into a
  compact state representation.
- **Bandit Policy**: Learns from feedback and selects optimal limit actions
  (Thompson Sampling, UCB, Epsilon-Greedy).
- **Reward Engine**: Computes monthly interchange revenue and default
  penalties; applies 3-month feedback delay.
- **Evaluation Module**: Computes cumulative regret, revenue lift, policy
  comparisons, and cohort analysis.

## 📁 Project Structure

```text
credit-limit-bandit/
│
├── src/                          # Core library
│   ├── simulator.py              # Synthetic user generator
│   ├── context.py                # 10-dim context vector
│   ├── reward.py                 # Reward engine + buffer
│   ├── reward_constants.py       # Shared normalization helpers
│   ├── evaluate.py               # Metrics and lift analysis
│   ├── simulate_run.py           # Main simulation loop
│   └── bandits/
│       ├── base.py               # Bandit interface
│       ├── thompson.py          # Thompson Sampling
│       ├── ucb.py                # UCB1
│       └── epsilon_greedy.py     # Epsilon-Greedy decay
│
├── dashboard/
│   └── app.py                    # Streamlit app
│
├── tests/
│   ├── conftest.py               # Shared fixtures
│   ├── test_context.py           # Context vector tests
│   ├── test_reward.py            # Reward engine tests
│   └── test_integration.py       # End-to-end tests
│
├── notebooks/
│   ├── 01_simulation_eda.ipynb   # Synthetic data validation
│   ├── 02_bandit_training.ipynb  # Learning curves
│   └── 03_evaluation.ipynb       # Regret analysis
│
├── docs/                         # Technical docs
│   ├── architecture/             # Design and data flow
│   ├── algorithms/               # Bandit deep dives
│   ├── components/               # Module docs
│   ├── concepts/                 # Core MAB concepts
│   ├── guides/                   # Setup and usage
│   └── results/                  # Metrics and interpretation
│
├── data/
│   ├── synthetic_users.csv       # 10,000 generated users
│   ├── simulation_results.csv    # All policy results
│   ├── cohort_results.csv        # Thompson cohort breakdown
│   └── cohort_by_tier.csv        # Results by risk tier
│
├── Dockerfile                    # Reproducible container
├── pyproject.toml                # Package config
├── requirements.txt              # Dependencies
└── .github/workflows/ci.yml      # CI pipeline
```

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Git
- 4 GB RAM minimum (8 GB recommended for full 10k user run)
- ~200 MB disk space for simulation outputs

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/adityagangwani30/credit-limit-bandit
cd credit-limit-bandit
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.venv\Scripts\Activate.ps1
```

**3. Install the package**

```bash
pip install -e .
pip install -r requirements.txt
```

**4. Verify installation**

```bash
pytest tests/ -v
# Expected: 22 passed
```

### Quick start (3 commands)

```bash
python src/simulate_run.py
streamlit run dashboard/app.py
```

## 🛣️ Running the Full Pipeline

The project has 4 runnable stages. Run them in order on a fresh clone.

### Stage 1 - Generate synthetic users

```bash
python src/simulator.py
# Output: data/synthetic_users.csv (10,000 users)
# Runtime: ~5 seconds
```

### Stage 2 - Run the simulation

```bash
# Full run - all 5 policies, 12 months (recommended)
python src/simulate_run.py --policy all --n_months 12 --seed 42

# Quick dev run - Thompson only, 3 months, 1,000 users
python src/simulate_run.py --policy thompson_sampling \
  --n_months 3 --seed 42

# With economic shock disabled
python src/simulate_run.py --policy all --economic_shock False
```

Available `--policy` values:
`thompson_sampling` · `ucb_bandit` · `epsilon_greedy_bandit` ·
`static_baseline` · `oracle_practical` · `all`

Expected runtime by configuration:

| Config | Users | Runtime |
|---|---:|---:|
| `--policy thompson_sampling --n_months 3` | 10,000 | ~1 min |
| `--policy all --n_months 12` | 10,000 | ~5-8 min |

### Stage 3 - Explore notebooks

```bash
jupyter notebook
# Open notebooks in order:
# 01_simulation_eda.ipynb    -> validate synthetic data
# 02_bandit_training.ipynb   -> learning curves and comparisons
# 03_evaluation.ipynb        -> regret and cohort analysis
```

### Stage 4 - Launch the dashboard

```bash
streamlit run dashboard/app.py
# Opens at http://localhost:8501
```

### Using Docker

```bash
docker build -t credit-limit-bandit .
docker run -p 8501:8501 credit-limit-bandit
# Dashboard available at http://localhost:8501
```

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected: 22 passed.

## 🖥️ Dashboard Guide

The Streamlit dashboard has 4 pages accessible from the left sidebar.

### 💼 Page 1 - Portfolio Overview

The main analytics page. Shows:

- **4 metric cards**: Total revenue (Thompson), revenue lift vs static,
  portfolio default rate, convergence month
- **Cumulative revenue chart**: All 5 policies over 12 months with a
  vertical marker at the month-6 economic shock
- **Action distribution**: How often each action (keep/+10/+20/+50) was
  taken each month
- **Default rate by risk tier**: Prime through Deep-Subprime

### 👤 Page 2 - Per-User Deep Dive

Inspect any individual user's journey. Enter a user ID or click **Random
User** to:

- See the user's risk tier, CIBIL score, income bucket, and initial credit
  limit
- View their **credit limit trajectory** over 12 months with action markers
- See **monthly reward** (positive = revenue, negative = default)
- Browse the **month-by-month decision table** showing exactly what the
  bandit chose and why

### 📊 Page 3 - Policy Comparison

Side-by-side comparison of all policies:

- **Reward curves**: Cumulative revenue, all 5 policies
- **Regret curves**: Thompson, UCB, Epsilon-Greedy vs practical oracle -
  shows learning speed visually
- **Summary table**: All 7 metrics for all policies with best-value
  highlighting

### ⚡ Page 4 - Live Simulation

Re-run the simulation with custom parameters:

- **Economic stress** (1.0-3.0): multiplier on default rates
- **Fraud spike toggle**: doubles Deep-Subprime defaults at month 9
- **Risk distribution sliders**: adjust % of each tier
- **Horizon**: 6, 9, or 12 months

Click **▶ Run Simulation** to see results update in real time.

## 🧰 Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Simulation & Data Processing | NumPy, Pandas | Synthetic portfolio generation, feature engineering |
| Algorithms | Pure Python | Thompson Sampling, UCB, Epsilon-Greedy implementations |
| Evaluation & Metrics | Scikit-learn, SciPy | Statistical testing, regret computation |
| Dashboard & UI | Streamlit | Interactive portfolio analytics and policy simulation |
| Data Visualization | Plotly | Real-time charts, regret curves, policy comparisons |
| Deployment | Streamlit Community Cloud | Production-grade app hosting |
| Testing | Pytest | Unit and integration test suite |
| Containerization | Docker | Reproducible environment and deployment |

## 🤖 Bandit Algorithms Implemented

### Thompson Sampling (Primary Algorithm)

Thompson Sampling maintains a Bayesian belief (Beta distribution) over the
reward for each user-action pair. When a decision is needed, it samples from
the belief and greedily picks the action with the highest sample. Good
outcomes strengthen the belief that an action is rewarding; poor outcomes
weaken it. This naturally balances exploration and exploitation without manual
tuning.

```python
# Core idea: Sample from posterior, pick greedily
sampled_reward = beta.rvs(alpha, beta)
best_action = argmax([sampled_reward for each action])
```

### Upper Confidence Bound (UCB)

UCB combines exploitation with a confidence bonus. It picks the action with
the highest upper confidence bound, where the bound widens for actions with
less data. This encourages exploration of uncertain actions while exploiting
high-performers.

```python
# Core idea: Exploitation + optimism under uncertainty
ucb_value = mean_reward + sqrt(ln(t) / count)
best_action = argmax(ucb_value)
```

### Epsilon-Greedy

Epsilon-Greedy is the simplest exploration strategy: with probability ε,
pick a random action (explore); with probability 1-ε, pick the empirically
best action (exploit). While crude, it serves as a baseline.

```python
# Core idea: Fixed exploration rate
if random() < epsilon:
    best_action = random_action()
else:
    best_action = argmax(mean_reward)
```

### Algorithm Comparison

| Property | Thompson Sampling | UCB | Epsilon-Greedy |
|---|---|---|---|
| Exploration mechanism | Uncertainty-guided (samples Beta) | Count-guided (confidence bonus) | Random (fixed ε) |
| Cold start behaviour | Wide Beta(1,1) prior - naturally uncertain | Forces one pull per action first | ε-random until enough pulls |
| Adapts to non-stationarity | Slowly (old evidence persists) | Slowly (running mean) | Yes (via epsilon decay) |
| Tunable parameters | alpha_prior, beta_prior | c (exploration constant) | epsilon, decay, min_epsilon |
| Theoretical guarantee | Bayesian regret bounds | O(√T log T) regret | None |
| Performance in this project | **Best** (16.34% regret) | Good (25.32% regret) | Poor (40.01% regret) |

**Why Thompson Sampling wins here:** The Beta distribution naturally
represents uncertainty about each user-action pair. With 3-month delayed
feedback, Thompson stays appropriately uncertain during the lag window and
updates cleanly once evidence arrives. UCB's count-based bonus doesn't handle
delayed feedback as gracefully - it keeps exploring actions whose rewards
haven't been observed yet.

## 🧠 Key ML Concepts

- **Exploration vs. Exploitation**: The core tradeoff - whether to try new
  limits (gather data) or stick with known-good limits (maximize immediate
  reward).
- **Delayed Reward Feedback**: Reflects production reality: credit losses or
  defaults surface 3 months after the limit decision, making online learning
  challenging.
- **Cold Start Problem**: New users have no history. Early policies must make
  blind guesses; Thompson Sampling naturally handles this through
  uncertainty.
- **Non-Stationarity**: User behavior shifts over time (economic shocks,
  lifestyle changes). Policies must adapt continuously, not assume
  stationarity.
- **Contextual Decisions**: Limit should depend on user features (CIBIL,
  utilization, tenure, spend)—pure exploration/exploitation is too naive for
  a heterogeneous portfolio.

## ⚠️ Known Limitations & Future Work

| Limitation | Impact | Production fix |
|---|---|---|
| Synthetic data only | No real behavioural patterns or correlations | Replace simulator with historical transaction data |
| Beta distributions don't forget | Old evidence persists even as user risk changes | Sliding window or discounted Beta updates |
| No fairness constraints | Algorithm may implicitly discriminate on income or age proxies | Add fairness-aware bandit with demographic parity constraints |
| Single reward signal | Only revenue + default | Add customer lifetime value, churn risk, and regulatory capital cost |
| No A/B test infrastructure | Cannot safely roll out to live users | Implement importance sampling for offline policy evaluation |
| Exploration rate 37.6% | Higher than the 10-25% target | Tighten Beta priors (higher alpha_prior) or add an exploration budget cap |
| Convergence at Month 9 | Later than the Month 3-5 target | Increase learning rate via larger Beta update steps or reduce delay |

## 💡 Project Highlights

- Built a Contextual Multi-Armed Bandit (Thompson Sampling) optimizing
  credit limit decisions for 10,000 simulated users over 12 months
- Implemented 3 bandit policies from scratch (Thompson Sampling, UCB,
  Epsilon-Greedy) benchmarked against an oracle upper bound
- Modeled production challenges: 3-month delayed reward feedback, cold
  start, and economic shock non-stationarity
- Deployed interactive Streamlit dashboard with live simulation controls

## 🤝 Contributing

To add a new bandit algorithm:

1. Create `src/bandits/your_algorithm.py`
2. Inherit from `ContextualBandit` in `src/bandits/base.py`
3. Implement `select_action()`, `update()`, `get_stats()`, `reset()`
4. Register it in `src/simulate_run.py` policy registry
5. Add tests in `tests/test_integration.py`
6. See `docs/guides/contributing.md` for full instructions

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Made for ML Engineering portfolio · Domain: Fintech / Reinforcement
Learning**
