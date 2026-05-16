# 💳 Credit Limit Bandit

**Contextual Multi-Armed Bandit for Dynamic Credit Limit Optimization**

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![CI Status](https://img.shields.io/badge/CI-passing-brightgreen) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red?logo=streamlit)

---

## 🚀 Live Demo

🚀 [Launch Streamlit App](https://credit-limit-banditgit.streamlit.app/)

Explore real-time credit limit recommendations across 10,000 simulated users with interactive policy comparisons, regret analysis, and performance dashboards.

---

## Problem Statement

Every major Indian fintech—Cred, Slice, Jupiter, HDFC, ICICI, and others—manually reviews credit limits every 6 months. This creates a fundamental business challenge: **set limits too low and you lose interchange revenue; set them too high and defaults spike.**

Today's approach is reactive and static. Credit decisions are reviewed on a fixed calendar, heuristics are applied inconsistently, and there is no systematic way to learn from outcomes. The result is left money on the table and unnecessary risk exposure.

**Credit Limit Bandit automates this decision using Reinforcement Learning.** It continuously learns from user behavior—spending patterns, repayment history, utilization trends—and recommends personalized limit increases in real time. By framing credit limit optimization as a contextual multi-armed bandit problem, we balance exploration (testing higher limits to find the sweet spot) against exploitation (sticking with what we know works).

---

## How It Works

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

Every month, we observe a user's financial snapshot and feed it to a bandit policy. The policy samples from its learned beliefs about which limit would generate the best outcome, then makes a decision. After a 3-month lag (reflecting real-world delayed feedback), we observe whether the user defaulted and calculate the true reward. That reward updates the policy's belief, making it smarter over time.

---

## Results

### Policy Comparison (12-month simulation, 10,000 users)

| Policy | Revenue (INR) | Lift vs Static | Default Rate | Exploration |
|---|---:|---:|---:|---:|
| Thompson Sampling | ₹43.9Cr | +370.8% | 3.38% | 75.4% |
| UCB | ₹24.0Cr | +157.2% | 3.38% | 75.0% |
| Epsilon-Greedy | ₹9.2Cr | −1.1% | 3.38% | 11.6% |
| Static Baseline | ₹9.3Cr | 0% | 3.38% | 0% |
| Oracle Upper Bound | ₹703.9Cr | +7455.2% | 3.38% | 80.5% |

> **Thompson Sampling met 3 of 4 targets from the project spec:**
> - Revenue lift >30% ✓ (achieved 370.8%)
> - Default rate <4% ✓ (achieved 3.38%)
> - Exploration rate 10-25% ✗ (achieved 75.4% — too aggressive)
> - Convergence by month 5 ✓ (policy stabilized in early months)

### Cohort Analysis — Which users benefit most?

| Risk Tier | Revenue (Thompson) | Revenue (Static) | Lift | Default Rate |
|---|---:|---:|---:|---:|
| Prime | ₹56.81Cr | ₹11.72Cr | +384.7% | 0.39% |
| Near-Prime | ₹0.64Cr | ₹0.43Cr | +46.1% | 1.85% |
| Subprime | ₹−7.18Cr | ₹−1.52Cr | Loss | 5.87% |
| Deep-Subprime | ₹−6.40Cr | ₹−1.31Cr | Loss | 15.32% |

> **Key insight:** Prime users show the highest revenue lift (+384.7%) because they have pristine credit history (0.39% default rate), allowing the bandit to confidently increase limits with minimal downside risk. Near-Prime users follow (+46.1%), showing the sweet spot where learning speed meets profitability. Lower tiers generate net losses due to default penalties outweighing interchange revenue.

### Production Resilience

| Challenge | Result |
|---|---|
| **Cold start** (months 1–3) | Thompson bootstraps from ₹0/user to ₹1,790/user by month 3 |
| **Economic shock** (month 6) | Default spike to 3.67%, recovered within 1 month |
| **Non-stationarity** | Policy adapted continuously with zero manual retuning |
| **Portfolio-wide revenue** | 438.6M INR (370.8% lift vs static baseline) |

---

## System Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│ User Simulator  │ ────→   │ Context Builder  │ ────→   │ Bandit Policy   │
└─────────────────┘         └──────────────────┘         └─────────────────┘
   Monthly CIBIL,             10-feature vector           Thompson / UCB /
   spend, utilization         (normalized)                Epsilon-Greedy
                                      │                           │
                                      └──────────┬────────────────┘
                                                 ↓
                                        ┌──────────────────┐
                                        │  Reward Engine   │
                                        └──────────────────┘
                                           Interchange +
                                           Default Loss
                                                 │
                                                 ↓
                                        ┌──────────────────┐
                                        │ Evaluation Module│
                                        └──────────────────┘
                                           Regret, Lift,
                                           Metrics, Charts
```

**Component Breakdown:**
- **User Simulator**: Generates synthetic 10,000-user portfolio with dynamic spend, utilization, and default behavior.
- **Context Builder**: Extracts and normalizes 10 financial features into a compact state representation.
- **Bandit Policy**: Learns from feedback and selects optimal limit actions (Thompson Sampling, UCB, Epsilon-Greedy).
- **Reward Engine**: Computes monthly interchange revenue and default penalties; applies 3-month feedback delay.
- **Evaluation Module**: Computes cumulative regret, revenue lift, policy comparisons, and cohort analysis.

---

## Tech Stack

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

---

## Bandit Algorithms Implemented

### Thompson Sampling (Primary Algorithm)

Thompson Sampling maintains a Bayesian belief (Beta distribution) over the reward for each user-action pair. When a decision is needed, it samples from the belief and greedily picks the action with the highest sample. Good outcomes strengthen the belief that an action is rewarding; poor outcomes weaken it. This naturally balances exploration and exploitation without manual tuning.

```python
# Core idea: Sample from posterior, pick greedily
sampled_reward = beta.rvs(alpha, beta)  # Alpha = wins, Beta = losses
best_action = argmax([sampled_reward for each action])
```

### Upper Confidence Bound (UCB)

UCB combines exploitation with a confidence bonus. It picks the action with the highest upper confidence bound, where the bound widens for actions with less data. This encourages exploration of uncertain actions while exploiting high-performers.

```python
# Core idea: Exploitation + optimism under uncertainty
ucb_value = mean_reward + sqrt(ln(t) / count) 
best_action = argmax(ucb_value)
```

### Epsilon-Greedy

Epsilon-Greedy is the simplest exploration strategy: with probability ε, pick a random action (explore); with probability 1-ε, pick the empirically best action (exploit). While crude, it serves as a baseline.

```python
# Core idea: Fixed exploration rate
if random() < epsilon:
    best_action = random_action()
else:
    best_action = argmax(mean_reward)
```

---

## Key ML Concepts

- **Exploration vs. Exploitation**: The core tradeoff—whether to try new limits (gather data) or stick with known-good limits (maximize immediate reward).
- **Delayed Reward Feedback**: Reflects production reality: credit losses or defaults surface 3 months after the limit decision, making online learning challenging.
- **Cold Start Problem**: New users have no history. Early policies must make blind guesses; Thompson Sampling naturally handles this through uncertainty.
- **Non-Stationarity**: User behavior shifts over time (economic shocks, lifestyle changes). Policies must adapt continuously, not assume stationarity.
- **Contextual Decisions**: Limit should depend on user features (CIBIL, utilization, tenure, spend)—pure exploration/exploitation is too naive for a heterogeneous portfolio.

---

## Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/credit-limit-bandit
cd credit-limit-bandit
python -m venv venv && source venv/bin/activate
pip install -e .
python src/simulate_run.py
streamlit run dashboard/app.py
```

**macOS/Linux (activate venv):**
```bash
source venv/bin/activate
```

**Windows PowerShell (activate venv):**
```powershell
.venv\Scripts\Activate.ps1
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Project Highlights (for Resume)

- Built a Contextual Multi-Armed Bandit (Thompson Sampling) optimizing credit limit decisions for 10,000 simulated users over 12 months
- Implemented 3 bandit policies from scratch (Thompson Sampling, UCB, Epsilon-Greedy) benchmarked against an oracle upper bound
- Modeled production challenges: 3-month delayed reward feedback, cold start, and economic shock non-stationarity
- Deployed interactive Streamlit dashboard with live simulation controls

---

**Made for ML Engineering portfolio · Domain: Fintech / Reinforcement Learning**
