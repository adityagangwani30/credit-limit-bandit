# Credit-Limit Bandit 💳🎰

A **contextual multi-armed bandit system** for optimising credit-card limit offers. The project simulates a lending environment where each customer is characterised by contextual features (income, credit score, tenure, spending behaviour), and the bandit agent must learn — through exploration and exploitation — which credit-limit tier maximises a composite reward signal that balances revenue generation, card utilisation, and default risk. Three classical bandit strategies (Thompson Sampling, UCB, ε-Greedy) are implemented, evaluated against cumulative regret, and visualised in an interactive Streamlit dashboard.

---

## Tech Stack

| Layer             | Technology                          |
|-------------------|-------------------------------------|
| Language          | Python 3.10+                        |
| Core ML / Stats   | NumPy, SciPy, scikit-learn          |
| Data Handling      | Pandas                              |
| Visualisation      | Plotly, Matplotlib                   |
| Dashboard          | Streamlit                            |
| Notebooks          | Jupyter, IPyKernel                   |
| Build System       | PEP 517 (setuptools)                |

---

## Project Structure

```
credit-limit-bandit/
├── data/                        # Raw & processed datasets
├── src/
│   ├── simulator.py             # Customer simulation engine
│   ├── reward.py                # Reward function definitions
│   ├── context.py               # Feature engineering
│   ├── evaluate.py              # Offline policy evaluation
│   └── bandits/
│       ├── base.py              # Abstract bandit interface
│       ├── thompson.py          # Thompson Sampling
│       ├── ucb.py               # Upper Confidence Bound
│       └── epsilon_greedy.py    # Epsilon-Greedy
├── notebooks/
│   ├── 01_simulation_eda.ipynb
│   ├── 02_bandit_training.ipynb
│   └── 03_evaluation.ipynb
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/adityagangwani30/credit-limit-bandit.git
cd credit-limit-bandit
```

### 2. Create & activate a virtual environment

```bash
python -m venv .env

# Windows
.env\Scripts\activate

# macOS / Linux
source .env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the project in editable mode (optional)

```bash
pip install -e ".[dev]"
```

### 5. Launch Jupyter notebooks

```bash
jupyter notebook notebooks/
```

### 6. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

---

## License

MIT
