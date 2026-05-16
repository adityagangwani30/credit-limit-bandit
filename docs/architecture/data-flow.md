---
title: Data Flow — From User Profile to Reward Signal
category: architecture
file_reference: src/context.py
---
# Data Flow — From User Profile to Reward Signal

Linear flow

raw user features -> context vector -> bandit decision -> simulator outcome -> reward buffer -> bandit update

Arrow details
- raw user features -> context vector
  - Data structure: pandas.DataFrame row or dict
  - Example: {user_id: 1234, cibil: 720, limit: 50000, util: 0.45}
  - Shape: single-row dict or (n_users, n_features) ndarray

- context vector -> bandit decision
  - Data structure: numpy.ndarray float32, shape (n_users, d) where d=10
  - Example values: [0.72, 0.5, 1.0, 0.2, ...]

- bandit decision -> simulator outcome
  - Data structure: action log (CSV rows) with fields (user_id, month, action_code)
  - Example action_code: 2 (increase by one tier)

- simulator outcome -> reward buffer
  - Data structure: dict with keys (user_id, month, spend, outstanding, default_flag)
  - Example: {'user_id':1234,'month':4,'spend':35000.0,'outstanding':20000.0,'default':False}

- reward buffer -> bandit update
  - Data structure: list of (user_id, action, reward_norm) released when month >= action_month+3

Delayed feedback flow

Diagram (delay)
``text
month T action -> simulator produces intermediate signals -> RewardBuffer stores outcome
RewardBuffer releases at T+3 -> bandit.update() receives normalized reward
``

Data schema examples (dtype/shape)
- Context vector: dtype=float32, shape=(10,) per user, values in [0,1]
- Action log CSV: columns: user_id:int, month:int, action:int, context_json:text
- Reward record: user_id:int, action_month:int, release_month:int, raw_reward:float, norm_reward:float

State maintained vs recomputed
- Maintained across months:
  - Bandit posterior (e.g., Beta parameters) per user-action pair
  - RewardBuffer with pending outcomes
- Recomputed each month:
  - Context vectors (derived from last 3 months of history)
  - Spending volatility metrics and rolling aggregates

Related docs
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/components/reward-engine.md](docs/components/reward-engine.md)
