---
title: Adding a New Bandit Algorithm
category: guide
file_reference: none
---
# Adding a New Bandit Algorithm

This guide explains how to add a new contextual bandit algorithm to the codebase.

Step-by-step
1. Create `src/bandits/your_algorithm.py`.
2. Inherit from `ContextualBandit` (see `src/bandits/base.py`).
3. Implement required methods:
   - `select_action(contexts: np.ndarray) -> np.ndarray` — returns action indices for batch contexts
   - `update(actions, rewards, contexts=None)` — perform posterior or statistics updates
   - `get_stats()` — return diagnostic counters
   - `reset()` — reset internal state for experiments
4. Add your algorithm to the policy registry in `src/simulate_run.py` so it is invocable via `--policy`.
5. Add visualization hooks in `dashboard/app.py` if you want per-policy tracing.
6. Write unit tests in `tests/test_your_algorithm.py` covering select/update/reset behavior.

Interface contract
- `select_action` receives a 2D numpy array `(n, d)` and returns a 1D integer array of length `n` of action indices.
- `update` will be called with the same action codes and normalized rewards in [0,1]. Implementations must be resilient to delayed updates (i.e., handle updates arriving out-of-order by month).

Example skeleton (LinUCB)
```python
class LinUCB(ContextualBandit):
    def __init__(self, d, alpha=1.0):
        # initialize A and b per action
        pass
    def select_action(self, contexts):
        # compute p = theta.T x + alpha * uncertainty
        pass
    def update(self, actions, rewards, contexts=None):
        # ridge update for selected arms
        pass
```

Testing & integration
- Add tests that run `simulate_run.py` with `--n_users 1000` and assert metrics are within expected ranges for trivial deterministic scenarios.

Related docs
- [docs/guides/getting-started.md](docs/guides/getting-started.md)
- [docs/algorithms/comparison.md](docs/algorithms/comparison.md)
