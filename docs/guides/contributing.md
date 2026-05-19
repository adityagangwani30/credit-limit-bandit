# Contributing: Add a New Bandit Algorithm

Step-by-step guide to implement a new algorithm (e.g., LinUCB) and integrate into the system.

## 6 Steps to Add New Algorithm

### Step 1: Create Algorithm File

Create `src/bandits/your_algorithm.py`:

```python
from src.bandits.base import ContextualBandit
import numpy as np

class YourBandit(ContextualBandit):
    def __init__(self):
        self.reset()
    
    def select_action(self, context, user_id, actions):
        # Your implementation
        return actions[0]  # placeholder
    
    def update(self, user_id, action, reward, context):
        # Your implementation
        pass
    
    def get_stats(self):
        return {"algorithm": "your_algorithm"}
    
    def reset(self):
        pass
```

### Step 2: Implement Interface Contract

| Method | Signature | Purpose |
|--------|-----------|---------|
| select_action | (context: np.ndarray, user_id: str, actions: list[str]) → str | Return action name |
| update | (user_id: str, action: str, reward: float, context: np.ndarray) → None | Update from delayed reward |
| get_stats | () → dict | Return stats dict with "algorithm" key |
| reset | () → None | Reset to initial state |

### Step 3: Integration

Edit `src/simulate_run.py`:

```python
from src.bandits.your_algorithm import YourBandit

if policy_name == "your_algorithm":
    bandit = YourBandit()
```

### Step 4: Testing

```bash
pytest tests/test_your_algorithm.py -v
```

Write tests covering: initialization, select_action, update, cold-start behavior.

### Step 5: Benchmarking

```bash
python src/simulate_run.py --policy your_algorithm --n_users 1000 --n_months 12
```

Compare revenue vs Thompson (₹12.13Cr), UCB (₹10.83Cr).

### Step 6: Documentation

Document in `docs/algorithms/your_algorithm.md` with:
- Algorithm intuition
- Hyperparameters
- Pros/cons
- Performance results

## LinUCB Skeleton Example

LinUCB (Contextual UCB) uses linear models + confidence bounds:

```python
class LinUCBBandit(ContextualBandit):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        self.A = {}  # Design matrix per action
        self.b = {}  # Reward vector per action
    
    def select_action(self, context, user_id, actions):
        scores = {}
        for action in actions:
            # A_inv @ b estimates theta
            # theta · context is mean reward
            # + alpha * sqrt(context^T A_inv context) is UCB bound
            scores[action] = self._ucb_score(context, action)
        return max(actions, key=lambda a: scores[a])
    
    def update(self, user_id, action, reward, context):
        # Update A, b for action's linear model
        pass
```

## Related Docs

- [thompson-sampling.md](../algorithms/thompson-sampling.md)
- [ucb.md](../algorithms/ucb.md)
