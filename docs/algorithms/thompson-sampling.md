---
title: Thompson Sampling
category: algorithm
file_reference: src/bandits/thompson.py
---
# Thompson Sampling

Thompson Sampling is a Bayesian, uncertainty-driven algorithm that selects actions by sampling from posterior distributions and choosing the action with the highest sampled value. Intuitively, it prefers actions that might be optimal while naturally decaying exploration as evidence accumulates.

## Intuition
Thompson implements "optimism under uncertainty" by drawing a possible outcome for each action from its posterior and choosing the action that looks best in that hypothetical world. Without math: actions with uncertain but potentially high returns get chosen more often early on.

## How it works (steps)
1. Maintain Beta(α, β) for each (user, action) pair.
2. At decision time, sample θ_a ~ Beta(α_a, β_a) for each action a.
3. Select action a* = argmax_a θ_a.
4. After observing the normalized reward r_norm in {0,1}, update:
   - α_a* ← α_a* + r_norm
   - β_a* ← β_a* + (1 - r_norm)

## The math
``text
Beta(α,β) ∝ x^{α-1} (1-x)^{β-1}
sample: θ ~ Beta(α,β)
update: α ← α + r_norm, β ← β + (1 - r_norm)
``

### Why Beta?
Beta is the conjugate prior for Bernoulli outcomes; it is computationally cheap and interpretable. In this project we treat normalized rewards as a Bernoulli-like success (after sigmoid normalization), which justifies Beta updates.

### Reward normalization
Raw INR rewards (spend-derived) are continuous and heavy-tailed. Before updating Beta counts we apply a sigmoid-like transform to map raw reward to [0,1] and treat it as r_norm. This preserves ordering while keeping updates stable.

### Cold start behavior
With α=1 and β=1 the Beta prior is uniform. Early actions are exploratory: sampling will encourage balanced testing across actions until evidence accumulates.

### Convergence
As observations grow, the posterior concentrates around the empirical mean; exploration naturally decays because samples from tight posteriors rarely exceed the current best.

### Implementation notes
See `src/bandits/thompson.py` for the concrete implementation. Key lines:
- initialization of alpha/beta per action
- sampling in `select_action()`
- Beta updates in `update()` after reward release

Pros and cons
| Pros | Cons |
| --- | --- |
| Principled uncertainty-driven exploration | Stochastic choices complicate deterministic reproductions |
| Strong empirical performance in small action spaces | Assumes Bernoulli-like normalized rewards |

When it underperforms
- In highly non-stationary settings without forgetting; in extremely large action spaces where per-arm posteriors are costly to maintain.

Related docs
- [docs/algorithms/comparison.md](docs/algorithms/comparison.md)
- [docs/components/reward-engine.md](docs/components/reward-engine.md)
