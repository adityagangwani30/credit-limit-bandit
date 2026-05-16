---
title: What is a Multi-Armed Bandit?
category: concept
file_reference: none
---
# What is a Multi-Armed Bandit?

The multi-armed bandit (MAB) is a formalism for sequential decision-making under uncertainty. Imagine a row of slot machines ("arms"); each has unknown payout distribution. You can pull one arm per timestep and observe its payout; your objective is to maximize cumulative reward over a horizon by balancing exploration (learning about arms) and exploitation (choosing arms believed to be best).

Casino analogy → credit limits
- Arms: actions (discrete limit change options) applied to users.
- Pull: selecting an action for a specific user at timestep t.
- Reward: net economic outcome (interchange revenue minus default penalty) observed after a delay.

Why "bandit" fits
- The problem is sequential with partial feedback: each decision yields feedback only for the chosen action, unlike supervised datasets with complete labels.

Contextual vs non-contextual bandits
- Non-contextual: choose the same arm population-wide (no personalization).
- Contextual: observe a context vector x_t for each user at time t and choose action a_t = π(x_t). This project uses contextual bandits to tailor credit limits to individual users.

Formal problem definition
- At each timestep t:
  1. Observe context x_t
  2. Choose action a_t from finite action set A
  3. Receive reward r_t (possibly delayed)
  4. Update policy to maximize sum_{t=1..T} r_t

Why not supervised learning or full RL
- Supervised learning predicts labels but does not produce data for counterfactual actions and suffers from selection bias when deployed.
- Full RL models state transitions and long-term effects; here, actions affect near-term rewards with delayed observation but do not require full MDP formalism for this demonstration.

Related docs
- [docs/concepts/exploration-exploitation.md](docs/concepts/exploration-exploitation.md)
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
