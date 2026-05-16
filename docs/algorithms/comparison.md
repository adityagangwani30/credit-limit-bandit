---
title: Algorithm Comparison
category: algorithm
file_reference: none
---
# Algorithm Comparison

This page compares Thompson Sampling, UCB, and Epsilon-Greedy across properties relevant to production credit-limit decisioning.

| Property | Thompson Sampling | UCB | Epsilon-Greedy |
| --- | --- | --- | --- |
| Exploration mechanism | Posterior sampling (stochastic) | Confidence bonus (deterministic) | Random exploration (ε) |
| Convergence speed | Fast in moderate action spaces | Fast if variance low | Slower; depends on ε decay |
| Handles cold start | Yes (uniform priors) | Needs initial pulls | Needs initial exploration policy |
| Handles non-stationarity | Adapts but retains old evidence | Slower without forgetting | Depends on decay/reset strategy |
| Computational cost | Low (sampling per arm) | Low (counts and sqrt) | Lowest (random + argmax) |
| Tunable parameters | Prior, reward normalization | c (exploration constant) | ε_0, decay, ε_min |
| Theoretical guarantees | Bayes-regret bounds under assumptions | Regret bounds (UCB1) | No rigorous guarantees except empirical bounds |
| Best for | Small-to-moderate action spaces with uncertainty | Well-calibrated, stationary problems | Baseline or constrained compute |

Results comparison (placeholder values)
| Metric | Thompson | UCB | Epsilon-Greedy | Oracle |
| --- | ---: | ---: | ---: | ---: |
| Total 12-month revenue | 1,200,000 | 1,150,000 | 1,050,000 | 1,300,000 |
| Default rate | 3.2% | 3.4% | 3.6% | 2.8% |
| Regret vs Oracle | 100,000 | 150,000 | 250,000 | 0 |
| Convergence month | 4 | 5 | 6 | - |
| Exploration % (avg) | 18% | 12% | 25% | 0% |

Which to use in production?
Thompson Sampling is recommended as a first production policy for this project due to its uncertainty-aware exploration, good empirical performance, and simplicity of per-arm posteriors. UCB is a strong alternative when deterministic behavior is preferred and variance is well-understood. Epsilon-Greedy is suitable as a baseline or for extremely constrained deployments.

Interview question: "Why didn't you use a neural network?"
Neural networks can model complex, non-linear relationships but require labeled counterfactuals or offline policy evaluation to safely deploy for decisioning; they are also less interpretable and require significantly more data. For sequential decision-making with limited labels and the need for online exploration, contextual bandits provide a simpler, safer framework that directly optimizes decisions under partial feedback.

Related docs
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
- [docs/architecture/overview.md](docs/architecture/overview.md)
