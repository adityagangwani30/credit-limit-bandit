---
title: The Cold Start Problem
category: concept
file_reference: none
---
# The Cold Start Problem

Cold start refers to decisions for new users who have little or no transaction history. In credit products, 10–15% of active users may be new in a month; handling them safely and profitably is crucial.

Why it matters
- New users lack personalized empirical reward data; naive policies must either be conservative (losing revenue) or aggressive (risking defaults).

Project approach
- Use context-based features available at onboarding (e.g., `cibil_score`, `income_bucket`, `employment_type`) to proxy expected behavior.
- Thompson Sampling uses a uniform Beta(1,1) prior for unknown arms; the context vector directs the bandit to select actions with higher expected reward for similar users.

Cold start vector example
- `cibil_score`: median population value (e.g., 0.7 normalized)
- `income_bucket`: onboarding-provided category (one-hot)
- `avg_spend_3m`, `volatility_3m`: population mean and std imputed

Why context helps
- A high-cibil, salaried new user is far more likely to behave like low-risk cohorts than a low-cibil gig-worker; context enables transfer of knowledge without prior individual history.

Limitations
- Context only helps if features are predictive and not adversarially noisy. If onboarding features are poor, exploration is still required and may be costly.

Related docs
- [docs/components/context-builder.md](docs/components/context-builder.md)
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
