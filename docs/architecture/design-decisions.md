---
title: Design Decisions & Rationale
category: architecture
file_reference: none
---
# Design Decisions & Rationale

Below are key design choices made for the project, alternatives considered, rationale, and tradeoffs accepted.

1. Decision: Use a Multi-Armed Bandit (MAB) instead of supervised classification
   - Alternatives: supervised classifier predicting default probability or expected revenue
   - Why chosen: bandits optimize sequential decision-making under partial feedback and actively balance exploration vs exploitation; supervised models require labeled counterfactuals and do not learn from deployed decisions.
   - Tradeoff: MABs provide online adaptation but less interpretability on feature weightings compared to some classifiers.

2. Decision: Thompson Sampling as the primary algorithm
   - Alternatives: UCB, epsilon-greedy, contextual bandits with linear models
   - Why chosen: Thompson offers probabilistic, uncertainty-aware exploration that performs well empirically with small-to-moderate action spaces.
   - Tradeoff: stochastic decisions may complicate explainability and reproducibility relative to deterministic policies.

3. Decision: Synthetic data over public real datasets
   - Alternatives: anonymized bank dataset, open datasets with different distributions
   - Why chosen: credit limit decisions require domain-specific reward and delay modeling not present in public datasets; synthetic data allows controlled shocks and validation targets.
   - Tradeoff: external validity is limited; results are for demonstration and engineering evaluation, not regulatory submission.

4. Decision: 3-month reward delay
   - Alternatives: 1-month, 6-month delays
   - Why chosen: three months balances realistic time-to-default for revolving credit and keeps simulation tractable for a 12-month horizon.
   - Tradeoff: shorter delays underestimate late defaults; longer delays require larger buffer logic and slower iteration.

5. Decision: Beta distribution for Thompson Sampling
   - Alternatives: Gaussian approximations, bootstrapped posteriors
   - Why chosen: the reward model is framed as (normalized) Bernoulli-like success for default vs productive spend, making Beta conjugate and computationally trivial to update.
   - Tradeoff: Beta assumes Bernoulli outcomes; continuous reward signals require normalization and approximate handling.

6. Decision: Streamlit for the dashboard
   - Alternatives: Flask + custom frontend, FastAPI + React
   - Why chosen: Streamlit enables rapid interactive visualizations, less engineering overhead, and ideal for demos and interviews.
   - Tradeoff: not optimized for production-scale deployments or multi-user dashboards without extra engineering.

Related docs
- [docs/architecture/overview.md](docs/architecture/overview.md)
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
