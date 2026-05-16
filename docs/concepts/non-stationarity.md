---
title: Non-Stationarity & Economic Shocks
category: concept
file_reference: none
---
# Non-Stationarity & Economic Shocks

Non-stationarity occurs when the reward distribution changes over time due to external factors (economic shocks, regulation, seasonality). This project simulates a shock at month 6 by doubling default probabilities to illustrate adaptation challenges.

Real-world examples
- COVID-19 causing sudden rise in defaults
- Interest rate changes reducing consumer spending

Project simulation
- At month 6: `economic_stress = 2.0` doubles the default rates across tiers, creating a sudden increase in realized defaults and a stress test for policies.

Why static models fail
- Classifiers trained on months 1–5 will be miscalibrated post-shock and may recommend overly aggressive limits.

How bandits adapt
- Thompson Sampling updates posterior with new defaults and reduces optimistic sampling for risky arms over subsequent months.

Limitations of Beta posteriors
- Beta accumulates evidence and does not forget; without discounting older observations the model is slow to adapt in persistent non-stationarity.

Production fixes
- Sliding-window updates: only keep the last W months of evidence.
- Discounted updating: apply exponential decay to older counts.
- Change-point detection: trigger model reset or conservative policy when abrupt shifts detected.

Related docs
- [docs/architecture/design-decisions.md](docs/architecture/design-decisions.md)
- [docs/results/interpreting-results.md](docs/results/interpreting-results.md)
