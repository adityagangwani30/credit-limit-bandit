---
title: Delayed Reward Feedback
category: concept
file_reference: src/reward.py
---
# Delayed Reward Feedback

In many financial products, the full consequences of a decision (e.g., increasing a credit limit) are only visible after several months. This file explains why delayed feedback breaks naive bandit updates and how this project addresses it with a `RewardBuffer`.

The problem
- Action at month T can cause spending that looks positive in month T+1 but the borrower might default in month T+3, erasing earlier gains. If a bandit updates immediately on spend, it can receive false positive signals.

Project approach
- Use a 3-month delay: outcomes associated with an action at T are released at T+3. The `RewardBuffer` stores (action, outcome) tuples with a `release_month` and only feeds updates when due.

Diagram (delay)
``text
T: choose action -> simulator logs intermediate spends
T+1/T+2: intermediate signals available but not used for update
T+3: final outcome (default/no-default, outstanding) released -> update()
``

Why 3 months
- Three months capture most early default outcomes for revolving credit while keeping an actionable simulation horizon of 12 months.

Algorithmic implications
- Policies must continue to act during the buffer window without fresh reward feedback; priors and uncertainty must guide decisions.
- Thompson Sampling naturally keeps posterior variance high until rewards are released; UCB relies on counts and can be overly confident if naive immediate counts are used.

Production notes
- For longer delays in production, techniques include importance weighting, off-policy evaluation using logged propensities, and discounting older evidence.

Related docs
- [docs/components/reward-engine.md](docs/components/reward-engine.md)
- [docs/concepts/non-stationarity.md](docs/concepts/non-stationarity.md)
