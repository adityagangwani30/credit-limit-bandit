---
title: Reward Engine & Delayed Feedback
category: component
file_reference: src/reward.py
---
# Reward Engine & Delayed Feedback

The Reward Engine converts simulator outcomes into net rewards the bandit can learn from, while the RewardBuffer enforces a 3-month delay between action and learning signal.

Reward formula
- net_reward = (amount_spent × 0.018) - (outstanding_amount × did_default)

Explanation
- `amount_spent × 0.018`: estimated interchange revenue (1.8%) from spend charged to the card.
- `outstanding_amount × did_default`: full outstanding loss when a default occurs (conservative penalty).

Examples
- Prime user spends ₹50,000 and does not default → reward = 50,000 × 0.018 = ₹900
- Subprime user defaults with ₹20,000 outstanding → reward = -20,000

Why 1.8%
- Reflects a representative interchange-like net revenue rate for card transactions in the project context; the value is a domain assumption and can be tuned in `src/reward.py`.

Delayed feedback handling
- Actions in month T produce outcomes that may only be fully attributable at month T+3. To avoid premature updates, the `RewardBuffer` stores outcome rows with `release_month = action_month + 3`.
- When the simulation advances to month R, the buffer releases rewards whose `release_month <= R` to the bandit `update()` API.

What happens if a user defaults during the delay
- The default flag is stored in the buffered outcome. When released, the reward will include the negative penalty and the bandit updates accordingly.

Normalization for bandit updates
- Raw INR rewards are heavy-tailed; before updating Beta posteriors we apply a sigmoid normalization to map rewards into (0,1). This preserves reward ordering and stabilizes Beta-count updates.

Edge cases
- First 3 months: the buffer contains no released rewards — policies must operate without updates (pure prior-driven exploration).
- Missing outcomes: if simulator fails to produce a row, the buffer logs and skips the update; tests assert full coverage.

Related docs
- [docs/architecture/data-flow.md](docs/architecture/data-flow.md)
- [docs/algorithms/thompson-sampling.md](docs/algorithms/thompson-sampling.md)
