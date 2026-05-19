# Multi-Armed Bandit Concept

A bandit is a decision-making framework where an agent repeatedly chooses from K actions, receives reward, and learns which actions are best.

## Casino Slot Machine Analogy

- **Slots:** 4 machines (keep, plus_10, plus_20, plus_50 credit limit increases)
- **Reward:** Payout (interchange ₹900, or loss −₹24,550 if default)
- **Goal:** Maximize total payout
- **Challenge:** Don't know which machine is best without trying (exploration needed)

In credit: trying plus_50 for a risky user might default (−₹2M loss). Exploration is costly.

## Formal Problem Definition

State: user context (10 features) | Action: a ∈ {keep, plus_10, plus_20, plus_50} | Reward: r ∈ [−₹27M, ₹700K] | Delay: 3 months

Objective: maximize cumulative reward over 12 months = Σ rewards across all users and actions, subject to default rate constraint (3.38%).

## Contextual vs Non-Contextual

- **Contextual:** Action selection depends on user context (CIBIL, income, utilization). Thompson Sampling is contextual (different posteriors per risk segment).
- **Non-contextual:** Action selection ignores context (epsilon-greedy in basic form). Pure MAB, not tailored to user risk profile.

Credit demands contextual: Prime users should get plus_50 more often; Deep-Subprime should get keep only.

## Why Not Supervised Learning?

Supervised learning (predict default probability) is NOT the same as credit limit optimization. Classifying "will this user default?" ≠ "what limit maximizes portfolio revenue?" Bandit directly optimizes revenue; classification optimizes a proxy.

## Why Not A/B Testing?

A/B tests fix a control group on old policy, test group on new. Bandits adapt continuously per user. Over 12 months, A/B wastes months 1-6 on control, then slowly ramps test. Bandits learn in 1 month and adapt.

## Why Not Full Reinforcement Learning (MDP)?

Full RL (Markov Decision Process) models future states: "if I increase limit now, user's utilization changes next month." Credit defaults are nearly independent of limit history (driven by external income/unemployment shocks). State transitions are weak, so MDP complexity is unjustified. Contextual bandit (single-step, delayed reward) is the right abstraction.

## Related Docs

- [exploration-exploitation.md](exploration-exploitation.md)
- [delayed-feedback.md](delayed-feedback.md)
