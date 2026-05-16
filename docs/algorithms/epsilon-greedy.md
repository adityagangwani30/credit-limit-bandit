---
title: Epsilon-Greedy
category: algorithm
file_reference: src/bandits/epsilon_greedy.py
---
# Epsilon-Greedy

Epsilon-Greedy is the simplest exploration strategy: with probability ε take a random action (explore), otherwise choose the current best (exploit). It serves as a baseline and is easy to implement and reason about.

## Intuition
Randomly explore a fraction of the time to discover potentially better actions while mostly exploiting the observed best.

## Algorithm
- With probability ε_t: choose a random action
- Otherwise: choose argmax of empirical mean reward

## Epsilon decay
We use a geometric decay:
``text
ε_t = max(ε_min, ε_0 * decay^t)
``
Project parameters: ε_0=0.15, decay=0.995, ε_min=0.01.

## Why decay matters
Constant ε keeps exploring indefinitely; decay reduces wasted exploration as confidence grows.

## Implementation notes
See `src/bandits/epsilon_greedy.py` for the implementation. Key functions:
- maintain empirical means per action
- compute ε_t at each time step
- sample random action vs exploit

Pros and cons
| Pros | Cons |
| --- | --- |
| Simple and fast | Explores randomly, not guided by uncertainty |
| Useful baseline | Performance typically worse than Thompson or tuned UCB |

Related docs
- [docs/algorithms/comparison.md](docs/algorithms/comparison.md)
- [docs/guides/contributing.md](docs/guides/contributing.md)
