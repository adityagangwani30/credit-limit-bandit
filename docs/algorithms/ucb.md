---
title: Upper Confidence Bound (UCB)
category: algorithm
file_reference: src/bandits/ucb.py
---
# Upper Confidence Bound (UCB)

UCB is a deterministic exploration strategy that selects actions by adding a confidence bonus to empirical means. The policy prefers arms with high mean reward or high uncertainty (few pulls).

## Intuition
"Be optimistic in the face of uncertainty": treat untried or sparsely-tried actions as potentially better by inflating their score with an exploration bonus.

## UCB1 formula
``text
score_a = mean_reward_a + c * sqrt( log(t) / n_a )
``
- mean_reward_a: average observed reward for action a
- t: total number of decision rounds elapsed
- n_a: number of times action a was selected
- c: exploration constant (set to 2.0 in this project)

## Differences vs Thompson Sampling
- UCB is deterministic given counts and means; Thompson is stochastic via sampling.
- UCB uses a principled, concentration-bound bonus; Thompson uses posterior sampling.

## First-pull problem
UCB assigns infinite or very large bonus to untried actions by using n_a=0; in practice we initialize n_a=1 or perform an initial round of forced exploration to avoid infinities.

## Implementation notes
See `src/bandits/ucb.py`. Key behaviors:
- track counts and average rewards per action
- compute score per action using UCB1 formula
- select highest score; update counts and mean after reward release

Pros and cons
| Pros | Cons |
| --- | --- |
| Deterministic and interpretable | Requires careful handling of untried arms |
| Strong theoretical guarantees for stationary problems | Less robust to misspecified reward variance and non-stationarity |

Related docs
- [docs/algorithms/comparison.md](docs/algorithms/comparison.md)
- [docs/concepts/exploration-exploitation.md](docs/concepts/exploration-exploitation.md)
