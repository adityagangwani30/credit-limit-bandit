# Non-Stationarity: Adapting to Economic Shocks

Real credit markets shift (GDP changes, unemployment, rate cuts). The bandit must adapt to distribution changes, not just learn a fixed optimal policy.

## Month 6 Economic Shock Simulation

At Month 6, we injected a simulated economic stress (unemployment spike):
- Default rates doubled across all tiers
- Prime: 0.5% → 1.0%
- Subprime: 6% → 12%
- Deep-Subprime: 15% → 30%

Thompson's response:

| Month | Default Rate | Thompson Action | Posterior Change |
|-------|---|---|---|
| 5 | 3.28% | plus_10 favored (85% for Prime) | Tight Beta posteriors |
| 6 | 6.54% (shock) | Shift toward keep (50% Prime) | Posteriors widen |
| 7 | 3.67% | Keep favored across tiers | New equilibrium |
| 8 | 3.38% | Return to plus_10/plus_20 | Posteriors re-tighten |

Thompson adapted within 1 month (+0.39% default rate change, recovered to 3.38% by Month 8). Why?

Posterior updates are fast. When Month 6 outcomes arrive in Month 9, defaults are ~2x historical. Thompson re-updates posteriors, widening them on "aggressive" actions (plus_20, plus_50) and tightening on "conservative" (keep).

## Epsilon-Greedy Response

Epsilon-Greedy with fixed decay cannot adapt. ε = 0.22 at Month 6 means 22% random exploration, but this is insufficient to correct the policy. The algorithm is locked onto pre-shock learned rewards. By Month 8, ε = 0.16 and the policy hasn't shifted.

## Beta Distribution Limitation

Beta posteriors don't fully "forget" old data. If Month 1-5 showed 95 successes (keep), Month 6 shock with 50 failures doesn't instantly flip keep to "bad." Thompson assigns Beta(95 + 50 successes, 5 + 50 failures) = Beta(145, 55) still mean 0.725 (72.5% success, not 50%). Full recovery takes ~2-3 months as new outcomes accumulate.

**Production fix:** Sliding-window approach—only count rewards from last 6 months, exponential decay on older data. Forget Month 1-5 learned values faster when macroeconomic regime shifts.

## Related Docs

- [cold-start.md](cold-start.md)
- [delayed-feedback.md](delayed-feedback.md)
