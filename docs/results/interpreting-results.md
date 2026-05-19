# Interpreting Results

What good results look like. Red flags (5 failure patterns with causes/fixes). Cohort heatmap. Epsilon-Greedy underperformance Q&A.

## What Good Results Look Like

| Criterion | Target | Thompson |
|-----------|--------|----------|
| Revenue Lift | +30%+ | +39.09% ✓ |
| Regret % | <20% | 16.34% ✓ |
| Convergence | Month <6 | Month 5 ✓ |
| Default Rate | 3-4% | 3.38% ✓ |
| Exploration % | 1-5% | 1.8% ✓ |

Thompson is **green across all metrics:** high revenue, low regret, fast convergence, risk controlled, efficient exploration.

## 5 Red Flags

### Flag 1: Default Rate Spike (>4%)

**Pattern:** Default rate jumps to 5%+ suddenly.

**Likely Cause:** Economic shock (Month 6 simulation) or policy too aggressive (over-exploration in Deep-Subprime).

**Fix:** Adjust safety-score thresholds (restrict plus_50 access). Widen posteriors on risky actions. Monitor macroeconomic indicators (unemployment, GDP).

### Flag 2: Regret Climbing (not flat by Month 5)

**Pattern:** Regret remains 30%+ at Month 8.

**Likely Cause:** Algorithm stuck in local optimum (Epsilon-Greedy decay too fast, random early action won). Insufficient exploration.

**Fix:** Use Thompson Sampling (faster convergence). Reduce epsilon decay constant (0.995 → 0.98). Extend cold-start uniform exploration phase.

### Flag 3: Action Distribution Imbalanced

**Pattern:** Plus_50 selected 0% for Prime users by Month 8.

**Likely Cause:** Early bad experiences with plus_50, posterior tightened to 5% success (unlucky sample). Algorithm lock-in.

**Fix:** Check historical outcomes for plus_50 Prime (should be 85%+). If actually low, plus_50 is bad; if unlucky sampling, use Thompson to re-explore.

### Flag 4: Revenue Plateaus Before Month 8

**Pattern:** Monthly revenue peaks Month 4, stays flat 5-8.

**Likely Cause:** Algorithm converged to suboptimal action too early. Early randomness favored inferior action.

**Fix:** Check posterior shapes. If actions have tight, similar posteriors (e.g., all Beta(80, 20)), algorithm confused. Add more features or longer cold-start exploration.

### Flag 5: Divergent Cohort Performance

**Pattern:** Prime tier revenue up 60%, Deep-Subprime down 20%.

**Likely Cause:** Policy tier-specific miscalibration. Deep-Subprime over-explored (too many plus_10+ attempts). Prime under-explored (locked on keep).

**Fix:** Check safety-score formula. Prime threshold too high? Deep-Subprime threshold too low? Rebalance tier-specific exploration budgets.

## Cohort Heatmap

Revenue lift by (policy, risk_tier):

| Policy | Prime | Near-Prime | Subprime | Deep-Subprime |
|--------|-------|-----------|----------|--------------|
| Thompson | +42% | +38% | +32% | +18% |
| UCB | +28% | +24% | +18% | +12% |
| Epsilon | −2% | +1% | −1% | −3% |
| Static | 0% | 0% | 0% | 0% |

Thompson green across all tiers. Epsilon weak/negative. Deep-Subprime shows smallest lift (higher default risk limits upside).

## Epsilon-Greedy Underperformance: Q&A

### Q1: Why did Epsilon-Greedy underperform (−0.26% vs baseline)?

**A:** Epsilon-Greedy's decay 0.995^t is independent of when rewards actually arrive (Month 4). By Month 4, ε = 0.32 (68% exploitation). If early randomness (Months 1-3) favored "keep" action, the policy locks there at Month 4 despite "plus_10" being better for Prime users. Thompson's posterior sampling would continue exploring plus_10 (high posterior variance) and discover its superiority.

### Q2: What should we have done differently with Epsilon-Greedy?

**A:** Two fixes: (1) Increase initial ε to 0.5 and slow decay to 0.98^t (ε = 0.28 at Month 12, more exploration). (2) Implement adaptive epsilon that increases when feedback arrives (Month 4: bump ε back to 0.4 when first outcomes land, re-explore). Neither was implemented; we used fixed decay.

### Q3: Why not just increase Epsilon and slow decay?

**A:** Even with ε = 0.5 forever (pure 50/50 exploration), you lose ₹500K/user from not concentrating on best action by Month 8. Thompson's principled uncertainty-based exploration is fundamentally more sample-efficient than random exploration at any fixed ε.

## Related Docs

- [metrics-explained.md](metrics-explained.md)
- [../concepts/delayed-feedback.md](../concepts/delayed-feedback.md)
