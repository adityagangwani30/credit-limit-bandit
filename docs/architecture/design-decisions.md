# Architecture Design Decisions

Six critical decisions with tradeoffs: two-branch linear normalization vs sigmoid, Thompson as primary algorithm, 3-month feedback delay, risk tier stratification, four discrete actions, explicit column naming.

## Decision 1: Two-Branch Linear Normalization

**Choice:** Linear (x-min)/(max-min) instead of sigmoid.

**Why:** Sigmoid squashes tails (300 CIBIL → 0.05, 850 → 0.95). Thompson needs to distinguish high/low credit quality. Linear preserves: 300→0.0, 850→0.92. Bandit learns different posteriors for different scores.

**Tradeoff:** Less robust to distribution shifts. Production monitoring needed.

## Decision 2: Thompson Sampling Primary Algorithm

**Choice:** Use Thompson Sampling despite complexity.

**Results:** ₹12.13Cr (+39.09% lift, 16.34% regret) vs UCB (₹10.83Cr, 25.32%) vs Epsilon (₹8.70Cr, −0.26%).

**Why:** 8.9% regret gap (16.34% vs 25.32%) justifies implementation complexity. Thompson's posterior-driven exploration outperforms UCB's fixed √(ln(n)/count) schedule.

**Tradeoff:** Harder to explain to non-technical stakeholders. Sampling adds randomness to decisions.

## Decision 3: 3-Month Feedback Delay

**Choice:** Wait 90 days for default outcomes instead of using proxy signals.

**Why:** Default is the economically critical outcome. Proxies (missed payments) are correlated but imperfect. 3 months aligns with credit industry standard and regulatory definitions.

**Tradeoff:** Months 1-3 have zero reward signal, forcing cold-start exploration. Cold-start costs ~₹50M on Deep-Subprime cohort (testing aggressive actions on risky users).

## Decision 4: Risk Tier Stratification (40/30/20/10)

**Choice:** Prime 40% / Near-Prime 30% / Subprime 20% / Deep-Subprime 10%.

**Why:** Reflects real bank portfolios. Uniform distribution would overweight risky users and distort learned policies.

**Tradeoff:** Requires synthetic data generation. Benefit: learned policies applicable to realistic portfolios.

## Decision 5: Four Discrete Actions

**Choice:** keep / plus_10 / plus_20 / plus_50 instead of continuous [0, 100K] increase.

**Why:** Reduces exploration from ∞ to 4 finite options. Thompson Sampling scales naturally to 4 actions. Four choices span key business levers.

**Tradeoff:** Some information loss (can't express "plus_25"). Benefit: exploration efficiency gains outweigh this loss.

## Decision 6: Explicit Column Names

**Choice:** reward_received, action_taken, policy (vs abbreviated: reward, action, algo).

**Why:** Clear, unambiguous column names benefit analysts/engineers/compliance. Policy values: thompson_sampling, ucb, epsilon_greedy.

**Tradeoff:** +0.5% CSV file size. Negligible; clarity worth it.

## Related Docs

- [overview.md](overview.md)
- [data-flow.md](data-flow.md)
