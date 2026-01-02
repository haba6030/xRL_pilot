# Executive Summary: Rollout Method Comparison

This document provides an overview of three methods for estimating planning depth from behavioral data: random rollout, rollout-free posterior, and opponent model rollout.

## Core Question

Can we accurately estimate human planning depth h from observed actions in game-playing?

Answer: Yes, but the estimation method matters enormously. Random rollout creates a +38% bias by mismatching training and inference distributions. The rollout-free method eliminates this artifact by using actual game futures.

## Three Methods Compared

| Method | Train Distribution | Inference Distribution | Bias | E[h] |
|--------|-------------------|----------------------|------|------|
| Random Rollout | (s_t, s_{t+h}^human) | (s_t, s_{t+h}^random) | +1.09 | 2.87 |
| Rollout-Free | (s_t, s_{t+h}^human) | (s_t, s_{t+h}^human) | None | 1.78 |
| Opponent Model | (s_t, s_{t+h}^human) | (s_t, s_{t+h}^learned) | +0.84 | 2.62 |

Random rollout trains on actual human game continuations but simulates futures with random opponent moves during inference. This distribution mismatch causes longer-horizon models to benefit disproportionately from the diversity of random futures, inflating h estimates.

Rollout-free eliminates simulation entirely, using actual future states from the game records. Training and inference distributions match exactly.

Opponent model is a middle ground, simulating futures with a learned human-like opponent policy. More realistic than random but still introduces some mismatch.

## Key Findings

Random rollout has a massive artifact: +1.09 step overestimation (38% bias). The distribution shift is dramatic:
- Random: P(h=1)=13%, P(h=2)=23%, P(h=3)=30%, P(h=4)=35%
- Rollout-free: P(h=1)=47%, P(h=2)=24%, P(h=3)=19%, P(h=4)=10%

Probability mass shifts from h=1 (reactive) to h=4 (far-sighted), making humans appear to plan much deeper than they actually do.

Humans plan myopically. The rollout-free estimate shows E[h] = 1.78, with nearly half of all moves best explained by h=1 (reactive planning). Only 10% of moves involve h=4 (far-sighted planning).

Planning depth does not predict expertise. All three methods show the same null result:
- Random rollout: r(Elo, E[h]) = -0.12, p = 0.47
- Rollout-free: r(Elo, E[h]) = -0.01, p = 0.94
- Opponent model: r(Elo, E[h]) = -0.03, p = 0.86

Expert h ≈ Novice h across all methods. This robust null result indicates planning depth is orthogonal to expertise in this task.

## Understanding the Distribution Mismatch

Why does random rollout overestimate h?

Training: Models learn P(action | state_t, state_{t+h}^human) where future states reflect actual strategic choices constrained by opponent behavior.

Inference with random rollout: We compute P(action | state_t, state_{t+h}^random) where futures are simulated with random opponent moves.

Random futures are more diverse than human futures. The h=4 model sees four steps of random exploration, which provides more information than four steps of constrained human play. The discriminator learns to associate "diverse futures" with "high h," creating systematic overestimation.

The rollout-free method sidesteps this by using actual futures: h=1 model sees one step of real constrained play, h=4 sees four steps of real constrained play. No artificial diversity.

## Comparison with van Opheusden PV Depth

van Opheusden reported that players explore 6-7 steps in their search trees (PV depth), with experts having lower PV depth (6.23 steps) than novices (7.29 steps), r = -0.50, p < 0.01.

Our finding: behavioral planning depth h = 1.78 steps, much shallower than PV depth, with Expert h = Novice h (no difference).

Both findings are compatible. PV depth measures search tree exploration breadth (how widely you search to verify choices). Behavioral h measures decision-relevant horizon (how far ahead determines your choice).

Decomposition: PV depth = behavioral h + verification depth

You might explore 6 steps to verify your choice is good (PV = 6), but only the first 2 steps determine which move you make (h = 2). The remaining 4 steps are pruning and verification.

Experts have lower PV depth because better heuristics enable efficient pruning. But the decision-relevant horizon is similar for everyone because task demands are similar.

## Practical Recommendations

Use rollout-free methods whenever you have access to actual behavioral data with observable futures. They eliminate distribution mismatch, require no simulation, provide Bayesian posteriors with uncertainty quantification, and run efficiently.

If you must simulate futures (off-policy evaluation, real-time inference in novel situations), use a learned opponent model rather than random rollout. While not perfect, it substantially reduces bias (+0.84 steps vs +1.09 steps).

Don't expect planning depth to predict expertise, at least in strategic games like 4-in-a-row. The null relationship is robust across all three methods with different biases. Expertise likely reflects heuristic quality rather than planning depth.

When reporting h estimates, always specify the inference method. The choice creates a 38% difference in this analysis. Method comparisons should be standard practice.

## Implications

For IRL: Model planning depth h explicitly as a latent confounder (different h produces distinguishable behavior), but don't use it to predict expertise. Use heuristic quality measures instead.

For cognitive science: PV depth and behavioral h measure different aspects of planning. Tree exploration breadth (PV depth) reflects computational effort and pruning efficiency. Decision horizon (behavioral h) reflects how far ahead matters for choices. Expertise manifests in better heuristics enabling efficient pruning, not in deeper decision horizons.

For methodology: Always match training and inference distributions when estimating latent variables from behavior. Distribution mismatch creates systematic artifacts that can be larger than real effects.

## Summary

Three methods for estimating planning depth produce systematically different results (1.78 to 2.87) due to train-inference distribution mismatch. Random rollout overestimates by +38%, opponent model by +32%, while rollout-free provides unbiased estimates.

Humans plan myopically (E[h] = 1.78, 47% of moves reactive h=1), much shallower than tree exploration metrics (PV depth = 6-7) suggest. The difference reflects that PV depth measures verification breadth while h measures decision horizon.

Planning depth does not predict expertise across all three methods (robust null result). Expertise likely reflects heuristic quality, consistent with van Opheusden's finding that better heuristics enable efficient pruning (lower PV depth for experts).

The rollout-free method is recommended for future work on planning depth estimation from behavioral data.

**Last updated**: 2025-12-31
