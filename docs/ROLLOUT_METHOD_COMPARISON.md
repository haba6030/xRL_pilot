# Comparing Three Methods for Planning Depth Estimation

We tested three approaches for estimating planning depth h from human game-playing behavior. The methods differ in how they handle future states during inference, and this choice creates dramatic differences in results.

## Overview

| Method | Mean E[h] | Elo Correlation | Expert E[h] | Novice E[h] | Difference |
|--------|-----------|----------------|-------------|-------------|------------|
| Random Rollout | 2.866 ± 0.075 | r=-0.117, p=0.471 | 2.804 | 2.842 | -1.089 steps |
| Rollout-Free | 1.777 ± 0.118 | r=-0.012, p=0.943 | 1.769 | 1.768 | baseline |
| Opponent Model | 2.621 ± 0.104 | r=-0.029, p=0.858 | 2.608 | 2.626 | +0.844 steps |

The three methods produce systematically different h estimates despite analyzing the same human data. This reveals a substantial methodological issue: how you simulate futures during inference dramatically affects your conclusions.

## Method 1: Random Rollout

The random rollout approach trains h-specific inverse models on actual human game data, then generates trajectories by simulating futures with random opponent moves. A discriminator trained on these generated trajectories is applied to the real human data to infer h.

Training distribution: P(action_t | state_t, state_{t+h}^human)
Inference distribution: P(action_t | state_t, state_{t+h}^random)

This creates a mismatch. Training sees constrained, strategic futures that actually occurred in human games. Inference sees diverse, exploratory futures from random simulation. The h=4 model benefits disproportionately from this diversity—four steps of random exploration provides more information than four steps of constrained human play.

Results:
- E[h] = 2.87 (substantially inflated)
- P(h=4) = 34.9% (too high)
- P(h=1) = 12.8% (too low)
- Expert E[h] = Novice E[h] (no discrimination)

The +1.09 step bias is consistent across all 40 players, suggesting a systematic artifact rather than random noise.

## Method 2: Rollout-Free Posterior

The rollout-free approach eliminates simulation entirely. Instead of generating trajectories, it uses actual future states from the game records. For each observed move at time t, we extract the actual states at t+1, t+2, t+3, t+4 from the game record. We then compute P(action_t | state_t, state_{t+h}) under each h-model and apply Bayes rule to get a posterior distribution over h.

Training distribution: P(action_t | state_t, state_{t+h}^human)
Inference distribution: P(action_t | state_t, state_{t+h}^human)

This matches training and inference distributions exactly, eliminating the mismatch artifact.

Results:
- E[h] = 1.78 (unbiased estimate)
- P(h=1) = 47.3% (myopic majority)
- P(h=4) = 9.7% (rare far-sightedness)
- Expert E[h] = Novice E[h] (no discrimination)

The null result on expertise persists, indicating it's not a methodological artifact.

## Method 3: Opponent Model Rollout

The opponent model approach is a middle ground. We train a human-like opponent policy from the game data (logistic regression on board features), then use this learned policy to simulate futures. This should be more realistic than random rollout but still involves simulation rather than using actual futures.

Training distribution: P(action_t | state_t, state_{t+h}^human)
Inference distribution: P(action_t | state_t, state_{t+h}^learned)

The learned opponent approximates human behavior but doesn't perfectly match actual futures.

Results:
- E[h] = 2.62 (between random and rollout-free)
- P(h=1) = 23.7%
- P(h=4) = 23.9%
- Expert E[h] = Novice E[h] (no discrimination)

As expected, this falls between the other two methods. The +0.84 step bias relative to rollout-free is substantial but smaller than random rollout's +1.09 bias.

## Understanding the Rollout Artifact

Why does simulation bias the estimates upward?

During training, all three methods learn from the same data: actual human game continuations. These futures are constrained by strategic considerations and opponent responses. A human h=1 player makes moves based on the next state, which is predictable given tactical constraints.

During inference, random rollout simulates futures that are much more diverse. The random opponent explores unlikely paths, creating novel board configurations. The h=4 model, which looks four steps ahead, sees four steps of this diverse exploration. This provides more information than four steps of constrained human play, making the h=4 model's predictions spuriously better.

The discriminator learns to associate "diverse, exploratory futures" with "high h." When applied to human data simulated with random rollout, it systematically overestimates h.

The opponent model reduces but doesn't eliminate this problem. The learned policy is more realistic than random but still explores more widely than actual human opponents in these specific game situations.

The rollout-free method sidesteps the issue entirely by using actual futures. The h=1 model sees one step of actual constrained play, the h=4 model sees four steps of actual constrained play. No artificial diversity is injected.

## Per-Player Estimates

| Participant | Random Rollout | Rollout-Free | Opponent Model |
|-------------|----------------|--------------|----------------|
| 1 | 2.834 | 1.892 | 2.618 |
| 2 | 2.841 | 1.877 | 2.605 |
| 3 | 2.913 | 1.972 | 2.698 |
| 4 | 2.850 | 1.904 | 2.631 |
| ... | ... | ... | ... |
| Mean | 2.866 | 1.777 | 2.621 |
| Std | 0.075 | 0.118 | 0.104 |

Every player shows the same pattern: random rollout highest, rollout-free lowest, opponent model intermediate. The consistency suggests these are methodological biases, not meaningful individual differences.

## Posterior Distributions

| h | Random Rollout | Rollout-Free | Opponent Model | Shift (RF→RR) |
|---|----------------|--------------|----------------|---------------|
| h=1 | 12.8% | 47.3% | 23.7% | +34.5% |
| h=2 | 22.6% | 24.0% | 27.7% | +1.4% |
| h=3 | 29.7% | 19.0% | 24.6% | -10.7% |
| h=4 | 34.9% | 9.7% | 23.9% | -25.2% |

Random rollout shifts probability mass from h=1 to h=4 (from myopic to far-sighted). The rollout-free distribution is more myopic: nearly half of moves are best explained by h=1.

## Expertise Analysis Across Methods

None of the three methods show a relationship between planning depth and expertise:

Correlation with Elo rating:
- Random rollout: r = -0.117, p = 0.471
- Rollout-free: r = -0.012, p = 0.943
- Opponent model: r = -0.029, p = 0.858

Group comparison (Expert vs Novice):
- Random rollout: 2.804 vs 2.842, p > 0.05
- Rollout-free: 1.769 vs 1.768, p > 0.05
- Opponent model: 2.608 vs 2.626, p > 0.05

This consistency is informative. If planning depth predicted expertise, we'd expect at least one method to detect it. The robust null result across methods with different biases suggests planning depth genuinely doesn't correlate with skill in this task.

## Myopic Planning is the Norm

The rollout-free estimate (E[h] = 1.78) is substantially lower than random rollout (E[h] = 2.87). Nearly half of all moves are best explained by h=1 (reactive planning). Only 10% require h=4 (far-sighted planning).

This is much shallower than van Opheusden's finding that players explore 6-7 steps in their search trees (PV depth). The difference likely reflects the distinction between exploration breadth (how widely you search) and decision horizon (how far ahead matters for your choice).

You might explore 6-7 steps to verify your preferred move is good, but only the next 2 steps actually determine which move you choose. The remaining 4-5 steps are verification and pruning, not decision-making.

## Practical Guidance

When estimating planning depth from behavioral data, use rollout-free methods whenever possible. They eliminate distribution mismatch, require no simulation, provide Bayesian posteriors with uncertainty quantification, and run efficiently.

If you must simulate futures (e.g., for off-policy evaluation or real-time inference in novel situations), use a learned opponent model rather than random rollout. While not perfect, it reduces bias substantially.

Don't use planning depth to predict expertise, at least in strategic games like 4-in-a-row. The null relationship is robust across methods. Expertise more likely reflects heuristic quality (van Opheusden features achieve 84% accuracy) than planning depth.

When reporting h estimates, always specify the inference method. The choice matters enormously—a 38% difference in this case. Method comparisons should be standard practice for planning depth estimation.

## Reconciling Different Planning Depth Measures

Our behavioral estimate (h ≈ 1.8) differs dramatically from van Opheusden's tree exploration depth (PV depth ≈ 6-7). Both can be correct if they measure different aspects of planning:

PV depth: How many steps you explore in your search tree (breadth)
Behavioral h: How many steps determine your choice (decision horizon)

These decompose as: PV depth = h_behavioral + h_verification

Example: You explore 6 steps (PV = 6), but only the first 2 steps determine your choice (h = 2), and the remaining 4 steps verify your choice is good (verification = 4).

Experts have lower PV depth (more efficient search via better pruning) but similar behavioral h (similar task demands). Everyone needs to look roughly 2 steps ahead to play competently. Experts just don't need to verify as widely because their heuristics are better.

## Summary

Three methods for estimating planning depth produce systematically different results due to train-inference distribution mismatch. Random rollout overestimates h by +1.09 steps (38%) by introducing artificial diversity during simulation. Opponent model rollout reduces this to +0.84 steps. Rollout-free eliminates the artifact by using actual game futures.

Humans plan myopically in 4-in-a-row (E[h] = 1.78, with 47% of moves reactive h=1 planning), much shallower than tree exploration metrics suggest. This likely reflects the difference between exploration breadth and decision horizon.

Planning depth does not predict expertise across all three methods. The robust null result suggests expertise reflects heuristic quality rather than planning depth, consistent with van Opheusden's finding that board evaluation features predict skill with 84% accuracy.

The rollout-free method is recommended for future work on planning depth estimation from behavioral data.

**Last updated**: 2025-12-31
