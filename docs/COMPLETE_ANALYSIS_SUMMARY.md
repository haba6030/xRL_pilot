# Integrated Analysis: Planning Depth and Expertise

**Date**: 2025-12-29

This document integrates findings from three h estimation methods (random rollout, rollout-free, opponent model) and compares planning depth with van Opheusden's heuristic features for predicting expertise.

## Core Question

Can planning depth h identify expertise in 4-in-a-row gameplay?

Answer: No. Planning depth is identifiable from behavior but orthogonal to expertise. Van Opheusden's heuristic features (17-dimensional board evaluation metrics) predict expertise with 84% AUC, while planning depth performs at chance level (53% AUC).

## Complete Results

Method comparison for h estimation:

| Method | Mean E[h] | Expert h | Novice h | Elo r | Win Rate r | Expertise AUC |
|--------|-----------|----------|----------|-------|------------|---------------|
| Random Rollout | 2.87 ± 0.08 | 2.80 | 2.84 | -0.12 (ns) | -0.43** | ~0.53 |
| Rollout-Free | 1.78 ± 0.12 | 1.77 | 1.77 | -0.01 (ns) | +0.08 (ns) | 0.53 |
| Opponent Model | 2.62 ± 0.10 | 2.61 | 2.63 | -0.03 (ns) | +0.06 (ns) | ~0.53 |

All three methods show h does not predict expertise.

Feature-based alternative:

| Method | Dimension | Individual r | Combined AUC | Interpretation |
|--------|-----------|-------------|--------------|----------------|
| van Opheusden Features | 17-dim | 0.035 (mean) | 0.84 | Strong predictor |
| Planning Depth h | 1-dim | 0.012 | 0.53 | Chance level |

Features predict expertise through a multivariate pattern. Individual features are weak (mean |r| = 0.035, 0/17 significant), but their combination strongly discriminates experts from novices.

## Finding 1: Rollout Method Matters

Random rollout overestimates h by +1.09 steps (38% bias). This happens because training uses actual human game continuations (constrained, strategic) while inference simulates futures with random moves (diverse, exploratory). Longer-horizon models benefit disproportionately from diverse futures, inflating P(h=4).

Distribution shift:
- Random Rollout: P(h=1)=13%, P(h=2)=23%, P(h=3)=30%, P(h=4)=35%
- Rollout-Free: P(h=1)=47%, P(h=2)=24%, P(h=3)=19%, P(h=4)=10%

The rollout-free method eliminates this artifact by using actual game futures during both training and inference. Opponent model rollout reduces but doesn't eliminate the bias (+0.84 steps relative to rollout-free).

Recommendation: Use rollout-free methods whenever actual futures are observable in the data.

## Finding 2: Humans Plan Myopically

Rollout-free estimate: E[h] = 1.78 ± 0.12

Distribution:
- 47% of moves: h=1 (reactive, immediate response)
- 24% of moves: h=2 (short-term planning)
- 19% of moves: h=3 (medium-term planning)
- 10% of moves: h=4 (far-sighted planning)

This is much shallower than van Opheusden's finding that players explore 6-7 steps in their search trees (PV depth). The difference reflects the distinction between tree exploration breadth (how widely you search) and decision horizon (how far ahead determines your choice).

PV depth = behavioral h + verification depth

Example: You explore 6 steps (PV = 6) to verify your choice is good, but only the first 2 steps determine which move you make (h = 2). The remaining 4 steps are pruning and verification, not decision-making.

## Finding 3: Expertise Paradox is Robust

Planning depth shows no relationship with expertise across all three methods:
- Random rollout: r = -0.12, p = 0.47, Expert h = Novice h
- Rollout-free: r = -0.01, p = 0.94, Expert h = Novice h
- Opponent model: r = -0.03, p = 0.86, Expert h = Novice h

This null result persists even after removing the rollout artifact. It represents a genuine phenomenon: h is orthogonal to expertise in this task.

## Finding 4: Features Predict Expertise Strongly

Van Opheusden's 17-dimensional features achieve AUC = 0.84 for expert classification, compared to h's AUC = 0.53 (chance level). The difference of +0.31 represents a 58.5% improvement.

Individual features are weak (mean |r| = 0.035 with Elo, 0/17 significant at p < 0.05), but their multivariate combination is strong. This indicates expertise reflects a balanced heuristic profile across multiple dimensions rather than excellence in any single aspect.

Top features by correlation magnitude:
1. 4-in-a-row horizontal: r = -0.244, p = 0.130 (not significant)
2. Connected 2-in-a-row diag1: r = +0.054, p = 0.742
3. 3-in-a-row diag1: r = +0.049, p = 0.765

Even the strongest individual feature doesn't reach significance. Expertise is about the pattern, not individual components.

## Reconciling with van Opheusden et al. (2023)

van Opheusden reported "Expertise increases planning depth in human gameplay" but their actual finding is that expert PV depth (6.23 steps) is lower than novice PV depth (7.29 steps), r = -0.50, p < 0.01. They concluded experts search more efficiently with shallower trees.

Our finding: Expert behavioral h = 1.769, Novice h = 1.768, r = -0.012, p = 0.94. Planning depth shows no relationship with expertise.

Both findings are compatible because PV depth and behavioral h measure different constructs:

PV depth (van Opheusden):
- Search tree exploration metric measuring computational effort
- Lower for experts (efficient pruning via better heuristics)

Behavioral h (our work):
- Decision-relevant horizon measuring how far ahead determines choices
- Same across skill levels (task demands are similar)

Both support the same conclusion: expertise comes from better heuristics (what you evaluate), not deeper planning (how far you look ahead). Experts search more efficiently because better heuristics enable aggressive pruning, but everyone needs to look roughly 2 steps ahead to play competently.

## Expertise Mechanism Model

Novice behavior:
- Poor heuristics across multiple dimensions
- Deep search to compensate (brute-force, high PV depth)
- Low performance
- h ≈ 1.8 (same as expert)

Expert behavior:
- High-quality heuristics across multiple dimensions
- Shallow search sufficient (efficient pruning, low PV depth)
- High performance
- h ≈ 1.8 (same as novice)

The key insight: h is controlled by task demands, not expertise. Expertise manifests in heuristic quality, not planning depth.

## Implications for IRL

Model h explicitly as a latent confounder when inferring rewards from behavior. Different h values produce distinguishable behavioral patterns (93.8% discriminator accuracy), so failing to model h can confound reward estimates.

However, don't use h to predict expertise. The correlation is essentially zero across all methods. Use van Opheusden features (AUC = 0.84) instead.

When you have access to actual behavioral data with observable futures, use rollout-free methods. They eliminate distribution mismatch (+38% bias removed) and provide unbiased h estimates with Bayesian uncertainty quantification.

## Implications for Cognitive Science

The finding that individual features are weak but their combination is strong suggests expertise involves developing a balanced heuristic profile rather than maximizing any single dimension. Interventions should target heuristic quality broadly rather than isolated skills.

Planning depth appears state-dependent rather than trait-like. All players cluster around E[h] ≈ 1.77 (narrow between-player variance of 0.38 steps), but individual moves vary widely within each player. This suggests h varies more by game context than by individual identity.

If planning depth is context-dependent, the relevant question is not "how far ahead does this person plan" but "when does this person plan deeply vs. shallowly." For clinical applications (anxiety, ADHD), analyze sensitivity of h to context rather than average h.

## Summary

Three methods for estimating planning depth produce different absolute values (1.78 to 2.87) due to rollout artifacts, but all show the same null relationship with expertise. Rollout-free is the most accurate method (eliminates +38% bias).

Humans plan myopically in 4-in-a-row (E[h] = 1.78, with 47% of moves reactive h=1 planning). This is much shallower than tree exploration metrics (PV depth = 6-7) because PV depth measures verification breadth while behavioral h measures decision horizon.

Planning depth does not predict expertise (AUC = 53%, chance level), but van Opheusden's heuristic features do (AUC = 84%). Expertise reflects multivariate heuristic quality, not planning depth. This reconciles with van Opheusden's finding that experts have lower PV depth: both support the view that expertise comes from better heuristics enabling efficient search.

For IRL, model h as a latent confounder but use features (not h) to predict expertise.

**Last updated**: 2025-12-31
