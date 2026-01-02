# Features vs Planning Depth: What Predicts Expertise?

We compared two approaches to predicting expertise in 4-in-a-row: van Opheusden's 17-dimensional heuristic features versus our estimated planning depth h. The question is whether skill reflects better heuristics (what you evaluate) or deeper planning (how far you look ahead).

## Summary

| Metric | Feature-Based | h-Based | Difference |
|--------|--------------|---------|------------|
| AUC (Expertise) | 0.840 | 0.530 | +58.5% |
| Accuracy | 77.5% | 75.0% | +2.5% |
| Mean \|r\| with Elo | 0.035 | 0.012 | Negligible |
| Significant predictors | 0/17 individual | 0/1 | Both weak individually |
| Multivariate pattern | Strong | Chance level | Features win |

Features are 58.5% better than planning depth for expertise prediction. Planning depth provides essentially no information about skill (AUC = 0.53, barely above chance).

## Individual Feature Correlations

Surprisingly, no single feature correlates significantly with Elo rating:

| Rank | Feature | r (Elo) | p-value | Significant? |
|------|---------|---------|---------|--------------|
| 1 | 4-in-a-row horizontal | -0.244 | 0.130 | No |
| 2 | Connected 2-in-a-row diag1 | +0.054 | 0.742 | No |
| 3 | 3-in-a-row diag1 | +0.049 | 0.765 | No |
| 4 | 4-in-a-row vertical | -0.035 | 0.831 | No |
| 5 | 4-in-a-row diag2 | +0.033 | 0.840 | No |
| ... | ... | ... | ... | ... |
| 17 | Center control | +0.015 | 0.925 | No |

Statistics:
- Significant features (p < 0.05): 0 / 17
- Mean |correlation|: 0.035 (very weak)
- Range: -0.244 to +0.054

This suggests expertise is not about excelling at any single aspect of play. Instead, it reflects a balanced profile across multiple heuristic dimensions.

## Multivariate Pattern: Combining Features

While individual features don't correlate with expertise, their combination does:

Method: Logistic Regression with all 17 features

Results:
- AUC: 0.840 (strong discrimination)
- Accuracy: 77.5%
- Classification: 10 experts vs 30 non-experts

The large gap between individual (weak) and multivariate (strong) performance indicates expertise is about the pattern of heuristics, not any single dimension. Experts don't just excel at one thing—they have better heuristics across the board, and the combination is what matters.

## Planning Depth Performance

Method: Logistic Regression with E[h] only

Results:
- AUC: 0.530 (chance level)
- Accuracy: 75.0%
- Correlation with Elo: r = -0.012, p = 0.943

Planning depth provides no information about expertise. The AUC of 0.53 is barely above 0.5 (random classifier). The near-zero correlation with Elo (r = -0.012) is robust.

## Comparing AUCs

The AUC difference is 0.840 - 0.530 = +0.310, a 58.5% improvement.

Statistical interpretation:
- Features: Strong discriminator (AUC >> 0.5)
- h: Random classifier (AUC ≈ 0.5)

Practical interpretation:
If you want to predict whether someone is an expert, use their heuristic features (84% AUC). Using planning depth is no better than flipping a coin (53% AUC).

## Reconciling with van Opheusden et al. (2023)

van Opheusden reported that expert PV depth (6.23 steps) is significantly lower than novice PV depth (7.29 steps), with r = -0.50, p < 0.01. They titled their paper "Expertise increases planning depth in human gameplay" but their actual finding is that experts search more efficiently with shallower trees.

Our finding: Expert h = 1.769, Novice h = 1.768, r = -0.012, p = 0.94. Planning depth shows no relationship with expertise.

These findings are compatible because PV depth and behavioral h measure different things:

PV depth (van Opheusden):
- Search tree exploration metric
- Measures computational effort and pruning efficiency
- Lower for experts (efficient search, better pruning)

Planning depth h (our work):
- Behavioral decision horizon
- Measures how far ahead determines your choice
- Same across skill levels (task demands are similar)

Both findings support the same conclusion: expertise comes from better heuristics (what you evaluate), not deeper planning (how far you look ahead). Experts search more efficiently because better heuristics enable aggressive pruning, but the decision-relevant horizon is similar for everyone.

## What This Means for Expertise

Expertise in 4-in-a-row reflects a multivariate heuristic pattern, not any single feature or planning depth. Individual features are weak predictors (mean |r| = 0.035), but their combination strongly predicts skill (AUC = 0.84).

This is similar to chess expertise, where skill comes from a balanced repertoire across positions rather than excellence in one opening or endgame type. Novices have poor heuristics across the board. Experts have high-quality heuristics in multiple dimensions simultaneously.

Planning depth h is orthogonal to expertise. All skill levels use h ≈ 1.8 on average. This robust null result has important implications for IRL: model h explicitly as a latent confounder, but don't use it as an expertise marker.

## Context-Dependent Planning Hypothesis

Everyone averages h ≈ 1.8, but individual moves vary widely within players. This suggests h might vary by game context rather than player identity.

Testable predictions:
- High-threat situations → lower h (reactive, block immediate loss)
- Low-threat situations → higher h (strategic, setup future advantage)
- Opening moves → higher h (exploratory, establish position)
- Time pressure → lower h (forced to respond quickly)

Testing this requires analyzing move-level h estimates against game state features (threat level, board complexity, game phase). If confirmed, it would suggest planning depth is context-adaptive rather than a stable cognitive trait.

## Planning Depth Across All Three Methods

| Method | Mean E[h] | Expert E[h] | Novice E[h] | Elo r | Expertise AUC |
|--------|-----------|-------------|-------------|-------|---------------|
| Random Rollout | 2.87 ± 0.08 | 2.80 | 2.84 | -0.12 (ns) | ~0.53 |
| Rollout-Free | 1.78 ± 0.12 | 1.77 | 1.77 | -0.01 (ns) | 0.53 |
| Opponent Model | 2.62 ± 0.10 | 2.61 | 2.63 | -0.03 (ns) | ~0.53 |
| Features (17-dim) | N/A | N/A | N/A | 0.035 (ns) | 0.84 |

All three h estimation methods show the same null relationship with expertise (AUC ≈ 0.53). This consistency across methods with different biases makes the negative result robust. Features are the only strong predictor.

## Implications for IRL

Planning depth h should be modeled explicitly in IRL because it's an identifiable latent variable that confounds reward inference (different h produces different behavior). However, don't interpret h as a skill measure or use it to predict expertise.

When inferring rewards from behavior:
- Model h as latent confounder (control for it)
- Use heuristic features to predict expertise (not h)
- Interpret reward differences after controlling for h

This deconfounding is conceptually important even though h doesn't correlate with skill. Two people might choose differently because they have different rewards OR because they plan different depths ahead. Explicitly modeling h separates these factors.

## Implications for Cognitive Science

The finding that individual features are weak but their combination is strong suggests expertise involves developing a balanced heuristic profile rather than maximizing any single dimension. Interventions should target heuristic quality broadly rather than focusing on isolated skills.

Planning depth appears state-dependent rather than trait-like (narrow between-player variance, high within-player variance). If confirmed, this means planning "capacity" is not fixed—people adapt how far they look ahead based on the situation. Clinical applications should focus on when people plan deeply vs. shallowly (adaptive control) rather than average planning depth.

## Summary

Van Opheusden's 17-dimensional heuristic features predict expertise with 84% AUC through a multivariate pattern (individual features are weak, combination is strong). Planning depth provides no expertise information (AUC = 53%, essentially chance).

This null result is robust across three h estimation methods (random rollout, rollout-free, opponent model), all showing Expert h ≈ Novice h with correlations near zero.

The findings are compatible with van Opheusden's report that expert PV depth is lower than novice: both support the conclusion that expertise reflects better heuristics (what you evaluate) rather than deeper planning (how far you look ahead). Experts search more efficiently with better pruning, but the decision-relevant horizon is similar for everyone.

For IRL applications, model h explicitly as a latent confounder but use features (not h) to predict expertise.

**Last updated**: 2025-12-31
