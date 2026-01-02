# Rollout-Free Posterior Method for Planning Depth Estimation

**Date**: 2025-12-29

## Motivation

The random rollout method showed unexpected results: experts appeared to plan less deeply than novices (Expert E[h] = 2.80 < Novice E[h] = 2.86), with a negative correlation between planning depth and win rate (r = -0.43, p = 0.006). This raised a methodological concern: random rollout creates a distribution mismatch between training and inference.

During training, the inverse models learn from actual human game continuations—constrained, strategic sequences. During inference with random rollout, we simulate futures using random opponent moves—diverse, exploratory sequences. This mismatch could bias h estimates upward, particularly for longer horizons that benefit more from seeing diverse futures.

The rollout-free method eliminates this bias by using actual future states from the game records during both training and inference.

## Approach

The training phase is identical to random rollout. For each h ∈ {1,2,3,4}, we extract (state_t, state_{t+h}, action_t) tuples from human games and train inverse models P(action_t | state_t, state_{t+h}).

The inference phase differs crucially:

Random rollout approach:
1. Observe state_t and action_t
2. Simulate future s_{t+h}^sim using random or opponent model
3. Compute P(a_t | s_t, s_{t+h}^sim) ← distribution mismatch

Rollout-free approach:
1. Observe state_t and action_t
2. Extract actual future s_{t+h}^actual from game record
3. Compute P(a_t | s_t, s_{t+h}^actual) ← matches training distribution

## Bayesian Posterior Computation

For each move t in player i's games, we compute a posterior distribution over h:

**Step 1**: Extract actual future states from the game record.

```
Game record: [s_10, s_11, s_12, s_13, s_14, ...]
Current move: t=10, action=24

Futures available:
  h=1: s_11 (actual board after 1 move)
  h=2: s_12 (actual board after 2 moves)
  h=3: s_13 (actual board after 3 moves)
  h=4: s_14 (actual board after 4 moves)
```

**Step 2**: Compute likelihoods under each h-model.

```python
for h in [1,2,3,4]:
    state_future = game_record[t+h]  # actual human future
    likelihood[h] = model_h.predict_proba(
        np.concatenate([state_t, state_future])
    )[action_t]
```

**Step 3**: Apply Bayes rule with uniform prior.

```
Prior: P(h) = [0.25, 0.25, 0.25, 0.25]
Posterior: P(h|move_t) ∝ likelihood[h] × P(h)
Normalize: P(h|move_t) = P(h|move_t) / Σ_h P(h|move_t)
```

**Step 4**: Aggregate across all moves to get per-player distribution.

```
For player i:
  P_avg(h) = mean over all moves t of P(h|move_t)
  E[h]_i = Σ_{h=1}^4 h × P_avg(h)
```

### Example Calculation

```
Move 10: Player places piece at position 24
Actual game continuation: s_10 → s_11 → s_12 → s_13 → s_14

Likelihoods from models:
  P(a=24 | s_10, s_11, h=1 model) = 0.52
  P(a=24 | s_10, s_12, h=2 model) = 0.68  ← highest
  P(a=24 | s_10, s_13, h=3 model) = 0.41
  P(a=24 | s_10, s_14, h=4 model) = 0.28

Posterior (uniform prior):
  P(h=1|move) = 0.52 / (0.52+0.68+0.41+0.28) = 0.28
  P(h=2|move) = 0.68 / 1.89 = 0.36  ← most likely
  P(h=3|move) = 0.22
  P(h=4|move) = 0.15

Interpretation: This move is best explained by h=2 planning
```

## Results

Analyzing 5,157 moves from 40 players:

```
Mean E[h]: 1.777 ± 0.118
Range: [1.594, 1.972]
Median: 1.767

Posterior distribution:
  P(h=1): 47.3%  (myopic/reactive)
  P(h=2): 24.0%  (short-term)
  P(h=3): 19.0%  (medium-term)
  P(h=4):  9.7%  (far-sighted)
```

Comparison with random rollout:

| Method | Mean E[h] | P(h=1) | P(h=4) | Difference |
|--------|-----------|--------|--------|------------|
| Random rollout | 2.866 | 12.8% | 34.9% | - |
| Rollout-free | 1.777 | 47.3% | 9.7% | -1.089 steps |
| Change | -38% | +34.5% | -25.2% | Large |

The random rollout method overestimates planning depth by inflating probability mass at longer horizons.

## Expertise Analysis

Correlations with skill measures:

```
Elo vs E[h]: r = -0.012, p = 0.943 (no correlation)
Win rate vs E[h]: r = 0.080, p = 0.623 (no correlation)
```

Group comparison:

```
Expert (top 33%): E[h] = 1.769 ± 0.117
Intermediate: E[h] = 1.786 ± 0.120
Novice (bottom 33%): E[h] = 1.768 ± 0.126

Expert vs Novice: Mann-Whitney p = 0.850, d = 0.006
```

All expertise levels show nearly identical planning depth (E[h] ≈ 1.77). The null relationship between planning depth and expertise persists even after removing the rollout artifact.

## Understanding the Distribution Mismatch

Why does random rollout overestimate h by +1.09 steps?

During training, models learn P(a | s_t, s_{t+h}^human) where s_{t+h}^human reflects actual strategic choices constrained by the opponent's actual moves. These futures are predictable and constrained.

During inference with random rollout, we compute P(a | s_t, s_{t+h}^random) where s_{t+h}^random explores many unlikely paths. These diverse futures provide more information—particularly for longer horizons. The h=4 model sees four steps of random exploration, which is more informative than four steps of constrained human play.

This asymmetry inflates P(h=4). The longer the horizon, the more diverse the random futures, the better the model's predictions appear.

The rollout-free method eliminates this by using actual human futures during inference, matching the training distribution. With constrained futures, h=1 becomes sufficient for near-term prediction. The true distribution shows P(h=1) = 47.3%.

## Myopic Planning and State-Dependence

Nearly half of all moves (47.3%) use h=1, suggesting reactive rather than far-sighted planning. This raises the question: is planning depth a stable individual trait, or does it vary by situation?

Evidence for state-dependence:
- Large within-player variance in move-level posterior P(h|move)
- Narrow between-player variance: E[h] ∈ [1.59, 1.97] (only 0.38 step range)
- All players cluster around E[h] ≈ 1.77 regardless of skill

If h were a stable trait, we'd expect wider differences between players. Instead, moves vary widely within each player while average h is similar across players. This suggests h varies more by game context than by individual identity.

Hypothesis: Planning depth adapts to game situation.
- High-threat situations → h=1 (block immediate loss)
- Low-threat situations → h=2 (setup future advantage)
- Opening moves → h=3,4 (strategic positioning)

Testing this requires analyzing move-level h estimates against game state features (threat level, board complexity, game phase).

## Reconciling with van Opheusden et al. (2023)

van Opheusden reported that experts explore 6-7 steps in their search trees (PV depth) while novices explore 7-8 steps. They concluded experts search more efficiently with better pruning.

Our finding: behavioral planning depth is 1.8 steps, much shallower than PV depth, and identical across expertise levels.

These findings are compatible. PV depth measures tree exploration breadth—how widely you search to verify your choice is good. Behavioral h measures the decision-relevant horizon—how far ahead actually matters for choosing which move to make.

You might explore 6-7 steps to check options (PV depth = 7), but only the next 2 steps determine your choice (behavioral h = 2). The remaining 5 steps are verification, not decision-making.

Experts have lower PV depth because better heuristics allow more aggressive pruning. They don't need to search as widely to verify their choice. But the decision-relevant horizon (behavioral h) is similar because task demands are similar—everyone needs to look roughly 2 steps ahead to play competently.

## Implications for IRL

Planning depth h is an identifiable latent variable that confounds reward inference. Model it explicitly when inferring rewards from behavior. Different h values produce distinguishable behavioral patterns (93.8% discriminator accuracy).

However, don't use h to predict expertise. The correlation is essentially zero across all methods. Expertise likely reflects heuristic quality (what you evaluate) rather than planning depth (how far you look ahead). Use van Opheusden features for expertise prediction instead.

When you have access to actual behavioral data with observable futures, use rollout-free methods. They eliminate distribution mismatch and provide unbiased estimates. Random rollout creates a systematic +38% upward bias for human data.

Interpret h as state-dependent rather than a fixed individual trait. The narrow between-player variance (0.38 steps) and high within-player variance suggest context drives h more than individual differences.

## Implications for Cognitive Science

The finding supports a model where expertise comes from better position evaluation (heuristics) rather than deeper planning. All skill levels plan approximately 2 steps ahead on average. Experts win because they evaluate positions better, not because they look farther ahead.

Planning depth appears context-adaptive. People don't have a fixed "planning capacity"—they adapt how far they look ahead based on the situation. Interventions targeting planning should focus on when to plan deeply vs. shallowly (adaptive control) rather than increasing average planning depth.

This matters for conditions like anxiety (might over-plan in threatening situations) or ADHD (might under-plan in calm situations). The relevant measure is not average h but sensitivity of h to context.

## Summary

The rollout-free method eliminates distribution mismatch by using actual game futures for inference. Results show E[h] = 1.78 ± 0.12, substantially lower than the random rollout estimate of 2.87 (a -1.09 step difference).

Planning depth does not predict expertise. Expert E[h] = Novice E[h] = 1.77. This null result is robust and represents a genuine phenomenon, not a methodological artifact.

Humans plan myopically (47% of moves use h=1), and planning depth appears state-dependent rather than trait-like. The decision-relevant horizon is much shallower than tree exploration depth, reconciling our findings with van Opheusden's observation that experts search 6-7 steps in their trees.

**Implementation**: `estimate_player_h_rollout_free.py`
**Results**: `results/human_h_rollout_free_estimates.csv`

**Last updated**: 2025-12-31
