# Rollout-Free Posterior Method for Planning Depth Estimation

**Date**: 2025-12-29

## Problem Statement

Random rollout method showed unexpected correlation between planning depth and expertise:
- Expert E[h] = 2.804 < Intermediate E[h] = 2.859
- Negative correlation with win rate (r = -0.426, p = 0.006)
- No correlation with Elo rating (r = -0.117, p = 0.471)

**Hypothesis**: Random rollout creates distribution mismatch, biasing h estimates upward.

**Solution**: Rollout-free posterior estimation using actual human futures from data.

---

## Method: Rollout-Free Posterior

### Core Innovation

Eliminate simulation by using observed future states from game records.

**Training phase** (same as random rollout):
```
For each h ∈ {1,2,3,4}:
  Extract (state_t, state_{t+h}, action_t) from human games
  Train inverse model: P(action_t | state_t, state_{t+h})
```

**Inference phase** (different from random rollout):
```
Random rollout approach:
  1. Observe state_t and action_t
  2. Simulate future: s_{t+h}^sim using random/opponent model
  3. Compute P(a_t | s_t, s_{t+h}^sim) ← distribution mismatch!

Rollout-free approach:
  1. Observe state_t and action_t
  2. Extract actual future: s_{t+h}^actual from game record
  3. Compute P(a_t | s_t, s_{t+h}^actual) ← matches training distribution
```

### Bayesian Posterior Computation

For each move t in player i's games:

**Step 1**: Extract actual future states
```
Game record: [s_10, s_11, s_12, s_13, s_14, ...]
Current move: t=10, action=24

Futures available:
  h=1: s_11 (actual board after 1 move)
  h=2: s_12 (actual board after 2 moves)
  h=3: s_13 (actual board after 3 moves)
  h=4: s_14 (actual board after 4 moves)
```

**Step 2**: Compute likelihoods under each h-model
```python
for h in [1,2,3,4]:
    state_future = game_record[t+h]  # actual human future
    likelihood[h] = model_h.predict_proba(
        np.concatenate([state_t, state_future])
    )[action_t]
```

**Step 3**: Apply Bayes rule
```
Prior: P(h) = uniform = [0.25, 0.25, 0.25, 0.25]

Posterior: P(h|move_t) ∝ likelihood[h] × P(h)

Normalize: P(h|move_t) = P(h|move_t) / Σ_h P(h|move_t)
```

**Step 4**: Aggregate per player
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

---

## Results

### Overall Distribution

**Rollout-free posterior** (40 players, 5,157 moves):
```
Mean E[h]: 1.777 ± 0.118
Range: [1.594, 1.972]
Median: 1.767

Posterior distribution:
  P(h=1): 47.3%  ████████████████████ (myopic/reactive)
  P(h=2): 24.0%  ██████████ (short-term)
  P(h=3): 19.0%  ████████ (medium-term)
  P(h=4):  9.7%  ████ (far-sighted)
```

**Comparison with random rollout**:

| Method | Mean E[h] | P(h=1) | P(h=4) | Shift |
|--------|-----------|--------|--------|-------|
| Random rollout | 2.866 | 12.8% | 34.9% | - |
| Rollout-free | 1.777 | 47.3% | 9.7% | -1.089 steps |
| Difference | -38% | +34.5% | -25.2% | Large |

**Interpretation**: Random rollout overestimates planning depth by inflating long-horizon probability mass.

### Expertise Analysis

**Correlation with skill** (rollout-free method):
```
Elo vs E[h]: r = -0.012, p = 0.943 (no correlation)
Win rate vs E[h]: r = 0.080, p = 0.623 (no correlation)
```

**Group comparison**:
```
Expert (top 33%): E[h] = 1.769 ± 0.117
Intermediate: E[h] = 1.786 ± 0.120
Novice (bottom 33%): E[h] = 1.768 ± 0.126

Expert vs Novice: Mann-Whitney p = 0.850, d = 0.006 (no difference)
```

**Finding**: All expertise levels plan identically (E[h] ≈ 1.77). The expertise paradox persists even after removing rollout artifact.

---

## Interpretation

### Finding 1: Distribution Mismatch Causes +1.09 Step Bias

Random rollout overestimates h because:
1. Training uses actual human futures (constrained, predictable)
2. Inference uses simulated futures (diverse, exploratory)
3. Higher h models benefit more from diverse futures
4. Result: Systematic bias toward h=4

**Mechanistic explanation**:
```
Training: P(a | s_t, s_{t+h}^human)
  - s_{t+h}^human reflects actual strategic choices
  - Constrained by opponent's actual moves

Inference (random rollout): P(a | s_t, s_{t+h}^random)
  - s_{t+h}^random explores many unlikely paths
  - h=4 model sees more diverse futures → better predictions
  - Inflates P(h=4)

Inference (rollout-free): P(a | s_t, s_{t+h}^human)
  - Matches training distribution
  - h=1 sufficient for near-term prediction
  - True distribution: P(h=1) = 47.3%
```

### Finding 2: Humans Plan Myopically (E[h] ≈ 1.8)

Nearly half of all moves (47.3%) use h=1 (reactive planning). This suggests:

**State-dependent planning hypothesis**:
- High-threat situations → reactive (h=1): Block immediate loss
- Low-threat situations → short-term (h=2): Setup future advantage
- Opening moves → may use h=3,4: Strategic positioning

**Evidence for state-dependence**:
- Large within-player variance in posterior P(h|move)
- E[h] range across players: only [1.59, 1.97] (narrow)
- Move-by-move variation: high

**Contrast with trait-like planning**:
- If h were a stable trait, expect wider between-player variance
- Actual: similar E[h] across all players (~ 1.77)
- Suggests h varies more by game context than by player identity

### Finding 3: Expertise Paradox is Not an Artifact

**Hypothesis tested**: "Random rollout underestimates expert planning due to unrealistic futures"

**Result**: Rejected
- Rollout-free method shows identical E[h] for experts and novices
- Both methods show no h-expertise correlation
- Artifact removal does not reveal hidden relationship

**Conclusion**: Planning depth h is orthogonal to expertise.

### Finding 4: PV Depth ≠ Behavioral Planning Depth

**van Opheusden (2023) reported**:
- PV depth: 6-7 steps (search tree exploration)
- Experts have lower PV depth (more efficient search)

**Our finding**:
- Behavioral h: 1.8 steps (decision-relevant lookahead)
- Experts and novices have identical behavioral h

**Reconciliation**:
```
PV depth = h_behavioral + h_verification

Example:
  PV depth = 6 steps (tree exploration)
  h_behavioral = 2 steps (decision horizon)
  h_verification = 4 steps (checking/pruning)

Experts:
  - Lower PV depth (efficient pruning)
  - Same h_behavioral (task demands)
  - Lower h_verification (better heuristics)
```

---

## Implications

### For Planning-Aware IRL

**DO**:
- Model h as latent confounder (confirmed: different h → different behavior)
- Use rollout-free methods when training data has actual futures
- Interpret h as state-dependent variable, not trait

**DON'T**:
- Use h to predict expertise (correlation ≈ 0)
- Use random rollout for human data (distribution mismatch)
- Assume fixed h per individual (likely state-dependent)

### For Cognitive Modeling

**Supported**:
- Humans use shallow planning (E[h] = 1.8, not 6-7)
- Planning is likely state-dependent (high move-level variance)
- Expertise comes from heuristic quality, not planning depth

**New hypotheses**:
- Context-dependent h: Vary by threat level, board density, game phase
- Skill manifests in when to plan deep, not average depth

---

## Summary

**Method**: Rollout-free posterior estimation using actual human futures

**Key results**:
1. E[h] = 1.78 ± 0.12 (vs 2.87 from random rollout)
2. P(h=1) = 47.3% (myopic planning dominates)
3. Expert E[h] = Novice E[h] (no expertise relationship)
4. Random rollout overestimates by +1.09 steps (38%)

**Main finding**: Planning depth is identifiable from behavior but does not predict expertise. The expertise paradox is a genuine phenomenon, not a methodological artifact.

**Files**:
- `estimate_player_h_rollout_free.py`: Implementation
- `results/human_h_rollout_free_estimates.csv`: Per-player estimates

**Last updated**: 2025-12-31
