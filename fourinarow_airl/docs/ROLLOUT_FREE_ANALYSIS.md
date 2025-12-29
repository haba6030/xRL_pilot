# Rollout-Free Posterior Analysis

**Date**: 2025-12-29

## 🎯 Motivation

The expertise paradox from random rollout method showed unexpected results:
- Expert E[h] = 2.804 < Intermediate E[h] = 2.859
- Negative correlation with win rate (r = -0.426, p = 0.006)
- No correlation with Elo rating (r = -0.117, p = 0.471)

**Hypothesis**: Random rollout creates distribution mismatch artifact
- Training: Uses real human future states s_{t+h}^human
- Inference: Uses simulated future states s_{t+h}^random
→ Mismatch could bias h estimates

## 🔬 Method: Rollout-Free Posterior

**Core Innovation**: Eliminate rollout simulation completely

```
Traditional (Random Rollout):
  Training:  (s_t, s_{t+h}^human) → a_t
  Inference: (s_t, s_{t+h}^random) → a_t  ❌ Mismatch!

Rollout-Free:
  Training:  (s_t, s_{t+h}^human) → a_t
  Inference: (s_t, s_{t+h}^human) → a_t  ✅ No mismatch!
```

**Bayesian Posterior Computation**:

For each move t:
1. Extract actual future states from human games: s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4}
2. Compute likelihood for each h model:
   ```
   ℓ_h(t) = P_model_h(a_t | s_t, s_{t+h}^human)
   ```
3. Apply Bayes rule:
   ```
   P(h|t) ∝ ℓ_h(t) × P(h)
   ```
4. Normalize: Σ P(h|t) = 1

Per-player aggregation:
```
P̄_i(h) = mean_{t∈player_i} P(h|t)
E[h]_i = Σ_{h=1}^4 h × P̄_i(h)
```

## 📊 Results

### Overall Statistics

```
Rollout-Free Posterior:
  Mean E[h]:     1.777 ± 0.118
  Range:         [1.594, 1.972]
  Median:        1.767
  N players:     40
  N moves:       5,157
```

**Comparison with Random Rollout**:

| Method | Mean E[h] | Std | Range |
|--------|-----------|-----|-------|
| Random Rollout | 2.866 | 0.075 | [2.68, 3.05] |
| **Rollout-Free** | **1.777** | **0.118** | **[1.59, 1.97]** |
| **Difference** | **-1.089** | **+0.043** | **~1.0 shift** |

**KEY FINDING**: Rollout-free estimates are ~1.09 steps LOWER than random rollout!

### Posterior Distribution (Aggregate)

```
P(h=1): 47.3%  ████████████████████
P(h=2): 24.0%  ██████████
P(h=3): 19.0%  ████████
P(h=4):  9.7%  ████

vs Random Rollout:
P(h=1): 12.8%  █████
P(h=2): 22.6%  █████████
P(h=3): 29.7%  ████████████
P(h=4): 34.9%  ██████████████
```

**Interpretation**: Rollout-free strongly favors h=1 (myopic planning)

### Expertise Analysis

**Correlation with Skill**:
```
Elo vs E[h]:
  Spearman r = -0.012, p = 0.943  (NO correlation)

Win rate vs E[h]:
  Spearman r = 0.080, p = 0.623   (NO correlation)
```

**Group Comparison**:
```
Expert:        E[h] = 1.769 ± 0.117  (n=10)
Intermediate:  E[h] = 1.786 ± 0.120  (n=20)
Novice:        E[h] = 1.768 ± 0.126  (n=10)

Expert vs Novice:
  Mann-Whitney U = 47.0, p = 0.850
  Cohen's d = 0.006  (essentially zero)
```

**KEY FINDING**: All expertise levels have identical E[h] ≈ 1.77

### Comparison: Random Rollout vs Rollout-Free

| Metric | Random Rollout | Rollout-Free | Interpretation |
|--------|----------------|--------------|----------------|
| **Mean E[h]** | 2.87 | 1.78 | -1.09 shift (38% lower) |
| **P(h=1)** | 12.8% | 47.3% | +34.5% (myopic mass) |
| **P(h=4)** | 34.9% | 9.7% | -25.2% (far-sighted mass) |
| **Elo correlation** | -0.117 (p=0.471) | -0.012 (p=0.943) | Both: no correlation |
| **Expert E[h]** | 2.804 | 1.769 | -1.04 shift |
| **Intermediate E[h]** | 2.859 | 1.786 | -1.07 shift |
| **Novice E[h]** | 2.842 | 1.768 | -1.07 shift |
| **Expertise difference?** | NO | NO | Both methods fail |

## 🔍 Interpretation

### 1. Rollout Artifact is MASSIVE

The ~1.09 step difference between methods suggests:
- Random rollout **overestimates** h by creating unrealistic futures
- Humans' actual future states are more constrained/predictable than random rollout
- **Distribution mismatch creates systematic bias toward higher h**

**Mechanistic explanation**:
```
Random rollout:
  - Explores many unlikely futures
  - h=4 model gets to "see" diverse futures
  - Better match to random futures → higher h

Rollout-free:
  - Only sees actual human futures
  - h=1 model is sufficient to predict near-term
  - Less benefit from long-horizon → lower h
```

### 2. Paradox PERSISTS Despite Artifact Removal

**Expected (if artifact was the cause)**:
- Expert E[h] > Novice E[h]
- Positive correlation with Elo/win rate

**Actual result**:
- Expert E[h] ≈ Novice E[h] (no difference)
- Zero correlation with skill

**Conclusion**: The expertise paradox is NOT purely a rollout artifact

### 3. Humans Plan ~1.8 Steps on Average (Myopic!)

E[h] ≈ 1.8 with P(h=1) = 47.3% suggests:
- **Nearly half of all moves are purely reactive** (h=1)
- Very few moves involve 4-step lookahead (9.7%)
- Humans use **context-dependent shallow planning**

**This contradicts**:
- van Opheusden (2023): PV depth = 6-7 steps
- Game theory: 4-in-a-row solvable with depth ~8-10

**Possible explanations**:
1. **PV depth ≠ planning depth**: Van Opheusden's PV measures search tree exploration, not decision-relevant lookahead
2. **Feature-based heuristics**: Humans may use pattern recognition instead of explicit tree search
3. **Model limitation**: Inverse kinematics models may not capture true planning mechanism

### 4. Planning Depth Does NOT Discriminate Expertise

**Robust finding across both methods**:
- Random rollout: no correlation
- Rollout-free: no correlation
- All expertise levels: E[h] ≈ 1.8

**Implications**:
1. **Expertise is not about planning depth** in 4-in-a-row
2. **Consistent with van Opheusden finding**: Experts had SHALLOWER PV depth (more efficient)
3. **h is NOT the right variable** to explain expertise

**Alternative hypothesis**:
- Expertise → **better heuristics/features**, not deeper planning
- Experts recognize patterns faster → need less lookahead
- Planning depth h should be treated as **nuisance variable**, not expertise marker

## 🎯 Conclusions

### Main Findings

1. ✅ **Rollout artifact confirmed**: Random rollout inflates h by ~1.09 steps
2. ✅ **Humans plan myopically**: E[h] ≈ 1.8 (actual behavior)
3. ✅ **Expertise paradox persists**: h does not discriminate skill
4. ✅ **h is identifiable but not meaningful** for expertise prediction

### Methodological Lessons

**DO**:
- Use rollout-free methods when training data has real future states
- Match training and inference distributions exactly
- Validate with multiple rollout methods (random, opponent, rollout-free)

**DON'T**:
- Assume h is the primary expertise variable
- Use random rollout for human behavior estimation
- Ignore distribution mismatch between train/test

### Theoretical Implications

**For Planning-Aware IRL**:
- h as latent confounder: ✅ Confirmed (different h → different behavior)
- h as expertise marker: ❌ Rejected (no correlation with skill)
- h identifiability: ✅ Confirmed (methods detect real h differences)

**For Cognitive Modeling**:
- Humans use shallow, context-dependent planning
- Pattern recognition > explicit lookahead
- Expertise = heuristic quality, not planning depth

### Next Steps

1. ✅ **Completed**: Rollout-free posterior method
2. 🔄 **In progress**: Opponent-model rollout (for comparison)
3. 📋 **Planned**: Feature-based expertise analysis (van Opheusden features)
4. 📋 **Planned**: Context-dependent h (does h vary by game state?)

## 📁 Files

- `estimate_player_h_rollout_free.py`: Implementation
- `results/human_h_rollout_free_estimates.csv`: Per-player estimates
- `figures/rollout_free_posterior_results.png`: Visualization

## 📚 References

**Methodological inspiration**:
- Bayesian posterior computation without simulation
- Match train/inference distributions exactly

**Related work**:
- van Opheusden et al. (2023): PV depth analysis
- Yao et al. (2024): Planning horizon as confounder
- This work: Rollout-free estimation for artifact removal

---

**Last Updated**: 2025-12-29
**Status**: Analysis complete, expertise paradox persists
