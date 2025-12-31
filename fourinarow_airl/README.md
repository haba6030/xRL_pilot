# Planning Depth Estimation from Human Game-Playing Behavior

This project estimates how many steps ahead humans plan when playing the 4-in-a-row board game. We use behavioral data from van Opheusden et al. (2023) and develop three methods to infer planning depth (h) from observed actions.

---

## Research Questions

### RQ1: Can we identify planning depth from behavior alone?

**Method**: Train inverse models to predict action from current state and future state h steps ahead: P(action | state_current, state_future). Use discriminator to classify h from (state, action) pairs.

**Result**: Planning depth is identifiable (discriminator accuracy: 93.8%).

**Example**:
- Given board position at move 10 and board position at move 14
- Inverse model predicts: P(action=24 | state_10, state_14) = 0.62
- This likelihood differs for h=1,2,3,4, allowing discrimination

---

### RQ2: What is human planning depth in 4-in-a-row?

**Method**: Apply three estimation approaches to 40 human players:
1. Random rollout: Simulate future with random opponent
2. Opponent model: Simulate future with learned opponent policy
3. Rollout-free: Use actual human futures from data

**Results**:

| Method | Mean E[h] | Interpretation |
|--------|-----------|----------------|
| Random rollout | 2.87 ± 0.08 | Overestimated (+1.09 bias) |
| Opponent model | 2.62 ± 0.09 | More realistic |
| Rollout-free | 1.78 ± 0.12 | Unbiased estimate |

**Distribution (rollout-free)**:
- 47% of moves: h=1 (immediate response)
- 24% of moves: h=2 (short-term planning)
- 19% of moves: h=3 (medium-term planning)
- 10% of moves: h=4 (long-term planning)

**Interpretation**: Humans plan approximately 2 steps ahead on average, with considerable move-by-move variation. The rollout-free method eliminates distribution mismatch between training (actual futures) and inference.

---

### RQ3: Does planning depth correlate with expertise?

**Method**: Correlate estimated E[h] with Elo ratings and win rates.

**Result**: Planning depth does NOT predict expertise.

| Method | Elo correlation | Win rate correlation | Expert vs Novice E[h] |
|--------|----------------|---------------------|----------------------|
| Random rollout | r = -0.12, p = 0.47 | r = -0.43, p = 0.006 | 2.80 vs 2.84 (ns) |
| Rollout-free | r = -0.01, p = 0.94 | r = +0.08, p = 0.63 | 1.77 vs 1.77 (identical) |

**Alternative approach - Feature-based expertise**:
- Van Opheusden features (17-dim: center control, threats, connected pieces)
- Logistic regression: AUC = 0.84 (expert classification)
- Comparison: Features predict expertise strongly, h does not (AUC = 0.53, chance level)

**Interpretation**: Expertise comes from heuristic quality (better position evaluation) rather than planning depth. Experts and novices plan similarly deep but evaluate positions differently.

---

### RQ4: Can planning depth explain clinical variability?

**Status**: Future work. Requires clinical population data (anxiety, ADHD, etc.).

**Hypothesis**: Clinical traits may affect planning parameters (depth, consistency, noise) rather than reward preferences. For example, anxiety might reduce planning depth due to computational load under stress.

---

## Methods

### Overview

This analysis is part of a larger pipeline for planning-aware inverse reinforcement learning (IRL). The full pipeline has 3 stages:

**Stage 1** (this project): Estimate planning depth h from behavior
**Stage 2** (future): Learn reward function conditioned on h: r(state, action, h)
**Stage 3** (future): Apply to safety-critical domains (pedestrian crossing)

We have completed Stage 1 with three methods:

### Method 1: Random Rollout (Multi-Step Inverse Kinematics)

**Training phase**:
1. Extract (state_t, state_{t+h}, action_t) tuples from human game data
2. Train separate inverse models for each h: P(action | state_t, state_{t+h})
3. Example: For h=2, use board at move 10 and board at move 12 to predict action at move 10

**Inference phase**:
1. For each human move, simulate h-step futures using random opponent
2. Compute likelihood under each h-specific model
3. Use Bayesian posterior: P(h | move) ∝ P(move | h) · P(h)

**Problem**: Random opponent creates unrealistic futures, causing +1.09 step overestimation.

**Example**:
```
Move 10: Player places piece at position 24
- Simulate random opponent for h=1: likelihood = 0.45
- Simulate random opponent for h=2: likelihood = 0.62  ← highest
- Simulate random opponent for h=3: likelihood = 0.38
- Simulate random opponent for h=4: likelihood = 0.21
→ Estimated h=2 for this move
```

---

### Method 2: Opponent Model Rollout

Same as Method 1, but opponent policy is learned from human data instead of random.

**Improvement**: More realistic futures, reduces bias to E[h] = 2.62.

**Limitation**: Still requires simulation, computational cost higher.

---

### Method 3: Rollout-Free Posterior (Main Innovation)

**Key idea**: Use actual human futures from data instead of simulating.

**Training**: Identical to Method 1 (train inverse models on real data)

**Inference** (different):
1. For move at timestep t, extract actual board state at t+h from game record
2. Compute likelihood directly: P(action_t | state_t, state_{t+h}^actual)
3. No simulation needed

**Advantage**: Eliminates distribution mismatch, unbiased h estimates.

**Example**:
```
Move 10: Player places piece at position 24
Game record shows board at move 11, 12, 13, 14 (actual human game)

- Use actual board at move 11: P(action=24 | state_10, state_11) = 0.52
- Use actual board at move 12: P(action=24 | state_10, state_12) = 0.68  ← highest
- Use actual board at move 13: P(action=24 | state_10, state_13) = 0.41
- Use actual board at move 14: P(action=24 | state_10, state_14) = 0.28

Bayesian posterior:
P(h=1|move) = 0.31
P(h=2|move) = 0.49  ← most likely
P(h=3|move) = 0.15
P(h=4|move) = 0.05

→ E[h] = 1*0.31 + 2*0.49 + 3*0.15 + 4*0.05 = 1.74
```

**Why this works**: Inverse models were trained on (state_t, state_{t+h}^human, action_t), so inference should use human futures too, not random/simulated ones.

---

## Key Findings

### Finding 1: Rollout Method Matters

Random rollout overestimates h by +1.09 steps (38%) due to distribution mismatch. Rollout-free eliminates this artifact.

### Finding 2: Humans Plan Myopically

E[h] ≈ 1.8 (rollout-free estimate). Most moves use h=1 or h=2, not deep lookahead.

Comparison with van Opheusden search depth:
- Their PV depth (search tree metric): 6-7 steps
- Our behavioral h (decision horizon): 1.8 steps
- Interpretation: Humans explore deeply but commit decisions based on shallow lookahead.

### Finding 3: Expertise ≠ Planning Depth

Planning depth does not correlate with skill (Elo, win rate). Expert and novice h estimates are nearly identical across all three methods.

Alternative: Van Opheusden features (heuristic quality) predict expertise with AUC = 0.84.

**Implication**: Skill comes from what you evaluate (heuristics), not how far you look ahead (depth).

### Finding 4: Feature-Based Expertise is Multivariate

Individual features show weak correlations with Elo (mean |r| = 0.035, none significant). Combined features show strong discrimination (AUC = 0.84).

**Interpretation**: Expertise is a balanced pattern across multiple heuristics, not one dominant feature.

---

## Repository Structure

### Core Analysis Scripts

```
estimate_player_h_rollout_free.py    - Main analysis: rollout-free h estimation
estimate_player_h_multiclass.py      - Random rollout h estimation
estimate_player_h_opponent.py        - Opponent model h estimation
analyze_feature_based_expertise.py   - Feature-based expertise analysis
generate_trajectories_opponent_model.py - Opponent policy training
```

### Supporting Code

```
env.py            - 4-in-a-row game environment
features.py       - Van Opheusden 17-feature extraction
data_loader.py    - Load human game data
```

### Data

```
data/
├── multistep_ik/               - Training data for inverse models
│   ├── ik_pairs_h1.pkl         - (state_t, state_{t+1}, action_t) pairs
│   ├── ik_pairs_h2.pkl         - (state_t, state_{t+2}, action_t) pairs
│   ├── ik_pairs_h3.pkl
│   └── ik_pairs_h4.pkl
└── human_elo_ratings.csv       - Elo ratings for 40 players
```

### Results

```
results/
├── human_h_rollout_free_estimates.csv    - E[h] per player (rollout-free)
├── human_h_multiclass_estimates.csv      - E[h] per player (random rollout)
├── human_h_opponent_estimates.csv        - E[h] per player (opponent model)
├── player_van_opheusden_features.csv     - 17-dim features per player
└── feature_elo_correlations.csv          - Feature-Elo correlations
```

### Documentation

```
docs/
├── ROLLOUT_FREE_ANALYSIS.md              - Rollout-free method details
├── ROLLOUT_METHOD_COMPARISON.md          - Three-method comparison
├── FEATURE_VS_H_COMPARISON.md            - Feature vs h analysis
├── COMPLETE_ANALYSIS_SUMMARY.md          - Integrated summary (EN)
├── 완전_분석_요약_KR.md                  - Integrated summary (KR)
└── van_Opheusden_비교_논의_KR.md         - van Opheusden reconciliation (KR)
```

### Deprecated Files (in backup/)

Previous approaches and intermediate analyses have been moved to `backup/` folder.

---

## Installation

```bash
pip install numpy pandas scipy matplotlib seaborn
pip install scikit-learn joblib
```

---

## Usage Example

```bash
# Run rollout-free h estimation (recommended method)
python estimate_player_h_rollout_free.py

# Output: results/human_h_rollout_free_estimates.csv
# Contains E[h] per player, P(h=1,2,3,4) per move

# Run feature-based expertise analysis
python analyze_feature_based_expertise.py

# Output: results/player_van_opheusden_features.csv
#         figures/feature_based_expertise_analysis.png
```

---

## Future Work: Planning-Aware IRL

The current analysis (Stage 1) enables future work on planning-aware inverse reinforcement learning:

**Stage 2**: Learn reward function conditioned on h
- Standard IRL assumes fixed h, causing confounded reward estimates
- Planning-aware IRL: r(state, action, h) accounts for varying planning depths
- Hypothesis: Deconfounding h improves reward identifiability

**Stage 3**: Apply to pedestrian crossing domain
- Safety-critical behavior (different from game-playing)
- Test if planning depth varies with clinical traits (anxiety, ADHD)
- Hypothesis: Anxiety → altered planning depth → risky behavior

---

## References

**van Opheusden, B., Acerbi, L., & Ma, W. J. (2023).** Expertise increases planning depth in human gameplay. *Nature*, 618(7965), 1000-1005.
- Source of human data (40 players, 318 games, 5,482 moves)
- 17-dimensional features for position evaluation

**Mhammedi, Z., Helou, D., Grau-Moya, J., Bou-Ammar, H., & Whiteson, S. (2023).** Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL. *International Conference on Machine Learning (ICML)*.
- Multi-step inverse kinematics framework
- Adapted for behavior analysis (originally for representation learning)

**Yao, W., Zhao, P., Qiao, Y., Abbeel, P., & Ding, Y. (2024).** Inverse Reinforcement Learning with Multiple Planning Horizons. *Conference on Learning Theory (COLT)*.
- Planning horizon as latent confounder in IRL
- Theoretical motivation for explicit h modeling

---

**Last Updated**: 2024-12-31
**Status**: Stage 1 complete (h estimation), Stage 2-3 future work
**Main Finding**: Planning depth is identifiable (93.8% accuracy) but does not predict expertise (AUC = 0.53). Expertise comes from heuristic quality, not planning depth.
