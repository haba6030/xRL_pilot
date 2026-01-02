# Planning Depth Estimation from Human Behavior

Bayesian inference of decision-relevant temporal horizons from human strategic gameplay. Analysis of 40 players (5,482 moves) from van Opheusden et al. (Nature 2023) 4-in-a-row dataset.

---

## Research Questions

This project addresses four research questions on planning-aware inverse reinforcement learning:

### RQ1: Can planning depth be estimated as an explicit behavioral variable?

**Question**: Can we define and infer planning depth h as an explicit factor in human decision-making, distinct from reward functions?

**Theoretical motivation**: IRL theory shows planning horizon acts as a latent confounder—if ignored, behavioral variation from different planning depths gets misattributed to reward differences, breaking identifiability (Yao et al., 2024).

**Operational approach**: Estimate planning depth h ∈ {1,2,3,4} from observed actions by training horizon-specific models π_h(a_t | s_t, s_{t+h}) and computing Bayesian posterior P(h | player's moves).

### RQ2: Does planning depth discriminate expertise levels?

**Question**: Do experts plan deeper than novices in strategic games?

**Prior expectation**: Cognitive modeling suggests expertise reflects deeper planning rather than fundamentally different heuristics (van Opheusden et al., 2023 reports experts have shallower tree exploration, suggesting more efficient planning).

**Test**: Compare estimated E[h] across expertise levels (Elo ratings, win rates, expert/novice groups).

### RQ3: How does explicit planning modeling improve IRL interpretability?

**Question**: Does modeling planning depth as a latent variable improve our ability to interpret behavioral variation in IRL settings?

**Approach**: Compare two explanations for behavioral differences:
- Standard IRL: All variation attributed to reward differences
- Planning-aware: Variation decomposed into reward differences + planning depth differences

**Test**: TBA with Yao(2024) approach

### RQ4: Can planning mechanisms explain individual differences beyond rewards?

**Question**: Do individual differences (expertise, clinical traits) reflect planning mechanisms rather than only reward function differences?

**Test**: TBA

---

## Current Analysis Scope

This analysis addresses **RQ1** and **RQ2** :

**RQ1 (Planning depth as explicit variable)**:
- ✓ Developed estimation method using future-state-conditioned inverse models
- ✓ Demonstrated behavioral identifiability (93.8% discriminator accuracy h=1 vs h=4)
- ✓ Estimated per-player planning depths: E[h] = 1.78 ± 0.12
- ⚠ Limitation: Doesn't directly measure planning as it's not forward simulation

**RQ2 (Planning depth and expertise)**:
- ✓ Tested correlation: Elo vs. E[h], r = -0.01, p = 0.94
- ✓ Tested correlation: Win rate vs. E[h], r = 0.08, p = 0.62
- ✓ Tested group differences: Experts E[h] = 1.77, Novices E[h] = 1.77
- ✓ Robustness: Null result consistent across three estimation methods
- ✓ Finding: Planning depth does NOT discriminate expertise

---

## Motivation

### IRL identifiability problem

Inverse reinforcement learning (IRL) typically assumes fixed planning horizon across all agents. However, behavioral differences can arise from:
- Different reward functions (what agents value)
- Different planning depths (temporal scope of decision-making)

Without explicitly modeling planning depth h, reward estimates become confounded: variation in h gets misattributed to variation in rewards, breaking identifiability (Yao et al., 2024).

### van Opheusden's tree exploration findings

van Opheusden et al. (2023) reported:
- Experts: 6-7 steps of tree exploration (principal variation depth)
- Novices: 7-8 steps of tree exploration
- Conclusion: Experts plan more efficiently (better pruning, not deeper search)

However, tree exploration depth (how widely you search) differs from decision-relevant horizon (how far ahead influences your choice). A player might explore 7 steps to verify a move is safe, but only use 2-step information to decide which move to make.
Also, as those planning depths are from heuristics, implying computational modeling may validate such results. 

### Our approach

We estimate decision-relevant horizon from behavioral data using future-state-conditioned inverse models. For each planning depth h, we train:

```
π_h(a_t | s_t, s_{t+h})
```

where s_{t+h} is the board state h steps into the actual game continuation. This differs from goal-conditioned policies: s_{t+h} is observed outcome, not specified goal.

Inference uses Bayesian framework:
```
P(h | player's moves) ∝ ∏_moves P(action | s_t, s_{t+h})^h × P(h)
```

This idea is from Mhammedi(2023), using inverse kinematics, estimating policy by inversely constructing policies.  

---

## Data

**Source**: van Opheusden et al. (2023)

Statistics:
- 40 human players (all reasonably skilled)
- 318 games (human vs. human)
- 5,482 moves with complete board states
- Elo ratings: 1464–1535 (narrow range limits expertise tests)

Game properties:
- 6×6 board
- Two-player zero-sum
- Full observability (no hidden information)
- Average game length: 17.2 moves

---

## Methods

### Training: h-specific inverse models

**Data extraction**:
```
For each h ∈ {1, 2, 3, 4}:
  For each game:
    For each timestep t where t+h < game_length:
      Extract tuple: (s_t, s_{t+h}, a_t)

Example (h=2):
  Move 10: board = s_10, action = place at position 24
  Move 12: board = s_12 (actual outcome after player + opponent moves)
  Training pair: input = concat(s_10, s_12), label = 24
```

**Model architecture**:
- Input: concatenate(s_t, s_{t+h}) = 178-dim
  - s_t: 89-dim (36 black pieces + 36 white pieces + 17 van Opheusden features)
  - s_{t+h}: 89-dim (same structure)
- Architecture: MLP (256-128-64 hidden layers)
- Output: softmax over 36 board positions
- Loss: cross-entropy

**Training details**:
- Split: 80% train / 20% validation
- Optimizer: Adam
- Early stopping on validation accuracy
- Final accuracy: ~50-55% (4× chance, reasonable for complex game)

### Inference: Bayesian depth estimation

For each player's moves in test data:

```python
# Example: Player 5, Move 10
s_t = current_board_state
a_t = observed_action  # e.g., position 24

# Get actual game continuations
s_{t+1}, s_{t+2}, s_{t+3}, s_{t+4} = actual_future_states

# Compute likelihoods
P(a_t | s_t, s_{t+1}) = model_h1.predict_proba([s_t, s_{t+1}])[a_t] = 0.52
P(a_t | s_t, s_{t+2}) = model_h2.predict_proba([s_t, s_{t+2}])[a_t] = 0.68  ← highest
P(a_t | s_t, s_{t+3}) = model_h3.predict_proba([s_t, s_{t+3}])[a_t] = 0.41
P(a_t | s_t, s_{t+4}) = model_h4.predict_proba([s_t, s_{t+4}])[a_t] = 0.28

# Bayesian posterior (uniform prior)
P(h=1 | move) = 0.52 / 1.89 = 0.28
P(h=2 | move) = 0.68 / 1.89 = 0.36  ← most likely
P(h=3 | move) = 0.41 / 1.89 = 0.22
P(h=4 | move) = 0.28 / 1.89 = 0.15

# Interpretation: h=2 model best predicts this action
```

Aggregate over all moves for per-player distribution:
```
E[h]_player = Σ_moves P(h | move) × h
```

### Three estimation methods

**Method 1: Rollout-free**
- Use actual game continuations for s_{t+h}
- Matches training distribution
- No simulation bias
- Limitation: Only works for retrospective analysis (requires complete game records)

**Method 2: Random rollout (standard baseline)**
- Simulate s_{t+h} by random opponent moves
- Fast, no opponent model needed
- Problem: Distribution mismatch (training=human futures, inference=random futures)
- Result: Systematic overestimation (+1.09 steps, 38% bias)

**Method 3: Opponent model rollout**
- Simulate s_{t+h} using learned human opponent policy
- Trained: LogisticRegression on all human moves (55% accuracy)
- More realistic than random, still has some mismatch
- Result: Intermediate bias (E[h]=2.62 vs. 1.78 rollout-free)

---

## Results

### 1. Estimated planning depths

**Rollout-free method** (unbiased):
```
Overall: E[h] = 1.78 ± 0.12

Per-move distribution:
  h=1: 47% (reactive/immediate)
  h=2: 24% (short-term)
  h=3: 19% (medium-term)
  h=4: 10% (far-sighted)

Per-player range: 1.59 to 1.97 (0.38 step spread)
```

Comparison to van Opheusden's principal variation (PV) depth:
- van Opheusden PV depth: 6-7 steps (tree exploration)
- Our decision horizon: 1.78 steps (decision-relevant scope)
- Interpretation: Participants could've not planned as deep as reported by observing state as a whole rather than heuristic based. 

### 2. Planning depth vs. expertise

**Correlation analysis**:
```
Elo rating vs. E[h]:
  r = -0.01, p = 0.94 (no correlation)

Win rate vs. E[h]:
  r = 0.08, p = 0.62 (not significant)
```

**Group comparison** (tertile split):
```
Experts (top 33%):    E[h] = 1.77
Intermediate (mid):   E[h] = 1.78
Novices (bottom 33%): E[h] = 1.77

F-test: F(2,37) = 0.02, p = 0.98 (no difference)
```

Robustness check across all three methods:
```
                    Experts   Novices   Diff
Rollout-free:       1.77      1.77      0.00
Random rollout:     2.86      2.88     +0.02
Opponent rollout:   2.61      2.63     +0.02
```

Null result persists across estimation methods.

### 3. Rollout method bias

Comparison of estimation methods:
```
Method              E[h]    Bias vs. rollout-free
Rollout-free:       1.78    —
Opponent rollout:   2.62    +0.84 (+47%)
Random rollout:     2.87    +1.09 (+61%)
```

Mechanism:
1. Training data: Human games (constrained, strategic continuations)
2. Random rollout: Simulated futures (diverse, exploratory)
3. Longer-horizon models see more diverse futures → benefit more

However, each method need extra validation as they have several underlying implications.

### 4. Within-player vs. between-player variation

```
Between-player variance: σ²_between = 0.012
Within-player variance:  σ²_within = 0.089

Ratio: 7.4× more variation within players than between
```

Interpretation: Planning depth varies more by move context than by player identity. Suggests state-dependent adaptation rather than stable trait.

### 5. van Opheusden features predict expertise

Feature-based classification (using 17 van Opheusden features):
```
Logistic regression accuracy: 84%
Top 3 predictive features:
  1. 3-in-a-row detection (coefficient: +2.4)
  2. Center control (coefficient: +1.8)
  3. Connected 2-in-a-row (coefficient: +1.3)
```

Suggests expertise reflects heuristic quality (what you evaluate) rather than planning depth (how far you look).

---

## Methodological Limitations

### What we measure vs. what we claim

**We measure**: Statistical association between actions and (s_t, s_{t+h}) pairs.

**We infer**: Decision-relevant temporal horizon h.

**We assume**: This reflects planning depth via forward simulation.

**We cannot distinguish**:
- Forward planning: "I simulate 2 steps ahead to choose action"
- Pattern recognition: "I recognize 2-step patterns and respond"

Example ambiguity:
```
Situation: Opponent has 3-in-a-row, one empty space
Action: Block the threat (h=1 has highest likelihood)

Possible interpretation A:
  Player simulated 1 step: "If I don't block, opponent wins"
  → Forward planning with h=1

Possible interpretation B:
  Player recognized threat pattern: "3-in-a-row = must block"
  → No explicit simulation, heuristic response

Our method: Cannot distinguish these cases
```

### Two-player confound

Critical issue: s_{t+h} is not determined by a_t alone.

In two-player games:
```
s_{t+h} = f(s_t, a_t, a_t^opp, a_{t+1}, a_{t+1}^opp, ..., a_{t+h-1}^opp)
                   ↑      ↑       ↑         ↑              ↑
                 player opponent player  opponent    opponent
```

Consequence: "Action to reach s_{t+h}" is ill-defined. The same a_t can lead to different s_{t+h} depending on opponent responses.

What we actually capture: Statistical dependency between (a_t, s_{t+h}) conditional on s_t, where s_{t+h} reflects both players' actions.

Assumption: This dependency structure varies systematically with planning depth h despite opponent confounding.

### Model architecture limitations

**Simple concatenation**:
- Input: [s_t || s_{t+h}] (naïve concatenation)
- No explicit temporal structure
- No attention mechanism to weight features by horizon
- Network must discover temporal relationships implicitly

Alternative approaches (not implemented):
- Trajectory encoding: model (s_t → s_{t+1} → ... → s_{t+h})
- Attention over time steps
- Separate encoders for current vs. future state

**Linear combination**:
- MLP learns weighted combinations of features
- May not capture complex interactions during actual planning
- Sufficient for statistical discrimination (93.8% accuracy) but not necessarily mechanistic

---

## Interpretation and Implications

### For inverse reinforcement learning

Planning depth h has dual nature:

**As latent variable**:
- Identifiable from behavior (discriminator accuracy 93.8%)
- Varies within individuals (state-dependent)
- Should be modeled explicitly to avoid confounding

Practical guidance for IRL:
```
Recommended: Condition reward learning on estimated h
  r(s, a | h) instead of r(s, a)
```

---

## Repository Structure

```
Main analysis:
  estimate_player_h_rollout_free.py       # Recommended method
  estimate_player_h_multiclass.py         # Random rollout baseline
  estimate_player_h_opponent.py           # Opponent model method
  analyze_feature_based_expertise.py      # Feature-expertise analysis

Data preprocessing:
  preprocess_multistep_ik_data.py         # Extract (s_t, s_{t+h}, a_t)
  generate_trajectories_opponent_model.py # Train opponent policy

Environment and features:
  env.py                                  # 4-in-a-row game (6×6 board)
  features.py                             # 17 van Opheusden features
  data_loader.py                          # Load human game records

Data:
  data/multistep_ik/                      # Training data per h
  opendata/raw_data.csv                   # Original game records (5,482 moves)

Results:
  results/human_h_rollout_free_estimates.csv    # Per-player E[h]
  results/player_van_opheusden_features.csv     # Heuristic features

Documentation:
  EXECUTIVE_SUMMARY.md                    # One-page overview
  docs/ROLLOUT_FREE_ANALYSIS.md           # Method details
  docs/METHOD_COMPARISON.md               # Comparison to MusIK
  docs/FEATURE_VS_H_COMPARISON.md         # Expertise analysis
  docs/완전_분석_요약_KR.md              # Korean summary
```

---

## References

van Opheusden, B., Acerbi, L., & Ma, W. J. (2023). Expertise increases planning depth in human gameplay. *Nature*, 618(7965), 1000-1005.

Mhammedi, Z., Helou, D., Grau-Moya, J., Bou-Ammar, H., & Whiteson, S. (2023). Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL. *ICML*.

Yao, W., Zhao, P., Qiao, Y., Abbeel, P., & Ding, Y. (2024). Inverse Reinforcement Learning with Multiple Planning Horizons. *COLT*.

---

**Last updated**: 2026-01-02
