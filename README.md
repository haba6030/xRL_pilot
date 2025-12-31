# Planning Depth Estimation from Human Behavior

This project estimates how many steps ahead humans plan when making decisions in a board game. We analyze behavioral data from 40 players in 4-in-a-row (van Opheusden et al., Nature 2023) and develop methods to infer planning depth from observed actions.

---

## Motivation

### The IRL problem

Inverse reinforcement learning (IRL) infers reward functions from behavior. The standard approach assumes everyone plans the same number of steps ahead—but this assumption is unrealistic. If two people make different choices, it could mean they have different rewards OR they're planning different depths into the future.

Planning depth acts as a confounding variable: behavioral differences get wrongly attributed to reward differences when they actually reflect planning differences. This breaks reward identifiability.

### The cognitive question

van Opheusden et al. (2023) found that experts explore 6-7 steps in their search trees while novices explore 7-8 steps. They concluded experts plan more efficiently (shallower search, better pruning). But tree exploration depth is not the same as decision-relevant planning depth—how far ahead actually matters for your choice.

We want to know: How far ahead do people actually plan? Does planning depth vary with expertise? Is it a stable trait or does it change with game situation?

### Our approach

We use "inverse kinematics"—train models that predict actions from state transitions. If someone's action at time t is best explained by the board position at time t+2, we infer they planned 2 steps ahead for that move.

---

## Methods

### Data

van Opheusden et al. (2023) dataset:
- 40 human players
- 318 games (human vs. human)
- 5,482 moves with full board states
- Elo ratings ranging from 1464-1535

All players are reasonably skilled (no complete beginners), which limits our ability to test expertise effects but provides clean data on skilled play.

### Approach: Multi-step inverse modeling

The basic idea: train separate models for each planning depth h.

**Training phase**:
```
For h = 1, 2, 3, 4:
  Extract (state_t, state_{t+h}, action_t) from games
  Train model: P(action_t | state_t, state_{t+h})

Example for h=2:
  Move 10: board state = s_10, action = place piece at position 24
  Move 12: board state = s_12 (actual future after 2 moves)
  Model learns: P(action=24 | s_10, s_12)
```

Each h-model learns to predict what action a player took given they could "see" h steps into the future.

**Inference phase**:
```
For each move in test data:
  Compute likelihood under each h-model
  Use Bayes rule: P(h|move) ∝ P(move|h) × P(h)
  Aggregate across moves to get per-player distribution
```

### Three estimation methods (and why)

We tested three ways to get future states during inference:

**1. Random rollout** (standard approach)
- Simulate future states using random opponent moves
- Fast, no need for opponent model
- Problem: creates distribution mismatch (training used real futures, inference uses random futures)
- Result: Overestimates h by +1.09 steps (38% bias)

**2. Rollout-free** (our innovation)
- Use actual future states from game records
- No simulation needed
- Matches training distribution exactly
- Result: Unbiased estimates, E[h] = 1.78

**3. Opponent model** (middle ground)
- Simulate futures using learned human opponent model
- More realistic than random, but still requires simulation
- Result: E[h] = 2.62 (between random and rollout-free)

We compared all three to understand the role of simulation bias and verify findings are robust.

### Example calculation (rollout-free method)

```
Move 10: Player places piece at position 24
Game record shows actual continuation: s_10 → s_11 → s_12 → s_13 → s_14

Compute likelihoods:
  P(action=24 | s_10, s_11) using h=1 model = 0.52
  P(action=24 | s_10, s_12) using h=2 model = 0.68  ← highest
  P(action=24 | s_10, s_13) using h=3 model = 0.41
  P(action=24 | s_10, s_14) using h=4 model = 0.28

Bayesian posterior (assuming uniform prior):
  P(h=1|move) = 0.28
  P(h=2|move) = 0.36  ← most likely
  P(h=3|move) = 0.22
  P(h=4|move) = 0.15

Interpretation: This move is best explained by 2-step planning
```

---

## Main Findings

### 1. Humans plan approximately 2 steps ahead

Rollout-free estimates show E[h] = 1.78 ± 0.12 across all players.

Distribution of planning depths:
- 47% of moves: h=1 (reactive, immediate response)
- 24% of moves: h=2 (short-term planning)
- 19% of moves: h=3 (medium-term planning)
- 10% of moves: h=4 (far-sighted planning)

This is much shallower than van Opheusden's finding of 6-7 step tree exploration. The likely explanation: tree exploration measures how widely you search, while our h measures the decision-relevant horizon. You might explore 6-7 steps to verify your choice is good, but only the next 2 steps matter for deciding which move to make.

### 2. Planning depth does NOT predict expertise

This was surprising. We tested the relationship multiple ways:

Correlation analysis:
- Elo rating vs. E[h]: r = -0.01, p = 0.94 (no relationship)
- Win rate vs. E[h]: r = 0.08, p = 0.62 (not significant)

Group comparison:
- Experts (top 33%): E[h] = 1.77
- Novices (bottom 33%): E[h] = 1.77
- Difference: 0.00 (literally identical)

We initially suspected this might be an artifact of the random rollout method, but the null result persists with rollout-free and opponent model methods. It appears to be genuine.

Alternative explanation: van Opheusden features (heuristic quality measures like center control, threat detection) predict expertise with 84% accuracy. This suggests expertise comes from what you evaluate (heuristic quality), not how far you look ahead (planning depth).

### 3. Simulation method creates large bias

Random rollout overestimates planning depth by +1.09 steps (38%) compared to rollout-free.

Why this happens:
- Training: Models learn from actual human game continuations (constrained, strategic)
- Inference with random rollout: Simulated futures are random (diverse, exploratory)
- Longer-horizon models benefit more from seeing diverse futures
- Result: Systematic bias toward estimating h=4

The rollout-free method avoids this by using actual game outcomes for both training and inference.

### 4. Planning depth varies more by situation than by player

All players have similar average planning depths (range 1.59 to 1.97—only 0.38 step spread). But individual moves vary widely in their posterior P(h|move). This suggests planning depth is context-dependent rather than a stable individual trait.

Hypothesis (untested): Planning depth adapts to game situation
- Threatening positions → reactive (h=1): block immediate loss
- Calm positions → strategic (h=2,3): setup future advantage
- Opening moves → exploratory (h=3,4): establish position

This requires further analysis comparing h estimates to game state features.

---

## Implications

### For inverse reinforcement learning

Our findings clarify h's role in IRL:

What h IS:
- Identifiable from behavior (93.8% discriminator accuracy)
- A latent confounding variable (different h → different behavior)
- State-dependent (varies by context)

What h is NOT:
- A measure of expertise (no correlation with skill)
- A stable trait (varies within-player more than between-player)
- The primary difference between experts and novices

Practical guidance: Model h explicitly in IRL to deconfound behavioral variation, but don't interpret h as a skill marker. Use heuristic features (van Opheusden) to predict expertise instead.

### For cognitive modeling

The findings support a view where:
- Expertise reflects heuristic quality (better position evaluation) rather than planning depth
- Planning depth is adaptive to task demands, not a fixed cognitive capacity
- Tree exploration (PV depth) and decision horizon (behavioral h) are distinct constructs

This reconciles our null result with van Opheusden's finding: experts have shallower tree exploration (more efficient search) but similar decision horizons (task demands are similar for everyone).

### For clinical applications

If planning depth is state-dependent rather than trait-like, interventions should target:
- When to plan deeply vs. shallowly (adaptive control)
- Quality of position evaluation (building better heuristics)

Rather than:
- Average planning depth (shows little individual variation)

This matters for conditions like anxiety (might over-plan in threatening situations) or ADHD (might under-plan in calm situations). Testing requires analyzing h by game context.

---

## Repository Structure

Main analysis scripts:
```
estimate_player_h_rollout_free.py    - Rollout-free method (recommended)
estimate_player_h_multiclass.py      - Random rollout method
estimate_player_h_opponent.py        - Opponent model method
analyze_feature_based_expertise.py   - Feature vs. h comparison
preprocess_multistep_ik_data.py      - Extract training data
generate_trajectories_opponent_model.py - Train opponent model
```

Supporting code:
```
env.py            - 4-in-a-row game environment
features.py       - van Opheusden feature extraction (17 dimensions)
data_loader.py    - Load human game data
```

Data:
```
data/multistep_ik/       - Training data for inverse models
opendata/raw_data.csv    - Original game records (5,482 moves)
```

Results:
```
results/human_h_rollout_free_estimates.csv - Planning depth per player
results/player_van_opheusden_features.csv  - Heuristic features per player
```

Documentation:
```
EXECUTIVE_SUMMARY.md         - One-page overview (start here)
docs/ROLLOUT_FREE_ANALYSIS.md - Detailed method documentation
docs/FEATURE_VS_H_COMPARISON.md - Expertise analysis
docs/完전_분석_요약_KR.md    - Summary in Korean
```

---

## Quick Start

View results:
```bash
# Planning depth estimates
cat results/human_h_rollout_free_estimates.csv

# Feature-based expertise analysis
python analyze_feature_based_expertise.py
```

Reproduce analysis:
```bash
# Run rollout-free estimation (recommended method)
python estimate_player_h_rollout_free.py

# Output: results/human_h_rollout_free_estimates.csv
# Runtime: ~5 minutes
```

Full pipeline (regenerate from raw data):
```bash
# 1. Extract training data (5 min)
python preprocess_multistep_ik_data.py

# 2. Train opponent model (optional, 30 min)
python generate_trajectories_opponent_model.py

# 3. Estimate planning depth (5 min)
python estimate_player_h_rollout_free.py

# 4. Analyze expertise (2 min)
python analyze_feature_based_expertise.py
```

---

## Next Steps

### Immediate (can do now)

Analyze state-dependence:
- Extract game state features (threat level, board complexity, game phase)
- Compute move-level posterior P(h|move)
- Test hypothesis: threatening situations → lower h, calm situations → higher h

Analyze heuristic features:
- Which van Opheusden features best predict expertise?
- Decision tree analysis for interpretable expertise profiles
- Test if experts use different features vs. weight features differently

### Near-term (requires new data collection)

Apply to pedestrian crossing:
- Different domain (safety-critical instead of game)
- Clinical populations (anxiety, ADHD)
- Test if planning depth varies with clinical traits
- Test if expertise relationship reverses in high-stakes domain

### Long-term (enabled by this work)

Develop planning-aware IRL:
- Condition reward learning on estimated h: r(state, action, h)
- Test if deconfounding h improves reward recovery
- Compare expert vs. novice rewards after controlling for h

---

## References

van Opheusden, B., Acerbi, L., & Ma, W. J. (2023). Expertise increases planning depth in human gameplay. Nature, 618(7965), 1000-1005.
- Source of behavioral data (40 players, 318 games)
- van Opheusden features for position evaluation (17 dimensions)

Mhammedi, Z., Helou, D., Grau-Moya, J., Bou-Ammar, H., & Whiteson, S. (2023). Representation Learning with Multi-Step Inverse Kinematics: An Efficient and Optimal Approach to Rich-Observation RL. International Conference on Machine Learning.
- Multi-step inverse kinematics framework
- We adapt from representation learning to behavior analysis

Yao, W., Zhao, P., Qiao, Y., Abbeel, P., & Ding, Y. (2024). Inverse Reinforcement Learning with Multiple Planning Horizons. Conference on Learning Theory.
- Planning horizon as latent confounder in IRL
- Theoretical motivation for explicit h modeling

---

## Summary

We can estimate how many steps ahead people plan from their choices. Humans plan about 2 steps ahead on average in 4-in-a-row, much shallower than tree search metrics suggest. Surprisingly, planning depth does not predict expertise—all skill levels plan similarly deep. This null result is robust across estimation methods. The findings suggest expertise comes from better position evaluation (what you evaluate) rather than deeper planning (how far you look ahead). For IRL, planning depth should be modeled as a confounding variable but not used as an expertise marker.

**Last updated**: 2025-12-31
