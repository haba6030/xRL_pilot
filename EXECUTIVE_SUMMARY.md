# Executive Summary: Planning Depth Estimation from Human Behavior

**Authors**: Analysis completed 2025-12-31

---

## The Problem

Inverse reinforcement learning (IRL) infers reward functions from behavior. Standard IRL assumes people plan the same number of steps ahead—but humans vary in how far they look ahead when making decisions. This variation confounds reward estimates: two people might choose differently because they have different rewards OR because they plan different depths into the future.

We need to estimate planning depth from observed behavior to disentangle these factors.

---

## What We Did

We analyzed 5,482 moves from 40 human players in a 4-in-a-row board game (data from van Opheusden et al., Nature 2023).

**Approach**: Train "inverse models" that predict which action a player took given their current board position and the board position h steps later. If a player's action is best explained by looking h=2 steps ahead, we infer they planned 2 steps for that move.

**Three estimation methods tested**:
1. Random rollout - simulate futures with random opponent
2. Opponent model - simulate futures with learned human opponent
3. Rollout-free - use actual game outcomes (no simulation)

We compared these methods and tested whether planning depth correlates with skill level.

---

## What We Found

### Finding 1: Humans plan approximately 2 steps ahead

Using the rollout-free method (which avoids simulation bias):
- Average planning depth: 1.78 steps
- 47% of moves are reactive (1-step lookahead)
- Only 10% of moves involve 4-step lookahead

This is much shallower than van Opheusden's finding that people explore 6-7 steps in their search trees. The difference likely reflects the distinction between tree exploration (how widely you search) and decision horizon (how far ahead matters for your choice).

### Finding 2: Planning depth does not predict expertise

We found no relationship between planning depth and skill level:
- Correlation with Elo rating: r = -0.01, p = 0.94 (essentially zero)
- Correlation with win rate: r = 0.08, p = 0.62 (not significant)
- Expert average: 1.77 steps, Novice average: 1.77 steps (identical)

This was surprising. We tested it multiple ways and the null result is robust.

### Finding 3: Simulation method matters enormously

Random rollout overestimates planning depth by 38% (+1.09 steps). This happens because:
- Training uses actual human game continuations
- Inference simulates futures with random moves
- Random futures are more diverse than human futures
- Longer-horizon models benefit from seeing diverse futures

The rollout-free method eliminates this bias by using actual game outcomes for both training and inference.

### Finding 4: Planning depth appears state-dependent, not trait-like

All players have similar average planning depths (range 1.59-1.97), but individual moves vary widely. This suggests planning depth changes based on game situation rather than being a stable individual characteristic:
- Threatening positions → reactive (h=1)
- Calm positions → strategic (h=2,3)
- Opening moves → exploratory (h=3,4)

This is a hypothesis requiring further testing.

---

## Why This Matters

### For inverse reinforcement learning

Standard IRL assumes everyone plans with the same depth. Our findings show:
- Planning depth h is identifiable from behavior (we can estimate it accurately)
- Planning depth varies across situations, so it should be modeled as a latent variable
- But planning depth does NOT predict expertise, so it's a nuisance parameter, not a skill marker

This means IRL should condition reward learning on estimated h to avoid confounding, but should not interpret h as a measure of competence.

### For understanding human decision-making

The null result (planning depth ≠ expertise) points toward a different view of skill:
- Expertise comes from what you evaluate (heuristic quality), not how far you look ahead
- van Opheusden features (position evaluation heuristics) predict expertise with 84% accuracy
- Planning depth is context-adaptive rather than a stable cognitive trait

### For clinical applications

If planning depth is state-dependent rather than trait-like, clinical interventions should focus on:
- When people plan deeply vs. shallowly (context-dependence)
- Quality of position evaluation (heuristics)
- Not average planning depth (which shows little individual variation)

This matters for conditions like anxiety or ADHD where planning deficits are hypothesized.

---

## Next Steps

### Short-term (completed)
- Estimate planning depth from human game data
- Test three estimation methods
- Analyze relationship with expertise

### Medium-term (feasible now)
- Test state-dependence hypothesis: analyze planning depth by game situation (threat level, board complexity, game phase)
- Analyze which van Opheusden features best predict expertise
- Apply to pedestrian crossing domain (different task, clinical populations)

### Long-term (enabled by this work)
- Develop planning-aware IRL that conditions rewards on estimated h
- Test whether deconfounding h improves reward recovery
- Compare expert vs. novice reward functions after controlling for h

---

## Bottom Line

We can estimate how many steps ahead people plan from their choices. Humans plan about 2 steps ahead on average—much shallower than tree search metrics suggest. Surprisingly, planning depth shows no relationship with skill level. This null result is robust across methods and has important implications: IRL should model planning depth as a confounding variable but should not use it as an expertise marker. The finding suggests expertise comes from better position evaluation (what you evaluate) rather than deeper planning (how far you look ahead).

---

**Data**: van Opheusden et al. (2023) 4-in-a-row dataset, 40 players, 5,482 moves

**Code**: `estimate_player_h_rollout_free.py` (recommended method)

**Results**: `results/human_h_rollout_free_estimates.csv`

**Documentation**: See `README.md` and `docs/` for technical details
