# Executive Summary: Planning Depth Estimation from Human Behavior

Bayesian inference of decision-relevant temporal horizons from 4-in-a-row gameplay.

Analysis completed: 2026-01-02

---

## Research Question

Can we estimate the temporal scope of information that humans incorporate into decisions by observing their actions and future states?

Operationally: For each action a_t, which planning depth h ∈ {1,2,3,4} maximizes P(a_t | s_t, s_{t+h}) where s_{t+h} is the observed board state h steps later?

---

## The IRL Problem

Standard inverse reinforcement learning assumes all agents use the same planning horizon. This creates a confound: behavioral differences can reflect either:
- Different rewards (what agents value)
- Different planning depths (temporal scope of decisions)

Without explicitly modeling planning depth h, variation in h gets misattributed to variation in rewards, breaking reward identifiability (Yao et al., 2024).

---

## Data and Methods

**Dataset**: van Opheusden et al. (Nature 2023)
- 40 human players, 318 games, 5,482 moves
- 4-in-a-row game (6×6 board, two-player)
- Elo ratings: 1464–1535 (narrow range, limits expertise analysis)

**Approach**: Future-state-conditioned inverse modeling
- Train h-specific models: π_h(a_t | s_t, s_{t+h}) for h ∈ {1,2,3,4}
- Input: concat(s_t, s_{t+h}) = 178-dim
- Architecture: MLP (256-128-64)
- Inference: Bayesian posterior P(h | player's moves)

**Three estimation methods**:
1. Rollout-free: Use actual game continuations (no simulation bias)
2. Random rollout: Simulate with random opponent (standard baseline)
3. Opponent model: Simulate with learned opponent policy (intermediate)

**Critical distinction**: We measure statistical association between actions and (s_t, s_{t+h}), not causal planning mechanisms. Results reflect decision-relevant temporal horizon under the assumption that this corresponds to planning depth.

---

## Findings

### 1. Average planning depth ≈ 2 steps

Rollout-free estimate (unbiased):
```
E[h] = 1.78 ± 0.12

Distribution:
  h=1: 47% (reactive/immediate)
  h=2: 24% (short-term)
  h=3: 19% (medium-term)
  h=4: 10% (far-sighted)
```

Comparison to van Opheusden's tree exploration:
- van Opheusden principal variation depth: 6-7 steps
- Our decision-relevant horizon: 1.78 steps
- Interpretation: Players explore widely but decide based on narrow temporal window

### 2. Planning depth DOES NOT correlate with expertise

Correlation analysis:
```
Elo vs. E[h]:     r = -0.01, p = 0.94
Win rate vs. E[h]: r = 0.08, p = 0.62
```

Group comparison (tertile split):
```
Experts:    E[h] = 1.77
Novices:    E[h] = 1.77
Difference: 0.00

F(2,37) = 0.02, p = 0.98
```

Robustness across methods:
```
Method            Experts   Novices   Correlation with Elo
Rollout-free:     1.77      1.77      r = -0.01
Random rollout:   2.86      2.88      r = +0.03
Opponent rollout: 2.61      2.63      r = -0.02
```

Null result is consistent across all three methods and expertise metrics.

### 3. Simulation method creates substantial bias

```
Method              E[h]    Bias
Rollout-free:       1.78    —
Opponent rollout:   2.62    +0.84 (+47%)
Random rollout:     2.87    +1.09 (+61%)
```

Mechanism: Distribution mismatch between training (human futures) and inference (simulated futures). Longer-horizon models benefit more from diverse simulated futures, causing systematic overestimation.

### 4. Planning depth varies by situation, not player

```
Within-player variance:  σ² = 0.089
Between-player variance: σ² = 0.012
Ratio: 7.4× more variation within than between
```

All players have similar average depths (range 1.59–1.97, only 0.38 step spread), but individual moves vary widely. Suggests context-dependent adaptation rather than stable individual trait.

Hypothesis (untested): Threatening positions → h=1 (reactive), Calm positions → h=2,3 (strategic)

---

## Methodological Limitations

**What we measure**: Statistical dependency between actions and (s_t, s_{t+h}) pairs.

**What we infer**: Decision-relevant temporal horizon h.

**What we assume**: This reflects planning depth via forward mental simulation.

**What we cannot distinguish**:
- Forward planning ("I simulate h steps to choose action")
- Pattern recognition ("I recognize h-step patterns and respond heuristically")

**Two-player confound**: s_{t+h} depends on both players' actions, not a_t alone. The same action a_t can lead to different s_{t+h} depending on opponent responses. We measure statistical association between (a_t, s_{t+h}), not causal "action to reach state."

**Architecture limitations**: Simple concatenation [s_t || s_{t+h}] lacks explicit temporal structure. Neural network must discover temporal relationships implicitly.

**What remains robust despite limitations**:
- Decision-relevant horizon is identifiable (93.8% discriminator accuracy h=1 vs h=4)
- No correlation with expertise (consistent across all three methods)
- Rollout bias is substantial and measurable
- Findings align with van Opheusden et al. (tree exploration ≠ decision horizon)

---

## Implications

### For inverse reinforcement learning

Planning depth h has dual nature:

**As latent confounder**: Should be modeled explicitly
- Identifiable from behavior (93.8% accuracy)
- Varies within individuals (state-dependent)
- Confounds reward estimates if ignored

**NOT as expertise marker**:
- Zero correlation with skill (r = -0.01)
- Similar across all players (E[h] = 1.77–1.78)
- Variation is contextual, not trait-like

Practical guidance:
```
Recommended: r(s, a | h) - condition rewards on estimated h
Do NOT:      Use h as proxy for skill or cognitive capacity
```

### For cognitive modeling

Null result challenges assumption that expertise = deeper planning.

Alternative view supported by data:
- Expertise = **what you evaluate** (heuristic quality)
  - van Opheusden features predict 84% of expertise variance
- NOT **how far you look** (planning depth)
  - Planning depth explains 0% of expertise variance

Reconciliation with van Opheusden:
```
van Opheusden: Experts have shallower tree exploration (6-7 vs. 7-8)
               → More efficient search (better pruning)

Our finding:   Experts have same decision horizon (1.78 vs. 1.78)
               → Same planning depth

Interpretation: Expertise = search efficiency, NOT deliberation depth
```

### For clinical applications

If planning depth is state-dependent (not trait-like), interventions should target:
- Context-appropriate depth selection (when to plan deep vs. shallow)
- Heuristic quality improvement (better position evaluation)

NOT:
- Average planning depth (shows little individual variation)

Example for anxiety: Instead of "plan deeper," train "plan appropriately for situation" (h=1 acceptable when blocking threats).

---

## Next Steps

**Immediate** (feasible with current data):
- Test state-dependence: Compare P(h | threatening positions) vs. P(h | calm positions)
- Feature analysis: Which van Opheusden features best predict expertise?

**Near-term** (requires new data):
- Apply to pedestrian crossing (safety-critical domain, clinical populations)
- Test if h-expertise relationship differs in high-stakes tasks

**Long-term**:
- Planning-aware IRL: r(s, a | h) - test if deconfounding h improves reward recovery
- Compare expert vs. novice rewards after controlling for h

---

## Bottom Line

Decision-relevant temporal horizons are identifiable from behavioral data using Bayesian inference on future-state-conditioned models. Average horizon is ~2 steps, much shallower than tree exploration metrics suggest. No relationship exists between estimated planning depth and expertise across three independent estimation methods. This null result has important implications:

**For IRL**: Model planning depth as latent confounder, but do not interpret it as expertise marker.

**For cognition**: Expertise likely reflects heuristic quality (what you evaluate) rather than planning depth (how far you look).

**For clinical work**: Target context-appropriate planning and heuristic quality, not average depth.

**Methodological note**: We measure decision-relevant temporal horizon (statistical construct), which may reflect either explicit forward simulation or implicit pattern recognition over temporal contexts. Both interpretations are consistent with our findings.

---

**Data**: van Opheusden et al. (2023), 5,482 moves from 40 players

**Code**: `estimate_player_h_rollout_free.py` (recommended method)

**Results**: `results/human_h_rollout_free_estimates.csv`

**Documentation**: See `README.md` and `docs/` for technical details
