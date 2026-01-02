# Documentation

Analysis documentation for planning depth estimation from human 4-in-a-row gameplay.

## Core Documents

### Method Documentation

**METHOD_COMPARISON.md** - Comparison with MusIK framework
- MusIK algorithm (Mhammedi et al. 2023) vs. our approach
- What we borrowed: Multi-step inverse modeling concept
- What we did NOT implement: IKDP, layer-by-layer construction, policy cover
- Methodological assumptions and limitations
- Conservative interpretation: Decision-relevant temporal horizon
- Recommended framing for PI presentation

**ROLLOUT_FREE_ANALYSIS.md** - Rollout-free method details
- Uses actual game continuations (no simulation)
- Eliminates distribution mismatch artifact
- Result: E[h] = 1.78 ± 0.12
- State-dependent planning hypothesis

**ROLLOUT_METHOD_COMPARISON.md** - Three-method comparison
- Random rollout vs. Opponent model vs. Rollout-free
- Distribution mismatch creates +1.09 step bias (61% overestimate)
- Mechanism: Random futures more diverse than human futures

### Results Documentation

**FEATURE_VS_H_COMPARISON.md** - Expertise analysis
- Planning depth does NOT predict expertise (r = -0.01, p = 0.94)
- Van Opheusden heuristic features DO predict expertise (AUC = 0.84)
- Interpretation: Expertise = heuristic quality, not planning depth

**COMPLETE_ANALYSIS_SUMMARY.md** - Integrated summary (English)

**ROLLOUT_COMPARISON_SUMMARY.md** - Executive summary of method comparison

### Korean Documentation

**완전_분석_요약_KR.md** - Complete analysis summary (Korean)

**van_Opheusden_비교_논의_KR.md** - van Opheusden comparison (Korean)

## Main Findings

1. **Decision-relevant horizon is identifiable**: Discriminator accuracy 93.8% (h=1 vs h=4)

2. **Estimated average depth**: E[h] = 1.78 steps
   - Distribution: 47% h=1, 24% h=2, 19% h=3, 10% h=4
   - Per-player range: 1.59 to 1.97 (0.38 step spread)

3. **No correlation with expertise**: r = -0.01, p = 0.94 (Elo vs. E[h])
   - Robust across all three estimation methods
   - Experts: E[h] = 1.77, Novices: E[h] = 1.77

4. **Heuristic features predict expertise**: Van Opheusden features achieve AUC = 0.84
   - Top predictors: 3-in-a-row detection, center control, connected 2-in-a-row

5. **Simulation method creates bias**: Random rollout overestimates by +1.09 steps (61%)
   - Rollout-free eliminates this artifact

6. **Within-player variance > between-player variance**: 7.4× more variation within than between
   - Suggests state-dependent adaptation rather than stable trait

## For PI Review

Start with these in order:
1. **README.md** (project root) - Overview and main findings
2. **EXECUTIVE_SUMMARY.md** - One-page summary
3. **METHOD_COMPARISON.md** - Critical methodological discussion
4. **FEATURE_VS_H_COMPARISON.md** - Null result on expertise

Key methodological points:
- We measure decision-relevant temporal horizon (statistical construct)
- Cannot distinguish forward simulation from pattern recognition
- Two-player confound: s_{t+h} depends on both players' actions
- Conservative interpretation more defensible than planning depth claim

---

**Last updated**: 2026-01-02
