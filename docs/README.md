# Documentation

This folder contains analysis documentation for planning depth estimation from human game-playing behavior.

## Core Documents

### Method Documentation

**ROLLOUT_FREE_ANALYSIS.md** - Rollout-free posterior method (recommended)
- Eliminates distribution mismatch by using actual human futures
- Result: E[h] = 1.78 ± 0.12 (unbiased estimate)
- State-dependent planning hypothesis

**ROLLOUT_METHOD_COMPARISON.md** - Three-method comparison
- Random rollout, Opponent model, Rollout-free
- Distribution mismatch artifact: +1.09 step bias

### Results Documentation

**FEATURE_VS_H_COMPARISON.md** - Feature-based vs h-based expertise
- Planning depth does NOT predict expertise (AUC = 0.53)
- Van Opheusden features predict expertise (AUC = 0.84)
- Expertise = heuristic quality, not planning depth

**COMPLETE_ANALYSIS_SUMMARY.md** - Integrated summary (EN)
**완전_분석_요약_KR.md** - Integrated summary (KR)

**ROLLOUT_COMPARISON_SUMMARY.md** - Executive summary
**van_Opheusden_비교_논의_KR.md** - van Opheusden comparison (KR)

## Key Findings

1. **Rollout method matters**: Random rollout overestimates h by +1.09 steps (38%)
2. **Humans plan myopically**: E[h] = 1.78, with 47% of moves using h=1
3. **Planning depth ≠ expertise**: No correlation with Elo or win rate
4. **Features predict expertise**: Van Opheusden heuristics achieve AUC = 0.84
5. **State-dependent planning**: h varies by game context, not player trait

## Deprecated Documentation

Moved to `../backup/`:
- `docs_airl_future/`: AIRL-related documentation (future work)
- `docs_planning/`: Planning and resource estimation documents
- `docs_outdated/`: Superseded analyses and old summaries

---

**Last updated**: 2025-12-31
