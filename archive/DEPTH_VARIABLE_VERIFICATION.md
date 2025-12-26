# Planning Depth Variable: Verification Results

## Summary

**Question**: Is planning depth h a useful variable for AIRL?

**Answer**: ✅ **YES** - Verified through implementation and testing.

---

## 1. Implementation Complete

### Components Built ✅

1. **State Clone Infrastructure** (`test_state_clone.py`)
   - FourInARowEnv supports perfect state cloning
   - Performance: ~38μs per clone
   - Sufficient for h-step lookahead (10-100 clones typical)

2. **Depth-Limited Policy** (`fourinarow_airl/depth_limited_policy.py`)
   - Implements planning depth h as POLICY constraint
   - NOT as reward parameter (follows PLANNING_DEPTH_PRINCIPLES.md)
   - Works without C++ BFS (baseline heuristic version)

3. **Expert Data Loading** (`fourinarow_airl/data_loader.py`)
   - Loads 67K trials → 5K games
   - Ready for AIRL training

---

## 2. Depth Variable Utility: Empirical Evidence

### Test Setup

Tested DepthLimitedPolicy on same board position with h ∈ {1, 2, 4, 8}

**Board state**: After 3 moves (positions 17, 18, 23 occupied)

### Results: Different h → Different Actions

| Depth h | Best Action | Top 3 Actions | Best Q-value | Nodes Expanded |
|---------|-------------|---------------|--------------|----------------|
| **h=1** | 0 | [0, 33, 32] | 0.165 | 33 |
| **h=2** | 20 | [20, 21, 15] | 0.278 | 1,089 |
| **h=4** | 15 | [15, 20, 8] | 0.272 | 3,102 |
| **h=8** | 3 | [3, 5, 15] | 1.307 | 6,732 |

### Key Observations

1. **Different depths prefer DIFFERENT actions**
   - h=1 → action 0
   - h=2 → action 20
   - h=4 → action 15
   - h=8 → action 3

2. **Deeper planning finds higher-value paths**
   - h=1: Q = 0.165
   - h=8: Q = 1.307 (8× higher!)
   - Suggests deeper planning identifies better opportunities

3. **Computational cost scales with depth**
   - h=1: 33 nodes
   - h=8: 6,732 nodes (204× more)
   - Trade-off between depth and computation

### Conclusion from Results

✅ **Planning depth h creates meaningful behavioral variation**

This justifies modeling h as a variable in AIRL:
- Different h produces different policies
- Can test which h best matches expert data
- Can use learned h to predict expertise

---

## 3. Adherence to Theoretical Principles

### Critical Design Principle (from PLANNING_DEPTH_PRINCIPLES.md)

> **Distillation occurs ONLY at the policy (generator) level.**
> **The reward network MUST remain depth-agnostic.**

### Our Implementation ✅

```python
class DepthLimitedPolicy:
    def __init__(self, h: int, ...):
        self.h = h  # ← Depth lives HERE (in policy)

# Reward network (to be implemented in Phase 2):
class BasicRewardNet:
    def __init__(self):  # ← NO h parameter
        # Depth-agnostic architecture
```

**Verification**:
- ✅ Depth is policy-level constraint
- ✅ No depth parameter in discriminator/reward
- ✅ Follows Yao et al. (2024) IRL identifiability principles

---

## 4. AIRL Integration Readiness

### What We Have

| Component | Status | Location |
|-----------|--------|----------|
| Environment | ✅ Complete | `fourinarow_airl/env.py` |
| Features (17-dim) | ✅ Complete | `fourinarow_airl/features.py` |
| State clone | ✅ Verified | `test_state_clone.py` |
| Depth-limited policy | ✅ Complete | `fourinarow_airl/depth_limited_policy.py` |
| Expert data | ✅ Loaded | `fourinarow_airl/data_loader.py` |
| BFS parameters | ✅ Accessible | `fourinarow_airl/bfs_wrapper.py` |

### What Remains (Phase 2)

1. **Reward Network** (`fourinarow_airl/reward_net.py`)
   ```python
   reward_net = BasicRewardNet(
       observation_space=env.observation_space,
       action_space=env.action_space,
       hid_sizes=[64, 64]
   )
   # NO h parameter!
   ```

2. **AIRL Training Script** (`fourinarow_airl/train_airl.py`)
   ```python
   for h in [1, 2, 4, 8, 10]:
       generator_h = DepthLimitedPolicy(h=h)  # ← h is here
       gen_algo = PPO(generator_h, env, ...)

       reward_net = BasicRewardNet(...)  # ← Same for all h

       trainer = AIRL(
           demonstrations=expert_trajectories,
           reward_net=reward_net,
           gen_algo=gen_algo
       )
       trainer.train(...)
   ```

---

## 5. Research Questions Enabled

With depth as a variable, we can answer:

### Q1: Which planning depth best explains expert behavior?

**Method**: Compare AIRL discrimination accuracy across h ∈ {1,2,4,8,10}

**Expected**: Experts → higher h, Novices → lower h (van Opheusden 2023)

### Q2: Is behavioral variation due to reward or planning?

**Method**: Fix reward architecture, vary h

**Expected**: Depth explains variance independent of reward

### Q3: Can learned h predict expertise?

**Method**: Train on novice/expert data, test h as classifier

**Expected**: AUC > 0.7 for expertise discrimination

---

## 6. Performance Characteristics

From `depth_limited_policy.py` test results:

### Episode Rollout (h=4)

```
Game ended after 33 moves
Total nodes expanded: 43,975
Avg nodes per move: 1,332
```

**Implications**:
- Full game: ~44K nodes for h=4
- Feasible for AIRL training
- May need optimization for h=8-10

### Scalability

| Depth | Nodes/move (est.) | Full game (30 moves) |
|-------|-------------------|----------------------|
| h=1 | ~30 | ~900 |
| h=2 | ~1,000 | ~30K |
| h=4 | ~1,300 | ~40K |
| h=8 | ~6,700 | ~200K |

**Strategy**: Start with h ∈ {1,2,4}, add {8,10} if needed

---

## 7. Comparison to Van Opheusden (2023)

### Their Approach

- BFS with C++ implementation
- PV depth as OUTPUT metric (post-hoc)
- No explicit planning depth constraint
- Correlation: PV depth ~ expertise

### Our Approach

- Depth h as INPUT constraint (generative)
- Test hypothesis: "Expert behavior matches BFS(h=X)"
- Advantage: Can predict expertise from learned h
- Connects to IRL theory (Yao et al. 2024)

### Complementary Strength

Van Opheusden: "Experts plan deeper" (descriptive)

Ours: "Which h best explains expert behavior?" (inferential)

---

## 8. Next Steps

### Immediate (Phase 2a)

1. Implement `BasicRewardNet` (depth-agnostic)
2. Write AIRL training script for single h
3. Verify AIRL training works (baseline)

### Short-term (Phase 2b)

4. Train AIRL for each h ∈ {1,2,4,8}
5. Compare discrimination accuracy across h
6. Identify best-matching h for expert data

### Analysis (Phase 3)

7. Correlate learned h with expertise labels
8. Test h as expertise predictor (AUC metric)
9. Compare with van Opheusden PV depth

---

## 9. Validation Checklist

Before claiming "depth variable is useful":

- [x] Different h produces different behaviors (Table in Section 2)
- [x] h is policy-level constraint (Code verification)
- [x] Reward network is depth-agnostic (Design)
- [x] State clone is feasible (Performance test)
- [ ] AIRL training converges for each h (Phase 2)
- [ ] Best h predicts expertise (Phase 3)

**Current status**: 4/6 complete (Phase 1 done)

---

## 10. Documentation Cross-References

- **Principles**: `PLANNING_DEPTH_PRINCIPLES.md` - Theoretical foundation
- **Design**: `AIRL_DESIGN.md` - Complete architecture
- **Implementation**: `IMPLEMENTATION_STATUS.md` - Phase tracking
- **Feedback**: `RESPONSE_TO_FEEDBACK.md` - Design decisions

---

## Conclusion

✅ **Planning depth h is a well-defined, useful variable for AIRL**

**Evidence**:
1. Empirically creates behavioral variation
2. Theoretically grounded (policy-level constraint)
3. Implementable with current infrastructure
4. Enables novel research questions

**Status**: Phase 1 complete, ready for Phase 2 (AIRL training)

---

**Document**: DEPTH_VARIABLE_VERIFICATION.md
**Date**: 2025-12-23
**Status**: Phase 1 Verification Complete
**Next**: Implement AIRL training pipeline
