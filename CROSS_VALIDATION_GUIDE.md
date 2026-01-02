# Cross-Validation for Planning Depth Estimation

## Problem Addressed

**Original concern**: Train/test data overlap
- Models trained on all players
- Inference on same players
- Risk: Overfitting to specific 40 participants

**Solution**: K-Fold Cross-Validation
- 5 folds (already defined in raw_data.csv)
- Train on 4 folds, test on 1 held-out fold
- Repeat for all 5 folds
- **No overlap** between training and test data

---

## Quick Start (오늘 내 완료 가능)

### Option 1: Automated Parallel Execution (추천)

```bash
# Single command to run entire pipeline (~25-30 min total)
./run_kfold_cv_parallel.sh
```

**Timeline:**
- Step 1: Data preprocessing (~5 min, sequential)
- Step 2: Training 5 folds (~15-20 min, **parallel**)
- Step 3: Estimation 5 folds (~5 min, **parallel**)

**Results:**
- `results/kfold_cv/fold_{1-5}_estimates.csv` - Individual folds
- `results/kfold_cv/combined_estimates.csv` - All folds combined

---

### Option 2: Manual Step-by-Step

**Step 1: Preprocess data with CV info**
```bash
python preprocess_multistep_ik_data_cv.py  # ~5 min
```

**Step 2a: Train all folds (sequential, ~1.5 hours)**
```bash
python train_h_models_kfold_cv.py --all
```

**OR Step 2b: Train in parallel (추천, ~20 min)**

Terminal 1:
```bash
python train_h_models_kfold_cv.py --fold 1 --h 1 2 3 4
```

Terminal 2:
```bash
python train_h_models_kfold_cv.py --fold 2 --h 1 2 3 4
```

Terminal 3:
```bash
python train_h_models_kfold_cv.py --fold 3 --h 1 2 3 4
```

Terminal 4:
```bash
python train_h_models_kfold_cv.py --fold 4 --h 1 2 3 4
```

Terminal 5:
```bash
python train_h_models_kfold_cv.py --fold 5 --h 1 2 3 4
```

**Step 3: Estimate planning depths**
```bash
python estimate_h_kfold_cv.py --all  # ~5 min
```

---

## Expected Results

### Validation of Original Findings

If our original analysis is robust, we expect:

```
Original (no CV):       E[h] = 1.78 ± 0.12
K-Fold CV (expected):   E[h] ≈ 1.75-1.85

Correlation (CV vs. Original): r > 0.8
```

### What Success Looks Like

✅ **E[h] similar across CV and original** → Estimates are robust
✅ **High correlation** → Player-level estimates are reliable
✅ **Null expertise correlation persists** → Finding is not artifact

### What Failure Looks Like

❌ **E[h] substantially different** → Original method was overfit
❌ **Low correlation** → High variance, unstable estimates
❌ **Expertise correlation emerges in CV** → Original was suppressed by overfitting

---

## Key Differences from Original

| Aspect | Original | K-Fold CV |
|--------|----------|-----------|
| **Training data** | All players | 4/5 folds (~32 players) |
| **Test data** | Same players | Held-out fold (~8 players) |
| **Overlap** | 100% | 0% |
| **Generalization** | Not tested | Directly tested |
| **Time** | ~30 min | ~30 min (parallel) |

---

## Why This Addresses the Concern

### Original Method
```python
# Potential issue
Train on: Players 1-40
Test on:  Players 1-40  # Same!
→ Overfitting risk
```

### K-Fold CV
```python
# Fold 1
Train on: Players in folds 2,3,4,5
Test on:  Players in fold 1  # Completely held-out

# Fold 2
Train on: Players in folds 1,3,4,5
Test on:  Players in fold 2  # Completely held-out

# ... etc
```

**No player appears in both training and test for any fold**

---

## Comparison with Proposed Alternatives

### Proposed Method 1: MusIK-inspired
- **Pros**: Trajectory-based
- **Cons**: Reintroduces random rollout (distribution mismatch), complex, 3-5 days work
- **Verdict**: ❌ Not worth the cost

### Proposed Method 2: AIRL + PPO
- **Pros**: Directly fits depth
- **Cons**: Massive computational cost, 1-2 weeks work, data insufficient
- **Verdict**: ❌ Infeasible for today

### K-Fold CV (Our Solution)
- **Pros**: ✅ Simple, ✅ Fast (오늘 완료), ✅ Directly addresses concern, ✅ Works for pedestrian
- **Cons**: Slightly lower training data per fold (minor)
- **Verdict**: ✅✅✅ Best solution

---

## Applying to Pedestrian Data

Same framework applies directly:

```bash
# 1. Add CV groups to pedestrian raw data (if not present)
# 2. Preprocess with CV info
python preprocess_pedestrian_multistep_ik_data_cv.py

# 3. Train with CV
python train_pedestrian_h_models_kfold_cv.py --all

# 4. Estimate with CV
python estimate_pedestrian_h_kfold_cv.py --all
```

**Timeline for pedestrian**: Same ~30 min (parallel)

---

## Monitoring Progress

**Check training progress:**
```bash
tail -f logs/train_fold_1.log  # Real-time monitoring
```

**Check intermediate results:**
```bash
ls models/kfold_cv/fold_*/  # Models being created
```

**Quick analysis:**
```bash
# After completion
python -c "
import pandas as pd
df = pd.read_csv('results/kfold_cv/combined_estimates.csv')
print(f'E[h] = {df[\"E[h]\"].mean():.3f} ± {df[\"E[h]\"].std():.3f}')
"
```

---

## Troubleshooting

**Issue: "File not found: data/multistep_ik_cv/ik_pairs_h1.pkl"**
→ Run preprocessing first: `python preprocess_multistep_ik_data_cv.py`

**Issue: "Model not found: models/kfold_cv/fold_1/model_h1.pkl"**
→ Run training first: `python train_h_models_kfold_cv.py --fold 1`

**Issue: Parallel jobs killed (out of memory)**
→ Run sequentially or reduce to 2-3 parallel jobs

**Issue: Different results from original**
→ Expected! This tests generalization. Report both.

---

## Integration with Main Analysis

After CV completion:

```bash
# Compare with original
python -c "
import pandas as pd
from scipy import stats

original = pd.read_csv('results/human_h_rollout_free_estimates.csv')
cv = pd.read_csv('results/kfold_cv/combined_estimates.csv')

merged = original.merge(cv[['participant', 'E[h]']],
                         on='participant',
                         suffixes=('_orig', '_cv'))

print(f'Original: E[h] = {merged[\"E[h]_orig\"].mean():.3f}')
print(f'CV:       E[h] = {merged[\"E[h]_cv\"].mean():.3f}')

r, p = stats.pearsonr(merged['E[h]_orig'], merged['E[h]_cv'])
print(f'Correlation: r = {r:.3f}, p = {p:.3e}')
"
```

**Update documentation:**
- README.md: Add CV validation results
- EXECUTIVE_SUMMARY.md: Note generalization tested
- METHOD_COMPARISON.md: Add CV methodology section

---

**Last updated**: 2026-01-02
