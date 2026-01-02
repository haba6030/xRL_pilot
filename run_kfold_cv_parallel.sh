#!/bin/bash
# Parallel execution of K-Fold CV for planning depth estimation
# Addresses train/test overlap concern via proper cross-validation

set -e  # Exit on error

echo "=============================================================================="
echo "K-Fold Cross-Validation Pipeline (Parallel Execution)"
echo "=============================================================================="

# Step 1: Data preprocessing (sequential, ~5 min)
echo ""
echo "[Step 1/3] Data preprocessing with CV information..."
python3 preprocess_multistep_ik_data_cv.py

# Step 2: Training (parallel, ~15-20 min total)
echo ""
echo "[Step 2/3] Training h-models for each fold (parallel)..."
echo "Launching 5 parallel training jobs..."

# Launch parallel jobs
python3 train_h_models_kfold_cv.py --fold 1 > logs/train_fold_1.log 2>&1 &
PID1=$!
python3 train_h_models_kfold_cv.py --fold 2 > logs/train_fold_2.log 2>&1 &
PID2=$!
python3 train_h_models_kfold_cv.py --fold 3 > logs/train_fold_3.log 2>&1 &
PID3=$!
python3 train_h_models_kfold_cv.py --fold 4 > logs/train_fold_4.log 2>&1 &
PID4=$!
python3 train_h_models_kfold_cv.py --fold 5 > logs/train_fold_5.log 2>&1 &
PID5=$!

echo "Training jobs launched:"
echo "  Fold 1: PID $PID1 (logs/train_fold_1.log)"
echo "  Fold 2: PID $PID2 (logs/train_fold_2.log)"
echo "  Fold 3: PID $PID3 (logs/train_fold_3.log)"
echo "  Fold 4: PID $PID4 (logs/train_fold_4.log)"
echo "  Fold 5: PID $PID5 (logs/train_fold_5.log)"

# Wait for all training jobs
echo ""
echo "Waiting for training jobs to complete..."
wait $PID1 && echo "  ✓ Fold 1 complete"
wait $PID2 && echo "  ✓ Fold 2 complete"
wait $PID3 && echo "  ✓ Fold 3 complete"
wait $PID4 && echo "  ✓ Fold 4 complete"
wait $PID5 && echo "  ✓ Fold 5 complete"

echo "✅ All training jobs complete!"

# Step 3: Inference (parallel, ~5 min total)
echo ""
echo "[Step 3/3] Estimating planning depths on held-out folds (parallel)..."

python3 estimate_h_kfold_cv.py --fold 1 > logs/estimate_fold_1.log 2>&1 &
PID1=$!
python3 estimate_h_kfold_cv.py --fold 2 > logs/estimate_fold_2.log 2>&1 &
PID2=$!
python3 estimate_h_kfold_cv.py --fold 3 > logs/estimate_fold_3.log 2>&1 &
PID3=$!
python3 estimate_h_kfold_cv.py --fold 4 > logs/estimate_fold_4.log 2>&1 &
PID4=$!
python3 estimate_h_kfold_cv.py --fold 5 > logs/estimate_fold_5.log 2>&1 &
PID5=$!

echo "Estimation jobs launched:"
echo "  Fold 1: PID $PID1 (logs/estimate_fold_1.log)"
echo "  Fold 2: PID $PID2 (logs/estimate_fold_2.log)"
echo "  Fold 3: PID $PID3 (logs/estimate_fold_3.log)"
echo "  Fold 4: PID $PID4 (logs/estimate_fold_4.log)"
echo "  Fold 5: PID $PID5 (logs/estimate_fold_5.log)"

echo ""
echo "Waiting for estimation jobs to complete..."
wait $PID1 && echo "  ✓ Fold 1 complete"
wait $PID2 && echo "  ✓ Fold 2 complete"
wait $PID3 && echo "  ✓ Fold 3 complete"
wait $PID4 && echo "  ✓ Fold 4 complete"
wait $PID5 && echo "  ✓ Fold 5 complete"

echo "✅ All estimation jobs complete!"

# Combined analysis
echo ""
echo "Running combined analysis..."
python3 -c "
import pandas as pd
from pathlib import Path

# Load all fold results
results = []
for fold in [1, 2, 3, 4, 5]:
    df = pd.read_csv(f'results/kfold_cv/fold_{fold}_estimates.csv')
    results.append(df)

combined = pd.concat(results, ignore_index=True)

print('='*80)
print('K-FOLD CV FINAL RESULTS')
print('='*80)
print(f'Total participants: {len(combined)}')
print(f'Total moves: {combined[\"n_moves\"].sum()}')
print(f'Overall E[h]: {combined[\"E[h]\"].mean():.3f} ± {combined[\"E[h]\"].std():.3f}')
print(f'Range: [{combined[\"E[h]\"].min():.3f}, {combined[\"E[h]\"].max():.3f}]')

# Distribution
print(f'\nPlanning depth distribution:')
for h in [1, 2, 3, 4]:
    prob = combined[f'P(h={h})'].mean()
    print(f'  h={h}: {prob:.1%}')

# Per-fold summary
print(f'\nPer-fold results:')
for fold in [1, 2, 3, 4, 5]:
    fold_df = combined[combined['fold'] == fold]
    print(f'  Fold {fold}: E[h] = {fold_df[\"E[h]\"].mean():.3f}, n = {len(fold_df)}')

# Save
combined.to_csv('results/kfold_cv/combined_estimates.csv', index=False)
print(f'\n✅ Saved to results/kfold_cv/combined_estimates.csv')
"

echo ""
echo "=============================================================================="
echo "✅ K-FOLD CV PIPELINE COMPLETE!"
echo "=============================================================================="
echo ""
echo "Results:"
echo "  - Individual folds: results/kfold_cv/fold_{1-5}_estimates.csv"
echo "  - Combined: results/kfold_cv/combined_estimates.csv"
echo "  - Logs: logs/train_fold_*.log, logs/estimate_fold_*.log"
echo ""
echo "Next steps:"
echo "  1. Compare with original (no CV): results/human_h_rollout_free_estimates.csv"
echo "  2. Check if E[h] estimates are robust to train/test split"
echo "  3. Apply same CV framework to pedestrian data"
