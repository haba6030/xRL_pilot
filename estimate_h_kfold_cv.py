"""
K-Fold Cross-Validation Inference for Planning Depth

Tests generalization by evaluating on held-out CV folds.

For each fold:
    - Load models trained on other 4 folds
    - Evaluate on test fold (unseen data)
    - Compute Bayesian posterior P(h | moves)

Usage:
    python estimate_h_kfold_cv.py --fold 1  # Test on fold 1
    python estimate_h_kfold_cv.py --all     # Test all 5 folds
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import joblib
from pathlib import Path
from collections import defaultdict
from scipy import stats


def load_h_models_for_fold(fold, h_values=[1, 2, 3, 4]):
    """Load h-models trained excluding specified fold."""
    models = {}

    for h in h_values:
        model_path = f'models/kfold_cv/fold_{fold}/model_h{h}.pkl'
        data = joblib.load(model_path)
        models[h] = data['model']
        print(f"  Loaded h={h} model (exclude_fold={fold}, val_acc={data['val_acc']:.3f})")

    return models


def load_test_data_for_fold(fold, data_dir='data/multistep_ik_cv'):
    """
    Load test data for specified fold.

    Returns moves from the held-out fold with all h futures.
    """
    print(f"\nLoading test data for fold {fold}...")

    # Load all h datasets
    all_moves = defaultdict(lambda: {
        'participant': None,
        'cv_group': None,
        'game_id': None,
        't': None,
        'action': None,
        'state_current': None,
        'futures': {}  # {h: state_future}
    })

    for h in [1, 2, 3, 4]:
        filepath = Path(data_dir) / f'ik_pairs_h{h}.pkl'
        with open(filepath, 'rb') as f:
            pairs = pickle.load(f)

        for pair in pairs:
            if pair['cv_group'] != fold:
                continue  # Skip non-test fold

            # Create unique key for this move
            key = (pair['game_id'], pair['t'])

            move = all_moves[key]
            move['participant'] = pair['participant']
            move['cv_group'] = pair['cv_group']
            move['game_id'] = pair['game_id']
            move['t'] = pair['t']
            move['action'] = pair['action']
            move['state_current'] = pair['state_current']
            move['futures'][h] = pair['state_future']

    # Convert to list and filter complete moves
    test_moves = []
    for key, move in all_moves.items():
        # Only include moves with all h futures available
        if len(move['futures']) == 4:
            test_moves.append(move)

    print(f"  Loaded {len(test_moves)} test moves from fold {fold}")

    # Group by participant
    participants = defaultdict(list)
    for move in test_moves:
        participants[move['participant']].append(move)

    print(f"  Test participants: {len(participants)}")
    for p, moves in list(participants.items())[:5]:
        print(f"    Participant {p}: {len(moves)} moves")

    return test_moves, participants


def estimate_h_bayesian(move, models):
    """
    Compute Bayesian posterior P(h | move) for single move.

    Args:
        move: Dict with state_current, futures, action
        models: {h: model}

    Returns:
        posterior: {h: P(h | move)}
        likelihoods: {h: P(action | state, h)}
    """
    state_curr = move['state_current']
    action = move['action']

    likelihoods = {}

    for h in [1, 2, 3, 4]:
        if h not in move['futures']:
            likelihoods[h] = 0.0
            continue

        state_fut = move['futures'][h]

        # Concatenate states
        X = np.concatenate([state_curr, state_fut]).reshape(1, -1)

        # Get action probability
        probs = models[h].predict_proba(X)[0]
        likelihood = probs[action]
        likelihoods[h] = likelihood

    # Bayesian posterior (uniform prior)
    total = sum(likelihoods.values())
    if total > 0:
        posterior = {h: lik / total for h, lik in likelihoods.items()}
    else:
        # Fallback to uniform if all likelihoods are zero
        posterior = {h: 0.25 for h in [1, 2, 3, 4]}

    return posterior, likelihoods


def estimate_player_h(player_moves, models):
    """Estimate planning depth distribution for player."""
    posteriors = []

    for move in player_moves:
        post, _ = estimate_h_bayesian(move, models)
        posteriors.append(post)

    # Average posterior
    h_values = [1, 2, 3, 4]
    avg_posterior = {h: np.mean([p[h] for p in posteriors]) for h in h_values}

    # Expected h
    E_h = sum(h * avg_posterior[h] for h in h_values)

    return E_h, avg_posterior, posteriors


def run_fold_cv(fold, save_results=True):
    """Run CV evaluation for one fold."""
    print("=" * 80)
    print(f"K-Fold CV Evaluation: Fold {fold}")
    print("=" * 80)

    # Load models (trained on folds != fold)
    print(f"\n[1. Loading Models (trained on folds ≠ {fold})]")
    models = load_h_models_for_fold(fold)

    # Load test data (fold == fold)
    print(f"\n[2. Loading Test Data (fold = {fold})]")
    test_moves, participants = load_test_data_for_fold(fold)

    # Estimate h for each participant
    print(f"\n[3. Estimating Planning Depths]")
    results = []

    for participant, moves in participants.items():
        E_h, avg_post, _ = estimate_player_h(moves, models)

        results.append({
            'participant': participant,
            'fold': fold,
            'E[h]': E_h,
            'P(h=1)': avg_post[1],
            'P(h=2)': avg_post[2],
            'P(h=3)': avg_post[3],
            'P(h=4)': avg_post[4],
            'n_moves': len(moves)
        })

        print(f"  Participant {participant}: E[h] = {E_h:.3f} ({len(moves)} moves)")

    # Summary
    df = pd.DataFrame(results)

    print(f"\n{'='*80}")
    print(f"Fold {fold} Summary")
    print(f"{'='*80}")
    print(f"Participants tested: {len(df)}")
    print(f"Total moves: {df['n_moves'].sum()}")
    print(f"Mean E[h]: {df['E[h]'].mean():.3f} ± {df['E[h]'].std():.3f}")
    print(f"Range: [{df['E[h]'].min():.3f}, {df['E[h]'].max():.3f}]")

    # Save
    if save_results:
        output_dir = Path('results/kfold_cv')
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / f'fold_{fold}_estimates.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✅ Saved results to {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        folds = [1, 2, 3, 4, 5]
    elif args.fold is not None:
        folds = [args.fold]
    else:
        raise ValueError("Must specify --fold N or --all")

    all_results = []

    for fold in folds:
        df = run_fold_cv(fold, save_results=True)
        all_results.append(df)

    # Combined analysis
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("COMBINED K-FOLD CV RESULTS")
        print("=" * 80)

        combined = pd.concat(all_results, ignore_index=True)

        print(f"\nTotal participants: {len(combined)}")
        print(f"Total moves: {combined['n_moves'].sum()}")
        print(f"\nOverall E[h]: {combined['E[h]'].mean():.3f} ± {combined['E[h]'].std():.3f}")

        print(f"\nPer-fold summary:")
        for fold in folds:
            fold_df = combined[combined['fold'] == fold]
            print(f"  Fold {fold}: E[h] = {fold_df['E[h]'].mean():.3f} "
                  f"(n={len(fold_df)}, moves={fold_df['n_moves'].sum()})")

        # Save combined
        output_path = Path('results/kfold_cv/combined_estimates.csv')
        combined.to_csv(output_path, index=False)
        print(f"\n✅ Saved combined results to {output_path}")

        # Compare with original (no CV)
        try:
            original = pd.read_csv('results/human_h_rollout_free_estimates.csv')
            print(f"\n" + "=" * 80)
            print("Comparison: K-Fold CV vs. Original (no CV)")
            print("=" * 80)

            # Merge on participant
            merged = combined.merge(original[['participant', 'E[h]']],
                                     on='participant',
                                     suffixes=('_cv', '_orig'))

            print(f"\nMatched participants: {len(merged)}")
            print(f"K-Fold CV E[h]:  {merged['E[h]_cv'].mean():.3f} ± {merged['E[h]_cv'].std():.3f}")
            print(f"Original E[h]:    {merged['E[h]_orig'].mean():.3f} ± {merged['E[h]_orig'].std():.3f}")

            # Correlation
            r, p = stats.pearsonr(merged['E[h]_cv'], merged['E[h]_orig'])
            print(f"\nCorrelation: r = {r:.3f}, p = {p:.3e}")

            # Difference
            diff = merged['E[h]_cv'] - merged['E[h]_orig']
            print(f"Mean difference (CV - Original): {diff.mean():.3f} ± {diff.std():.3f}")

        except FileNotFoundError:
            print("\n⚠ Original results not found for comparison")

    print("\n✅ K-Fold CV evaluation complete!")


if __name__ == '__main__':
    main()
