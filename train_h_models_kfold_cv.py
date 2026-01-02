"""
K-Fold Cross-Validation for Planning Depth Estimation

Addresses train/test data overlap concern:
- Use existing 5-fold CV groups from raw_data.csv
- Train h-models on 4 folds, test on held-out fold
- Ensures generalization to unseen data

Usage:
    python train_h_models_kfold_cv.py --fold 1  # Train on folds 2-5, test on fold 1
    python train_h_models_kfold_cv.py --all     # Run all 5 folds
"""

import numpy as np
import pickle
import argparse
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def load_h_specific_data_with_cv(h, data_dir='data/multistep_ik_cv', exclude_fold=None):
    """
    Load data for specific h value, optionally excluding a CV fold.

    Args:
        h: Planning depth
        exclude_fold: If specified, exclude this fold (1-5)

    Returns:
        X: (N, 178) features [state_current (89), state_future (89)]
        y: (N,) actions
        metadata: fold labels for each sample
    """
    print(f"\nLoading h={h} data (exclude_fold={exclude_fold})...")

    filepath = Path(data_dir) / f'ik_pairs_h{h}.pkl'
    with open(filepath, 'rb') as f:
        pairs = pickle.load(f)

    X = []
    y = []
    cv_groups = []

    for pair in pairs:
        state_curr = pair['state_current']  # (89,)
        state_fut = pair['state_future']    # (89,)
        fold = pair.get('cv_group', -1)     # Get CV group if available

        # Skip if this is the excluded fold
        if exclude_fold is not None and fold == exclude_fold:
            continue

        features = np.concatenate([state_curr, state_fut])  # (178,)
        X.append(features)
        y.append(pair['action'])
        cv_groups.append(fold)

    X = np.array(X)
    y = np.array(y)
    cv_groups = np.array(cv_groups)

    print(f"  Loaded {len(X)} samples (exclude_fold={exclude_fold})")
    if exclude_fold is not None:
        print(f"  Excluded {sum([1 for p in pairs if p.get('cv_group', -1) == exclude_fold])} samples from fold {exclude_fold}")

    return X, y, cv_groups


def train_h_model_cv(h, exclude_fold, val_split=0.1):
    """
    Train h-specific model excluding one CV fold.

    Args:
        h: Planning depth
        exclude_fold: Fold to exclude (1-5)
        val_split: Validation split from training data
    """
    print("=" * 80)
    print(f"Training h={h} Model (K-Fold CV: exclude fold {exclude_fold})")
    print("=" * 80)

    # Load data excluding specified fold
    X, y, cv_groups = load_h_specific_data_with_cv(h, exclude_fold=exclude_fold)

    # Split into train/val (from the training folds only)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")

    # Train model (same architecture as before)
    print(f"\nTraining MLP...")

    model = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size=64,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\n{'='*80}")
    print(f"Results for h={h}, fold {exclude_fold} excluded")
    print(f"{'='*80}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")

    # Save model
    output_dir = Path(f'models/kfold_cv/fold_{exclude_fold}')
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f'model_h{h}.pkl'

    joblib.dump({
        'model': model,
        'h': h,
        'exclude_fold': exclude_fold,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'feature_dim': X.shape[1],
        'n_train': len(X_train),
        'n_val': len(X_val)
    }, model_path)

    print(f"\n✅ Saved model to {model_path}")

    return model, train_acc, val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to exclude (1-5)')
    parser.add_argument('--all', action='store_true',
                        help='Run all 5 folds')
    parser.add_argument('--h', type=int, nargs='+', default=[1, 2, 3, 4],
                        help='Which h values to train')
    args = parser.parse_args()

    if args.all:
        folds = [1, 2, 3, 4, 5]
    elif args.fold is not None:
        folds = [args.fold]
    else:
        raise ValueError("Must specify --fold N or --all")

    print("=" * 80)
    print("K-Fold Cross-Validation Training")
    print("=" * 80)
    print(f"Folds to process: {folds}")
    print(f"H values: {args.h}")

    results = {}

    for fold in folds:
        print(f"\n\n{'#'*80}")
        print(f"# Processing Fold {fold}")
        print(f"{'#'*80}\n")

        results[fold] = {}

        for h in args.h:
            model, train_acc, val_acc = train_h_model_cv(h, exclude_fold=fold)
            results[fold][h] = {
                'train_acc': train_acc,
                'val_acc': val_acc
            }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for fold in folds:
        print(f"\nFold {fold} (excluded):")
        for h in args.h:
            print(f"  h={h}: train_acc={results[fold][h]['train_acc']:.3f}, "
                  f"val_acc={results[fold][h]['val_acc']:.3f}")

    print(f"\n✅ K-Fold CV training complete!")
    print(f"Next step: python estimate_h_kfold_cv.py --all")


if __name__ == '__main__':
    main()
