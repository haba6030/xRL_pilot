"""
Train Multi-Step IK Policies using scikit-learn

Simpler alternative to PyTorch implementation.
Uses Random Forest or MLP Classifier to learn the mapping:
    (state_current, state_future, h) → action

Usage:
    python3 train_multistep_ik_sklearn.py --model mlp
    python3 train_multistep_ik_sklearn.py --model rf
"""

import numpy as np
import pickle
from pathlib import Path
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib


def load_all_ik_data(h_values=[1, 2, 3, 4]):
    """
    Load and prepare all multi-step IK data.

    Returns:
        X: (N, feature_dim) features [state_current, state_future, h_onehot]
        y: (N,) actions
        h_labels: (N,) planning depths (for analysis)
    """
    print("Loading multi-step IK data...")

    all_X = []
    all_y = []
    all_h = []

    for h in h_values:
        data_path = Path('data/multistep_ik') / f'ik_pairs_h{h}.pkl'
        with open(data_path, 'rb') as f:
            pairs = pickle.load(f)

        print(f"  h={h}: {len(pairs)} pairs")

        for pair in pairs:
            # Features: concatenate current state + future state + h one-hot
            state_curr = pair['state_current']  # (90,)
            state_fut = pair['state_future']    # (90,)

            # h one-hot encoding (4 values: 1,2,3,4)
            h_onehot = np.zeros(4)
            h_onehot[h - 1] = 1.0

            # Concatenate all features (89 + 89 + 4 = 182)
            features = np.concatenate([state_curr, state_fut, h_onehot])  # (182,)

            all_X.append(features)
            all_y.append(pair['action'])
            all_h.append(h)

    X = np.array(all_X)
    y = np.array(all_y)
    h_labels = np.array(all_h)

    print(f"\nTotal dataset:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Action distribution: {np.bincount(y, minlength=36)[:10]}... (showing first 10)")

    return X, y, h_labels


def train_model(X_train, y_train, X_val, y_val, model_type='mlp'):
    """Train scikit-learn model"""

    print(f"\nTraining {model_type.upper()} model...")

    if model_type == 'mlp':
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

    elif model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            verbose=1,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\nTraining accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")

    return model, train_acc, val_acc


def analyze_h_specific_performance(model, X, y, h_labels):
    """Analyze model performance for each h separately"""

    print("\n" + "=" * 80)
    print("H-Specific Performance Analysis")
    print("=" * 80)

    for h in [1, 2, 3, 4]:
        mask = (h_labels == h)
        X_h = X[mask]
        y_h = y[mask]

        if len(X_h) == 0:
            continue

        y_pred = model.predict(X_h)
        acc = accuracy_score(y_h, y_pred)

        print(f"\nh = {h}:")
        print(f"  Samples: {len(X_h)}")
        print(f"  Accuracy: {acc:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'rf'],
                        help='Model type: mlp or rf')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    args = parser.parse_args()

    print("=" * 80)
    print("Train Multi-Step IK Policy (scikit-learn)")
    print("=" * 80)

    # Load data
    X, y, h_labels = load_all_ik_data(h_values=[1, 2, 3, 4])

    # Train/val split
    X_train, X_val, y_train, y_val, h_train, h_val = train_test_split(
        X, y, h_labels,
        test_size=args.val_split,
        random_state=42,
        stratify=y  # Stratify by action to keep class balance
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")

    # Train model
    model, train_acc, val_acc = train_model(X_train, y_train, X_val, y_val, model_type=args.model)

    # Analyze h-specific performance
    analyze_h_specific_performance(model, X_val, y_val, h_val)

    # Save model
    output_dir = Path('models/multistep_ik')
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f'policy_{args.model}.pkl'

    joblib.dump({
        'model': model,
        'model_type': args.model,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'feature_dim': X.shape[1]
    }, model_path)

    print(f"\n✅ Saved model to {model_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(f"\nNext step: Generate trajectories using this policy")
    print(f"  python3 generate_trajectories_with_ik_policy.py --model {args.model}")


if __name__ == '__main__':
    main()
