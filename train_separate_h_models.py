"""
Train Separate Models for h=1 and h=4

Option B3: Separate Encoder per h

Key idea:
- Train h=1 model ONLY on h=1 data
- Train h=4 model ONLY on h=4 data
- No h_onehot needed (each model is h-specific)
- Eliminates h-interference problem

Expected benefit:
- Each model specializes on its h
- No competition for capacity
- Clearer h-dependent behavior

Usage:
    python3 train_separate_h_models.py
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib


def load_h_specific_data(h, data_dir='data/multistep_ik'):
    """
    Load data for specific h value.

    Returns:
        X: (N, 178) features [state_current (89), state_future (89)]
        y: (N,) actions
    """
    print(f"\nLoading h={h} data...")

    filepath = Path(data_dir) / f'ik_pairs_h{h}.pkl'
    with open(filepath, 'rb') as f:
        pairs = pickle.load(f)

    print(f"  Total pairs: {len(pairs)}")

    X = []
    y = []

    for pair in pairs:
        # Features: state_current + state_future (NO h_onehot)
        state_curr = pair['state_current']  # (89,)
        state_fut = pair['state_future']    # (89,)

        features = np.concatenate([state_curr, state_fut])  # (178,)

        X.append(features)
        y.append(pair['action'])

    X = np.array(X)
    y = np.array(y)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    return X, y


def train_h_specific_model(h, val_split=0.2):
    """Train MLP model for specific h"""

    print("=" * 80)
    print(f"Training h={h} Specific Model")
    print("=" * 80)

    # Load data
    X, y = load_h_specific_data(h)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_split,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")

    # Create model (same architecture as before)
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

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\n{'='*80}")
    print(f"Results for h={h}")
    print(f"{'='*80}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")

    # Save model
    output_dir = Path('models/separate_h')
    output_dir.mkdir(exist_ok=True, parents=True)
    model_path = output_dir / f'model_h{h}.pkl'

    joblib.dump({
        'model': model,
        'h': h,
        'train_acc': train_acc,
        'val_acc': val_acc,
        'feature_dim': X.shape[1]
    }, model_path)

    print(f"\n✅ Saved model to {model_path}")

    return model, train_acc, val_acc


def main():
    print("=" * 80)
    print("Train Separate Models for h=1 and h=4")
    print("Option B3: Eliminate h-interference")
    print("=" * 80)

    results = {}

    # Train h=1 model
    model_h1, train_acc_h1, val_acc_h1 = train_h_specific_model(h=1)
    results[1] = {'train_acc': train_acc_h1, 'val_acc': val_acc_h1}

    # Train h=4 model
    model_h4, train_acc_h4, val_acc_h4 = train_h_specific_model(h=4)
    results[4] = {'train_acc': train_acc_h4, 'val_acc': val_acc_h4}

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nh=1 Model:")
    print(f"  Train accuracy: {results[1]['train_acc']:.3f}")
    print(f"  Val accuracy: {results[1]['val_acc']:.3f}")

    print(f"\nh=4 Model:")
    print(f"  Train accuracy: {results[4]['train_acc']:.3f}")
    print(f"  Val accuracy: {results[4]['val_acc']:.3f}")

    print(f"\nKey difference from shared model:")
    print(f"  - No h_onehot encoding")
    print(f"  - Each model specialized on single h")
    print(f"  - Input: 178-dim (vs 182-dim before)")

    print(f"\nNext step:")
    print(f"  python3 generate_trajectories_separate_h.py --h 1")
    print(f"  python3 generate_trajectories_separate_h.py --h 4")
    print(f"  python3 compare_separate_h_distributions.py")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
