"""
Multi-Step IK Data Preprocessing with CV Group Information

Same as preprocess_multistep_ik_data.py but includes:
- cv_group (cross-validation fold 1-5)
- participant ID

This enables K-fold cross-validation training.

Usage:
    python preprocess_multistep_ik_data_cv.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.append('.')

from env import FourInARowEnv


def parse_board_state(black_pieces, white_pieces, current_player='Black'):
    """Convert binary string to board state observation."""
    try:
        from features import extract_van_opheusden_features
    except ImportError:
        features = np.zeros(17, dtype=np.float32)
        return np.concatenate([
            np.array([int(c) for c in black_pieces], dtype=np.float32),
            np.array([int(c) for c in white_pieces], dtype=np.float32),
            features
        ])

    black = np.array([int(c) for c in black_pieces], dtype=np.float32)
    white = np.array([int(c) for c in white_pieces], dtype=np.float32)

    features = extract_van_opheusden_features(black, white, current_player)

    observation = np.concatenate([
        black,      # 36-dim
        white,      # 36-dim
        features    # 17-dim
    ]).astype(np.float32)

    return observation


def load_human_trajectories_with_cv(data_path='opendata/raw_data.csv', max_games=None):
    """
    Load human trajectories WITH cv_group information.

    Returns:
        trajectories: List of dicts with:
            - observations, actions, participant, game_id
            - cv_group: Cross-validation fold (1-5)
    """
    print("Loading human game data with CV information...")
    df = pd.read_csv(data_path)

    print(f"Total moves: {len(df)}")
    print(f"Participants: {df['participant'].nunique()}")
    print(f"CV groups: {sorted(df['cross-validation group'].unique())}")

    trajectories = []
    current_game = None
    game_id = 0

    for idx, row in df.iterrows():
        participant = row['participant']
        cv_group = row['cross-validation group']
        black_pieces = row['black_pieces']
        white_pieces = row['white_pieces']
        color = row['color']

        # Detect new game
        is_empty_board = (black_pieces == '0' * 36 and white_pieces == '0' * 36)

        if is_empty_board or current_game is None:
            if current_game is not None and len(current_game['actions']) > 0:
                trajectories.append(current_game)
                game_id += 1

                if max_games and game_id >= max_games:
                    break

            current_game = {
                'observations': [],
                'actions': [],
                'participant': participant,
                'cv_group': cv_group,  # ← CV information added
                'game_id': game_id
            }

        state = parse_board_state(black_pieces, white_pieces, current_player=color)
        current_game['observations'].append(state)
        current_game['actions'].append(int(row['move']))

    # Save last game
    if current_game is not None and len(current_game['actions']) > 0:
        trajectories.append(current_game)

    # Convert to arrays
    for traj in trajectories:
        traj['observations'] = np.array(traj['observations'])
        traj['actions'] = np.array(traj['actions'])

    print(f"Loaded {len(trajectories)} games")
    print(f"Average game length: {np.mean([len(t['actions']) for t in trajectories]):.1f} moves")

    # CV group distribution
    cv_dist = pd.Series([t['cv_group'] for t in trajectories]).value_counts().sort_index()
    print(f"\nCV group distribution:")
    for cv, count in cv_dist.items():
        print(f"  Fold {cv}: {count} games")

    return trajectories


def create_multistep_ik_pairs_with_cv(trajectories, h):
    """
    Create multi-step IK pairs WITH cv_group and participant.

    Each pair includes:
        - state_current, state_future, action, h, game_id
        - cv_group, participant  ← Added for CV
    """
    pairs = []

    for traj in trajectories:
        observations = traj['observations']
        actions = traj['actions']
        game_id = traj['game_id']
        cv_group = traj['cv_group']  # ← CV info
        participant = traj['participant']  # ← Participant ID

        T = len(actions)
        for t in range(T - h):
            pair = {
                'state_current': observations[t],
                'state_future': observations[t + h],
                'action': actions[t],
                'h': h,
                'game_id': game_id,
                't': t,
                'cv_group': cv_group,          # ← CV fold
                'participant': participant      # ← Player ID
            }
            pairs.append(pair)

    return pairs


def main():
    print("=" * 80)
    print("Multi-Step IK Data Preprocessing (WITH CV INFORMATION)")
    print("=" * 80)

    h_values = [1, 2, 3, 4]
    output_dir = Path('data/multistep_ik_cv')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load trajectories with CV info
    print("\n[1. Loading Human Game Data with CV Groups]")
    trajectories = load_human_trajectories_with_cv(
        data_path='opendata/raw_data.csv',
        max_games=None  # Use ALL games
    )

    # Create datasets
    print(f"\n[2. Creating Multi-Step IK Datasets with CV Info]")

    for h in h_values:
        print(f"\n--- h = {h} ---")

        pairs = create_multistep_ik_pairs_with_cv(trajectories, h)

        print(f"  Total pairs: {len(pairs)}")
        print(f"  Pairs per game: {len(pairs) / len(trajectories):.1f}")

        # CV group distribution
        cv_dist = pd.Series([p['cv_group'] for p in pairs]).value_counts().sort_index()
        print(f"  CV distribution:")
        for cv, count in cv_dist.items():
            print(f"    Fold {cv}: {count} pairs")

        # Save
        output_path = output_dir / f'ik_pairs_h{h}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(pairs, f)

        print(f"  Saved to {output_path}")

        # Sample
        if len(pairs) > 0:
            sample = pairs[0]
            print(f"\n  Sample pair:")
            print(f"    state_current shape: {sample['state_current'].shape}")
            print(f"    state_future shape: {sample['state_future'].shape}")
            print(f"    action: {sample['action']}")
            print(f"    cv_group: {sample['cv_group']}")
            print(f"    participant: {sample['participant']}")

    print("\n" + "=" * 80)
    print("✅ Preprocessing complete!")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("  1. python train_h_models_kfold_cv.py --all")
    print("  2. python estimate_h_kfold_cv.py --all")


if __name__ == '__main__':
    main()
