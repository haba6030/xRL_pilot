"""
Multi-Step Inverse Kinematics Data Preprocessing

Based on Mhammedi et al. (2023) multi-step inverse kinematics approach.
No heuristics needed - planning depth h determines how many steps ahead to look.

Key idea:
- Load human game trajectories from opendata/raw_data.csv
- For each h ∈ {1, 2, 3, 4}, create training pairs: (state_t, state_{t+h}, action_t)
- This naturally creates h-dependent datasets without using any heuristics

Usage:
    python3 preprocess_multistep_ik_data.py
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict
import sys
sys.path.append('.')

from env import FourInARowEnv


def parse_board_state(black_pieces, white_pieces, current_player='Black'):
    """
    Convert binary string representation to board state observation.

    MUST match env._get_observation() format exactly!

    Args:
        black_pieces: 36-char binary string (6x6 board)
        white_pieces: 36-char binary string
        current_player: 'Black' or 'White'

    Returns:
        observation: (89,) array matching env._get_observation() format
    """
    try:
        from features import extract_van_opheusden_features
    except ImportError:
        # Fallback: use dummy features if not available
        # This should match the 17-dim van Opheusden features
        features = np.zeros(17, dtype=np.float32)
        return np.concatenate([
            np.array([int(c) for c in black_pieces], dtype=np.float32),  # 36
            np.array([int(c) for c in white_pieces], dtype=np.float32),  # 36
            features  # 17
        ])

    # Parse strings to numpy arrays
    black = np.array([int(c) for c in black_pieces], dtype=np.float32)
    white = np.array([int(c) for c in white_pieces], dtype=np.float32)

    # Extract van Opheusden features (same as env.py)
    features = extract_van_opheusden_features(
        black,
        white,
        current_player
    )

    # Create observation: [black (36), white (36), features (17)] = 89
    observation = np.concatenate([
        black,      # 36-dim
        white,      # 36-dim
        features    # 17-dim
    ]).astype(np.float32)

    return observation


def load_human_trajectories(data_path='../opendata/raw_data.csv', max_games=None):
    """
    Load human game trajectories from raw_data.csv.

    Returns:
        trajectories: List of dicts, each containing:
            - 'observations': (T+1, 90) array of states
            - 'actions': (T,) array of actions
            - 'participant': participant ID
            - 'game_id': unique game identifier
    """
    print("Loading human game data...")
    df = pd.read_csv(data_path)

    print(f"Total moves in dataset: {len(df)}")
    print(f"Unique participants: {df['participant'].nunique()}")

    # Group moves into games
    # Strategy: detect game boundaries by checking for empty boards
    trajectories = []
    current_game = None
    game_id = 0

    for idx, row in df.iterrows():
        participant = row['participant']
        black_pieces = row['black_pieces']
        white_pieces = row['white_pieces']
        color = row['color']  # 'Black' or 'White'

        # Check if this is a new game (empty board)
        is_empty_board = (black_pieces == '0' * 36 and white_pieces == '0' * 36)

        if is_empty_board or current_game is None:
            # Save previous game if exists
            if current_game is not None and len(current_game['actions']) > 0:
                trajectories.append(current_game)
                game_id += 1

                if max_games and game_id >= max_games:
                    break

            # Start new game
            current_game = {
                'observations': [],
                'actions': [],
                'participant': participant,
                'game_id': game_id
            }

        # Parse state with current player info
        state = parse_board_state(black_pieces, white_pieces, current_player=color)
        current_game['observations'].append(state)
        current_game['actions'].append(int(row['move']))

    # Save last game
    if current_game is not None and len(current_game['actions']) > 0:
        trajectories.append(current_game)

    # Convert lists to arrays
    for traj in trajectories:
        traj['observations'] = np.array(traj['observations'])
        traj['actions'] = np.array(traj['actions'])

    print(f"Loaded {len(trajectories)} game trajectories")
    print(f"Average game length: {np.mean([len(t['actions']) for t in trajectories]):.1f} moves")

    return trajectories


def create_multistep_ik_pairs(trajectories, h):
    """
    Create multi-step inverse kinematics training pairs.

    For each timestep t where t+h is valid:
        (state_t, state_{t+h}, action_t)

    Args:
        trajectories: List of game trajectories
        h: Planning depth (number of steps ahead to look)

    Returns:
        pairs: List of dicts with keys:
            - 'state_current': (90,) current state
            - 'state_future': (90,) state h steps ahead
            - 'action': action taken at current state
            - 'h': planning depth
            - 'game_id': which game this came from
    """
    pairs = []

    for traj in trajectories:
        observations = traj['observations']
        actions = traj['actions']
        game_id = traj['game_id']

        # For each valid timestep
        T = len(actions)
        for t in range(T - h):
            pair = {
                'state_current': observations[t],      # (90,)
                'state_future': observations[t + h],   # (90,)
                'action': actions[t],                   # scalar
                'h': h,
                'game_id': game_id,
                't': t
            }
            pairs.append(pair)

    return pairs


def main():
    print("=" * 80)
    print("Multi-Step Inverse Kinematics Data Preprocessing")
    print("=" * 80)

    # Configuration
    h_values = [1, 2, 3, 4]
    output_dir = Path('data/multistep_ik')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load human trajectories
    print("\n[1. Loading Human Game Data]")
    trajectories = load_human_trajectories(
        data_path='../opendata/raw_data.csv',
        max_games=100  # Start with 100 games for pilot
    )

    # Create multi-step IK datasets for each h
    print(f"\n[2. Creating Multi-Step IK Datasets]")

    for h in h_values:
        print(f"\n--- h = {h} ---")

        pairs = create_multistep_ik_pairs(trajectories, h)

        print(f"  Created {len(pairs)} training pairs")
        print(f"  Average pairs per game: {len(pairs) / len(trajectories):.1f}")

        # Save dataset
        output_path = output_dir / f'ik_pairs_h{h}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(pairs, f)

        print(f"  Saved to {output_path}")

        # Print sample
        if len(pairs) > 0:
            sample = pairs[0]
            print(f"\n  Sample pair:")
            print(f"    Game ID: {sample['game_id']}")
            print(f"    Timestep: t={sample['t']}")
            print(f"    Action: {sample['action']}")
            print(f"    State current shape: {sample['state_current'].shape}")
            print(f"    State future shape: {sample['state_future'].shape}")

    # Create summary
    print(f"\n[3. Summary]")
    print(f"\nDatasets created:")
    for h in h_values:
        path = output_dir / f'ik_pairs_h{h}.pkl'
        with open(path, 'rb') as f:
            pairs = pickle.load(f)
        print(f"  h={h}: {len(pairs)} pairs ({path})")

    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")

    print(f"""
Next steps:
1. Verify data quality: check state transitions make sense
2. Analyze action distributions per h (should differ due to different future states)
3. Train h-specific policies using multi-step IK objective
4. Compare behavioral distinguishability (KL/JS divergence)

Key difference from heuristic approach:
- No heuristic weights used!
- h naturally affects behavior through different (state_t, state_{{t+h}}) pairs
- Expected: Higher KL divergence between h values
""")


if __name__ == '__main__':
    main()
