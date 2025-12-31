"""
Data Loading for AIRL Training
Convert raw_data.csv expert trajectories to format needed for imitation library
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GameTrajectory:
    """Single game trajectory"""
    observations: np.ndarray  # (T+1, 89) - states including final state
    actions: np.ndarray       # (T,) - action indices
    rewards: np.ndarray       # (T,) - rewards (0 until final +1/-1)
    player_id: int            # Which player (0=Black, 1=White)
    game_id: int              # Unique game identifier
    participant_id: int       # Participant ID


def parse_board_string(black_str: str, white_str: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert board strings to binary arrays

    Args:
        black_str: 36-char string of '0' and '1'
        white_str: 36-char string of '0' and '1'

    Returns:
        black_pieces: 36-dim binary array
        white_pieces: 36-dim binary array
    """
    black_pieces = np.array([int(c) for c in black_str], dtype=np.float32)
    white_pieces = np.array([int(c) for c in white_str], dtype=np.float32)
    return black_pieces, white_pieces


def reconstruct_games(raw_data: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Reconstruct individual games from trial-level data

    Args:
        raw_data: DataFrame from raw_data.csv

    Returns:
        games: List of DataFrames, each representing one game
    """
    games = []

    # Group by participant and session
    # Need to infer game boundaries by detecting board resets

    for (participant, session), group in raw_data.groupby(['participant', 'session number']):
        group = group.sort_index()  # Ensure chronological order

        # Detect game boundaries: when board becomes emptier than before
        game_starts = [0]
        prev_piece_count = 0

        for idx, (_, row) in enumerate(group.iterrows()):
            black_count = str(row['black_pieces']).count('1')
            white_count = str(row['white_pieces']).count('1')
            total_count = black_count + white_count

            # If piece count decreased, it's a new game
            if total_count < prev_piece_count:
                game_starts.append(idx)

            prev_piece_count = total_count

        # Add end marker
        game_starts.append(len(group))

        # Extract each game
        for i in range(len(game_starts) - 1):
            start_idx = game_starts[i]
            end_idx = game_starts[i + 1]
            game_df = group.iloc[start_idx:end_idx].copy()

            if len(game_df) > 0:  # Skip empty games
                games.append(game_df)

    return games


def game_to_trajectory(
    game_df: pd.DataFrame,
    player_filter: Optional[int] = None
) -> Optional[GameTrajectory]:
    """
    Convert a game DataFrame to a trajectory

    Args:
        game_df: DataFrame for a single game
        player_filter: If provided (0 or 1), only include moves by that player

    Returns:
        GameTrajectory or None if filtering removes all moves
    """
    try:
        from .env import FourInARowEnv
    except ImportError:
        from env import FourInARowEnv

    # Filter by player if requested
    if player_filter is not None:
        color_name = 'Black' if player_filter == 0 else 'White'
        game_df = game_df[game_df['color'] == color_name].copy()

    if len(game_df) == 0:
        return None

    # Initialize environment to generate observations
    env = FourInARowEnv()
    obs, _ = env.reset()

    observations = [obs.copy()]
    actions = []
    rewards = []

    # Replay the game
    for idx, row in game_df.iterrows():
        # Parse board state (for verification)
        black_pieces, white_pieces = parse_board_string(
            row['black_pieces'],
            row['white_pieces']
        )

        # Get action
        action = int(row['move'])
        actions.append(action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        observations.append(obs.copy())

        # Rewards are 0 until game ends
        rewards.append(reward if terminated else 0.0)

        if terminated:
            break

    # Convert to numpy arrays
    observations = np.array(observations, dtype=np.float32)  # (T+1, 89)
    actions = np.array(actions, dtype=np.int64)              # (T,)
    rewards = np.array(rewards, dtype=np.float32)            # (T,)

    # Get metadata
    participant_id = game_df.iloc[0]['participant']
    game_id = id(game_df)  # Use object ID as unique identifier

    return GameTrajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        player_id=player_filter if player_filter is not None else 0,
        game_id=game_id,
        participant_id=participant_id
    )


def load_expert_trajectories(
    csv_path: str = 'opendata/raw_data.csv',
    player_filter: Optional[int] = 0,
    max_trajectories: Optional[int] = None
) -> List[GameTrajectory]:
    """
    Load expert trajectories from raw_data.csv

    Args:
        csv_path: Path to raw_data.csv
        player_filter: If provided, only include moves by that player
                      (0=Black, 1=White, None=both)
        max_trajectories: Maximum number of trajectories to load

    Returns:
        trajectories: List of GameTrajectory objects
    """
    print(f"Loading expert trajectories from {csv_path}...")

    # Load raw data
    raw_data = pd.read_csv(csv_path)
    print(f"Loaded {len(raw_data)} trials")

    # Reconstruct games
    games = reconstruct_games(raw_data)
    print(f"Reconstructed {len(games)} games")

    # Convert to trajectories
    trajectories = []
    for game_df in games:
        traj = game_to_trajectory(game_df, player_filter=player_filter)
        if traj is not None:
            trajectories.append(traj)

            if max_trajectories and len(trajectories) >= max_trajectories:
                break

    print(f"Created {len(trajectories)} trajectories")

    if len(trajectories) > 0:
        avg_length = np.mean([len(t.actions) for t in trajectories])
        print(f"Average trajectory length: {avg_length:.1f}")

    return trajectories


def trajectories_to_imitation_format(
    trajectories: List[GameTrajectory]
) -> List[Dict]:
    """
    Convert GameTrajectory objects to imitation library format

    Args:
        trajectories: List of GameTrajectory objects

    Returns:
        imitation_trajs: List of dicts with 'obs', 'acts', 'infos', 'terminal'
    """
    imitation_trajs = []

    for traj in trajectories:
        imitation_traj = {
            'obs': traj.observations,      # (T+1, 89)
            'acts': traj.actions,          # (T,)
            'infos': None,                 # Optional
            'terminal': True               # All games terminate
        }
        imitation_trajs.append(imitation_traj)

    return imitation_trajs


def test_data_loading():
    """Test data loading pipeline"""
    print("=" * 80)
    print("Testing Expert Trajectory Loading")
    print("=" * 80)

    # Load small sample
    import os
    if os.path.exists('opendata/raw_data.csv'):
        csv_path = 'opendata/raw_data.csv'
    elif os.path.exists('../opendata/raw_data.csv'):
        csv_path = '../opendata/raw_data.csv'
    else:
        print("ERROR: Cannot find raw_data.csv")
        return

    trajectories = load_expert_trajectories(
        csv_path=csv_path,
        player_filter=0,  # Black player only
        max_trajectories=10
    )

    if len(trajectories) == 0:
        print("ERROR: No trajectories loaded!")
        return

    # Print first trajectory details
    traj = trajectories[0]
    print(f"\nFirst trajectory:")
    print(f"  Participant: {traj.participant_id}")
    print(f"  Game ID: {traj.game_id}")
    print(f"  Player: {traj.player_id}")
    print(f"  Length: {len(traj.actions)} moves")
    print(f"  Observations shape: {traj.observations.shape}")
    print(f"  Actions: {traj.actions}")
    print(f"  Final reward: {traj.rewards[-1]}")

    # Convert to imitation format
    imitation_trajs = trajectories_to_imitation_format(trajectories)
    print(f"\nConverted to imitation format: {len(imitation_trajs)} trajectories")

    # Statistics
    lengths = [len(t.actions) for t in trajectories]
    print(f"\nTrajectory length statistics:")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Median: {np.median(lengths):.1f}")


if __name__ == '__main__':
    test_data_loading()
