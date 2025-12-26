"""
Step A: Generate h-specific Training Data

Generate trajectories using DepthLimitedPolicy(h) for Behavior Cloning.

CRITICAL VALIDATION CHECKPOINTS:
1. Observations are 89-dim (board + features, NO h information)
2. Actions are in range [0, 35]
3. 'h' is metadata only (NOT used in training)

Usage:
    python3 generate_training_data.py --h 4 --num_episodes 100 --seed 42
"""

import numpy as np
import os
import pickle
from typing import List, Dict
import argparse

try:
    from .env import FourInARowEnv
    from .depth_limited_policy import DepthLimitedPolicy
    from .bfs_wrapper import load_all_participant_parameters
except ImportError:
    from env import FourInARowEnv
    from depth_limited_policy import DepthLimitedPolicy
    from bfs_wrapper import load_all_participant_parameters


def generate_depth_limited_trajectories(
    h: int,
    num_episodes: int = 100,
    seed: int = 42,
    participant_id: int = 1,
    beta: float = 1.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate trajectories using DepthLimitedPolicy(h)

    Args:
        h: Planning depth
        num_episodes: Number of episodes to generate
        seed: Random seed
        participant_id: Which participant's parameters to use
        beta: Inverse temperature (default: 1.0)
        verbose: Print progress

    Returns:
        trajectories: List of trajectory dicts with:
            - 'observations': (T+1, 89) numpy array
            - 'actions': (T,) numpy array
            - 'length': int
            - 'h': int (metadata only!)

    VALIDATION CHECKPOINTS:
    ✓ Checkpoint 1: Observations are 89-dim (NO h)
    ✓ Checkpoint 2: Actions in range [0, 35]
    ✓ Checkpoint 3: 'h' is metadata only (not used in BC training)
    """
    if verbose:
        print("=" * 80)
        print(f"Generating Depth-Limited Trajectories (h={h})")
        print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Setup: Environment and Policy
    # ═══════════════════════════════════════════════════════
    env = FourInARowEnv()

    # Load expert parameters for heuristic weights
    params_file = os.path.join('opendata', 'model_fits_main_model.csv')
    if not os.path.exists(params_file):
        # Try parent directory
        params_file = os.path.join('..', 'opendata', 'model_fits_main_model.csv')

    if verbose:
        print(f"\n[Setup]")
        print(f"  Loading parameters from: {params_file}")

    params_dict = load_all_participant_parameters(params_file)
    expert_params = params_dict[participant_id]

    if verbose:
        print(f"  Participant ID: {participant_id}")
        print(f"  Pruning threshold: {expert_params.pruning_threshold:.3f}")
        print(f"  Lapse rate: {expert_params.lapse_rate:.3f}")

    # ═══════════════════════════════════════════════════════
    # CRITICAL: h는 여기서만 사용됨 (policy internal)
    # ═══════════════════════════════════════════════════════
    policy = DepthLimitedPolicy(
        h=h,                                    # ← h is HERE (policy internal)
        params=expert_params,
        beta=beta,
        lapse_rate=expert_params.lapse_rate
    )

    if verbose:
        print(f"\n[Policy]")
        print(f"  Planning depth: h={h}")
        print(f"  Beta (inverse temp): {beta}")
        print(f"  Lapse rate: {expert_params.lapse_rate:.3f}")

    # ═══════════════════════════════════════════════════════
    # Generate Trajectories
    # ═══════════════════════════════════════════════════════
    trajectories = []
    rng = np.random.default_rng(seed + h)  # h-specific seed

    if verbose:
        print(f"\n[Generation]")
        print(f"  Episodes to generate: {num_episodes}")
        print(f"  Random seed: {seed + h} (base {seed} + h offset)")

    total_steps = 0
    total_nodes_expanded = 0

    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset(seed=seed + h + episode * 1000)

        episode_obs = [obs.copy()]  # (T+1,) - includes initial state
        episode_acts = []           # (T,)

        done = False
        step_count = 0
        max_steps = 36  # Board size (6x6)

        while not done and step_count < max_steps:
            # ═══════════════════════════════════════════════════════
            # VALIDATION CHECKPOINT 1:
            # obs는 89-dim (board + features), NO h information
            # ═══════════════════════════════════════════════════════
            assert obs.shape == (89,), \
                f"Observation should be 89-dim, got {obs.shape}"

            # Select action using h-step planning
            action, planning_result = policy.select_action(env, rng)

            # ═══════════════════════════════════════════════════════
            # VALIDATION CHECKPOINT 2:
            # action은 0-35 범위
            # ═══════════════════════════════════════════════════════
            assert 0 <= action <= 35, \
                f"Action out of range: {action}"

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)

            episode_obs.append(obs.copy())
            episode_acts.append(action)

            done = terminated or truncated
            step_count += 1

            # Track nodes expanded
            total_nodes_expanded += planning_result.nodes_expanded

        # ═══════════════════════════════════════════════════════
        # Store trajectory
        # ═══════════════════════════════════════════════════════
        trajectory = {
            'observations': np.array(episode_obs, dtype=np.float32),  # (T+1, 89)
            'actions': np.array(episode_acts, dtype=np.int64),        # (T,)
            'length': len(episode_acts),
            'h': h  # ← Metadata only (NOT used in training!)
        }

        # ═══════════════════════════════════════════════════════
        # VALIDATION CHECKPOINT 3:
        # Verify final trajectory format
        # ═══════════════════════════════════════════════════════
        assert trajectory['observations'].shape == (len(episode_acts) + 1, 89), \
            f"Expected ({len(episode_acts) + 1}, 89), got {trajectory['observations'].shape}"
        assert trajectory['actions'].shape == (len(episode_acts),), \
            f"Expected ({len(episode_acts)},), got {trajectory['actions'].shape}"

        trajectories.append(trajectory)
        total_steps += step_count

        # Progress update
        if verbose and (episode + 1) % 10 == 0:
            print(f"  Episode {episode + 1}/{num_episodes} complete "
                  f"(avg length: {total_steps / (episode + 1):.1f})")

    # ═══════════════════════════════════════════════════════
    # Summary Statistics
    # ═══════════════════════════════════════════════════════
    avg_length = np.mean([t['length'] for t in trajectories])
    min_length = np.min([t['length'] for t in trajectories])
    max_length = np.max([t['length'] for t in trajectories])
    avg_nodes = total_nodes_expanded / num_episodes

    if verbose:
        print(f"\n[Results]")
        print(f"  Generated trajectories: {len(trajectories)}")
        print(f"  Total steps: {total_steps}")
        print(f"  Average length: {avg_length:.1f}")
        print(f"  Length range: [{min_length}, {max_length}]")
        print(f"  Average nodes expanded: {avg_nodes:.0f}")
        print(f"\n✓ All validation checkpoints passed")
        print(f"✓ Observations: 89-dim (NO depth information)")
        print(f"✓ Actions: [0, 35] range")
        print(f"✓ 'h' stored as metadata only")
        print("=" * 80)

    return trajectories


def save_trajectories(
    trajectories: List[Dict],
    h: int,
    output_dir: str = 'data/training_trajectories'
):
    """
    Save generated trajectories to disk

    Args:
        trajectories: List of trajectory dicts
        h: Planning depth (for filename)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'trajectories_h{h}.pkl')

    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"\n✓ Saved {len(trajectories)} trajectories to {output_path}")

    # Also save summary statistics
    summary_path = os.path.join(output_dir, f'summary_h{h}.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Trajectory Generation Summary (h={h})\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Number of trajectories: {len(trajectories)}\n")
        f.write(f"Average length: {np.mean([t['length'] for t in trajectories]):.1f}\n")
        f.write(f"Min length: {np.min([t['length'] for t in trajectories])}\n")
        f.write(f"Max length: {np.max([t['length'] for t in trajectories])}\n")
        f.write(f"Total steps: {sum(t['length'] for t in trajectories)}\n")
        f.write(f"\nObservation shape: (T+1, 89)\n")
        f.write(f"Action shape: (T,)\n")
        f.write(f"h metadata: {h} (NOT used in training)\n")

    print(f"✓ Saved summary to {summary_path}")


def load_trajectories(h: int, data_dir: str = 'data/training_trajectories') -> List[Dict]:
    """
    Load previously generated trajectories

    Args:
        h: Planning depth
        data_dir: Data directory

    Returns:
        trajectories: List of trajectory dicts
    """
    input_path = os.path.join(data_dir, f'trajectories_h{h}.pkl')

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Trajectories not found: {input_path}")

    with open(input_path, 'rb') as f:
        trajectories = pickle.load(f)

    print(f"✓ Loaded {len(trajectories)} trajectories from {input_path}")

    return trajectories


def generate_all_depths(
    depths: List[int] = [1, 2, 4, 8],
    num_episodes: int = 100,
    seed: int = 42,
    output_dir: str = 'data/training_trajectories'
):
    """
    Generate trajectories for all planning depths

    Args:
        depths: List of planning depths
        num_episodes: Episodes per depth
        seed: Random seed
        output_dir: Output directory
    """
    print("=" * 80)
    print("Generating Training Data for All Depths")
    print("=" * 80)
    print(f"Depths: {depths}")
    print(f"Episodes per depth: {num_episodes}")
    print(f"Output directory: {output_dir}")
    print()

    for h in depths:
        print(f"\n{'=' * 80}")
        print(f"Processing h={h}")
        print(f"{'=' * 80}\n")

        # Generate trajectories
        trajectories = generate_depth_limited_trajectories(
            h=h,
            num_episodes=num_episodes,
            seed=seed,
            verbose=True
        )

        # Save to disk
        save_trajectories(trajectories, h, output_dir)

    print(f"\n{'=' * 80}")
    print("✓ All depths processed successfully")
    print(f"{'=' * 80}")


def test_generation():
    """Test trajectory generation with a small sample"""
    print("=" * 80)
    print("Testing Trajectory Generation (Step A)")
    print("=" * 80)

    # Test with h=2, 5 episodes
    trajectories = generate_depth_limited_trajectories(
        h=2,
        num_episodes=5,
        seed=42,
        verbose=True
    )

    # Verify format
    print(f"\n[Verification]")
    for i, traj in enumerate(trajectories):
        print(f"\nTrajectory {i}:")
        print(f"  Observations shape: {traj['observations'].shape}")
        print(f"  Actions shape: {traj['actions'].shape}")
        print(f"  Length: {traj['length']}")
        print(f"  h (metadata): {traj['h']}")

        # Verify dtypes
        assert traj['observations'].dtype == np.float32
        assert traj['actions'].dtype == np.int64

        # Verify dimensions
        assert traj['observations'].shape == (traj['length'] + 1, 89)
        assert traj['actions'].shape == (traj['length'],)

        # Verify action range
        assert np.all(traj['actions'] >= 0) and np.all(traj['actions'] <= 35)

        print(f"  ✓ Format validated")

    print(f"\n{'=' * 80}")
    print("✓ Test passed: All trajectories have correct format")
    print("✓ Observations: 89-dim (NO h information)")
    print("✓ Actions: [0, 35] range")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate h-specific training trajectories (Step A)'
    )
    parser.add_argument('--h', type=int, default=None,
                        help='Planning depth (default: generate all [1,2,4,8])')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default='data/training_trajectories',
                        help='Output directory (default: data/training_trajectories)')
    parser.add_argument('--test', action='store_true',
                        help='Run test with 5 episodes (h=2)')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_generation()
    elif args.h is not None:
        # Generate single depth
        trajectories = generate_depth_limited_trajectories(
            h=args.h,
            num_episodes=args.num_episodes,
            seed=args.seed,
            verbose=True
        )
        save_trajectories(trajectories, args.h, args.output_dir)
    else:
        # Generate all depths
        generate_all_depths(
            depths=[1, 2, 4, 8],
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir
        )
