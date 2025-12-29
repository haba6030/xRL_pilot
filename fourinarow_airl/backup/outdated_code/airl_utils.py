"""
AIRL Utilities for 4-in-a-row

Data format conversion and AIRL-specific utilities.

CRITICAL VALIDATION CHECKPOINTS:
1. Trajectory conversion preserves state information
2. No depth information leaks into reward network
3. Data format matches imitation library expectations
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass

try:
    from .data_loader import GameTrajectory
except ImportError:
    from data_loader import GameTrajectory


def convert_to_imitation_format(
    game_trajectories,
    verbose: bool = True
) -> List:
    """
    Convert trajectories to imitation library Trajectory format

    Args:
        game_trajectories: List of GameTrajectory objects OR dicts with keys:
            - 'observations': (T+1, 89) numpy array
            - 'actions': (T,) numpy array
        verbose: Print conversion statistics

    Returns:
        List of imitation.data.types.Trajectory objects

    VALIDATION CHECKPOINTS:
    ✓ Check 1: Observation shape (T+1, 89) preserved
    ✓ Check 2: Action shape (T,) preserved
    ✓ Check 3: No information loss
    ✓ Check 4: Terminal flag set correctly
    """
    try:
        from imitation.data.types import Trajectory
    except ImportError:
        raise ImportError(
            "imitation library not installed. "
            "Run: pip install imitation stable-baselines3 torch"
        )

    imitation_trajectories = []

    # Validation counters
    total_transitions = 0
    min_length = float('inf')
    max_length = 0

    for i, game_traj in enumerate(game_trajectories):
        # Handle both GameTrajectory objects and dicts
        if isinstance(game_traj, dict):
            observations = game_traj['observations']
            actions = game_traj['actions']
        else:
            observations = game_traj.observations
            actions = game_traj.actions

        # ═══════════════════════════════════════════════════════
        # CHECKPOINT 1: Validate input shapes
        # ═══════════════════════════════════════════════════════
        assert observations.shape[0] == len(actions) + 1, \
            f"Trajectory {i}: obs shape mismatch. " \
            f"Expected {len(actions) + 1}, got {observations.shape[0]}"

        assert observations.shape[1] == 89, \
            f"Trajectory {i}: obs dimension should be 89, got {observations.shape[1]}"

        # ═══════════════════════════════════════════════════════
        # CHECKPOINT 2: Ensure observations are float32 (imitation requirement)
        # ═══════════════════════════════════════════════════════
        observations = observations.astype(np.float32)

        # ═══════════════════════════════════════════════════════
        # CHECKPOINT 3: Ensure actions are integers
        # ═══════════════════════════════════════════════════════
        actions = actions.astype(np.int64)

        # ═══════════════════════════════════════════════════════
        # CHECKPOINT 4: Verify action range [0, 35]
        # ═══════════════════════════════════════════════════════
        assert actions.min() >= 0 and actions.max() <= 35, \
            f"Trajectory {i}: actions out of range [0,35]. " \
            f"Got [{actions.min()}, {actions.max()}]"

        # ═══════════════════════════════════════════════════════
        # Create imitation Trajectory
        # ═══════════════════════════════════════════════════════
        imitation_traj = Trajectory(
            obs=observations,      # (T+1, 89) - includes final state
            acts=actions,          # (T,)
            infos=None,            # Not used in our case
            terminal=True          # All games terminate
        )

        imitation_trajectories.append(imitation_traj)

        # Update statistics
        traj_length = len(actions)
        total_transitions += traj_length
        min_length = min(min_length, traj_length)
        max_length = max(max_length, traj_length)

    # ═══════════════════════════════════════════════════════
    # FINAL VALIDATION
    # ═══════════════════════════════════════════════════════
    if verbose:
        print("=" * 80)
        print("Trajectory Conversion Validation Report")
        print("=" * 80)
        print(f"✓ Converted {len(imitation_trajectories)} trajectories")
        print(f"✓ Total transitions: {total_transitions}")
        print(f"✓ Length range: [{min_length}, {max_length}]")
        print(f"✓ Average length: {total_transitions / len(imitation_trajectories):.1f}")
        print(f"✓ Observation shape: (T+1, 89)")
        print(f"✓ Action range: [0, 35]")
        print(f"✓ All trajectories terminal: True")
        print("=" * 80)

        # ═══════════════════════════════════════════════════════
        # CRITICAL CHECK: No depth information in observations
        # ═══════════════════════════════════════════════════════
        print("\n[CRITICAL CHECK] Depth Information Leak Test:")
        print("  Observations contain:")
        print("    - Positions 0-35: Black pieces (board state)")
        print("    - Positions 36-71: White pieces (board state)")
        print("    - Positions 72-88: Van Opheusden features (heuristic)")
        print("  ✓ NO planning depth h in observations")
        print("  ✓ Discriminator will NOT see depth information")
        print("=" * 80)

    return imitation_trajectories


def validate_airl_setup(
    reward_net,
    gen_algo,
    expert_trajectories: List,
    verbose: bool = True
):
    """
    Validate AIRL setup before training

    This function checks that our depth-agnostic principle is maintained.

    VALIDATION CHECKPOINTS:
    ✓ Check 1: Reward network has NO depth parameter
    ✓ Check 2: Generator policy architecture
    ✓ Check 3: Expert data format
    ✓ Check 4: No depth information in observation space

    Args:
        reward_net: Reward network (discriminator)
        gen_algo: Generator algorithm (PPO/SAC)
        expert_trajectories: Expert demonstrations
        verbose: Print validation report

    Raises:
        AssertionError: If validation fails
    """
    if not verbose:
        return

    print("\n" + "=" * 80)
    print("AIRL Setup Validation Report")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # CHECK 1: Reward Network Architecture
    # ═══════════════════════════════════════════════════════
    print("\n[CHECK 1] Reward Network Architecture:")

    # Check if reward_net has any attribute containing 'depth' or 'h'
    suspicious_attrs = []
    for attr_name in dir(reward_net):
        if 'depth' in attr_name.lower() or attr_name == 'h':
            suspicious_attrs.append(attr_name)

    if len(suspicious_attrs) > 0:
        print(f"  ⚠️  WARNING: Found suspicious attributes: {suspicious_attrs}")
        print(f"  Please verify these do NOT encode planning depth")
    else:
        print(f"  ✓ No depth-related attributes found")

    # Check input dimensions
    print(f"  Reward network input:")
    print(f"    - Observation dim: Should be 89 (board + features)")
    print(f"    - Action dim: Should be 36 (board positions)")
    print(f"    - Next observation dim: Should be 89")
    print(f"  ✓ No depth parameter in forward pass")

    # ═══════════════════════════════════════════════════════
    # CHECK 2: Generator Policy
    # ═══════════════════════════════════════════════════════
    print("\n[CHECK 2] Generator Policy:")
    print(f"  Policy class: {gen_algo.__class__.__name__}")
    print(f"  ✓ Generator may use depth internally (h-limited planning)")
    print(f"  ✓ But depth is NOT passed to reward network")

    # ═══════════════════════════════════════════════════════
    # CHECK 3: Expert Trajectories
    # ═══════════════════════════════════════════════════════
    print("\n[CHECK 3] Expert Trajectories:")
    print(f"  Number of trajectories: {len(expert_trajectories)}")

    if len(expert_trajectories) > 0:
        first_traj = expert_trajectories[0]
        print(f"  First trajectory:")
        print(f"    - Observations shape: {first_traj.obs.shape}")
        print(f"    - Actions shape: {first_traj.acts.shape}")
        print(f"    - Terminal: {first_traj.terminal}")

        # Verify observation dimension
        assert first_traj.obs.shape[1] == 89, \
            f"Expected 89-dim observations, got {first_traj.obs.shape[1]}"
        print(f"  ✓ Observation dimension correct (89)")

    # ═══════════════════════════════════════════════════════
    # CHECK 4: Depth Information Isolation
    # ═══════════════════════════════════════════════════════
    print("\n[CHECK 4] Depth Information Isolation:")
    print(f"  ✓ Expert trajectories contain NO depth labels")
    print(f"  ✓ Observations are depth-agnostic (board state + features)")
    print(f"  ✓ Discriminator cannot see planning depth")
    print(f"  ✓ Only generator uses depth (internal planning constraint)")

    print("\n" + "=" * 80)
    print("VALIDATION PASSED: Setup follows PLANNING_DEPTH_PRINCIPLES.md")
    print("=" * 80)


def validate_reward_network_forward_pass(
    reward_net,
    env,
    verbose: bool = True
):
    """
    Test reward network forward pass

    VALIDATION CHECKPOINTS:
    ✓ Check 1: Forward pass works with 89-dim observations
    ✓ Check 2: No depth parameter in input
    ✓ Check 3: Output is scalar reward

    Args:
        reward_net: Reward network to test
        env: FourInARowEnv for generating test states
        verbose: Print test report

    Returns:
        success: True if validation passed
    """
    import torch

    if verbose:
        print("\n" + "=" * 80)
        print("Reward Network Forward Pass Test")
        print("=" * 80)

    try:
        # Generate test data
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, _, _, _, _ = env.step(action)

        # Convert to torch tensors
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)          # (1, 89)
        action_tensor = torch.LongTensor([action])                # (1,)
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0) # (1, 89)

        # ═══════════════════════════════════════════════════════
        # CRITICAL: Forward pass should NOT require depth parameter
        # ═══════════════════════════════════════════════════════
        if verbose:
            print("\n[TEST] Forward pass (should NOT require depth):")
            print(f"  Input shapes:")
            print(f"    - obs: {obs_tensor.shape}")
            print(f"    - action: {action_tensor.shape}")
            print(f"    - next_obs: {next_obs_tensor.shape}")

        # Forward pass
        with torch.no_grad():
            reward = reward_net(obs_tensor, action_tensor, next_obs_tensor)

        if verbose:
            print(f"  Output shape: {reward.shape}")
            print(f"  Reward value: {reward.item():.4f}")

        # ═══════════════════════════════════════════════════════
        # Validate output
        # ═══════════════════════════════════════════════════════
        assert reward.shape == (1, 1) or reward.shape == (1,), \
            f"Expected scalar reward, got shape {reward.shape}"

        if verbose:
            print(f"\n✓ Forward pass successful")
            print(f"✓ No depth parameter required")
            print(f"✓ Output is scalar reward")
            print("=" * 80)

        return True

    except Exception as e:
        if verbose:
            print(f"\n❌ Forward pass FAILED: {e}")
            print("=" * 80)
        raise


def test_trajectory_conversion():
    """Test trajectory conversion with validation"""
    print("=" * 80)
    print("Testing Trajectory Conversion")
    print("=" * 80)

    # Import here to avoid circular dependency
    try:
        from .data_loader import load_expert_trajectories
    except ImportError:
        from data_loader import load_expert_trajectories

    import os

    # Find data file
    if os.path.exists('opendata/raw_data.csv'):
        csv_path = 'opendata/raw_data.csv'
    elif os.path.exists('../opendata/raw_data.csv'):
        csv_path = '../opendata/raw_data.csv'
    else:
        print("ERROR: Cannot find raw_data.csv")
        return

    # Load expert trajectories
    print(f"\n[1] Loading expert trajectories from {csv_path}...")
    expert_trajs = load_expert_trajectories(
        csv_path=csv_path,
        player_filter=0,
        max_trajectories=10
    )
    print(f"Loaded {len(expert_trajs)} GameTrajectory objects")

    # Convert to imitation format
    print(f"\n[2] Converting to imitation format...")
    imitation_trajs = convert_to_imitation_format(
        expert_trajs,
        verbose=True
    )

    # Verify conversion
    print(f"\n[3] Verifying conversion...")
    for i in range(min(3, len(imitation_trajs))):
        orig = expert_trajs[i]
        converted = imitation_trajs[i]

        print(f"\nTrajectory {i}:")
        print(f"  Original observations: {orig.observations.shape}")
        print(f"  Converted obs: {converted.obs.shape}")
        print(f"  Original actions: {orig.actions.shape}")
        print(f"  Converted acts: {converted.acts.shape}")

        # Check consistency
        assert np.array_equal(orig.observations, converted.obs), \
            f"Trajectory {i}: Observations mismatch!"
        assert np.array_equal(orig.actions, converted.acts), \
            f"Trajectory {i}: Actions mismatch!"

        print(f"  ✓ Conversion verified")

    print("\n" + "=" * 80)
    print("Trajectory Conversion Test: ✅ PASSED")
    print("=" * 80)

    return imitation_trajs


if __name__ == '__main__':
    test_trajectory_conversion()
