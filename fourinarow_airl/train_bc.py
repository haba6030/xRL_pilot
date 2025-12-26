"""
Step B: Behavior Cloning (BC)

Train neural network to mimic DepthLimitedPolicy(h) behavior.

CRITICAL PRINCIPLE:
BC learns (state → action) mapping ONLY. Planning depth h is NOT used in training!

VALIDATION CHECKPOINTS:
1. Only observations and actions used (NO h)
2. BC policy has NO depth-related attributes
3. Input dimension is 89 (board + features)
4. Output dimension is 36 (actions)

Usage:
    python3 train_bc.py --h 4 --n_epochs 50
"""

import numpy as np
import os
import pickle
from typing import List, Dict
import argparse
import torch
import torch.nn as nn

try:
    from .env import FourInARowEnv
    from .generate_training_data import load_trajectories
except ImportError:
    from env import FourInARowEnv
    from generate_training_data import load_trajectories

# Import imitation library
try:
    from imitation.algorithms import bc
    from imitation.data import types as il_types
except ImportError:
    print("ERROR: imitation library not installed")
    print("Install with: pip install imitation stable-baselines3 torch")
    raise


def train_bc_policy(
    trajectories: List[Dict],
    env,
    h: int,  # For logging only!
    n_epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    verbose: bool = True
):
    """
    Train BC policy to mimic DepthLimitedPolicy(h)

    Args:
        trajectories: Generated from DepthLimitedPolicy(h)
        env: FourInARowEnv
        h: Planning depth (for logging/saving ONLY, NOT training!)
        n_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate (not used directly, but for reference)
        verbose: Print progress

    Returns:
        bc_trainer: Trained BC object

    VALIDATION CHECKPOINTS:
    ✓ Checkpoint 3: Convert to imitation format WITHOUT using h
    ✓ Checkpoint 4: BC policy architecture has NO h parameter
    """
    if verbose:
        print("=" * 80)
        print(f"Behavior Cloning Training (h={h})")
        print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 3:
    # Convert to imitation format WITHOUT using h
    # ═══════════════════════════════════════════════════════
    imitation_trajectories = []

    if verbose:
        print(f"\n[Data Conversion]")

    for i, traj in enumerate(trajectories):
        # ═══════════════════════════════════════════════════════
        # CRITICAL: Only use observations and actions
        # DO NOT use traj['h'] anywhere!
        # ═══════════════════════════════════════════════════════
        obs = traj['observations']   # (T+1, 89)
        acts = traj['actions']       # (T,)

        # Verify dimensions
        assert obs.shape[1] == 89, \
            f"Trajectory {i}: Expected 89-dim obs, got {obs.shape[1]}"
        assert acts.min() >= 0 and acts.max() <= 35, \
            f"Trajectory {i}: Actions out of range [{acts.min()}, {acts.max()}]"

        # Create imitation Trajectory
        imitation_traj = il_types.Trajectory(
            obs=obs,
            acts=acts,
            infos=None,
            terminal=True
        )
        imitation_trajectories.append(imitation_traj)

    if verbose:
        total_transitions = sum(len(t.acts) for t in imitation_trajectories)
        print(f"  Trajectories: {len(imitation_trajectories)}")
        print(f"  Total transitions: {total_transitions}")
        print(f"  Average trajectory length: {total_transitions / len(imitation_trajectories):.1f}")
        print(f"  ✓ Converted WITHOUT using h metadata")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 4:
    # BC policy architecture has NO h parameter
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n[BC Policy Architecture]")
        print(f"  Observation space: {env.observation_space} (89-dim)")
        print(f"  Action space: {env.action_space} (36 discrete)")
        print(f"  Network: MLP [64, 64] with Tanh activation")
        print(f"  ✓ NO h parameter in architecture")

    # Create RNG for BC
    rng = np.random.default_rng(seed=42)

    # Create BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,  # Box(89,) - NO h!
        action_space=env.action_space,            # Discrete(36)
        demonstrations=imitation_trajectories,
        rng=rng,
        batch_size=batch_size,
        # Note: Using default policy architecture (MLP)
        # No h parameter in policy!
    )

    # ═══════════════════════════════════════════════════════
    # VALIDATION: Verify policy has no h-related attributes
    # ═══════════════════════════════════════════════════════
    policy = bc_trainer.policy
    suspicious_attrs = [
        attr for attr in dir(policy)
        if 'depth' in attr.lower() or attr == 'h'
    ]

    if verbose:
        print(f"\n[Validation]")

    if len(suspicious_attrs) > 0:
        if verbose:
            print(f"  ⚠️  WARNING: Found suspicious attributes: {suspicious_attrs}")
            print(f"  Please verify these do NOT encode planning depth")
    else:
        if verbose:
            print(f"  ✓ Policy has no depth-related attributes")

    # Verify observation space
    assert policy.observation_space.shape == (89,), \
        f"Policy observation space should be (89,), got {policy.observation_space.shape}"

    if verbose:
        print(f"  ✓ Policy observation space: {policy.observation_space.shape}")
        print(f"  ✓ Policy action space: {policy.action_space.n} actions")

    # ═══════════════════════════════════════════════════════
    # Train
    # ═══════════════════════════════════════════════════════
    if verbose:
        print(f"\n[Training]")
        print(f"  Epochs: {n_epochs}")
        print(f"  Batch size: {batch_size}")

    # Train BC
    bc_trainer.train(n_epochs=n_epochs)

    if verbose:
        print(f"\n✓ BC training complete")
        print(f"✓ Policy mimics DepthLimitedPolicy(h={h}) behavior")
        print(f"✓ Policy is depth-agnostic (only sees 89-dim observations)")
        print("=" * 80)

    return bc_trainer


def save_bc_policy(
    bc_trainer,
    h: int,
    output_dir: str = 'models/bc_policies'
):
    """
    Save trained BC policy

    Args:
        bc_trainer: Trained BC object
        h: Planning depth (for filename only)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save policy
    policy_path = os.path.join(output_dir, f'bc_policy_h{h}.pt')
    torch.save(bc_trainer.policy.state_dict(), policy_path)
    print(f"\n✓ Saved BC policy to {policy_path}")

    # Save entire trainer (for later PPO wrapping)
    trainer_path = os.path.join(output_dir, f'bc_trainer_h{h}.pkl')
    with open(trainer_path, 'wb') as f:
        pickle.dump(bc_trainer, f)
    print(f"✓ Saved BC trainer to {trainer_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, f'metadata_h{h}.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"BC Policy Metadata (h={h})\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Planning depth: h={h} (policy mimics this depth)\n")
        f.write(f"Observation space: {bc_trainer.policy.observation_space}\n")
        f.write(f"Action space: {bc_trainer.policy.action_space}\n")
        f.write(f"Network architecture: [64, 64] MLP\n")
        f.write(f"Activation: Tanh\n")
        f.write(f"\nIMPORTANT:\n")
        f.write(f"- Policy does NOT take h as input\n")
        f.write(f"- Policy only sees 89-dim observations\n")
        f.write(f"- h is used only to label which depth this policy mimics\n")

    print(f"✓ Saved metadata to {metadata_path}")


def load_bc_trainer(h: int, model_dir: str = 'models/bc_policies'):
    """
    Load previously trained BC policy

    Args:
        h: Planning depth
        model_dir: Model directory

    Returns:
        bc_trainer: Loaded BC trainer
    """
    trainer_path = os.path.join(model_dir, f'bc_trainer_h{h}.pkl')

    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"BC trainer not found: {trainer_path}")

    with open(trainer_path, 'rb') as f:
        bc_trainer = pickle.load(f)

    print(f"✓ Loaded BC trainer from {trainer_path}")

    return bc_trainer


def train_all_depths(
    depths: List[int] = [1, 2, 4, 8],
    n_epochs: int = 50,
    data_dir: str = 'data/training_trajectories',
    output_dir: str = 'models/bc_policies'
):
    """
    Train BC policies for all planning depths

    Args:
        depths: List of planning depths
        n_epochs: Training epochs per depth
        data_dir: Directory with generated trajectories
        output_dir: Output directory for models
    """
    print("=" * 80)
    print("Training BC Policies for All Depths")
    print("=" * 80)
    print(f"Depths: {depths}")
    print(f"Epochs per depth: {n_epochs}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print()

    env = FourInARowEnv()

    for h in depths:
        print(f"\n{'=' * 80}")
        print(f"Processing h={h}")
        print(f"{'=' * 80}\n")

        # Load trajectories
        print(f"[Loading Data]")
        trajectories = load_trajectories(h, data_dir)
        print()

        # Train BC
        bc_trainer = train_bc_policy(
            trajectories=trajectories,
            env=env,
            h=h,
            n_epochs=n_epochs,
            verbose=True
        )

        # Save model
        save_bc_policy(bc_trainer, h, output_dir)

    print(f"\n{'=' * 80}")
    print("✓ All BC policies trained successfully")
    print(f"{'=' * 80}")


def test_bc_training():
    """Test BC training with generated test data"""
    print("=" * 80)
    print("Testing BC Training (Step B)")
    print("=" * 80)

    # Generate test trajectories
    print("\n[Generating Test Data]")
    from generate_training_data import generate_depth_limited_trajectories

    trajectories = generate_depth_limited_trajectories(
        h=2,
        num_episodes=10,
        seed=42,
        verbose=False
    )
    print(f"✓ Generated {len(trajectories)} test trajectories")

    # Train BC
    env = FourInARowEnv()
    bc_trainer = train_bc_policy(
        trajectories=trajectories,
        env=env,
        h=2,
        n_epochs=5,  # Quick test
        verbose=True
    )

    # Test forward pass
    print(f"\n[Testing Forward Pass]")
    obs, _ = env.reset(seed=42)
    action, _ = bc_trainer.policy.predict(obs, deterministic=True)

    print(f"  Test observation shape: {obs.shape}")
    print(f"  Predicted action: {action}")
    print(f"  Action in valid range: {0 <= action <= 35}")

    assert obs.shape == (89,), f"Observation shape mismatch"
    assert 0 <= action <= 35, f"Invalid action: {action}"

    print(f"\n✓ Forward pass test passed")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train BC policy to mimic DepthLimitedPolicy (Step B)'
    )
    parser.add_argument('--h', type=int, default=None,
                        help='Planning depth (default: train all [1,2,4,8])')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Training epochs (default: 50)')
    parser.add_argument('--data_dir', type=str, default='data/training_trajectories',
                        help='Directory with trajectories (default: data/training_trajectories)')
    parser.add_argument('--output_dir', type=str, default='models/bc_policies',
                        help='Output directory (default: models/bc_policies)')
    parser.add_argument('--test', action='store_true',
                        help='Run test with 10 episodes (h=2, 5 epochs)')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_bc_training()
    elif args.h is not None:
        # Train single depth
        env = FourInARowEnv()
        trajectories = load_trajectories(args.h, args.data_dir)
        bc_trainer = train_bc_policy(
            trajectories=trajectories,
            env=env,
            h=args.h,
            n_epochs=args.n_epochs,
            verbose=True
        )
        save_bc_policy(bc_trainer, args.h, args.output_dir)
    else:
        # Train all depths
        train_all_depths(
            depths=[1, 2, 4, 8],
            n_epochs=args.n_epochs,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
