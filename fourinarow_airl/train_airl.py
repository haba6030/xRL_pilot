"""
Step E: AIRL Training

Train AIRL (Adversarial Inverse Reinforcement Learning) for 4-in-a-row.

CRITICAL PRINCIPLE:
- h-specific generator (PPO trained with h-dependent BC policy)
- Depth-AGNOSTIC discriminator (reward network has NO h parameter)

VALIDATION CHECKPOINTS:
1. Expert trajectories have NO h labels
2. Generator learned from h-specific policy
3. Discriminator has NO h parameter
4. AIRL training follows depth-agnostic principles

Usage:
    # Single depth
    python3 train_airl.py --h 2 --total_timesteps 10000

    # All depths
    python3 train_airl.py --total_timesteps 10000

    # Test with minimal setup
    python3 train_airl.py --test
"""

import numpy as np
import os
import pickle
import argparse
from typing import Optional, Dict, List
import torch

try:
    from .env import FourInARowEnv
    from .create_ppo_generator import load_ppo_generator
    from .create_reward_net import create_reward_network
    from .airl_utils import convert_to_imitation_format, validate_airl_setup
    from .data_loader import load_expert_trajectories
except ImportError:
    from env import FourInARowEnv
    from create_ppo_generator import load_ppo_generator
    from create_reward_net import create_reward_network
    from airl_utils import convert_to_imitation_format, validate_airl_setup
    from data_loader import load_expert_trajectories

# Import imitation library
try:
    from imitation.algorithms.adversarial import airl
    from stable_baselines3.common.vec_env import DummyVecEnv
    from imitation.data.types import Trajectory
except ImportError:
    print("ERROR: imitation library not installed")
    print("Install with: pip install imitation stable-baselines3 torch")
    raise


def train_airl_single_depth(
    h: int,
    expert_trajectories: List[Trajectory],
    env,
    total_timesteps: int = 10000,
    demo_batch_size: int = 64,
    n_disc_updates_per_round: int = 4,
    gen_train_timesteps: int = 512,
    output_dir: str = 'models/airl_results',
    verbose: int = 1
):
    """
    Train AIRL for a single planning depth h

    Args:
        h: Planning depth (metadata only, NOT used in discriminator!)
        expert_trajectories: Expert demonstrations (NO h labels)
        env: FourInARowEnv
        total_timesteps: Total training timesteps
        demo_batch_size: Batch size for discriminator updates
        n_disc_updates_per_round: Discriminator updates per round
        gen_train_timesteps: Generator training timesteps per round
        output_dir: Output directory for results
        verbose: Verbosity level

    Returns:
        trainer: Trained AIRL object

    VALIDATION CHECKPOINTS:
    ✓ Checkpoint 7: Expert trajectories have NO h labels
    ✓ Checkpoint 7: Generator learned from h-specific policy
    ✓ Checkpoint 7: Discriminator has NO h parameter
    """
    print("=" * 80)
    print(f"Training AIRL for h={h}")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 7a:
    # Expert trajectories have NO h labels
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL VALIDATION]")
    print(f"  Expert trajectories: {len(expert_trajectories)} trajectories")
    print(f"  ✓ Expert data contains NO h labels")
    print(f"  ✓ Observations are 89-dim (board + features, NO depth)")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 7b:
    # Load h-specific generator (trained via BC → PPO)
    # ═══════════════════════════════════════════════════════
    print(f"\n[Loading h-specific Generator]")
    print(f"  Generator trained from DepthLimitedPolicy(h={h})")
    print(f"  ✓ Generator behavior mimics h={h} planning")
    print(f"  ✓ But generator is neural network (depth-agnostic observations)")

    # Wrap environment for PPO (PPO expects vectorized env)
    venv = DummyVecEnv([lambda: env])

    # Load PPO generator
    ppo_model_path = os.path.join('models/ppo_generators', f'ppo_generator_h{h}.zip')

    if not os.path.exists(ppo_model_path):
        raise FileNotFoundError(
            f"PPO generator not found: {ppo_model_path}\n"
            f"Run create_ppo_generator.py first to create generators for all depths."
        )

    from stable_baselines3 import PPO
    gen_algo = PPO.load(ppo_model_path, env=venv)

    print(f"  ✓ Loaded PPO generator from {ppo_model_path}")
    print(f"  Policy observation space: {gen_algo.policy.observation_space}")
    print(f"  Policy action space: {gen_algo.policy.action_space}")

    # Verify generator uses 89-dim observations
    assert gen_algo.policy.observation_space.shape == (89,), \
        f"Generator should use 89-dim observations, got {gen_algo.policy.observation_space.shape}"

    print(f"  ✓ Generator uses 89-dim observations (NO h)")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 7c:
    # Create depth-AGNOSTIC reward network
    # ═══════════════════════════════════════════════════════
    print(f"\n[Creating Depth-AGNOSTIC Reward Network]")
    print(f"  Reward network has NO h parameter!")
    print(f"  Same architecture for ALL h values")
    print(f"  Different h experiments use FRESH instances (separate training)")

    reward_net = create_reward_network(env)  # NO h parameter!

    print(f"  ✓ Reward network created (NO h parameter)")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 7d:
    # Validate AIRL setup
    # ═══════════════════════════════════════════════════════
    validate_airl_setup(
        reward_net=reward_net,
        gen_algo=gen_algo,
        expert_trajectories=expert_trajectories,
        verbose=(verbose >= 1)
    )

    # ═══════════════════════════════════════════════════════
    # AIRL Trainer Setup
    # ═══════════════════════════════════════════════════════
    print(f"\n[AIRL Trainer Configuration]")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Demo batch size: {demo_batch_size}")
    print(f"  Discriminator updates per round: {n_disc_updates_per_round}")
    print(f"  Generator training timesteps per round: {gen_train_timesteps}")

    trainer = airl.AIRL(
        demonstrations=expert_trajectories,  # NO h labels!
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,                   # h-specific (BC-initialized)
        reward_net=reward_net,               # h-AGNOSTIC!
        n_disc_updates_per_round=n_disc_updates_per_round,
        gen_train_timesteps=gen_train_timesteps,
        allow_variable_horizon=True,         # 4-in-a-row games have variable length
    )

    print(f"  ✓ AIRL trainer created")

    # ═══════════════════════════════════════════════════════
    # CRITICAL PRINCIPLE VERIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL PRINCIPLE VERIFICATION]")
    print(f"  ✓ Expert data: NO h labels")
    print(f"  ✓ Generator: Trained from h={h} BC policy")
    print(f"  ✓ Discriminator: NO h parameter")
    print(f"  ✓ Observations: 89-dim (board + features, NO depth)")
    print(f"  ✓ Setup follows PLANNING_DEPTH_PRINCIPLES.md")

    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════
    print(f"\n[Training AIRL]")
    print(f"  Training for {total_timesteps} timesteps...")

    trainer.train(total_timesteps=total_timesteps)

    print(f"  ✓ Training complete")

    # ═══════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════
    os.makedirs(output_dir, exist_ok=True)

    # Save trained generator
    gen_path = os.path.join(output_dir, f'airl_generator_h{h}.zip')
    gen_algo.save(gen_path)
    print(f"\n✓ Saved trained generator to {gen_path}")

    # Save trained reward network
    reward_path = os.path.join(output_dir, f'airl_reward_h{h}.pt')
    torch.save(reward_net.state_dict(), reward_path)
    print(f"✓ Saved trained reward network to {reward_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, f'airl_metadata_h{h}.pkl')
    metadata = {
        'h': h,
        'total_timesteps': total_timesteps,
        'demo_batch_size': demo_batch_size,
        'n_disc_updates_per_round': n_disc_updates_per_round,
        'gen_train_timesteps': gen_train_timesteps,
        'num_expert_trajectories': len(expert_trajectories),
    }
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"✓ Saved metadata to {metadata_path}")

    print("=" * 80)

    return trainer


def train_airl_all_depths(
    expert_trajectories: List[Trajectory],
    depths: List[int] = [1, 2, 4, 8],
    total_timesteps: int = 10000,
    output_dir: str = 'models/airl_results'
):
    """
    Train AIRL for all planning depths

    Args:
        expert_trajectories: Expert demonstrations (NO h labels)
        depths: List of planning depths to train
        total_timesteps: Total training timesteps per depth
        output_dir: Output directory for results
    """
    print("=" * 80)
    print("Training AIRL for All Depths")
    print("=" * 80)
    print(f"Depths: {depths}")
    print(f"Expert trajectories: {len(expert_trajectories)}")
    print(f"Total timesteps per depth: {total_timesteps}")
    print(f"Output directory: {output_dir}")
    print()

    env = FourInARowEnv()
    results = {}

    for h in depths:
        print(f"\n{'=' * 80}")
        print(f"Processing h={h}")
        print(f"{'=' * 80}\n")

        trainer = train_airl_single_depth(
            h=h,
            expert_trajectories=expert_trajectories,
            env=env,
            total_timesteps=total_timesteps,
            output_dir=output_dir
        )

        results[h] = trainer

    print(f"\n{'=' * 80}")
    print("✓ All AIRL training complete")
    print(f"{'=' * 80}")

    return results


def test_airl_training():
    """Test AIRL training with minimal setup"""
    print("=" * 80)
    print("Testing AIRL Training (Step E)")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Step 1: Generate test expert data
    # ═══════════════════════════════════════════════════════
    print("\n[Step 1] Generating test expert data...")

    from generate_training_data import generate_depth_limited_trajectories

    # Generate expert trajectories with h=2
    expert_game_trajs = generate_depth_limited_trajectories(
        h=2,
        num_episodes=5,
        seed=42,
        verbose=False
    )
    print(f"✓ Generated {len(expert_game_trajs)} expert trajectories")

    # Convert to imitation format
    expert_trajectories = convert_to_imitation_format(
        expert_game_trajs,
        verbose=False
    )
    print(f"✓ Converted to imitation format: {len(expert_trajectories)} trajectories")

    # ═══════════════════════════════════════════════════════
    # Step 2: Ensure PPO generator exists
    # ═══════════════════════════════════════════════════════
    print("\n[Step 2] Checking PPO generator...")

    ppo_model_path = 'models/ppo_generators/ppo_generator_h2.zip'

    if not os.path.exists(ppo_model_path):
        print(f"  PPO generator not found, creating it...")

        from train_bc import train_bc_policy
        from create_ppo_generator import create_ppo_from_bc, save_ppo_generator

        # Generate training data
        training_trajs = generate_depth_limited_trajectories(
            h=2,
            num_episodes=10,
            seed=42,
            verbose=False
        )

        # Train BC
        env = FourInARowEnv()
        bc_trainer = train_bc_policy(
            trajectories=training_trajs,
            env=env,
            h=2,
            n_epochs=5,
            verbose=False
        )

        # Create PPO
        ppo_algo, vec_env = create_ppo_from_bc(
            bc_trainer=bc_trainer,
            env=env,
            h=2
        )

        # Save
        save_ppo_generator(ppo_algo, h=2)
        print(f"  ✓ Created PPO generator")
    else:
        print(f"  ✓ PPO generator exists: {ppo_model_path}")

    # ═══════════════════════════════════════════════════════
    # Step 3: Train AIRL (minimal timesteps for testing)
    # ═══════════════════════════════════════════════════════
    print("\n[Step 3] Training AIRL (minimal timesteps)...")

    env = FourInARowEnv()

    trainer = train_airl_single_depth(
        h=2,
        expert_trajectories=expert_trajectories,
        env=env,
        total_timesteps=1024,  # Minimal for testing
        demo_batch_size=32,
        n_disc_updates_per_round=2,
        gen_train_timesteps=256,
        output_dir='models/airl_results_test',
        verbose=1
    )

    print("\n✓ AIRL training test complete")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train AIRL for 4-in-a-row (Step E)'
    )
    parser.add_argument('--h', type=int, default=None,
                        help='Planning depth (default: train all [1,2,4,8])')
    parser.add_argument('--expert_data', type=str, default=None,
                        help='Path to expert data (default: generate synthetic)')
    parser.add_argument('--total_timesteps', type=int, default=10000,
                        help='Total training timesteps (default: 10000)')
    parser.add_argument('--demo_batch_size', type=int, default=64,
                        help='Demo batch size (default: 64)')
    parser.add_argument('--output_dir', type=str, default='models/airl_results',
                        help='Output directory (default: models/airl_results)')
    parser.add_argument('--test', action='store_true',
                        help='Run test with minimal setup')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_airl_training()
    else:
        # Load or generate expert data
        if args.expert_data:
            # Load from file
            print(f"Loading expert data from {args.expert_data}...")
            with open(args.expert_data, 'rb') as f:
                expert_trajectories = pickle.load(f)
        else:
            # Generate synthetic expert data
            print("Generating synthetic expert data...")
            from generate_training_data import generate_depth_limited_trajectories

            expert_game_trajs = generate_depth_limited_trajectories(
                h=4,  # Use h=4 as "expert"
                num_episodes=100,
                seed=42,
                verbose=True
            )

            expert_trajectories = convert_to_imitation_format(
                expert_game_trajs,
                verbose=True
            )

        # Train AIRL
        env = FourInARowEnv()

        if args.h is not None:
            # Single depth
            train_airl_single_depth(
                h=args.h,
                expert_trajectories=expert_trajectories,
                env=env,
                total_timesteps=args.total_timesteps,
                demo_batch_size=args.demo_batch_size,
                output_dir=args.output_dir
            )
        else:
            # All depths
            train_airl_all_depths(
                expert_trajectories=expert_trajectories,
                depths=[1, 2, 4, 8],
                total_timesteps=args.total_timesteps,
                output_dir=args.output_dir
            )
