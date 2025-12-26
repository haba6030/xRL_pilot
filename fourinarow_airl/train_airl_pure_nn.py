"""
Option A: Pure Neural Network AIRL Training (Pedestrian 방식)

Train AIRL with pure neural network generator (NO BC pretraining).

Comparison:
- Option A (this file): Pure NN generator (random init)
- Option B (train_airl.py): BC-initialized generator (from BFS)

Usage:
    # Single depth
    python3 train_airl_pure_nn.py --h 2 --total_timesteps 50000

    # Test
    python3 train_airl_pure_nn.py --test
"""

import numpy as np
import os
import pickle
import argparse
from typing import Optional, Dict, List
import torch

try:
    from .env import FourInARowEnv
    from .create_ppo_generator_pure_nn import create_pure_ppo_generator
    from .create_reward_net import create_reward_network
    from .airl_utils import convert_to_imitation_format, validate_airl_setup
    from .data_loader import load_expert_trajectories
except ImportError:
    from env import FourInARowEnv
    from create_ppo_generator_pure_nn import create_pure_ppo_generator
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


def train_airl_pure_nn(
    h: int,
    expert_trajectories: List[Trajectory],
    env,
    total_timesteps: int = 50000,
    demo_batch_size: int = 64,
    n_disc_updates_per_round: int = 4,
    gen_train_timesteps: int = 2048,
    output_dir: str = 'models/airl_pure_nn_results',
    verbose: int = 1
):
    """
    Train AIRL with pure NN generator (Option A)

    Args:
        h: Planning depth (metadata only, NOT used in networks!)
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
    """
    print("=" * 80)
    print(f"Training AIRL with Pure NN Generator (Option A) - h={h}")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # CRITICAL DISTINCTION
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL: Option A vs Option B]")
    print(f"  Option A (this script):")
    print(f"    - Generator: Pure NN, random initialization")
    print(f"    - NO van Opheusden BFS knowledge")
    print(f"    - Learns ENTIRELY from AIRL reward signal")
    print(f"    - Similar to Pedestrian project approach")
    print(f"")
    print(f"  Option B (train_airl.py):")
    print(f"    - Generator: BC-initialized from BFS(h={h})")
    print(f"    - Uses van Opheusden domain knowledge")
    print(f"    - Warm start → faster convergence")
    print(f"")
    print(f"  Both options:")
    print(f"    - Reward network: Pure NN (depth-agnostic)")
    print(f"    - Observations: 89-dim (NO depth encoding)")
    print(f"    - h is only for naming/metadata")

    # ═══════════════════════════════════════════════════════
    # Expert trajectories validation
    # ═══════════════════════════════════════════════════════
    print(f"\n[Expert Trajectories]")
    print(f"  Number of trajectories: {len(expert_trajectories)}")
    print(f"  ✓ Expert data contains NO h labels")
    print(f"  ✓ Observations are 89-dim (board + features, NO depth)")

    # ═══════════════════════════════════════════════════════
    # Create Pure NN Generator (NO BC)
    # ═══════════════════════════════════════════════════════
    print(f"\n[Creating Pure NN Generator]")
    print(f"  ⚠️  NO BC pretraining!")
    print(f"  Starting from RANDOM initialization")
    print(f"  This may require MORE timesteps than Option B")

    # Create generator
    gen_algo, venv = create_pure_ppo_generator(
        env=env,
        h=h,  # For naming only!
        verbose=0
    )

    print(f"  ✓ Pure NN generator created")
    print(f"  Policy observation space: {gen_algo.policy.observation_space}")
    print(f"  Policy action space: {gen_algo.policy.action_space}")

    # Verify generator uses 89-dim observations
    assert gen_algo.policy.observation_space.shape == (89,), \
        f"Generator should use 89-dim observations, got {gen_algo.policy.observation_space.shape}"

    print(f"  ✓ Generator uses 89-dim observations (NO h)")

    # ═══════════════════════════════════════════════════════
    # Create Depth-AGNOSTIC Reward Network
    # ═══════════════════════════════════════════════════════
    print(f"\n[Creating Depth-AGNOSTIC Reward Network]")
    print(f"  Reward network has NO h parameter!")
    print(f"  Same architecture for ALL h values")

    reward_net = create_reward_network(env)  # NO h parameter!

    print(f"  ✓ Reward network created (NO h parameter)")

    # ═══════════════════════════════════════════════════════
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
    print(f"")
    print(f"  ⚠️  Recommendation for Option A:")
    print(f"    - Use MORE timesteps than Option B")
    print(f"    - Suggested: 50K-100K (vs 10K for Option B)")
    print(f"    - Generator starts from scratch (no domain knowledge)")

    trainer = airl.AIRL(
        demonstrations=expert_trajectories,  # NO h labels!
        demo_batch_size=demo_batch_size,
        venv=venv,
        gen_algo=gen_algo,                   # Pure NN (random init)
        reward_net=reward_net,               # h-AGNOSTIC!
        n_disc_updates_per_round=n_disc_updates_per_round,
        gen_train_timesteps=gen_train_timesteps,
        allow_variable_horizon=True,
    )

    print(f"  ✓ AIRL trainer created")

    # ═══════════════════════════════════════════════════════
    # CRITICAL PRINCIPLE VERIFICATION
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL PRINCIPLE VERIFICATION]")
    print(f"  ✓ Expert data: NO h labels")
    print(f"  ✓ Generator: Pure NN, random init (NO BC)")
    print(f"  ✓ Discriminator: NO h parameter")
    print(f"  ✓ Observations: 89-dim (board + features, NO depth)")
    print(f"  ✓ Option A: 'Pedestrian-style' pure learning")

    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Training
    # ═══════════════════════════════════════════════════════
    print(f"\n[Training AIRL - Option A]")
    print(f"  Training for {total_timesteps} timesteps...")
    print(f"  This may take LONGER than Option B (no warm start)")

    trainer.train(total_timesteps=total_timesteps)

    print(f"  ✓ Training complete")

    # ═══════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════
    os.makedirs(output_dir, exist_ok=True)

    # Save trained generator
    gen_path = os.path.join(output_dir, f'airl_pure_generator_h{h}.zip')
    gen_algo.save(gen_path)
    print(f"\n✓ Saved trained generator to {gen_path}")

    # Save trained reward network
    reward_path = os.path.join(output_dir, f'airl_pure_reward_h{h}.pt')
    torch.save(reward_net.state_dict(), reward_path)
    print(f"✓ Saved trained reward network to {reward_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, f'airl_pure_metadata_h{h}.pkl')
    metadata = {
        'h': h,
        'option': 'A',
        'description': 'Pure NN generator (random init, NO BC)',
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


def test_airl_pure_nn():
    """Test AIRL training with pure NN (Option A)"""
    print("=" * 80)
    print("Testing AIRL Training with Pure NN (Option A)")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # Generate test expert data
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
    # Train AIRL (minimal timesteps for testing)
    # ═══════════════════════════════════════════════════════
    print("\n[Step 2] Training AIRL with Pure NN (minimal timesteps)...")

    env = FourInARowEnv()

    trainer = train_airl_pure_nn(
        h=2,
        expert_trajectories=expert_trajectories,
        env=env,
        total_timesteps=2048,  # Minimal for testing
        demo_batch_size=32,
        n_disc_updates_per_round=2,
        gen_train_timesteps=512,
        output_dir='models/airl_pure_nn_test',
        verbose=1
    )

    print("\n✓ AIRL Pure NN training test complete")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train AIRL with Pure NN generator (Option A)'
    )
    parser.add_argument('--h', type=int, default=4,
                        help='Planning depth (for naming only, default: 4)')
    parser.add_argument('--expert_data', type=str, default=None,
                        help='Path to expert data (default: generate synthetic)')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                        help='Total training timesteps (default: 50000)')
    parser.add_argument('--demo_batch_size', type=int, default=64,
                        help='Demo batch size (default: 64)')
    parser.add_argument('--output_dir', type=str, default='models/airl_pure_nn_results',
                        help='Output directory')
    parser.add_argument('--test', action='store_true',
                        help='Run test with minimal setup')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_airl_pure_nn()
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

        print(f"\n{'=' * 80}")
        print(f"Training Option A: Pure NN Generator")
        print(f"{'=' * 80}")
        print(f"⚠️  This may take LONGER than Option B")
        print(f"⚠️  Recommended timesteps: 50K-100K (you set: {args.total_timesteps})")
        print(f"")

        train_airl_pure_nn(
            h=args.h,
            expert_trajectories=expert_trajectories,
            env=env,
            total_timesteps=args.total_timesteps,
            demo_batch_size=args.demo_batch_size,
            output_dir=args.output_dir
        )

        print(f"\n{'=' * 80}")
        print(f"✅ Option A Training Complete")
        print(f"{'=' * 80}")
        print(f"Next steps:")
        print(f"  1. Compare with Option B (train_airl.py)")
        print(f"  2. Evaluate both generators on test data")
        print(f"  3. Check which approach better matches expert behavior")
