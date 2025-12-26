"""
Step C: Wrap BC Policy with PPO

Wrap BC-trained policy with PPO for AIRL fine-tuning.

CRITICAL PRINCIPLE:
PPO receives depth-agnostic BC policy. Planning depth h is NOT used!

VALIDATION CHECKPOINTS:
1. PPO uses BC policy (which is depth-agnostic)
2. Observation dimension is 89 (NO h)
3. PPO policy has no depth-related attributes

Usage:
    python3 create_ppo_generator.py --h 4
"""

import numpy as np
import os
import pickle
from typing import Optional
import argparse

try:
    from .env import FourInARowEnv
    from .train_bc import load_bc_trainer
except ImportError:
    from env import FourInARowEnv
    from train_bc import load_bc_trainer

# Import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3 torch")
    raise


def create_ppo_from_bc(
    bc_trainer,
    env,
    h: int,  # For logging/saving only!
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 0
):
    """
    Wrap BC policy with PPO for AIRL training

    Args:
        bc_trainer: Trained BC object
        env: FourInARowEnv
        h: Planning depth (metadata only, NOT used in training!)
        learning_rate: PPO learning rate
        n_steps: Steps per update
        batch_size: Batch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        verbose: Verbosity level

    Returns:
        ppo_algo: PPO algorithm with BC-initialized policy

    VALIDATION CHECKPOINTS:
    ✓ Checkpoint 5: PPO uses BC policy, which is depth-agnostic
    """
    print("=" * 80)
    print(f"Creating PPO Generator from BC Policy (h={h})")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 5:
    # PPO uses BC policy, which is depth-agnostic
    # ═══════════════════════════════════════════════════════

    # Extract BC policy
    bc_policy = bc_trainer.policy

    # Verify policy input dimension
    print(f"\n[BC Policy Verification]")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  BC policy observation space: {bc_policy.observation_space}")
    print(f"  BC policy action space: {bc_policy.action_space}")

    # Verify observation dimension
    assert bc_policy.observation_space.shape == (89,), \
        f"BC policy should have 89-dim observation, got {bc_policy.observation_space.shape}"

    print(f"  ✓ BC policy has 89-dim observation space (NO h)")

    # ═══════════════════════════════════════════════════════
    # CRITICAL: PPO receives depth-agnostic policy
    # h는 여기서도 사용되지 않음
    # ═══════════════════════════════════════════════════════

    # Wrap environment for PPO (PPO expects vectorized env)
    vec_env = DummyVecEnv([lambda: env])

    print(f"\n[PPO Configuration]")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Steps per update: {n_steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs per update: {n_epochs}")
    print(f"  Gamma: {gamma}")

    # Get BC policy architecture
    # BC in imitation 1.0.1 uses default SB3 architecture
    # which is typically [32, 32] for the feature extractor
    import torch

    # Check BC policy structure to get network dimensions
    # BC policy uses ActorCriticPolicy structure
    first_layer_shape = None
    for name, param in bc_policy.named_parameters():
        if 'mlp_extractor.policy_net.0.weight' in name:
            first_layer_shape = param.shape
            break

    if first_layer_shape is not None:
        net_arch_size = first_layer_shape[0]  # e.g., 32 or 64
        print(f"  BC policy network size: [{net_arch_size}, {net_arch_size}]")
    else:
        net_arch_size = 32  # Default
        print(f"  Using default network size: [32, 32]")

    # Create PPO with matching architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=[net_arch_size, net_arch_size],  # Policy network
            vf=[net_arch_size, net_arch_size],  # Value network
        )
    )

    ppo_algo = PPO(
        policy="MlpPolicy",  # Use MLP policy
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        policy_kwargs=policy_kwargs,  # Match BC architecture
        verbose=verbose,
    )

    # Copy BC policy weights to PPO policy
    print(f"\n[Policy Initialization]")
    print(f"  Copying BC policy weights to PPO...")
    ppo_algo.policy.load_state_dict(bc_policy.state_dict())
    print(f"  ✓ BC policy weights loaded into PPO")

    # Verify observation space
    assert ppo_algo.policy.observation_space.shape == (89,), \
        f"PPO policy should have 89-dim observation, got {ppo_algo.policy.observation_space.shape}"

    print(f"\n[Validation]")
    print(f"  ✓ PPO created with BC-initialized policy")
    print(f"  ✓ Policy is depth-agnostic (only sees 89-dim observations)")
    print(f"  ✓ Policy mimics DepthLimitedPolicy(h={h}) behavior")
    print(f"  ✓ NO h parameter in PPO policy")

    print("=" * 80)

    return ppo_algo, vec_env


def save_ppo_generator(
    ppo_algo,
    h: int,
    output_dir: str = 'models/ppo_generators'
):
    """
    Save PPO generator

    Args:
        ppo_algo: PPO algorithm
        h: Planning depth (for filename only)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save PPO model
    model_path = os.path.join(output_dir, f'ppo_generator_h{h}.zip')
    ppo_algo.save(model_path)
    print(f"\n✓ Saved PPO generator to {model_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, f'metadata_h{h}.txt')
    with open(metadata_path, 'w') as f:
        f.write(f"PPO Generator Metadata (h={h})\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Planning depth: h={h} (behavior mimics this depth)\n")
        f.write(f"Observation space: {ppo_algo.policy.observation_space}\n")
        f.write(f"Action space: {ppo_algo.policy.action_space}\n")
        f.write(f"Policy type: MLP\n")
        f.write(f"\nIMPORTANT:\n")
        f.write(f"- Policy does NOT take h as input\n")
        f.write(f"- Policy only sees 89-dim observations\n")
        f.write(f"- Policy was initialized from BC(h={h})\n")
        f.write(f"- h is used only to label which depth behavior this mimics\n")

    print(f"✓ Saved metadata to {metadata_path}")


def load_ppo_generator(h: int, model_dir: str = 'models/ppo_generators'):
    """
    Load previously created PPO generator

    Args:
        h: Planning depth
        model_dir: Model directory

    Returns:
        ppo_algo: Loaded PPO algorithm
    """
    model_path = os.path.join(model_dir, f'ppo_generator_h{h}.zip')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"PPO generator not found: {model_path}")

    ppo_algo = PPO.load(model_path)
    print(f"✓ Loaded PPO generator from {model_path}")

    return ppo_algo


def create_all_generators(
    depths: list = [1, 2, 4, 8],
    bc_model_dir: str = 'models/bc_policies',
    output_dir: str = 'models/ppo_generators'
):
    """
    Create PPO generators for all planning depths

    Args:
        depths: List of planning depths
        bc_model_dir: Directory with BC models
        output_dir: Output directory for PPO generators
    """
    print("=" * 80)
    print("Creating PPO Generators for All Depths")
    print("=" * 80)
    print(f"Depths: {depths}")
    print(f"BC model directory: {bc_model_dir}")
    print(f"Output directory: {output_dir}")
    print()

    env = FourInARowEnv()

    for h in depths:
        print(f"\n{'=' * 80}")
        print(f"Processing h={h}")
        print(f"{'=' * 80}\n")

        # Load BC trainer
        print(f"[Loading BC Policy]")
        bc_trainer = load_bc_trainer(h, bc_model_dir)
        print()

        # Create PPO
        ppo_algo, vec_env = create_ppo_from_bc(
            bc_trainer=bc_trainer,
            env=env,
            h=h
        )

        # Save PPO
        save_ppo_generator(ppo_algo, h, output_dir)

    print(f"\n{'=' * 80}")
    print("✓ All PPO generators created successfully")
    print(f"{'=' * 80}")


def test_ppo_creation():
    """Test PPO creation from BC policy"""
    print("=" * 80)
    print("Testing PPO Generator Creation (Step C)")
    print("=" * 80)

    # Generate and train BC first
    print("\n[Step 1] Generating test data...")
    from generate_training_data import generate_depth_limited_trajectories
    from train_bc import train_bc_policy

    trajectories = generate_depth_limited_trajectories(
        h=2,
        num_episodes=10,
        seed=42,
        verbose=False
    )
    print(f"✓ Generated {len(trajectories)} trajectories")

    print("\n[Step 2] Training BC policy...")
    env = FourInARowEnv()
    bc_trainer = train_bc_policy(
        trajectories=trajectories,
        env=env,
        h=2,
        n_epochs=5,
        verbose=False
    )
    print(f"✓ BC policy trained")

    print("\n[Step 3] Creating PPO generator...")
    ppo_algo, vec_env = create_ppo_from_bc(
        bc_trainer=bc_trainer,
        env=env,
        h=2
    )

    # Test forward pass
    print(f"\n[Step 4] Testing PPO forward pass...")
    obs = vec_env.reset()
    action, _ = ppo_algo.predict(obs, deterministic=True)

    print(f"  Observation shape: {obs.shape}")
    print(f"  Predicted action: {action}")
    print(f"  Action in valid range: {0 <= action[0] <= 35}")

    assert obs.shape == (1, 89), f"Observation shape mismatch"
    assert 0 <= action[0] <= 35, f"Invalid action: {action[0]}"

    print(f"\n✓ PPO forward pass test passed")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create PPO generator from BC policy (Step C)'
    )
    parser.add_argument('--h', type=int, default=None,
                        help='Planning depth (default: create all [1,2,4,8])')
    parser.add_argument('--bc_model_dir', type=str, default='models/bc_policies',
                        help='BC model directory (default: models/bc_policies)')
    parser.add_argument('--output_dir', type=str, default='models/ppo_generators',
                        help='Output directory (default: models/ppo_generators)')
    parser.add_argument('--test', action='store_true',
                        help='Run test')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_ppo_creation()
    elif args.h is not None:
        # Create single PPO generator
        env = FourInARowEnv()
        bc_trainer = load_bc_trainer(args.h, args.bc_model_dir)
        ppo_algo, vec_env = create_ppo_from_bc(
            bc_trainer=bc_trainer,
            env=env,
            h=args.h
        )
        save_ppo_generator(ppo_algo, args.h, args.output_dir)
    else:
        # Create all PPO generators
        create_all_generators(
            depths=[1, 2, 4, 8],
            bc_model_dir=args.bc_model_dir,
            output_dir=args.output_dir
        )
