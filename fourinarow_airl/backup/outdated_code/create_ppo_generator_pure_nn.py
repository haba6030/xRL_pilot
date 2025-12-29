"""
Option A: Pure Neural Network Generator (Pedestrian 방식)

NO BC pretraining - start from random initialization.
This is the "pure" approach where the generator learns entirely from AIRL,
without using any domain knowledge from van Opheusden BFS.

Comparison:
- Option A (this file): PPO from scratch
- Option B (create_ppo_generator.py): BC(BFS) → PPO

Usage:
    python3 create_ppo_generator_pure_nn.py --h 4
"""

import numpy as np
import os
from typing import Optional
import argparse

try:
    from .env import FourInARowEnv
except ImportError:
    from env import FourInARowEnv

# Import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3 torch")
    raise


def create_pure_ppo_generator(
    env,
    h: int,  # For logging/saving only - NOT used in network!
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    verbose: int = 1
):
    """
    Create PPO generator from RANDOM initialization (Option A)

    NO BC pretraining - this is pure neural network approach.

    Args:
        env: FourInARowEnv
        h: Planning depth (metadata only, NOT used in architecture!)
        learning_rate: PPO learning rate
        n_steps: Steps per rollout
        batch_size: Batch size
        n_epochs: PPO epochs per update
        gamma: Discount factor
        verbose: Verbosity level

    Returns:
        ppo_algo: PPO algorithm with random initialization
        venv: Vectorized environment

    CRITICAL PRINCIPLE:
    - h is only used for NAMING/LOGGING (e.g., save as ppo_pure_h4.zip)
    - The network architecture has NO depth parameter
    - Observation is 89-dim (board + features, NO depth encoding)
    - This is "Option A: Pure NN" - no van Opheusden BFS knowledge
    """
    print("=" * 80)
    print(f"Creating Pure PPO Generator (Option A) - h={h}")
    print("=" * 80)
    print(f"\n⚠️  IMPORTANT: This is Option A (Pure NN)")
    print(f"  - NO BC pretraining from BFS")
    print(f"  - Random initialization")
    print(f"  - h={h} is only for logging/naming")
    print(f"  - Network has NO depth parameter")

    # ═══════════════════════════════════════════════════════
    # Verify environment
    # ═══════════════════════════════════════════════════════
    print(f"\n[Environment]")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  ✓ Observation is 89-dim (NO depth)")

    # ═══════════════════════════════════════════════════════
    # Create vectorized environment
    # ═══════════════════════════════════════════════════════
    venv = DummyVecEnv([lambda: env])

    # ═══════════════════════════════════════════════════════
    # Create PPO with RANDOM initialization
    # ═══════════════════════════════════════════════════════
    print(f"\n[Creating PPO]")
    print(f"  Policy: MlpPolicy (randomly initialized)")
    print(f"  Learning rate: {learning_rate}")
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  gamma: {gamma}")

    ppo_algo = PPO(
        "MlpPolicy",              # ✅ Pure MLP, no custom initialization
        venv,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        verbose=verbose,
        tensorboard_log=f"./tensorboard_logs/ppo_pure_h{h}/"
    )

    print(f"  ✓ PPO created with random initialization")

    # ═══════════════════════════════════════════════════════
    # VALIDATION: Verify depth-agnostic architecture
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL VALIDATION]")
    print(f"  Policy observation space: {ppo_algo.policy.observation_space}")
    print(f"  Policy action space: {ppo_algo.policy.action_space}")

    # Check observation dimension
    assert ppo_algo.policy.observation_space.shape == (89,), \
        f"Policy should have 89-dim observation, got {ppo_algo.policy.observation_space.shape}"
    print(f"  ✓ Policy has 89-dim observation (NO depth parameter)")

    # Check for depth-related attributes
    suspicious_attrs = [
        attr for attr in dir(ppo_algo.policy)
        if 'depth' in attr.lower() or attr == 'h'
    ]
    if len(suspicious_attrs) > 0:
        print(f"  ⚠️  WARNING: Found suspicious attributes: {suspicious_attrs}")
    else:
        print(f"  ✓ No depth-related attributes found")

    print(f"\n[Option A Summary]")
    print(f"  ✅ Pure neural network (NO domain knowledge)")
    print(f"  ✅ Random initialization (NO BC pretraining)")
    print(f"  ✅ Depth-agnostic observations (89-dim)")
    print(f"  ✅ h={h} is only metadata for naming")
    print(f"  ⚠️  Note: This may be HARDER to train than Option B")
    print(f"  ⚠️  Option B uses van Opheusden BFS as warm start")

    print("=" * 80)

    return ppo_algo, venv


def save_pure_ppo_generator(
    ppo_algo,
    h: int,
    output_dir: str = 'models/ppo_generators_pure'
):
    """
    Save pure PPO generator

    Args:
        ppo_algo: PPO algorithm
        h: Planning depth (for naming only)
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, f'ppo_pure_h{h}.zip')
    ppo_algo.save(model_path)

    print(f"\n✓ Saved pure PPO generator to {model_path}")
    print(f"  Note: This is Option A (Pure NN, NO BC)")

    return model_path


def load_pure_ppo_generator(
    h: int,
    env,
    input_dir: str = 'models/ppo_generators_pure'
):
    """
    Load pure PPO generator

    Args:
        h: Planning depth (for naming only)
        env: FourInARowEnv
        input_dir: Input directory

    Returns:
        ppo_algo: Loaded PPO algorithm
    """
    model_path = os.path.join(input_dir, f'ppo_pure_h{h}.zip')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pure PPO generator not found: {model_path}\n"
            f"Create it with create_ppo_generator_pure_nn.py"
        )

    # Wrap environment
    venv = DummyVecEnv([lambda: env])

    # Load model
    ppo_algo = PPO.load(model_path, env=venv)

    print(f"✓ Loaded pure PPO generator from {model_path}")
    print(f"  This is Option A (Pure NN, NO BC)")

    return ppo_algo, venv


def test_pure_ppo_generator():
    """Test pure PPO generator creation"""
    print("=" * 80)
    print("Testing Pure PPO Generator Creation (Option A)")
    print("=" * 80)

    # Create environment
    env = FourInARowEnv()

    # Create pure PPO generator
    h = 4  # arbitrary depth for naming
    ppo_algo, venv = create_pure_ppo_generator(
        env=env,
        h=h,
        verbose=1
    )

    # Test forward pass
    print(f"\n[Testing Policy Forward Pass]")
    obs = venv.reset()
    print(f"  Observation shape: {obs.shape}")

    action, _ = ppo_algo.predict(obs, deterministic=False)
    print(f"  Sampled action: {action}")
    print(f"  ✓ Forward pass successful")

    # Save
    model_path = save_pure_ppo_generator(ppo_algo, h=h)

    # Load
    print(f"\n[Testing Load]")
    ppo_loaded, _ = load_pure_ppo_generator(h=h, env=env)
    print(f"  ✓ Load successful")

    print("\n" + "=" * 80)
    print("Pure PPO Generator Test: ✅ PASSED")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create pure PPO generator (Option A)'
    )
    parser.add_argument('--h', type=int, default=4,
                        help='Planning depth (for naming only, default: 4)')
    parser.add_argument('--test', action='store_true',
                        help='Run test')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--output_dir', type=str, default='models/ppo_generators_pure',
                        help='Output directory')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_pure_ppo_generator()
    else:
        # Create pure PPO generator
        env = FourInARowEnv()

        ppo_algo, venv = create_pure_ppo_generator(
            env=env,
            h=args.h,
            learning_rate=args.learning_rate,
            verbose=1
        )

        # Save
        save_pure_ppo_generator(
            ppo_algo,
            h=args.h,
            output_dir=args.output_dir
        )

        print(f"\n✓ Pure PPO generator created and saved")
        print(f"  This is Option A: Pure NN (NO BC pretraining)")
        print(f"  Use this in AIRL training for 'Pedestrian-style' approach")
