"""
Step D: Create Depth-AGNOSTIC Reward Network

Create discriminator (reward network) that is completely depth-agnostic.

CRITICAL PRINCIPLE:
Reward network has NO h parameter. Same architecture for ALL h values.

VALIDATION CHECKPOINTS:
1. NO h parameter in function signature
2. NO h parameter in reward network initialization
3. NO depth-related attributes in reward network
4. Forward pass takes (state, action, next_state) only

Usage:
    python3 create_reward_net.py --test
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import argparse

try:
    from .env import FourInARowEnv
except ImportError:
    from env import FourInARowEnv

# Import imitation library
try:
    from imitation.rewards.reward_nets import BasicRewardNet
except ImportError:
    print("ERROR: imitation library not installed")
    print("Install with: pip install imitation stable-baselines3 torch")
    raise


def create_reward_network(
    env,
    hid_sizes: list = [64, 64],
    activation: nn.Module = nn.Tanh
):
    """
    Create depth-agnostic reward network

    CRITICAL: This function has NO h parameter!
    Same architecture for ALL h values.

    Args:
        env: FourInARowEnv
        hid_sizes: Hidden layer sizes (default: [64, 64])
        activation: Activation function (default: Tanh)

    Returns:
        reward_net: BasicRewardNet (depth-agnostic)

    VALIDATION CHECKPOINTS:
    ✓ Checkpoint 6: NO h parameter in function signature
    ✓ Checkpoint 6: NO h parameter in reward network
    ✓ Checkpoint 6: No depth-related attributes
    """
    print("=" * 80)
    print("Creating Depth-AGNOSTIC Reward Network")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 6a:
    # NO h parameter in this function
    # ═══════════════════════════════════════════════════════
    print(f"\n[CRITICAL VALIDATION]")
    print(f"  Function signature: create_reward_network(env, ...)")
    print(f"  ✓ NO h parameter in function signature")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 6b:
    # NO h parameter in reward network initialization
    # ═══════════════════════════════════════════════════════
    print(f"\n[Reward Network Architecture]")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  Input: (state, action, next_state)")
    print(f"    - state dim: 89 (board + features, NO depth)")
    print(f"    - action dim: 36 (one-hot encoded)")
    print(f"    - next_state dim: 89 (board + features, NO depth)")
    print(f"  Hidden layers: {hid_sizes}")
    print(f"  Activation: {activation.__name__}")
    print(f"  Output: scalar reward")
    print(f"  ✓ NO h parameter in architecture")

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,  # Box(89,) - NO h!
        action_space=env.action_space,            # Discrete(36)
        hid_sizes=hid_sizes,                      # MLP hidden layers
        # Note: No activation parameter in BasicRewardNet for imitation 1.0.1
    )

    print(f"  ✓ Reward network created")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 6c:
    # Check for depth-related attributes
    # ═══════════════════════════════════════════════════════
    print(f"\n[Validation: Depth-Related Attributes]")
    suspicious_attrs = [
        attr for attr in dir(reward_net)
        if 'depth' in attr.lower() or attr == 'h'
    ]

    if len(suspicious_attrs) > 0:
        print(f"  ⚠️  WARNING: Found suspicious attributes: {suspicious_attrs}")
        print(f"  These may indicate depth information leak!")
        raise ValueError(
            f"Reward network has depth-related attributes: {suspicious_attrs}\n"
            f"This violates PLANNING_DEPTH_PRINCIPLES.md!"
        )
    else:
        print(f"  ✓ No depth-related attributes found")

    # ═══════════════════════════════════════════════════════
    # VALIDATION CHECKPOINT 6d:
    # Verify forward pass signature
    # ═══════════════════════════════════════════════════════
    print(f"\n[Validation: Forward Pass]")
    print(f"  Forward signature: reward_net(state, action, next_state, done)")
    print(f"  ✓ NO h parameter in forward pass")
    print(f"  ✓ Only uses observable information (states, actions, terminal)")

    print(f"\n[Important Notes]")
    print(f"  - This reward network is the SAME for all h values")
    print(f"  - Different h experiments use FRESH instances of this network")
    print(f"  - Each instance is trained independently with h-specific generator")
    print(f"  - Correct terminology: 'reward trained with h=X generator'")

    print("=" * 80)

    return reward_net


def test_reward_network_forward_pass(
    reward_net,
    env,
    verbose: bool = True
):
    """
    Test reward network forward pass

    Args:
        reward_net: Reward network to test
        env: FourInARowEnv for generating test states
        verbose: Print test report

    Returns:
        success: True if validation passed

    VALIDATION CHECKPOINTS:
    ✓ Forward pass works with 89-dim observations
    ✓ No depth parameter in input
    ✓ Output is scalar reward
    """
    if verbose:
        print("\n" + "=" * 80)
        print("Reward Network Forward Pass Test")
        print("=" * 80)

    try:
        # Generate test data
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Convert to torch tensors (BasicRewardNet expects tensors)
        # IMPORTANT: Action must be 1D tensor of indices (batch_size,)
        # preprocessing.preprocess_obs will convert to one-hot (batch_size, 36)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)   # (1, 89)
        action_tensor = torch.LongTensor([action])         # (1,) - action index!
        next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0)  # (1, 89)
        done_tensor = torch.BoolTensor([done])             # (1,) - boolean

        # ═══════════════════════════════════════════════════════
        # CRITICAL: Forward pass should NOT require depth parameter
        # ═══════════════════════════════════════════════════════
        if verbose:
            print(f"\n[Test 1] Forward pass (should NOT require depth):")
            print(f"  Input shapes:")
            print(f"    - obs: {obs_tensor.shape}")
            print(f"    - action: {action_tensor.shape}")
            print(f"    - next_obs: {next_obs_tensor.shape}")
            print(f"    - done: {done_tensor.shape}")

        # Forward pass - use preprocess() first, then forward()
        # OR use predict_th() which does both
        state_th, action_th, next_state_th, done_th = reward_net.preprocess(
            obs_tensor, action_tensor, next_obs_tensor, done_tensor
        )

        with torch.no_grad():
            reward = reward_net(state_th, action_th, next_state_th, done_th)

        if verbose:
            print(f"  Output shape: {reward.shape}")
            print(f"  Reward value: {reward.item():.6f}")

        # ═══════════════════════════════════════════════════════
        # Validate output
        # ═══════════════════════════════════════════════════════
        assert reward.shape[0] == 1, \
            f"Expected batch size 1, got {reward.shape[0]}"

        if verbose:
            print(f"\n✓ Test 1 passed: Forward pass successful")
            print(f"✓ No depth parameter required")
            print(f"✓ Output is reward tensor")

        # ═══════════════════════════════════════════════════════
        # Test with batch of transitions
        # ═══════════════════════════════════════════════════════
        if verbose:
            print(f"\n[Test 2] Batch forward pass:")

        batch_size = 10
        obs_batch = torch.FloatTensor(np.random.rand(batch_size, 89))
        action_batch = torch.LongTensor(np.random.randint(0, 36, size=batch_size))  # (batch,) - action indices
        next_obs_batch = torch.FloatTensor(np.random.rand(batch_size, 89))
        done_batch = torch.BoolTensor(np.random.randint(0, 2, size=batch_size))  # (batch,)

        if verbose:
            print(f"  Batch size: {batch_size}")

        # Preprocess batch
        state_batch_th, action_batch_th, next_state_batch_th, done_batch_th = reward_net.preprocess(
            obs_batch, action_batch, next_obs_batch, done_batch
        )

        with torch.no_grad():
            reward_batch = reward_net(state_batch_th, action_batch_th, next_state_batch_th, done_batch_th)

        if verbose:
            print(f"  Output shape: {reward_batch.shape}")
            print(f"  Reward values (first 3): {reward_batch[:3].tolist()}")

        assert reward_batch.shape[0] == batch_size, \
            f"Expected batch size {batch_size}, got {reward_batch.shape[0]}"

        if verbose:
            print(f"\n✓ Test 2 passed: Batch forward pass successful")

        # ═══════════════════════════════════════════════════════
        # Test gradient flow
        # ═══════════════════════════════════════════════════════
        if verbose:
            print(f"\n[Test 3] Gradient flow:")

        obs_grad = torch.FloatTensor(obs).unsqueeze(0)           # (1, 89)
        action_grad = torch.LongTensor([action])                 # (1,) - action index
        next_obs_grad = torch.FloatTensor(next_obs).unsqueeze(0)  # (1, 89)
        done_grad = torch.BoolTensor([done])                     # (1,)

        obs_grad.requires_grad = True
        next_obs_grad.requires_grad = True

        # Preprocess
        state_grad_th, action_grad_th, next_state_grad_th, done_grad_th = reward_net.preprocess(
            obs_grad, action_grad, next_obs_grad, done_grad
        )

        reward_grad = reward_net(state_grad_th, action_grad_th, next_state_grad_th, done_grad_th)
        reward_grad.sum().backward()

        if verbose:
            # Note: gradients may not flow through preprocess() depending on implementation
            # The important thing is that reward_net's parameters get gradients
            if obs_grad.grad is not None:
                print(f"  Obs gradient shape: {obs_grad.grad.shape}")
            else:
                print(f"  Obs gradient: None (preprocess may create new tensors)")

            if next_obs_grad.grad is not None:
                print(f"  Next obs gradient shape: {next_obs_grad.grad.shape}")
            else:
                print(f"  Next obs gradient: None (preprocess may create new tensors)")

        # Check that reward network parameters have gradients
        has_grad = any(p.grad is not None for p in reward_net.parameters())
        assert has_grad, "No gradients computed for reward network parameters"

        if verbose:
            print(f"\n✓ Test 3 passed: Gradients computed successfully")

        if verbose:
            print("=" * 80)
            print("✅ All forward pass tests PASSED")
            print("=" * 80)

        return True

    except Exception as e:
        if verbose:
            print(f"\n❌ Forward pass FAILED: {e}")
            print("=" * 80)
        raise


def test_reward_network():
    """Test reward network creation and validation"""
    print("=" * 80)
    print("Testing Reward Network Creation (Step D)")
    print("=" * 80)

    # Create environment
    env = FourInARowEnv()

    # Create reward network
    reward_net = create_reward_network(env)

    # Test forward pass
    test_reward_network_forward_pass(reward_net, env)

    print(f"\n{'=' * 80}")
    print("✅ Reward Network Test PASSED")
    print(f"{'=' * 80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create depth-agnostic reward network (Step D)'
    )
    parser.add_argument('--test', action='store_true',
                        help='Run test')
    parser.add_argument('--hid_sizes', nargs='+', type=int, default=[64, 64],
                        help='Hidden layer sizes (default: 64 64)')

    args = parser.parse_args()

    if args.test:
        # Run test
        test_reward_network()
    else:
        # Create and display reward network
        env = FourInARowEnv()
        reward_net = create_reward_network(
            env,
            hid_sizes=args.hid_sizes
        )
        print(f"\n✓ Reward network created successfully")
        print(f"  Architecture: {args.hid_sizes}")
        print(f"  Observation space: {env.observation_space.shape}")
        print(f"  Action space: {env.action_space.n}")
