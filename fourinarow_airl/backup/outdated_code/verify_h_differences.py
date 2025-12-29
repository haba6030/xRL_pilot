#!/usr/bin/env python3
"""
Verify that different planning depths (h) produce different behavioral patterns

This is a CRITICAL validation step before proceeding with Planning-Aware AIRL.
If h doesn't create meaningful differences, the whole approach is invalid.

Usage:
    python3 verify_h_differences.py --num_episodes 50

Output:
    - Action distribution comparison
    - State visitation frequency
    - Trajectory similarity metrics
    - Statistical tests (KL divergence, Chi-square)
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict
from scipy.stats import chi2_contingency, entropy
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import argparse

from env import FourInARowEnv
from depth_limited_policy import DepthLimitedPolicy
from bfs_wrapper import load_all_participant_parameters


def collect_trajectories(env, h_value, num_episodes=50, seed=42, participant_id=1):
    """Collect trajectories using BFS with specific depth h"""
    rng = np.random.default_rng(seed + h_value)

    trajectories = []
    action_counts = defaultdict(int)  # Count actions across all states
    state_action_pairs = []  # (state_hash, action) pairs

    # Load expert parameters
    import os
    params_file = os.path.join('..', 'opendata', 'model_fits_main_model.csv')
    if not os.path.exists(params_file):
        params_file = os.path.join('opendata', 'model_fits_main_model.csv')

    params_dict = load_all_participant_parameters(params_file)
    expert_params = params_dict[participant_id]

    # Create depth-limited policy
    policy = DepthLimitedPolicy(
        h=h_value,
        params=expert_params,
        beta=1.0,
        lapse_rate=expert_params.lapse_rate
    )

    for ep in range(num_episodes):
        obs, _ = env.reset()
        trajectory = {
            'observations': [],
            'actions': [],
            'state_hashes': []
        }

        done = False
        while not done:
            # Get action from depth-limited policy
            action, _ = policy.select_action(env, rng)

            # Record
            state_hash = hash(obs.tobytes())
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action)
            trajectory['state_hashes'].append(state_hash)

            action_counts[action] += 1
            state_action_pairs.append((state_hash, action))

            # Step
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        trajectories.append(trajectory)

    return {
        'trajectories': trajectories,
        'action_counts': action_counts,
        'state_action_pairs': state_action_pairs,
        'h': h_value,
        'num_episodes': num_episodes
    }


def compute_action_distribution(data):
    """Compute normalized action distribution"""
    action_counts = data['action_counts']
    total = sum(action_counts.values())

    # Create distribution over all 36 actions
    dist = np.zeros(36)
    for action, count in action_counts.items():
        dist[action] = count / total

    return dist


def compute_state_action_distribution(data):
    """Compute state-action co-occurrence distribution"""
    state_action_pairs = data['state_action_pairs']

    # Create distribution
    counts = defaultdict(int)
    for state_hash, action in state_action_pairs:
        counts[(state_hash, action)] += 1

    total = sum(counts.values())
    dist = {k: v/total for k, v in counts.items()}

    return dist


def compare_action_distributions(dist_h1, dist_h2):
    """Compare two action distributions"""
    # KL divergence (add small epsilon to avoid log(0))
    epsilon = 1e-10
    dist_h1_smooth = dist_h1 + epsilon
    dist_h2_smooth = dist_h2 + epsilon

    # Normalize after smoothing
    dist_h1_smooth /= dist_h1_smooth.sum()
    dist_h2_smooth /= dist_h2_smooth.sum()

    kl_div = entropy(dist_h1_smooth, dist_h2_smooth)
    js_div = jensenshannon(dist_h1_smooth, dist_h2_smooth)

    # Chi-square test
    # Create contingency table
    counts_h1 = dist_h1 * dist_h1.sum()  # Convert back to counts (approximate)
    counts_h2 = dist_h2 * dist_h2.sum()

    # Remove zero columns
    mask = (counts_h1 > 0) | (counts_h2 > 0)
    if mask.sum() > 1:
        contingency = np.array([counts_h1[mask], counts_h2[mask]])
        chi2, p_value, dof, expected = chi2_contingency(contingency)
    else:
        chi2, p_value = None, None

    return {
        'kl_divergence': kl_div,
        'js_divergence': js_div,
        'chi2': chi2,
        'p_value': p_value
    }


def compute_trajectory_similarity(traj1, traj2):
    """Compute similarity between two trajectories"""
    actions1 = traj1['actions']
    actions2 = traj2['actions']

    # Longest common subsequence (LCS) length
    # Simple approach: just count matching actions at same positions
    min_len = min(len(actions1), len(actions2))
    matching = sum(1 for i in range(min_len) if actions1[i] == actions2[i])

    similarity = matching / max(len(actions1), len(actions2))
    return similarity


def visualize_results(results, output_dir='figures'):
    """Visualize comparison results"""
    Path(output_dir).mkdir(exist_ok=True)

    h_values = sorted(results.keys())

    # 1. Action distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Action Distributions by Planning Depth h', fontsize=16)

    for idx, h in enumerate(h_values):
        ax = axes[idx // 2, idx % 2]
        dist = results[h]['action_distribution']

        ax.bar(range(36), dist)
        ax.set_title(f'h = {h}')
        ax.set_xlabel('Action (0-35)')
        ax.set_ylabel('Probability')
        ax.set_ylim([0, max([results[h]['action_distribution'].max() for h in h_values]) * 1.1])

    plt.tight_layout()
    plt.savefig(f'{output_dir}/action_distributions.png', dpi=150)
    print(f"✅ Saved {output_dir}/action_distributions.png")

    # 2. Pairwise comparisons
    print("\n" + "=" * 60)
    print("Pairwise Action Distribution Comparisons")
    print("=" * 60)

    for i, h1 in enumerate(h_values):
        for h2 in h_values[i+1:]:
            dist1 = results[h1]['action_distribution']
            dist2 = results[h2]['action_distribution']

            comparison = compare_action_distributions(dist1, dist2)

            print(f"\nh={h1} vs h={h2}:")
            print(f"  KL divergence:  {comparison['kl_divergence']:.4f}")
            print(f"  JS divergence:  {comparison['js_divergence']:.4f}")
            if comparison['p_value'] is not None:
                print(f"  Chi-square:     χ²={comparison['chi2']:.2f}, p={comparison['p_value']:.4f}")
                if comparison['p_value'] < 0.05:
                    print(f"  → Significantly different! (p < 0.05)")
                else:
                    print(f"  → Not significantly different (p >= 0.05)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes per h value')
    parser.add_argument('--h_values', type=int, nargs='+', default=[1, 2, 4, 8],
                        help='Planning depth values to compare')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Verifying Planning Depth h Differences")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  h values:      {args.h_values}")
    print(f"  Episodes/h:    {args.num_episodes}")
    print(f"  Random seed:   {args.seed}")

    # Create environment
    env = FourInARowEnv()

    # Collect data for each h
    print("\n📊 Collecting trajectories...")
    results = {}

    for h in args.h_values:
        print(f"\n  h={h}:")
        data = collect_trajectories(env, h, num_episodes=args.num_episodes, seed=args.seed)

        # Compute distributions
        action_dist = compute_action_distribution(data)

        results[h] = {
            'data': data,
            'action_distribution': action_dist
        }

        total_actions = sum(data['action_counts'].values())
        unique_actions = len(data['action_counts'])
        print(f"    Total actions: {total_actions}")
        print(f"    Unique actions used: {unique_actions}/36")
        print(f"    Entropy: {entropy(action_dist + 1e-10):.3f}")

    # Visualize and compare
    visualize_results(results)

    # Save results
    output_file = 'data/h_verification_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\n✅ Saved results to {output_file}")

    print("\n" + "=" * 60)
    print("✅ Verification complete!")
    print("=" * 60)

    print("\n📝 Interpretation:")
    print("""
    If KL/JS divergence is HIGH (> 0.1) and p-value is LOW (< 0.05):
      → h creates meaningful behavioral differences
      → Planning-Aware AIRL is valid
      → Proceed with Phase 2

    If KL/JS divergence is LOW (< 0.05) and p-value is HIGH (> 0.05):
      → h does NOT create meaningful differences
      → Planning-Aware AIRL may not work
      → Need to reconsider approach
    """)


if __name__ == '__main__':
    main()
