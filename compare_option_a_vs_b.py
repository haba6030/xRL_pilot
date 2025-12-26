"""
Compare Option A vs Option B

Evaluate and compare two AIRL approaches:
- Option A: Pure NN generator (random init)
- Option B: BC-initialized generator (from BFS)

Usage:
    python3 compare_option_a_vs_b.py --h 4 --num_episodes 50
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
import argparse
import os

from fourinarow_airl import FourInARowEnv
from fourinarow_airl.data_loader import load_expert_trajectories, GameTrajectory
from fourinarow_airl.airl_utils import convert_to_imitation_format

# Import stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("ERROR: stable-baselines3 not installed")
    raise


def load_trained_generator(option: str, h: int, env):
    """
    Load trained AIRL generator

    Args:
        option: 'A' or 'B'
        h: Planning depth
        env: FourInARowEnv

    Returns:
        gen_algo: Trained PPO generator
    """
    if option == 'A':
        model_path = f'models/airl_pure_nn_results/airl_pure_generator_h{h}.zip'
    elif option == 'B':
        model_path = f'models/airl_results/airl_generator_h{h}.zip'
    else:
        raise ValueError(f"Invalid option: {option}. Must be 'A' or 'B'")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained generator not found: {model_path}\n"
            f"Train it first with:\n"
            f"  Option A: python3 fourinarow_airl/train_airl_pure_nn.py --h {h}\n"
            f"  Option B: python3 fourinarow_airl/train_airl.py --h {h}"
        )

    # Load model
    venv = DummyVecEnv([lambda: env])
    gen_algo = PPO.load(model_path, env=venv)

    print(f"✓ Loaded Option {option} generator from {model_path}")

    return gen_algo


def generate_trajectories(
    gen_algo,
    env,
    num_episodes: int,
    seed: int = 42
) -> List[GameTrajectory]:
    """
    Generate trajectories using trained generator

    Args:
        gen_algo: Trained PPO generator
        env: FourInARowEnv
        num_episodes: Number of episodes to generate
        seed: Random seed

    Returns:
        trajectories: List of GameTrajectory objects
    """
    trajectories = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)

        observations = [obs.copy()]
        actions = []
        rewards = []

        for step in range(36):  # Max game length
            action, _ = gen_algo.predict(obs, deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)

            actions.append(action)
            observations.append(obs.copy())
            rewards.append(reward if terminated else 0.0)

            if terminated or truncated:
                break

        # Create GameTrajectory
        traj = GameTrajectory(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            player_id=0,
            game_id=episode,
            participant_id=-1
        )
        trajectories.append(traj)

    return trajectories


def compute_action_distribution(trajectories: List[GameTrajectory]) -> np.ndarray:
    """Compute action distribution over 36 positions"""
    action_counts = np.zeros(36)
    for traj in trajectories:
        for action in traj.actions:
            action_counts[action] += 1

    if action_counts.sum() > 0:
        return action_counts / action_counts.sum()
    else:
        return action_counts


def compute_kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL(p || q)"""
    p = p + 1e-10
    q = q + 1e-10
    return np.sum(p * np.log(p / q))


def compute_win_rate(trajectories: List[GameTrajectory]) -> float:
    """Calculate win rate"""
    wins = sum(1 for t in trajectories if t.rewards.sum() > 0)
    return wins / len(trajectories) if len(trajectories) > 0 else 0.0


def compare_options(
    h: int,
    expert_trajectories: List[GameTrajectory],
    num_episodes: int = 50,
    seed: int = 42
):
    """
    Compare Option A vs Option B

    Args:
        h: Planning depth
        expert_trajectories: Expert demonstrations
        num_episodes: Number of episodes to generate per option
        seed: Random seed
    """
    print("=" * 80)
    print(f"Comparing Option A vs Option B (h={h})")
    print("=" * 80)

    env = FourInARowEnv()

    # ═══════════════════════════════════════════════════════
    # Load trained generators
    # ═══════════════════════════════════════════════════════
    print(f"\n[Loading Trained Generators]")

    try:
        gen_a = load_trained_generator('A', h, env)
    except FileNotFoundError as e:
        print(f"⚠️  Option A not found: {e}")
        gen_a = None

    try:
        gen_b = load_trained_generator('B', h, env)
    except FileNotFoundError as e:
        print(f"⚠️  Option B not found: {e}")
        gen_b = None

    if gen_a is None and gen_b is None:
        print("\n❌ No trained generators found. Train them first!")
        return

    # ═══════════════════════════════════════════════════════
    # Generate trajectories
    # ═══════════════════════════════════════════════════════
    print(f"\n[Generating Trajectories]")
    print(f"  Episodes per option: {num_episodes}")

    results = {}

    if gen_a is not None:
        print(f"\n  Generating Option A trajectories...")
        trajs_a = generate_trajectories(gen_a, env, num_episodes, seed)
        results['A'] = trajs_a
        print(f"    ✓ Generated {len(trajs_a)} trajectories")
    else:
        trajs_a = None

    if gen_b is not None:
        print(f"\n  Generating Option B trajectories...")
        trajs_b = generate_trajectories(gen_b, env, num_episodes, seed)
        results['B'] = trajs_b
        print(f"    ✓ Generated {len(trajs_b)} trajectories")
    else:
        trajs_b = None

    # ═══════════════════════════════════════════════════════
    # Compute metrics
    # ═══════════════════════════════════════════════════════
    print(f"\n[Computing Metrics]")

    # Expert metrics
    expert_action_dist = compute_action_distribution(expert_trajectories)
    expert_win_rate = compute_win_rate(expert_trajectories)
    expert_lengths = [len(t.actions) for t in expert_trajectories]

    print(f"\n  Expert:")
    print(f"    Win rate: {expert_win_rate:.3f}")
    print(f"    Avg trajectory length: {np.mean(expert_lengths):.1f}")

    metrics = {}

    if trajs_a is not None:
        action_dist_a = compute_action_distribution(trajs_a)
        kl_a = compute_kl_divergence(expert_action_dist, action_dist_a)
        win_rate_a = compute_win_rate(trajs_a)
        lengths_a = [len(t.actions) for t in trajs_a]

        metrics['A'] = {
            'kl_divergence': kl_a,
            'win_rate': win_rate_a,
            'avg_length': np.mean(lengths_a),
            'std_length': np.std(lengths_a),
            'action_dist': action_dist_a
        }

        print(f"\n  Option A (Pure NN):")
        print(f"    KL divergence from expert: {kl_a:.4f}")
        print(f"    Win rate: {win_rate_a:.3f} (diff: {win_rate_a - expert_win_rate:+.3f})")
        print(f"    Avg trajectory length: {np.mean(lengths_a):.1f} ± {np.std(lengths_a):.1f}")

    if trajs_b is not None:
        action_dist_b = compute_action_distribution(trajs_b)
        kl_b = compute_kl_divergence(expert_action_dist, action_dist_b)
        win_rate_b = compute_win_rate(trajs_b)
        lengths_b = [len(t.actions) for t in trajs_b]

        metrics['B'] = {
            'kl_divergence': kl_b,
            'win_rate': win_rate_b,
            'avg_length': np.mean(lengths_b),
            'std_length': np.std(lengths_b),
            'action_dist': action_dist_b
        }

        print(f"\n  Option B (BC-initialized):")
        print(f"    KL divergence from expert: {kl_b:.4f}")
        print(f"    Win rate: {win_rate_b:.3f} (diff: {win_rate_b - expert_win_rate:+.3f})")
        print(f"    Avg trajectory length: {np.mean(lengths_b):.1f} ± {np.std(lengths_b):.1f}")

    # ═══════════════════════════════════════════════════════
    # Comparison
    # ═══════════════════════════════════════════════════════
    if trajs_a is not None and trajs_b is not None:
        print(f"\n[Direct Comparison]")
        print(f"  KL divergence (lower is better):")
        print(f"    Option A: {metrics['A']['kl_divergence']:.4f}")
        print(f"    Option B: {metrics['B']['kl_divergence']:.4f}")
        if metrics['A']['kl_divergence'] < metrics['B']['kl_divergence']:
            print(f"    → Option A matches expert better")
        else:
            print(f"    → Option B matches expert better")

        print(f"\n  Win rate:")
        print(f"    Expert: {expert_win_rate:.3f}")
        print(f"    Option A: {metrics['A']['win_rate']:.3f}")
        print(f"    Option B: {metrics['B']['win_rate']:.3f}")

    # ═══════════════════════════════════════════════════════
    # Visualization
    # ═══════════════════════════════════════════════════════
    print(f"\n[Creating Visualization]")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Trajectory length distribution
    ax = axes[0, 0]
    data_to_plot = [expert_lengths]
    labels = ['Expert']
    colors = ['coral']

    if trajs_a is not None:
        data_to_plot.append(lengths_a)
        labels.append('Option A')
        colors.append('skyblue')

    if trajs_b is not None:
        data_to_plot.append(lengths_b)
        labels.append('Option B')
        colors.append('lightgreen')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    for i, color in enumerate(colors):
        bp['boxes'][i].set_facecolor(color)

    ax.set_ylabel('Trajectory Length (moves)', fontsize=12)
    ax.set_title('Trajectory Length Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Plot 2: Win rate comparison
    ax = axes[0, 1]
    win_rates = [expert_win_rate]
    labels_wr = ['Expert']
    colors_wr = ['coral']

    if trajs_a is not None:
        win_rates.append(metrics['A']['win_rate'])
        labels_wr.append('Option A')
        colors_wr.append('skyblue')

    if trajs_b is not None:
        win_rates.append(metrics['B']['win_rate'])
        labels_wr.append('Option B')
        colors_wr.append('lightgreen')

    ax.bar(range(len(win_rates)), win_rates, color=colors_wr)
    ax.set_xticks(range(len(win_rates)))
    ax.set_xticklabels(labels_wr)
    ax.set_ylabel('Win Rate', fontsize=12)
    ax.set_title('Win Rate Comparison', fontsize=14, fontweight='bold')
    ax.axhline(expert_win_rate, color='red', linestyle='--', alpha=0.5, label='Expert baseline')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 3: KL divergence
    ax = axes[1, 0]
    if trajs_a is not None or trajs_b is not None:
        kl_values = []
        kl_labels = []
        kl_colors = []

        if trajs_a is not None:
            kl_values.append(metrics['A']['kl_divergence'])
            kl_labels.append('Option A')
            kl_colors.append('skyblue')

        if trajs_b is not None:
            kl_values.append(metrics['B']['kl_divergence'])
            kl_labels.append('Option B')
            kl_colors.append('lightgreen')

        ax.bar(range(len(kl_values)), kl_values, color=kl_colors)
        ax.set_xticks(range(len(kl_values)))
        ax.set_xticklabels(kl_labels)
        ax.set_ylabel('KL(Expert || Generated)', fontsize=12)
        ax.set_title('Action Distribution Similarity to Expert', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"Comparison Summary (h={h})\n\n"
    summary_text += f"Expert:\n"
    summary_text += f"  Win rate: {expert_win_rate:.3f}\n"
    summary_text += f"  Avg length: {np.mean(expert_lengths):.1f}\n\n"

    if trajs_a is not None:
        summary_text += f"Option A (Pure NN):\n"
        summary_text += f"  KL divergence: {metrics['A']['kl_divergence']:.4f}\n"
        summary_text += f"  Win rate: {metrics['A']['win_rate']:.3f}\n"
        summary_text += f"  Avg length: {metrics['A']['avg_length']:.1f}\n\n"

    if trajs_b is not None:
        summary_text += f"Option B (BC-init):\n"
        summary_text += f"  KL divergence: {metrics['B']['kl_divergence']:.4f}\n"
        summary_text += f"  Win rate: {metrics['B']['win_rate']:.3f}\n"
        summary_text += f"  Avg length: {metrics['B']['avg_length']:.1f}\n\n"

    if trajs_a is not None and trajs_b is not None:
        summary_text += "Winner:\n"
        if metrics['A']['kl_divergence'] < metrics['B']['kl_divergence']:
            summary_text += "  Option A (lower KL)\n"
        else:
            summary_text += "  Option B (lower KL)\n"

    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    output_path = f'figures/option_a_vs_b_h{h}.png'
    os.makedirs('figures', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

    print("\n" + "=" * 80)
    print("Comparison Complete")
    print("=" * 80)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare Option A vs Option B'
    )
    parser.add_argument('--h', type=int, default=4,
                        help='Planning depth (default: 4)')
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes to generate per option (default: 50)')
    parser.add_argument('--expert_data', type=str, default='opendata/raw_data.csv',
                        help='Path to expert data (default: opendata/raw_data.csv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # Load expert data
    print(f"Loading expert data from {args.expert_data}...")
    expert_trajectories = load_expert_trajectories(
        csv_path=args.expert_data,
        player_filter=0,
        max_trajectories=100
    )
    print(f"Loaded {len(expert_trajectories)} expert trajectories")

    # Compare
    compare_options(
        h=args.h,
        expert_trajectories=expert_trajectories,
        num_episodes=args.num_episodes,
        seed=args.seed
    )
