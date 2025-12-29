"""
Estimate Player Planning Depth (h) using Multi-Class Discriminator - FIXED

CRITICAL FIX: Separate each player's moves in human-vs-human games.
Previous version incorrectly averaged both players' moves together.

Usage:
    python3 estimate_player_h_multiclass_fixed.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_multiclass_discriminator(model_path='models/multiclass_discriminator.pt'):
    """Load the trained multi-class discriminator"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Import model class
    from train_multiclass_discriminator import MultiClassDiscriminator

    # Get model parameters from checkpoint
    h_values = checkpoint['h_values']
    num_classes = checkpoint['num_classes']

    # Create model
    model = MultiClassDiscriminator(
        state_dim=89,
        action_dim=36,
        num_classes=num_classes,
        hidden_dims=[256, 128, 64]
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded multi-class discriminator: {h_values}")
    print(f"Test accuracy: {checkpoint['accuracy']:.3f}")

    return model, h_values


def load_human_games(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """Load human game data from van Opheusden dataset"""
    df = pd.read_csv(data_path)

    # Filter for human vs human games
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print(f"Loaded {len(df)} moves from human games")
    print(f"Unique participants: {df['participant'].nunique()}")

    return df


def parse_board_state(black_str, white_str):
    """Convert board string to observation vector"""
    from env import FourInARowEnv

    # Create environment
    env = FourInARowEnv()

    # Parse board strings (36 characters, '1' means piece present)
    black_pieces = np.array([int(c) for c in black_str])
    white_pieces = np.array([int(c) for c in white_str])

    # Set board state
    env.board = np.zeros(36, dtype=np.int8)
    env.board[black_pieces == 1] = 1   # Black = 1
    env.board[white_pieces == 1] = -1  # White = -1

    # Get observation (includes van Opheusden features)
    obs = env._get_observation()

    return obs


def estimate_h_for_participant_moves(participant_moves, discriminator, h_values):
    """
    Estimate h distribution for a specific participant's moves only

    Args:
        participant_moves: List of moves by this participant
        discriminator: Trained discriminator model
        h_values: [1, 2, 3, 4]

    Returns:
        h_expected: E[h] for this participant
        h_probs: probability distribution over h values
    """
    observations = []
    actions = []

    for move_data in participant_moves:
        # Parse board state
        obs = parse_board_state(move_data['black_pieces'], move_data['white_pieces'])
        action = move_data['move']

        observations.append(obs)
        actions.append(action)

    if len(observations) == 0:
        return None, None

    # Convert to tensors
    states = np.array(observations, dtype=np.float32)
    states = torch.FloatTensor(states)
    action_indices = torch.LongTensor(actions)

    # Get discriminator predictions
    with torch.no_grad():
        logits = discriminator(states, action_indices)
        probs = torch.softmax(logits, dim=1).numpy()  # (T, num_classes)

    # Average probabilities across all moves
    h_probs = np.mean(probs, axis=0)  # (num_classes,)

    # Expected value E[h]
    h_expected = np.sum(h_probs * np.array(h_values))

    return h_expected, h_probs


def estimate_all_players(df, discriminator, h_values):
    """
    Estimate h for all players, separating each player's moves

    CRITICAL: Only analyze moves made by each specific participant
    """
    print("\n" + "=" * 80)
    print("Estimating h for All Players (FIXED: Player-Specific Moves)")
    print("=" * 80)

    results = []

    # Group by participant
    for participant in df['participant'].unique():
        # Get only this participant's moves
        participant_moves = df[df['participant'] == participant].to_dict('records')

        print(f"\nParticipant {participant}: {len(participant_moves)} moves")

        # Estimate E[h] for this participant only
        h_expected, h_probs = estimate_h_for_participant_moves(
            participant_moves, discriminator, h_values
        )

        if h_expected is None:
            continue

        # Mode (argmax)
        h_pred_mode = h_values[np.argmax(h_probs)]

        results.append({
            'participant': participant,
            'num_moves': len(participant_moves),
            'h_expected_mean': h_expected,
            'h_pred_mode': h_pred_mode,
            # Individual h probabilities
            'P(h=1)': h_probs[0],
            'P(h=2)': h_probs[1],
            'P(h=3)': h_probs[2],
            'P(h=4)': h_probs[3],
        })

        print(f"  E[h] = {h_expected:.3f}, Mode h = {h_pred_mode}")

    results_df = pd.DataFrame(results)

    print(f"\nProcessed {len(results_df)} participants")

    return results_df


def analyze_results(results_df, h_values):
    """Analyze and summarize results"""
    print("\n" + "=" * 80)
    print("Analysis of Human Planning Depth (Multi-Class, FIXED)")
    print("=" * 80)

    # Overall statistics
    h_mean = results_df['h_expected_mean'].mean()
    h_std = results_df['h_expected_mean'].std()
    h_min = results_df['h_expected_mean'].min()
    h_max = results_df['h_expected_mean'].max()

    print(f"\nOverall Statistics (N={len(results_df)}):")
    print(f"  E[h] mean: {h_mean:.3f} ± {h_std:.3f}")
    print(f"  E[h] range: [{h_min:.3f}, {h_max:.3f}]")

    # Distribution of mode predictions
    print(f"\nMode Classification:")
    for h in h_values:
        count = np.sum(results_df['h_pred_mode'] == h)
        pct = count / len(results_df) * 100
        print(f"  h={h}: {count} participants ({pct:.1f}%)")

    # Average probability distribution
    avg_probs = results_df[['P(h=1)', 'P(h=2)', 'P(h=3)', 'P(h=4)']].mean()
    print(f"\nAverage Probability Distribution:")
    for i, h in enumerate(h_values):
        print(f"  P(h={h}) = {avg_probs[i]:.3f}")

    return {
        'h_mean': h_mean,
        'h_std': h_std,
        'avg_probs': avg_probs.values
    }


def visualize_results(results_df, stats):
    """Create visualizations"""
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. E[h] distribution
    ax = axes[0, 0]
    ax.hist(results_df['h_expected_mean'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(stats['h_mean'], color='red', linestyle='--', linewidth=2,
               label=f'Mean = {stats["h_mean"]:.3f}')
    ax.set_xlabel('E[h] (Planning Depth)')
    ax.set_ylabel('Number of participants')
    ax.set_title('Distribution of Planning Depth (FIXED)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 2. Probability distribution
    ax = axes[0, 1]
    h_values = [1, 2, 3, 4]
    avg_probs = stats['avg_probs']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

    bars = ax.bar(h_values, avg_probs, color=colors, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Planning Depth (h)')
    ax.set_ylabel('Average Probability')
    ax.set_title('Average Probability Distribution Across All Players')
    ax.set_xticks(h_values)
    ax.set_xticklabels([f'h={h}' for h in h_values])
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, prob in zip(bars, avg_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}',
                ha='center', va='bottom', fontsize=10)

    # 3. Mode classification
    ax = axes[1, 0]
    mode_counts = results_df['h_pred_mode'].value_counts().sort_index()
    ax.bar(mode_counts.index, mode_counts.values, color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Predicted h (Mode)')
    ax.set_ylabel('Number of participants')
    ax.set_title('Mode Classification Distribution')
    ax.set_xticks(h_values)
    ax.set_xticklabels([f'h={h}' for h in h_values])
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Scatter: Number of moves vs E[h]
    ax = axes[1, 1]
    ax.scatter(results_df['num_moves'], results_df['h_expected_mean'],
              alpha=0.6, s=80, color='steelblue')

    # Correlation
    from scipy.stats import spearmanr
    corr, p_val = spearmanr(results_df['num_moves'], results_df['h_expected_mean'])
    ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Number of moves')
    ax.set_ylabel('E[h] (Planning Depth)')
    ax.set_title('Planning Depth vs Sample Size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = 'figures/human_h_multiclass_results_fixed.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")

    plt.show()


def save_results(results_df, output_csv='human_h_multiclass_estimates_fixed.csv'):
    """Save results to CSV"""
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results: {output_csv}")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("Estimating Human Planning Depth (h) - FIXED VERSION")
    print("=" * 80)

    # Load discriminator
    discriminator, h_values = load_multiclass_discriminator()

    # Load human games
    df = load_human_games()

    # Estimate h for all players (FIXED: player-specific moves)
    results_df = estimate_all_players(df, discriminator, h_values)

    # Analyze results
    stats = analyze_results(results_df, h_values)

    # Visualize
    visualize_results(results_df, stats)

    # Save
    save_results(results_df)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
