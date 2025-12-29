"""
Analyze Learning Effect: Early vs Late Games

Tests if players show learning effects by comparing E[h] in:
- Early period: First 30 games
- Late period: Last 30 games

Uses within-subject design (paired t-test) to detect changes in planning depth.

Research Question:
- Do players increase planning depth with experience?
- Or are they skilled from the start (selection effect)?

Usage:
    python3 analyze_learning_effect.py
"""

import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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


def detect_games_by_participant(df):
    """
    Detect individual games and organize by participant

    Returns:
        dict: {participant_id: [game1, game2, ...]}
              Each game is a list of move dictionaries
    """
    participant_games = defaultdict(list)

    # Add piece count column
    df['num_pieces'] = df['black_pieces'].apply(lambda x: x.count('1')) + \
                       df['white_pieces'].apply(lambda x: x.count('1'))

    # Group by participant first
    for participant in df['participant'].unique():
        participant_df = df[df['participant'] == participant].copy()
        participant_df = participant_df.sort_index()

        # Detect games within this participant's data
        current_game = []
        prev_pieces = 0

        for idx, row in participant_df.iterrows():
            num_pieces = row['num_pieces']

            # New game detected (piece count decreased)
            if num_pieces < prev_pieces:
                if len(current_game) > 0:
                    participant_games[participant].append(current_game)
                current_game = []

            current_game.append(row)
            prev_pieces = num_pieces

        # Add last game
        if len(current_game) > 0:
            participant_games[participant].append(current_game)

    # Print statistics
    total_games = sum(len(games) for games in participant_games.values())
    print(f"\nDetected {total_games} games across {len(participant_games)} participants")

    games_per_player = [len(games) for games in participant_games.values()]
    print(f"Games per participant: mean={np.mean(games_per_player):.1f}, "
          f"min={np.min(games_per_player)}, max={np.max(games_per_player)}")

    return participant_games


def estimate_h_for_moves(moves, discriminator, h_values):
    """
    Estimate h distribution for a set of moves

    Returns:
        h_expected: E[h] for this set of moves
        h_probs: probability distribution over h values
    """
    observations = []
    actions = []

    for move in moves:
        # Parse board state
        obs = parse_board_state(move['black_pieces'], move['white_pieces'])
        action = move['move']

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


def analyze_early_vs_late(participant_games, discriminator, h_values, n_games=30):
    """
    Compare E[h] in early vs late games for each participant

    Args:
        participant_games: dict {participant: [game1, game2, ...]}
        discriminator: trained discriminator model
        h_values: [1, 2, 3, 4]
        n_games: number of games to use for early/late (default: 30)

    Returns:
        results_df: DataFrame with per-participant early/late comparisons
    """
    print("\n" + "=" * 80)
    print(f"Analyzing Early (first {n_games}) vs Late (last {n_games}) Games")
    print("=" * 80)

    results = []

    for participant, games in participant_games.items():
        num_games = len(games)

        # Skip participants with insufficient games
        if num_games < n_games * 2:
            print(f"Participant {participant}: Only {num_games} games (need {n_games*2}), skipping")
            continue

        # Split into early and late
        early_games = games[:n_games]
        late_games = games[-n_games:]

        # Flatten games into moves
        early_moves = [move for game in early_games for move in game]
        late_moves = [move for game in late_games for move in game]

        # Estimate E[h] for each period
        h_early, probs_early = estimate_h_for_moves(early_moves, discriminator, h_values)
        h_late, probs_late = estimate_h_for_moves(late_moves, discriminator, h_values)

        if h_early is None or h_late is None:
            continue

        # Calculate change
        h_change = h_late - h_early

        results.append({
            'participant': participant,
            'num_games': num_games,
            'E[h]_early': h_early,
            'E[h]_late': h_late,
            'E[h]_change': h_change,
            'P(h=1)_early': probs_early[0],
            'P(h=2)_early': probs_early[1],
            'P(h=3)_early': probs_early[2],
            'P(h=4)_early': probs_early[3],
            'P(h=1)_late': probs_late[0],
            'P(h=2)_late': probs_late[1],
            'P(h=3)_late': probs_late[2],
            'P(h=4)_late': probs_late[3],
            'num_moves_early': len(early_moves),
            'num_moves_late': len(late_moves),
        })

        print(f"Participant {participant:3d}: E[h]_early={h_early:.3f}, "
              f"E[h]_late={h_late:.3f}, change={h_change:+.3f}")

    results_df = pd.DataFrame(results)

    print(f"\nAnalyzed {len(results_df)} participants with sufficient data")

    return results_df


def statistical_tests(results_df):
    """Perform statistical tests on early vs late comparison"""
    print("\n" + "=" * 80)
    print("Statistical Analysis: Early vs Late")
    print("=" * 80)

    # Paired t-test
    h_early = results_df['E[h]_early'].values
    h_late = results_df['E[h]_late'].values

    t_stat, p_value = stats.ttest_rel(h_late, h_early)

    # Effect size (Cohen's d for paired samples)
    differences = h_late - h_early
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff

    print(f"\nPaired t-test (Late vs Early):")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Mean difference: {mean_diff:+.4f}")
    print(f"  Std difference: {std_diff:.4f}")
    print(f"  Cohen's d: {cohens_d:.4f}")

    # Interpretation
    print(f"\nInterpretation:")
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"

    print(f"  Statistical significance: {sig} (p={p_value:.4f})")

    if abs(cohens_d) < 0.2:
        effect = "negligible"
    elif abs(cohens_d) < 0.5:
        effect = "small"
    elif abs(cohens_d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    print(f"  Effect size: {effect} (d={cohens_d:.4f})")

    if mean_diff > 0:
        direction = "increased"
    else:
        direction = "decreased"

    print(f"  Direction: Planning depth {direction} from early to late games")

    # Summary statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Early games E[h]: {np.mean(h_early):.3f} ± {np.std(h_early):.3f}")
    print(f"  Late games E[h]:  {np.mean(h_late):.3f} ± {np.std(h_late):.3f}")
    print(f"  Change range: [{np.min(differences):.3f}, {np.max(differences):.3f}]")
    print(f"  Positive changes: {np.sum(differences > 0)} / {len(differences)}")
    print(f"  Negative changes: {np.sum(differences < 0)} / {len(differences)}")
    print(f"  No change: {np.sum(differences == 0)} / {len(differences)}")

    return {
        't_stat': t_stat,
        'p_value': p_value,
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'cohens_d': cohens_d
    }


def visualize_results(results_df, stats_results, n_games=30):
    """Create visualizations for early vs late comparison"""
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Paired plot: Early vs Late for each participant
    ax = axes[0, 0]

    h_early = results_df['E[h]_early'].values
    h_late = results_df['E[h]_late'].values
    participants = results_df['participant'].values

    # Sort by change magnitude
    sort_idx = np.argsort(h_late - h_early)

    y_pos = np.arange(len(participants))
    ax.plot([h_early[sort_idx], h_late[sort_idx]], [y_pos, y_pos],
            'o-', color='gray', alpha=0.5, linewidth=1)
    ax.scatter(h_early[sort_idx], y_pos, color='steelblue', label='Early', s=50, zorder=3)
    ax.scatter(h_late[sort_idx], y_pos, color='coral', label='Late', s=50, zorder=3)

    ax.set_xlabel('E[h]')
    ax.set_ylabel('Participant (sorted by change)')
    ax.set_title(f'Planning Depth: Early vs Late Games\n(First {n_games} vs Last {n_games})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Distribution of changes
    ax = axes[0, 1]

    differences = h_late - h_early
    ax.hist(differences, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(np.mean(differences), color='orange', linestyle='-', linewidth=2,
               label=f'Mean = {np.mean(differences):+.3f}')

    ax.set_xlabel('E[h]_late - E[h]_early')
    ax.set_ylabel('Number of participants')
    ax.set_title(f'Distribution of Planning Depth Changes\n'
                 f't={stats_results["t_stat"]:.3f}, p={stats_results["p_value"]:.4f}, '
                 f'd={stats_results["cohens_d"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Scatter plot: Early vs Late
    ax = axes[1, 0]

    ax.scatter(h_early, h_late, alpha=0.6, s=80, color='steelblue')

    # Add diagonal line (no change)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1])
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='No change', zorder=1)

    # Add correlation
    corr = np.corrcoef(h_early, h_late)[0, 1]
    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('E[h] Early')
    ax.set_ylabel('E[h] Late')
    ax.set_title('Early vs Late Planning Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # 4. Probability distribution changes
    ax = axes[1, 1]

    # Average probabilities
    probs_early = results_df[['P(h=1)_early', 'P(h=2)_early', 'P(h=3)_early', 'P(h=4)_early']].mean()
    probs_late = results_df[['P(h=1)_late', 'P(h=2)_late', 'P(h=3)_late', 'P(h=4)_late']].mean()

    x = np.arange(4)
    width = 0.35

    ax.bar(x - width/2, probs_early.values, width, label='Early', color='steelblue', alpha=0.7)
    ax.bar(x + width/2, probs_late.values, width, label='Late', color='coral', alpha=0.7)

    ax.set_xlabel('Planning Depth (h)')
    ax.set_ylabel('Average Probability')
    ax.set_title('Probability Distribution: Early vs Late')
    ax.set_xticks(x)
    ax.set_xticklabels(['h=1', 'h=2', 'h=3', 'h=4'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save figure
    output_path = 'figures/learning_effect_analysis.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")

    plt.show()


def save_results(results_df, stats_results, n_games=30):
    """Save results to CSV and pickle"""
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)

    # Save DataFrame
    csv_path = f'results/learning_effect_early{n_games}_vs_late{n_games}.csv'
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save complete results including stats
    pkl_path = f'results/learning_effect_early{n_games}_vs_late{n_games}.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'results_df': results_df,
            'stats': stats_results,
            'n_games': n_games
        }, f)
    print(f"Saved pickle: {pkl_path}")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("Learning Effect Analysis: Early vs Late Games")
    print("=" * 80)

    # Load discriminator
    discriminator, h_values = load_multiclass_discriminator()

    # Load human games
    df = load_human_games()

    # Detect games by participant
    participant_games = detect_games_by_participant(df)

    # Analyze early vs late (first 10 vs last 10)
    # Using n_games=10 to maximize sample size (need at least 20 games total)
    results_df = analyze_early_vs_late(participant_games, discriminator, h_values, n_games=10)

    if len(results_df) == 0:
        print("\nERROR: No participants with sufficient games!")
        print("Try reducing n_games further or using a different analysis approach.")
        return

    # Statistical tests
    stats_results = statistical_tests(results_df)

    # Visualize
    visualize_results(results_df, stats_results, n_games=10)

    # Save results
    save_results(results_df, stats_results, n_games=10)

    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

    # Key findings summary
    print("\nKEY FINDINGS:")
    print(f"  Sample size: {len(results_df)} participants")
    print(f"  Mean E[h] change: {stats_results['mean_diff']:+.4f}")
    print(f"  Statistical significance: p = {stats_results['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {stats_results['cohens_d']:.4f}")

    if stats_results['p_value'] < 0.05:
        if stats_results['mean_diff'] > 0:
            print("\n  CONCLUSION: Significant INCREASE in planning depth with experience")
            print("              → Evidence for LEARNING EFFECT")
        else:
            print("\n  CONCLUSION: Significant DECREASE in planning depth with experience")
            print("              → Possible fatigue or strategy shift")
    else:
        print("\n  CONCLUSION: No significant change in planning depth")
        print("              → Evidence for SELECTION EFFECT (skilled from start)")


if __name__ == '__main__':
    main()
