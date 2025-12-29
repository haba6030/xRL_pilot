"""
Estimate Player Planning Depth (h) using Multi-Class Discriminator

Re-estimates all human players using the 4-class discriminator (h=1,2,3,4)
instead of the binary discriminator (h=1 vs h=4).

Expected improvements:
- Better calibration (intermediate values h=2,3)
- Finer-grained h distribution
- Reduced bias from binary decision boundary

Usage:
    python3 estimate_player_h_multiclass.py
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


def detect_games(df):
    """Detect individual games by tracking piece count resets"""
    games = []
    current_game = []

    df['num_pieces'] = df['black_pieces'].apply(lambda x: x.count('1')) + \
                       df['white_pieces'].apply(lambda x: x.count('1'))

    prev_pieces = 0
    current_game_id = 0

    for idx, row in df.iterrows():
        num_pieces = row['num_pieces']

        # New game detected (piece count decreased)
        if num_pieces < prev_pieces:
            if len(current_game) > 0:
                games.append({
                    'game_id': current_game_id,
                    'moves': current_game
                })
            current_game = []
            current_game_id += 1

        current_game.append(row)
        prev_pieces = num_pieces

    # Add last game
    if len(current_game) > 0:
        games.append({
            'game_id': current_game_id,
            'moves': current_game
        })

    print(f"Detected {len(games)} games")
    return games


def estimate_h_multiclass_for_game(game_moves, discriminator, h_values):
    """
    Estimate h distribution for a single game using multi-class discriminator

    Returns:
        h_probs: (4,) array with probabilities for each h value
        h_pred: predicted h value (1,2,3, or 4)
        h_expected: expected value E[h]
    """
    observations = []
    actions = []

    for move in game_moves:
        # Parse board state
        obs = parse_board_state(move['black_pieces'], move['white_pieces'])
        action = move['move']

        observations.append(obs)
        actions.append(action)

    # Convert to tensors
    states = np.array(observations, dtype=np.float32)
    states = torch.FloatTensor(states)
    action_indices = torch.LongTensor(actions)

    # Get discriminator predictions
    with torch.no_grad():
        logits = discriminator(states, action_indices)
        probs = torch.softmax(logits, dim=1).numpy()  # (T, 4)

    # Average probabilities across all moves in the game
    h_probs = np.mean(probs, axis=0)  # (4,)

    # Predicted h (argmax)
    h_pred_idx = np.argmax(h_probs)
    h_pred = h_values[h_pred_idx]

    # Expected value E[h]
    h_expected = np.sum(h_probs * np.array(h_values))

    return h_probs, h_pred, h_expected


def estimate_all_players(df, discriminator, h_values):
    """Estimate h for all players across all their games"""
    print("\n" + "=" * 80)
    print("Estimating h for All Players")
    print("=" * 80)

    results = []
    player_aggregated = defaultdict(lambda: {
        'games': [],
        'h_probs_all': [],
        'h_pred_all': [],
        'h_expected_all': []
    })

    # Detect games
    games = detect_games(df)

    # Process each game
    for game in games:
        game_moves = game['moves']

        if len(game_moves) == 0:
            continue

        # Get player ID (assume all moves in game from same player pair)
        # In human-vs-human, each game has two players
        # We'll aggregate by participant
        participants = set(move['participant'] for move in game_moves)

        h_probs, h_pred, h_expected = estimate_h_multiclass_for_game(
            game_moves, discriminator, h_values
        )

        # Store results
        for participant in participants:
            player_aggregated[participant]['games'].append(game['game_id'])
            player_aggregated[participant]['h_probs_all'].append(h_probs)
            player_aggregated[participant]['h_pred_all'].append(h_pred)
            player_aggregated[participant]['h_expected_all'].append(h_expected)

    # Aggregate per player
    print(f"\nProcessed {len(games)} games, {len(player_aggregated)} players")

    for participant, data in player_aggregated.items():
        # Average h probabilities across all games
        h_probs_mean = np.mean(data['h_probs_all'], axis=0)
        h_probs_std = np.std(data['h_probs_all'], axis=0)

        # Most common h prediction
        h_pred_mode = max(set(data['h_pred_all']), key=data['h_pred_all'].count)

        # Average expected value
        h_expected_mean = np.mean(data['h_expected_all'])
        h_expected_std = np.std(data['h_expected_all'])

        results.append({
            'participant': participant,
            'num_games': len(data['games']),
            'h_probs_mean': h_probs_mean,
            'h_probs_std': h_probs_std,
            'h_pred_mode': h_pred_mode,
            'h_expected_mean': h_expected_mean,
            'h_expected_std': h_expected_std,
            # Individual h probabilities for each class
            'P(h=1)': h_probs_mean[0],
            'P(h=2)': h_probs_mean[1],
            'P(h=3)': h_probs_mean[2],
            'P(h=4)': h_probs_mean[3],
        })

    return results


def analyze_results(results, h_values):
    """Analyze and summarize results"""
    print("\n" + "=" * 80)
    print("Analysis of Human Planning Depth (Multi-Class)")
    print("=" * 80)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Overall statistics
    print(f"\nOverall Statistics:")
    print(f"  Total players: {len(df_results)}")
    print(f"  Mean E[h]: {df_results['h_expected_mean'].mean():.3f} ± {df_results['h_expected_mean'].std():.3f}")

    # Distribution of predicted h (mode)
    print(f"\nDistribution of h predictions (mode):")
    for h in h_values:
        count = (df_results['h_pred_mode'] == h).sum()
        pct = count / len(df_results) * 100
        print(f"  h={h}: {count:2d} players ({pct:4.1f}%)")

    # Average probabilities across all players
    print(f"\nAverage probabilities across all players:")
    for i, h in enumerate(h_values):
        mean_prob = df_results[f'P(h={h})'].mean()
        std_prob = df_results[f'P(h={h})'].std()
        print(f"  P(h={h}): {mean_prob:.3f} ± {std_prob:.3f}")

    # Comparison with binary discriminator expectation
    print(f"\nComparison with Binary Discriminator (h=1 vs h=4):")
    print(f"  Binary discriminator found: All players h≈4 (mean=0.936)")
    print(f"  Multi-class E[h]: {df_results['h_expected_mean'].mean():.3f}")
    print(f"  Conclusion: ", end="")

    if df_results['h_expected_mean'].mean() > 3.5:
        print("Consistent with binary (most players are h=4)")
    elif df_results['h_expected_mean'].mean() < 2.0:
        print("DIFFERENT - players more myopic than binary suggested")
    else:
        print("REFINED - players distributed across h=2,3,4")

    return df_results


def visualize_results(df_results, h_values):
    """Visualize results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Distribution of E[h]
    ax1 = axes[0, 0]
    ax1.hist(df_results['h_expected_mean'], bins=20, edgecolor='black', alpha=0.7)
    ax1.axvline(df_results['h_expected_mean'].mean(), color='r',
                linestyle='--', linewidth=2, label=f'Mean={df_results["h_expected_mean"].mean():.2f}')
    ax1.set_xlabel('E[h] (expected planning depth)')
    ax1.set_ylabel('Number of players')
    ax1.set_title('Distribution of Expected Planning Depth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of h predictions (mode)
    ax2 = axes[0, 1]
    h_counts = [sum(df_results['h_pred_mode'] == h) for h in h_values]
    ax2.bar(h_values, h_counts, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted h (mode)')
    ax2.set_ylabel('Number of players')
    ax2.set_title('Distribution of h Predictions')
    ax2.set_xticks(h_values)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Average probabilities across all players
    ax3 = axes[0, 2]
    avg_probs = [df_results[f'P(h={h})'].mean() for h in h_values]
    std_probs = [df_results[f'P(h={h})'].std() for h in h_values]
    ax3.bar(h_values, avg_probs, yerr=std_probs, capsize=5,
            edgecolor='black', alpha=0.7)
    ax3.set_xlabel('h')
    ax3.set_ylabel('Average P(h)')
    ax3.set_title('Average Probability Distribution')
    ax3.set_xticks(h_values)
    ax3.axhline(y=0.25, color='gray', linestyle='--', label='Uniform (25%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Heatmap of P(h) for each player
    ax4 = axes[1, 0]
    prob_matrix = np.array([
        [df_results.iloc[i][f'P(h={h})'] for h in h_values]
        for i in range(len(df_results))
    ])
    sns.heatmap(prob_matrix.T, cmap='YlOrRd', cbar_kws={'label': 'P(h)'},
                yticklabels=[f'h={h}' for h in h_values],
                xticklabels=False, ax=ax4)
    ax4.set_xlabel('Player ID')
    ax4.set_ylabel('Planning depth h')
    ax4.set_title('Probability Distribution per Player')

    # Plot 5: Scatter E[h] vs std(h)
    ax5 = axes[1, 1]
    ax5.scatter(df_results['h_expected_mean'], df_results['h_expected_std'],
                alpha=0.6)
    ax5.set_xlabel('E[h] (mean)')
    ax5.set_ylabel('std(E[h])')
    ax5.set_title('Planning Depth: Mean vs Variability')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = f"""
    Multi-Class Discriminator Results

    Total players: {len(df_results)}

    E[h] across players:
      Mean: {df_results['h_expected_mean'].mean():.3f}
      Std:  {df_results['h_expected_mean'].std():.3f}
      Min:  {df_results['h_expected_mean'].min():.3f}
      Max:  {df_results['h_expected_mean'].max():.3f}

    h predictions (mode):
      h=1: {sum(df_results['h_pred_mode'] == 1)} ({sum(df_results['h_pred_mode'] == 1)/len(df_results)*100:.1f}%)
      h=2: {sum(df_results['h_pred_mode'] == 2)} ({sum(df_results['h_pred_mode'] == 2)/len(df_results)*100:.1f}%)
      h=3: {sum(df_results['h_pred_mode'] == 3)} ({sum(df_results['h_pred_mode'] == 3)/len(df_results)*100:.1f}%)
      h=4: {sum(df_results['h_pred_mode'] == 4)} ({sum(df_results['h_pred_mode'] == 4)/len(df_results)*100:.1f}%)

    Comparison:
      Binary: mean h_score = 0.936
      Multi-class: mean E[h] = {df_results['h_expected_mean'].mean():.3f}
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    output_path = Path('figures') / 'human_h_multiclass_results.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Estimate Human Planning Depth (Multi-Class Discriminator)")
    print("=" * 80)

    # Load discriminator
    discriminator, h_values = load_multiclass_discriminator()

    # Load human data
    df = load_human_games()

    # Estimate h for all players
    results = estimate_all_players(df, discriminator, h_values)

    # Analyze results
    df_results = analyze_results(results, h_values)

    # Visualize
    visualize_results(df_results, h_values)

    # Save results
    output_path = Path('human_h_multiclass_estimates.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✅ Saved results to {output_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
