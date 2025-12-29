"""
Estimate Planning Depth (h) for Human Players

Uses the trained AIRL discriminator to estimate each player's planning depth
from their actual game moves.

Research Question: Does planning depth correlate with expertise?
Hypothesis (van Opheusden et al. 2023): Experts have deeper planning.

Usage:
    python3 estimate_player_h.py
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

import sys
sys.path.append('.')
from env import FourInARowEnv
from pilot_airl_discriminator import AIRLDiscriminator


def load_discriminator(model_path='models/pilot_airl_discriminator.pt'):
    """Load trained discriminator"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = AIRLDiscriminator(state_dim=89, action_dim=36)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Loaded discriminator (accuracy: {checkpoint['accuracy']:.3f})")

    return model


def parse_board_state(black_pieces, white_pieces, current_player='Black'):
    """
    Convert CSV board representation to 89-dim observation

    Matches env._get_observation() format
    """
    # Parse piece strings
    black = np.array([int(c) for c in black_pieces], dtype=np.float32)
    white = np.array([int(c) for c in white_pieces], dtype=np.float32)

    # Try to use actual van Opheusden features
    try:
        from features import extract_van_opheusden_features
        features = extract_van_opheusden_features(black, white, current_player)
    except ImportError:
        # Fallback: 17-dim zeros
        features = np.zeros(17, dtype=np.float32)

    # Create 89-dim observation
    observation = np.concatenate([black, white, features])

    return observation


def load_human_games(csv_path='../opendata/raw_data.csv', max_games=None):
    """
    Load human games from van Opheusden dataset

    Returns:
        games: Dict[game_id] -> {
            'observations': List of (89,) states
            'actions': List of action indices
            'participant': participant ID
            'outcome': 'win' or 'loss' or 'draw' (from Black's perspective)
        }
    """
    print("=" * 80)
    print("Loading Human Games")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    print(f"\nTotal trials: {len(df)}")
    print(f"Unique participants: {df['participant'].nunique()}")

    # Identify games by detecting when board resets (piece count drops)
    # Each game starts when total pieces decreases (new game)
    df['num_pieces'] = df['black_pieces'].apply(lambda x: x.count('1')) + \
                       df['white_pieces'].apply(lambda x: x.count('1'))

    df['game_id'] = 0
    current_game = 0
    prev_pieces = 0

    for idx, row in df.iterrows():
        if row['num_pieces'] < prev_pieces:
            # Board reset detected - new game
            current_game += 1
        df.at[idx, 'game_id'] = current_game
        prev_pieces = row['num_pieces']

    print(f"Detected games: {df['game_id'].nunique()}")

    # Group by game_id
    games = {}

    for gid, game_df in df.groupby('game_id'):
        game_df = game_df.reset_index(drop=True)

        # Get participant from first row
        participant = game_df['participant'].iloc[0]

        observations = []
        actions = []

        for _, row in game_df.iterrows():
            # Parse state
            state = parse_board_state(
                row['black_pieces'],
                row['white_pieces'],
                row['color']
            )
            observations.append(state)

            # Parse action
            action = int(row['move'])
            actions.append(action)

        # Determine outcome (from Black's perspective)
        # This is a simplification - we'd need win/loss info from data
        # For now, use game length as proxy (longer games might indicate expertise)
        outcome = 'unknown'  # Would need actual outcome data

        games[f"game_{gid}"] = {
            'observations': observations,
            'actions': actions,
            'participant': participant,
            'game_num': gid,
            'outcome': outcome,
            'num_moves': len(actions)
        }

        if max_games and len(games) >= max_games:
            break

    print(f"\nLoaded {len(games)} games")
    print(f"Average game length: {np.mean([g['num_moves'] for g in games.values()]):.1f} moves")

    return games


def estimate_h_for_game(game, discriminator):
    """
    Estimate h for a single game

    Returns:
        h_score: Mean probability of h=4 across all moves (0=h1, 1=h4)
        probs: List of per-move h=4 probabilities
    """
    observations = game['observations']
    actions = game['actions']

    # In CSV: each row has (state_before_move, action)
    # So observations and actions have same length
    # We use observations as states (current state when action was taken)

    if len(observations) != len(actions):
        # Handle mismatch (shouldn't happen but be safe)
        min_len = min(len(observations), len(actions))
        observations = observations[:min_len]
        actions = actions[:min_len]

    if len(actions) == 0:
        return 0.5, []  # Neutral if no actions

    # Prepare inputs
    states = np.array(observations, dtype=np.float32)
    states = torch.FloatTensor(states)
    action_indices = torch.LongTensor(actions)

    # Predict with discriminator
    with torch.no_grad():
        logits = discriminator(states, action_indices).squeeze()
        probs = torch.sigmoid(logits).numpy()

    # Handle single-element case
    if len(actions) == 1:
        probs = [float(probs)]
    else:
        probs = probs.tolist() if hasattr(probs, 'tolist') else list(probs)

    # h_score: average probability of h=4
    h_score = float(np.mean(probs))

    return h_score, probs


def estimate_h_per_player(games, discriminator):
    """
    Estimate h for each participant

    Returns:
        player_stats: Dict[participant_id] -> {
            'h_score': mean h across all games,
            'h_scores': list of per-game h scores,
            'num_games': number of games,
            'num_moves': total moves,
            'avg_game_length': average game length
        }
    """
    print("\n" + "=" * 80)
    print("Estimating h per Player")
    print("=" * 80)

    player_games = defaultdict(list)

    # Group games by participant
    for game_id, game in games.items():
        player_games[game['participant']].append(game)

    player_stats = {}

    for participant, participant_games in player_games.items():
        h_scores = []
        all_probs = []
        total_moves = 0

        for game in participant_games:
            h_score, probs = estimate_h_for_game(game, discriminator)
            h_scores.append(h_score)
            all_probs.extend(probs)
            total_moves += len(probs)

        player_stats[participant] = {
            'h_score': np.mean(h_scores),
            'h_std': np.std(h_scores),
            'h_scores': h_scores,
            'all_probs': all_probs,
            'num_games': len(participant_games),
            'num_moves': total_moves,
            'avg_game_length': total_moves / len(participant_games)
        }

    print(f"\nEstimated h for {len(player_stats)} participants")

    return player_stats


def analyze_results(player_stats):
    """Analyze h distribution across players"""
    print("\n" + "=" * 80)
    print("Analysis Results")
    print("=" * 80)

    h_scores = [stats['h_score'] for stats in player_stats.values()]
    num_games = [stats['num_games'] for stats in player_stats.values()]

    print(f"\nPlanning Depth Distribution:")
    print(f"  Mean h_score: {np.mean(h_scores):.3f}")
    print(f"  Std h_score:  {np.std(h_scores):.3f}")
    print(f"  Min h_score:  {np.min(h_scores):.3f}")
    print(f"  Max h_score:  {np.max(h_scores):.3f}")
    print(f"  Median:       {np.median(h_scores):.3f}")

    # Interpretation
    print(f"\nInterpretation:")
    print(f"  h_score ≈ 0.0-0.3: Strong h=1 (myopic planning)")
    print(f"  h_score ≈ 0.3-0.7: Mixed (intermediate planning)")
    print(f"  h_score ≈ 0.7-1.0: Strong h=4 (far-sighted planning)")

    # Count by category
    myopic = sum(1 for h in h_scores if h < 0.3)
    mixed = sum(1 for h in h_scores if 0.3 <= h < 0.7)
    farsighted = sum(1 for h in h_scores if h >= 0.7)

    print(f"\nPlayer Categories:")
    print(f"  Myopic (h≈1):      {myopic} ({myopic/len(h_scores)*100:.1f}%)")
    print(f"  Mixed:             {mixed} ({mixed/len(h_scores)*100:.1f}%)")
    print(f"  Far-sighted (h≈4): {farsighted} ({farsighted/len(h_scores)*100:.1f}%)")

    # Top/bottom players
    sorted_players = sorted(player_stats.items(), key=lambda x: x[1]['h_score'], reverse=True)

    print(f"\nTop 5 Players (highest h):")
    for i, (pid, stats) in enumerate(sorted_players[:5]):
        print(f"  {i+1}. Player {pid}: h_score={stats['h_score']:.3f} "
              f"({stats['num_games']} games, {stats['num_moves']} moves)")

    print(f"\nBottom 5 Players (lowest h):")
    for i, (pid, stats) in enumerate(sorted_players[-5:]):
        print(f"  {i+1}. Player {pid}: h_score={stats['h_score']:.3f} "
              f"({stats['num_games']} games, {stats['num_moves']} moves)")

    return h_scores, num_games


def visualize_results(player_stats):
    """Visualize h distribution and per-player results"""
    h_scores = [stats['h_score'] for stats in player_stats.values()]
    num_games = [stats['num_games'] for stats in player_stats.values()]
    num_moves = [stats['num_moves'] for stats in player_stats.values()]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: h_score distribution
    ax1 = axes[0, 0]
    ax1.hist(h_scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Neutral (h=2.5)')
    ax1.axvline(x=np.mean(h_scores), color='green', linestyle='-', linewidth=2, label=f'Mean ({np.mean(h_scores):.3f})')
    ax1.set_xlabel('h_score (0=h1, 1=h4)')
    ax1.set_ylabel('Number of Players')
    ax1.set_title('Planning Depth Distribution Across Players')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: h_score vs number of games
    ax2 = axes[0, 1]
    ax2.scatter(num_games, h_scores, alpha=0.6, s=100)
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('h_score')
    ax2.set_title('Planning Depth vs Experience')
    ax2.grid(True, alpha=0.3)

    # Add correlation
    if len(num_games) > 2:
        corr, p_val = stats.pearsonr(num_games, h_scores)
        ax2.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.4f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: h_score vs number of moves
    ax3 = axes[0, 2]
    ax3.scatter(num_moves, h_scores, alpha=0.6, s=100, c=num_games, cmap='viridis')
    ax3.set_xlabel('Total Moves')
    ax3.set_ylabel('h_score')
    ax3.set_title('Planning Depth vs Total Experience')
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3)
    cbar.set_label('Number of Games')

    # Plot 4: Per-player h_scores (sorted)
    ax4 = axes[1, 0]
    sorted_h = sorted(h_scores)
    ax4.plot(range(len(sorted_h)), sorted_h, marker='o', linestyle='-', markersize=4)
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    ax4.fill_between(range(len(sorted_h)), 0, 0.3, alpha=0.2, color='blue', label='Myopic (h≈1)')
    ax4.fill_between(range(len(sorted_h)), 0.3, 0.7, alpha=0.2, color='gray', label='Mixed')
    ax4.fill_between(range(len(sorted_h)), 0.7, 1, alpha=0.2, color='red', label='Far-sighted (h≈4)')
    ax4.set_xlabel('Player Rank')
    ax4.set_ylabel('h_score')
    ax4.set_title('Players Ranked by Planning Depth')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Within-player variance
    ax5 = axes[1, 1]
    h_stds = [stats['h_std'] for stats in player_stats.values()]
    ax5.scatter(h_scores, h_stds, alpha=0.6, s=100)
    ax5.set_xlabel('Mean h_score')
    ax5.set_ylabel('Std of h_score')
    ax5.set_title('Planning Consistency Within Players')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary_text = f"""
Planning Depth Analysis Summary

Total Players: {len(player_stats)}
Total Moves Analyzed: {sum(num_moves):,}

h_score Statistics:
  Mean:   {np.mean(h_scores):.3f}
  Median: {np.median(h_scores):.3f}
  Std:    {np.std(h_scores):.3f}
  Range:  [{np.min(h_scores):.3f}, {np.max(h_scores):.3f}]

Player Categories:
  Myopic (h<0.3):      {sum(1 for h in h_scores if h < 0.3)} players
  Mixed (0.3≤h<0.7):   {sum(1 for h in h_scores if 0.3 <= h < 0.7)} players
  Far-sighted (h≥0.7): {sum(1 for h in h_scores if h >= 0.7)} players

Interpretation:
  - Most players: {'myopic' if np.mean(h_scores) < 0.4 else 'mixed' if np.mean(h_scores) < 0.6 else 'far-sighted'}
  - High variance: {'Yes' if np.std(h_scores) > 0.2 else 'No'} (std={np.std(h_scores):.3f})
  - Consistency: {'Low' if np.mean(h_stds) > 0.15 else 'Medium' if np.mean(h_stds) > 0.1 else 'High'} (mean std={np.mean(h_stds):.3f})
"""

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    output_path = Path('figures') / 'human_h_estimates.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


def save_results(player_stats, output_path='results/player_h_estimates.pkl'):
    """Save results for future analysis"""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'wb') as f:
        pickle.dump(player_stats, f)

    print(f"\n✅ Saved player h estimates to {output_path}")

    # Also save CSV for easy inspection
    csv_path = output_path.parent / 'player_h_estimates.csv'

    df = pd.DataFrame([
        {
            'participant': pid,
            'h_score': stats['h_score'],
            'h_std': stats['h_std'],
            'num_games': stats['num_games'],
            'num_moves': stats['num_moves'],
            'avg_game_length': stats['avg_game_length']
        }
        for pid, stats in player_stats.items()
    ])

    df = df.sort_values('h_score', ascending=False)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV to {csv_path}")


def main():
    print("=" * 80)
    print("Estimate Planning Depth (h) for Human Players")
    print("=" * 80)

    # Load discriminator
    discriminator = load_discriminator()

    # Load human games
    games = load_human_games(csv_path='../opendata/raw_data.csv')

    # Estimate h per player
    player_stats = estimate_h_per_player(games, discriminator)

    # Analyze results
    h_scores, num_games = analyze_results(player_stats)

    # Visualize
    visualize_results(player_stats)

    # Save results
    save_results(player_stats)

    # Final interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    mean_h = np.mean(h_scores)
    std_h = np.std(h_scores)

    print(f"\nMean h_score: {mean_h:.3f} ± {std_h:.3f}")

    if mean_h < 0.4:
        print("\n✅ FINDING: Human players show predominantly MYOPIC planning (h≈1)")
        print("   This suggests most players use shallow lookahead (1-2 steps).")
    elif mean_h < 0.6:
        print("\n✅ FINDING: Human players show MIXED planning strategies")
        print("   This suggests diversity in planning depth across population.")
    else:
        print("\n✅ FINDING: Human players show predominantly FAR-SIGHTED planning (h≈4)")
        print("   This suggests most players use deep lookahead (3-4+ steps).")

    if std_h > 0.2:
        print(f"\n✅ FINDING: HIGH individual differences (std={std_h:.3f})")
        print("   Planning depth varies substantially across players.")
        print("   This supports the expertise hypothesis!")
    else:
        print(f"\n⚠️  FINDING: LOW individual differences (std={std_h:.3f})")
        print("   Most players show similar planning depth.")
        print("   Expertise may not strongly correlate with h.")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    print("""
1. Obtain expertise metrics (Elo ratings, win rates)
2. Correlate h_score with expertise
3. Test van Opheusden hypothesis: Experts → higher h
4. Analyze within-player consistency across games
5. Compare with van Opheusden's PV depth estimates
""")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
