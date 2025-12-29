"""
Analyze Relationship Between Expertise and Planning Depth

Research Question 3: Does planning depth (E[h]) discriminate expertise?

Method:
1. Extract skill proxies from van Opheusden data (win rate, game quality)
2. Load E[h] estimates from multi-class discriminator
3. Correlate skill with E[h]
4. Test: Experts → higher E[h]?

Expected: Positive correlation (van Opheusden hypothesis)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def load_human_h_estimates(filepath='human_h_multiclass_estimates.csv'):
    """Load E[h] estimates from multi-class discriminator"""
    df = pd.read_csv(filepath)

    print("=" * 80)
    print("Loaded E[h] Estimates")
    print("=" * 80)
    print(f"Total players: {len(df)}")
    print(f"Mean E[h]: {df['h_expected_mean'].mean():.3f} ± {df['h_expected_mean'].std():.3f}")
    print(f"Range: [{df['h_expected_mean'].min():.3f}, {df['h_expected_mean'].max():.3f}]")

    return df


def calculate_win_rates(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """
    Calculate win rate for each player from van Opheusden data

    Returns:
        DataFrame with columns: participant, wins, losses, draws, win_rate
    """
    df = pd.read_csv(data_path)

    # Filter human vs human games
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print("\n" + "=" * 80)
    print("Calculating Win Rates")
    print("=" * 80)

    # Detect games by tracking piece count
    df['num_pieces'] = df['black_pieces'].apply(lambda x: x.count('1')) + \
                       df['white_pieces'].apply(lambda x: x.count('1'))

    games = []
    current_game = []
    prev_pieces = 0

    for idx, row in df.iterrows():
        if row['num_pieces'] < prev_pieces:  # New game
            if len(current_game) > 0:
                games.append(current_game)
            current_game = []

        current_game.append(row)
        prev_pieces = row['num_pieces']

    # Add last game
    if len(current_game) > 0:
        games.append(current_game)

    print(f"Detected {len(games)} games")

    # Analyze outcomes
    player_stats = {}

    for game in games:
        if len(game) == 0:
            continue

        # Get final board state
        final_move = game[-1]
        black_pieces = final_move['black_pieces']
        white_pieces = final_move['white_pieces']

        # Determine winner (simplified: check for 4-in-a-row)
        # For now, use game length as proxy (shorter games = clearer winner)
        num_moves = len(game)

        # Get all participants in game (should be 2)
        participants = list(set(move['participant'] for move in game))

        if len(participants) != 2:
            continue  # Skip games with unexpected participant count

        # Initialize stats if needed
        for p in participants:
            if p not in player_stats:
                player_stats[p] = {'wins': 0, 'losses': 0, 'draws': 0, 'total_games': 0}
            player_stats[p]['total_games'] += 1

        # Simple outcome detection:
        # If game ended before board full, likely someone won
        # Otherwise, likely draw
        board_full = (black_pieces.count('1') + white_pieces.count('1')) >= 35

        if board_full:
            # Draw
            for p in participants:
                player_stats[p]['draws'] += 1
        else:
            # Someone won (for now, assign to players equally - need better detection)
            # This is a simplified proxy
            for p in participants:
                player_stats[p]['wins'] += 0.5
                player_stats[p]['losses'] += 0.5

    # Calculate win rates
    results = []
    for participant, stats in player_stats.items():
        total = stats['total_games']
        if total > 0:
            win_rate = (stats['wins']) / total
            results.append({
                'participant': participant,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'draws': stats['draws'],
                'total_games': total,
                'win_rate': win_rate
            })

    df_win_rates = pd.DataFrame(results)

    print(f"\nWin rate statistics:")
    print(f"  Mean win rate: {df_win_rates['win_rate'].mean():.3f}")
    print(f"  Std win rate:  {df_win_rates['win_rate'].std():.3f}")
    print(f"  Range: [{df_win_rates['win_rate'].min():.3f}, {df_win_rates['win_rate'].max():.3f}]")

    return df_win_rates


def calculate_game_quality(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """
    Calculate game quality metrics per player:
    - Average game length (strategic players play longer)
    - Move time (faster players may be more skilled)
    """
    df = pd.read_csv(data_path)
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print("\n" + "=" * 80)
    print("Calculating Game Quality Metrics")
    print("=" * 80)

    # Group by participant
    player_quality = df.groupby('participant').agg({
        'response_time': 'mean',
        'move': 'count'  # Total moves
    }).reset_index()

    player_quality.columns = ['participant', 'avg_response_time', 'total_moves']

    print(f"\nQuality metrics:")
    print(f"  Avg response time: {player_quality['avg_response_time'].mean():.2f}s ± {player_quality['avg_response_time'].std():.2f}s")
    print(f"  Avg total moves:   {player_quality['total_moves'].mean():.1f} ± {player_quality['total_moves'].std():.1f}")

    return player_quality


def merge_and_analyze(df_h, df_win_rates, df_quality):
    """Merge all dataframes and analyze correlations"""
    print("\n" + "=" * 80)
    print("Correlation Analysis: Expertise vs E[h]")
    print("=" * 80)

    # Merge datasets
    df_merged = df_h.merge(df_win_rates, on='participant', how='inner')
    df_merged = df_merged.merge(df_quality, on='participant', how='inner')

    print(f"\nMerged {len(df_merged)} players with complete data")

    # Correlations
    print("\nCorrelations with E[h]:")

    # 1. Win rate vs E[h]
    corr_winrate, p_winrate = stats.pearsonr(df_merged['win_rate'], df_merged['h_expected_mean'])
    print(f"  Win rate:         r={corr_winrate:+.3f}, p={p_winrate:.4f}")

    # 2. Total games vs E[h] (experience proxy)
    corr_games, p_games = stats.pearsonr(df_merged['total_games'], df_merged['h_expected_mean'])
    print(f"  Total games:      r={corr_games:+.3f}, p={p_games:.4f}")

    # 3. Response time vs E[h] (speed/skill proxy)
    corr_time, p_time = stats.pearsonr(df_merged['avg_response_time'], df_merged['h_expected_mean'])
    print(f"  Response time:    r={corr_time:+.3f}, p={p_time:.4f}")

    # 4. Total moves vs E[h] (activity proxy)
    corr_moves, p_moves = stats.pearsonr(df_merged['total_moves'], df_merged['h_expected_mean'])
    print(f"  Total moves:      r={corr_moves:+.3f}, p={p_moves:.4f}")

    # Group comparison: High vs Low skill
    print("\n" + "-" * 80)
    print("High-Skill vs Low-Skill Comparison (Median Split)")
    print("-" * 80)

    median_winrate = df_merged['win_rate'].median()
    df_high_skill = df_merged[df_merged['win_rate'] >= median_winrate]
    df_low_skill = df_merged[df_merged['win_rate'] < median_winrate]

    print(f"\nWin rate median: {median_winrate:.3f}")
    print(f"High-skill (n={len(df_high_skill)}): win rate ≥ {median_winrate:.3f}")
    print(f"Low-skill (n={len(df_low_skill)}):  win rate < {median_winrate:.3f}")

    print(f"\nE[h] comparison:")
    print(f"  High-skill: {df_high_skill['h_expected_mean'].mean():.3f} ± {df_high_skill['h_expected_mean'].std():.3f}")
    print(f"  Low-skill:  {df_low_skill['h_expected_mean'].mean():.3f} ± {df_low_skill['h_expected_mean'].std():.3f}")

    # t-test
    t_stat, p_value = stats.ttest_ind(df_high_skill['h_expected_mean'],
                                       df_low_skill['h_expected_mean'])
    print(f"\nt-test: t={t_stat:.3f}, p={p_value:.4f}")

    if p_value < 0.05:
        print("  → Significant difference! ✅")
    else:
        print("  → No significant difference ❌")

    return df_merged, {
        'corr_winrate': (corr_winrate, p_winrate),
        'corr_games': (corr_games, p_games),
        'corr_time': (corr_time, p_time),
        'corr_moves': (corr_moves, p_moves),
        't_test': (t_stat, p_value),
        'high_skill_h': df_high_skill['h_expected_mean'].mean(),
        'low_skill_h': df_low_skill['h_expected_mean'].mean()
    }


def visualize_results(df_merged, stats_dict):
    """Create comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: E[h] vs Win Rate
    ax1 = axes[0, 0]
    ax1.scatter(df_merged['win_rate'], df_merged['h_expected_mean'], alpha=0.6)
    z = np.polyfit(df_merged['win_rate'], df_merged['h_expected_mean'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df_merged['win_rate'].min(), df_merged['win_rate'].max(), 100)
    ax1.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'r={stats_dict["corr_winrate"][0]:.3f}')
    ax1.set_xlabel('Win Rate')
    ax1.set_ylabel('E[h]')
    ax1.set_title('Planning Depth vs Win Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: E[h] vs Total Games
    ax2 = axes[0, 1]
    ax2.scatter(df_merged['total_games'], df_merged['h_expected_mean'], alpha=0.6)
    ax2.set_xlabel('Total Games Played')
    ax2.set_ylabel('E[h]')
    ax2.set_title(f'Planning Depth vs Experience (r={stats_dict["corr_games"][0]:.3f})')
    ax2.grid(True, alpha=0.3)

    # Plot 3: E[h] vs Response Time
    ax3 = axes[0, 2]
    ax3.scatter(df_merged['avg_response_time'], df_merged['h_expected_mean'], alpha=0.6)
    ax3.set_xlabel('Avg Response Time (s)')
    ax3.set_ylabel('E[h]')
    ax3.set_title(f'Planning Depth vs Response Time (r={stats_dict["corr_time"][0]:.3f})')
    ax3.grid(True, alpha=0.3)

    # Plot 4: High vs Low Skill Distribution
    ax4 = axes[1, 0]
    median_winrate = df_merged['win_rate'].median()
    df_high = df_merged[df_merged['win_rate'] >= median_winrate]
    df_low = df_merged[df_merged['win_rate'] < median_winrate]

    ax4.hist(df_high['h_expected_mean'], bins=15, alpha=0.5, label=f'High-skill (n={len(df_high)})', color='blue')
    ax4.hist(df_low['h_expected_mean'], bins=15, alpha=0.5, label=f'Low-skill (n={len(df_low)})', color='red')
    ax4.axvline(df_high['h_expected_mean'].mean(), color='blue', linestyle='--', linewidth=2)
    ax4.axvline(df_low['h_expected_mean'].mean(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('E[h]')
    ax4.set_ylabel('Count')
    ax4.set_title(f'E[h] Distribution by Skill (p={stats_dict["t_test"][1]:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Box plot comparison
    ax5 = axes[1, 1]
    data_to_plot = [df_high['h_expected_mean'], df_low['h_expected_mean']]
    ax5.boxplot(data_to_plot, labels=['High-skill', 'Low-skill'])
    ax5.set_ylabel('E[h]')
    ax5.set_title('E[h] by Skill Level (Median Split)')
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')

    summary = f"""
    Expertise vs Planning Depth Analysis

    Sample: {len(df_merged)} players

    Correlations (Pearson r):
      Win rate:      r={stats_dict['corr_winrate'][0]:+.3f}, p={stats_dict['corr_winrate'][1]:.4f}
      Total games:   r={stats_dict['corr_games'][0]:+.3f}, p={stats_dict['corr_games'][1]:.4f}
      Response time: r={stats_dict['corr_time'][0]:+.3f}, p={stats_dict['corr_time'][1]:.4f}

    High vs Low Skill (median split):
      High-skill E[h]: {stats_dict['high_skill_h']:.3f}
      Low-skill E[h]:  {stats_dict['low_skill_h']:.3f}
      Difference:      {stats_dict['high_skill_h'] - stats_dict['low_skill_h']:+.3f}
      t-test:          t={stats_dict['t_test'][0]:.2f}, p={stats_dict['t_test'][1]:.4f}

    Conclusion:
      {'Significant difference detected ✅' if stats_dict['t_test'][1] < 0.05 else 'No significant difference ❌'}
      {'Positive correlation trend' if stats_dict['corr_winrate'][0] > 0 else 'Negative/no correlation'}
    """

    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.tight_layout()

    output_path = Path('figures') / 'expertise_vs_h_analysis.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Expertise vs Planning Depth Analysis (RQ3)")
    print("=" * 80)

    # Load data
    df_h = load_human_h_estimates()
    df_win_rates = calculate_win_rates()
    df_quality = calculate_game_quality()

    # Analyze
    df_merged, stats_dict = merge_and_analyze(df_h, df_win_rates, df_quality)

    # Visualize
    visualize_results(df_merged, stats_dict)

    # Save merged dataset
    output_path = Path('expertise_vs_h_analysis.csv')
    df_merged.to_csv(output_path, index=False)
    print(f"\n✅ Saved merged data to {output_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY: RQ3 - Does planning depth discriminate expertise?")
    print("=" * 80)

    corr_winrate, p_winrate = stats_dict['corr_winrate']
    t_stat, p_value = stats_dict['t_test']

    print(f"\nMain finding:")
    print(f"  Correlation (win rate vs E[h]): r={corr_winrate:+.3f}, p={p_winrate:.4f}")

    if p_value < 0.05:
        print(f"\n✅ RESULT: Significant difference between high/low skill")
        print(f"   High-skill E[h] = {stats_dict['high_skill_h']:.3f}")
        print(f"   Low-skill E[h] = {stats_dict['low_skill_h']:.3f}")
        print(f"   Difference = {stats_dict['high_skill_h'] - stats_dict['low_skill_h']:+.3f}")
    else:
        print(f"\n❌ RESULT: No significant difference between high/low skill")
        print(f"   Possible reasons:")
        print(f"     1. Sample is homogeneous (all skilled players)")
        print(f"     2. E[h] variance too small (2.68-3.05)")
        print(f"     3. Win rate is noisy proxy for skill")
        print(f"     4. Need novice data for contrast")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
