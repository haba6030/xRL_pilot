"""
Analyze Elo Rating vs Planning Depth (E[h])

Compares Elo-based expertise with inferred planning depth to test:
- RQ3: Does planning depth discriminate expertise?

Uses Elo rating (objective skill measure) instead of win rate.

Usage:
    python3 analyze_elo_vs_h.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path


def load_elo_ratings(elo_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/data/human_elo_ratings.csv'):
    """Load Elo ratings for all participants"""
    df = pd.read_csv(elo_path)
    print(f"Loaded Elo ratings for {len(df)} participants")
    print(f"Elo range: {df['elo'].min():.1f} - {df['elo'].max():.1f}")
    print(f"Median Elo: {df['elo'].median():.1f}")
    return df


def load_h_estimates(h_path='human_h_opponent_estimates.csv'):
    """Load planning depth estimates from multi-class discriminator (OPPONENT MODEL)"""
    df = pd.read_csv(h_path)

    # Extract E[h] (already computed as h_expected_mean)
    print(f"\nLoaded E[h] estimates for {len(df)} participant entries")
    print(f"E[h] range: {df['h_expected_mean'].min():.3f} - {df['h_expected_mean'].max():.3f}")
    print(f"Mean E[h]: {df['h_expected_mean'].mean():.3f}")

    return df


def merge_data(elo_df, h_df):
    """Merge Elo and E[h] data"""
    # FIXED version: No duplicate participants, merge directly
    merged = elo_df.merge(h_df, on='participant', how='inner')

    print(f"\nMerged dataset: {len(merged)} participants")

    return merged


def correlation_analysis(df):
    """Compute correlations between Elo and E[h]"""
    print("\n" + "=" * 80)
    print("Correlation Analysis: Elo vs E[h]")
    print("=" * 80)

    elo = df['elo'].values
    h = df['h_expected_mean'].values

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(elo, h)

    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = stats.spearmanr(elo, h)

    print(f"\nPearson correlation:")
    print(f"  r = {r_pearson:.4f}")
    print(f"  p-value = {p_pearson:.4f}")

    print(f"\nSpearman correlation:")
    print(f"  r = {r_spearman:.4f}")
    print(f"  p-value = {p_spearman:.4f}")

    # Interpretation
    if p_spearman < 0.001:
        sig = "***"
    elif p_spearman < 0.01:
        sig = "**"
    elif p_spearman < 0.05:
        sig = "*"
    else:
        sig = "ns"

    print(f"\nStatistical significance: {sig} (p={p_spearman:.4f})")

    # Effect size interpretation
    abs_r = abs(r_spearman)
    if abs_r < 0.1:
        effect = "negligible"
    elif abs_r < 0.3:
        effect = "small"
    elif abs_r < 0.5:
        effect = "medium"
    else:
        effect = "large"

    print(f"Effect size: {effect} (|r|={abs_r:.3f})")

    return {
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman
    }


def group_comparison(df):
    """Compare E[h] across Elo-based expertise groups"""
    print("\n" + "=" * 80)
    print("Group Comparison: Expert vs Intermediate vs Novice")
    print("=" * 80)

    # Use tertile-based expertise from Elo file
    expert = df[df['expertise'] == 'expert']['h_expected_mean'].values
    intermediate = df[df['expertise'] == 'intermediate']['h_expected_mean'].values
    novice = df[df['expertise'] == 'novice']['h_expected_mean'].values

    print(f"\nExpert (n={len(expert)}): E[h] = {np.mean(expert):.3f} ± {np.std(expert):.3f}")
    print(f"Intermediate (n={len(intermediate)}): E[h] = {np.mean(intermediate):.3f} ± {np.std(intermediate):.3f}")
    print(f"Novice (n={len(novice)}): E[h] = {np.mean(novice):.3f} ± {np.std(novice):.3f}")

    # ANOVA (one-way)
    f_stat, p_anova = stats.f_oneway(expert, intermediate, novice)
    print(f"\nOne-way ANOVA:")
    print(f"  F-statistic = {f_stat:.4f}")
    print(f"  p-value = {p_anova:.4f}")

    # Pairwise t-tests
    print(f"\nPairwise t-tests:")

    # Expert vs Novice
    t_stat, p_val = stats.ttest_ind(expert, novice)
    cohens_d = (np.mean(expert) - np.mean(novice)) / np.sqrt((np.std(expert)**2 + np.std(novice)**2) / 2)
    print(f"  Expert vs Novice: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

    # Expert vs Intermediate
    t_stat, p_val = stats.ttest_ind(expert, intermediate)
    cohens_d = (np.mean(expert) - np.mean(intermediate)) / np.sqrt((np.std(expert)**2 + np.std(intermediate)**2) / 2)
    print(f"  Expert vs Intermediate: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

    # Intermediate vs Novice
    t_stat, p_val = stats.ttest_ind(intermediate, novice)
    cohens_d = (np.mean(intermediate) - np.mean(novice)) / np.sqrt((np.std(intermediate)**2 + np.std(novice)**2) / 2)
    print(f"  Intermediate vs Novice: t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.3f}")

    return {
        'expert_mean': np.mean(expert),
        'intermediate_mean': np.mean(intermediate),
        'novice_mean': np.mean(novice),
        'f_stat': f_stat,
        'p_anova': p_anova
    }


def binary_comparison(df):
    """Binary comparison using median split"""
    print("\n" + "=" * 80)
    print("Binary Comparison: High-Elo vs Low-Elo (Median Split)")
    print("=" * 80)

    median_elo = df['elo'].median()
    print(f"Median Elo: {median_elo:.1f}")

    high_elo = df[df['elo'] >= median_elo]
    low_elo = df[df['elo'] < median_elo]

    print(f"\nHigh-Elo (n={len(high_elo)}): E[h] = {high_elo['h_expected_mean'].mean():.3f} ± {high_elo['h_expected_mean'].std():.3f}")
    print(f"Low-Elo (n={len(low_elo)}): E[h] = {low_elo['h_expected_mean'].mean():.3f} ± {low_elo['h_expected_mean'].std():.3f}")

    # t-test
    t_stat, p_val = stats.ttest_ind(high_elo['h_expected_mean'], low_elo['h_expected_mean'])

    # Cohen's d
    mean_diff = high_elo['h_expected_mean'].mean() - low_elo['h_expected_mean'].mean()
    pooled_std = np.sqrt((high_elo['h_expected_mean'].std()**2 + low_elo['h_expected_mean'].std()**2) / 2)
    cohens_d = mean_diff / pooled_std

    print(f"\nt-test:")
    print(f"  t-statistic = {t_stat:.4f}")
    print(f"  p-value = {p_val:.4f}")
    print(f"  Mean difference = {mean_diff:.4f}")
    print(f"  Cohen's d = {cohens_d:.4f}")

    return {
        'high_elo_mean': high_elo['h_expected_mean'].mean(),
        'low_elo_mean': low_elo['h_expected_mean'].mean(),
        't_stat': t_stat,
        'p_val': p_val,
        'cohens_d': cohens_d
    }


def visualize_results(df, corr_stats, group_stats):
    """Create comprehensive visualizations"""
    print("\n" + "=" * 80)
    print("Creating Visualizations")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot: Elo vs E[h]
    ax = axes[0, 0]

    # Color by expertise
    colors = {'expert': 'red', 'intermediate': 'orange', 'novice': 'blue'}
    for expertise in ['expert', 'intermediate', 'novice']:
        subset = df[df['expertise'] == expertise]
        ax.scatter(subset['elo'], subset['h_expected_mean'],
                  label=expertise.capitalize(), alpha=0.7, s=80,
                  color=colors[expertise])

    # Regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df['elo'], df['h_expected_mean'])
    x_line = np.array([df['elo'].min(), df['elo'].max()])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.5, label=f'Linear fit (r={corr_stats["r_pearson"]:.3f})')

    ax.set_xlabel('Elo Rating')
    ax.set_ylabel('E[h] (Planning Depth)')
    ax.set_title(f'Elo vs Planning Depth\nr={corr_stats["r_spearman"]:.3f}, p={corr_stats["p_spearman"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Box plot: Expertise groups
    ax = axes[0, 1]

    expertise_order = ['novice', 'intermediate', 'expert']
    data_by_expertise = [df[df['expertise'] == exp]['h_expected_mean'].values for exp in expertise_order]

    bp = ax.boxplot(data_by_expertise, labels=['Novice', 'Intermediate', 'Expert'],
                    patch_artist=True)

    # Color boxes
    for patch, color in zip(bp['boxes'], [colors['novice'], colors['intermediate'], colors['expert']]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    ax.set_ylabel('E[h] (Planning Depth)')
    ax.set_title(f'Planning Depth by Expertise\nANOVA: F={group_stats["f_stat"]:.2f}, p={group_stats["p_anova"]:.4f}')
    ax.grid(True, alpha=0.3, axis='y')

    # 3. Histogram: Elo distribution
    ax = axes[1, 0]

    ax.hist(df['elo'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(df['elo'].median(), color='red', linestyle='--', linewidth=2, label='Median')
    ax.axvline(df['elo'].quantile(0.25), color='orange', linestyle='--', linewidth=2, label='Q1/Q3')
    ax.axvline(df['elo'].quantile(0.75), color='orange', linestyle='--', linewidth=2)

    ax.set_xlabel('Elo Rating')
    ax.set_ylabel('Number of participants')
    ax.set_title(f'Elo Distribution (Range: {df["elo"].min():.1f} - {df["elo"].max():.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Histogram: E[h] distribution by expertise
    ax = axes[1, 1]

    for expertise in ['expert', 'intermediate', 'novice']:
        subset = df[df['expertise'] == expertise]
        ax.hist(subset['h_expected_mean'], bins=15, alpha=0.5,
               label=expertise.capitalize(), color=colors[expertise])

    ax.set_xlabel('E[h] (Planning Depth)')
    ax.set_ylabel('Number of participants')
    ax.set_title('Planning Depth Distribution by Expertise')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = 'figures/elo_vs_h_analysis.png'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure: {output_path}")

    plt.show()


def compare_with_winrate(df):
    """Compare Elo-based vs Win-rate-based analysis"""
    print("\n" + "=" * 80)
    print("Comparison: Elo vs Win Rate")
    print("=" * 80)

    # Correlation with win rate
    r_winrate, p_winrate = stats.spearmanr(df['win_rate'], df['h_expected_mean'])

    # Correlation with Elo
    r_elo, p_elo = stats.spearmanr(df['elo'], df['h_expected_mean'])

    print(f"\nCorrelation with E[h]:")
    print(f"  Win Rate: r={r_winrate:.4f}, p={p_winrate:.4f}")
    print(f"  Elo Rating: r={r_elo:.4f}, p={p_elo:.4f}")

    # Which is better predictor?
    print(f"\nWhich is better predictor of E[h]?")
    if abs(r_elo) > abs(r_winrate):
        print(f"  → Elo rating (|r|={abs(r_elo):.3f} > {abs(r_winrate):.3f})")
    else:
        print(f"  → Win rate (|r|={abs(r_winrate):.3f} > {abs(r_elo):.3f})")

    # Elo vs Win rate correlation
    r_elo_winrate, p_elo_winrate = stats.spearmanr(df['elo'], df['win_rate'])
    print(f"\nElo vs Win Rate correlation:")
    print(f"  r={r_elo_winrate:.4f}, p={p_elo_winrate:.4f}")

    return {
        'r_winrate': r_winrate,
        'p_winrate': p_winrate,
        'r_elo': r_elo,
        'p_elo': p_elo
    }


def save_results(df, output_path='results/elo_vs_h_analysis.csv'):
    """Save merged dataset"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved merged data: {output_path}")


def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("Elo Rating vs Planning Depth Analysis")
    print("=" * 80)

    # Load data
    elo_df = load_elo_ratings()
    h_df = load_h_estimates()

    # Merge
    merged_df = merge_data(elo_df, h_df)

    # Correlation analysis
    corr_stats = correlation_analysis(merged_df)

    # Group comparison
    group_stats = group_comparison(merged_df)

    # Binary comparison
    binary_stats = binary_comparison(merged_df)

    # Compare with win rate
    winrate_comparison = compare_with_winrate(merged_df)

    # Visualize
    visualize_results(merged_df, corr_stats, group_stats)

    # Save
    save_results(merged_df)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: RQ3 - Does Planning Depth Discriminate Expertise?")
    print("=" * 80)

    print(f"\nCorrelation (Elo vs E[h]):")
    print(f"  Spearman r = {corr_stats['r_spearman']:.4f}, p = {corr_stats['p_spearman']:.4f}")

    print(f"\nGroup Differences:")
    print(f"  Expert: E[h] = {group_stats['expert_mean']:.3f}")
    print(f"  Intermediate: E[h] = {group_stats['intermediate_mean']:.3f}")
    print(f"  Novice: E[h] = {group_stats['novice_mean']:.3f}")
    print(f"  ANOVA: p = {group_stats['p_anova']:.4f}")

    print(f"\nBinary Split (High vs Low Elo):")
    print(f"  Difference = {binary_stats['high_elo_mean'] - binary_stats['low_elo_mean']:.4f}")
    print(f"  Cohen's d = {binary_stats['cohens_d']:.4f}")
    print(f"  p-value = {binary_stats['p_val']:.4f}")

    # Conclusion
    if corr_stats['p_spearman'] < 0.05:
        print(f"\n✅ CONCLUSION: Elo rating significantly correlates with planning depth")
        print(f"   → Planning depth DOES discriminate expertise (RQ3 supported)")
    else:
        print(f"\n❌ CONCLUSION: No significant correlation between Elo and planning depth")
        print(f"   → Planning depth does NOT discriminate expertise (RQ3 not supported)")


if __name__ == '__main__':
    main()
