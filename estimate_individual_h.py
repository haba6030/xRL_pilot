"""
Estimate individual planning depth h from human behavioral data

Two approaches:
1. Principal Variation (PV) depth from van Opheusden et al.
2. Behavioral model fitting (simpler: use response time patterns)

Goal: Test if expertise ↔ planning depth correlation exists
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from typing import Dict, List, Tuple
import os

# Load data
print("=" * 80)
print("Estimating Individual Planning Depth from Human Data")
print("=" * 80)

data_path = 'opendata/raw_data.csv'
print(f"\nLoading data from {data_path}...")

df = pd.read_csv(data_path)
print(f"Loaded {len(df)} trials")
print(f"Columns: {list(df.columns)}")

# Basic statistics
print(f"\nDataset overview:")
print(f"  Participants: {df['participant'].nunique()}")
print(f"  Experiments: {df['experiment'].unique()}")
print(f"  Total trials: {len(df)}")

# ═══════════════════════════════════════════════════════
# Approach 1: PV depth proxy (simplified)
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("Approach 1: Response Time as Planning Depth Proxy")
print(f"{'='*80}")

# Group by participant
participant_stats = df.groupby('participant').agg({
    'response_time': ['mean', 'std', 'median', 'count'],
    'experiment': 'first'
}).reset_index()

participant_stats.columns = ['participant', 'rt_mean', 'rt_std', 'rt_median', 'n_trials', 'experiment']

print(f"\nParticipant statistics:")
print(participant_stats.head(10))

# Simple heuristic: Longer response time → deeper planning
# Based on van Opheusden et al.: planning depth correlates with thinking time

# Normalize RT within experiment type (human-vs-human has different time pressure)
for exp in df['experiment'].unique():
    mask = participant_stats['experiment'] == exp
    if mask.sum() > 0:
        rt_values = participant_stats.loc[mask, 'rt_mean']
        # Z-score normalization
        participant_stats.loc[mask, 'rt_z'] = (rt_values - rt_values.mean()) / rt_values.std()

# Map RT z-score to estimated h (heuristic)
# Based on van Opheusden: human experts typically h=3-5
# We'll map:
#   RT z < -1: h=1-2 (very fast, shallow)
#   RT z -1 to 0: h=2-3 (below average)
#   RT z 0 to 1: h=3-4 (above average, expert-like)
#   RT z > 1: h=4-5 (very slow, deep thinking)

def rt_z_to_h(z):
    """Map response time z-score to estimated planning depth"""
    if z < -1:
        return 1.5
    elif z < -0.5:
        return 2
    elif z < 0:
        return 2.5
    elif z < 0.5:
        return 3
    elif z < 1:
        return 3.5
    else:
        return 4

participant_stats['h_estimated'] = participant_stats['rt_z'].apply(rt_z_to_h)

print(f"\nEstimated planning depth distribution:")
print(participant_stats['h_estimated'].describe())
print(f"\nh value counts:")
print(participant_stats['h_estimated'].value_counts().sort_index())

# ═══════════════════════════════════════════════════════
# Approach 2: Elo rating as expertise proxy
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("Approach 2: Elo Rating Calculation")
print(f"{'='*80}")

# For human-vs-human games, we can compute Elo ratings
# Filter to human-vs-human games
hvh_games = df[df['experiment'].str.contains('human', case=False, na=False)]

print(f"\nHuman-vs-human trials: {len(hvh_games)}")

if len(hvh_games) > 0:
    # Simplified Elo: count wins/losses per participant
    # Note: This is simplified - proper Elo requires game-level outcomes
    
    # For now, use response time and trial count as skill proxy
    # (Real Elo would require reconstructing games and outcomes)
    
    print("\nNote: Simplified expertise metric using trial performance")
    print("Proper Elo would require game-level outcome reconstruction")
    
    # Use number of trials as engagement/experience proxy
    participant_stats['experience'] = participant_stats['n_trials']
    
    # Expertise score: composite of experience and response time pattern
    # Hypothesis: Experts are fast (low RT) but not too fast (not random)
    # and have high engagement (many trials)
    
    # Normalize experience
    exp_norm = (participant_stats['experience'] - participant_stats['experience'].min()) / \
               (participant_stats['experience'].max() - participant_stats['experience'].min())
    
    # RT should be moderate (not too fast, not too slow)
    # Inverted U-shape: optimal around z=0
    rt_expertise = 1 - np.abs(participant_stats['rt_z'].fillna(0)) / 2
    rt_expertise = np.clip(rt_expertise, 0, 1)
    
    participant_stats['expertise_score'] = 0.6 * exp_norm + 0.4 * rt_expertise
    
    print(f"\nExpertise score distribution:")
    print(participant_stats['expertise_score'].describe())
else:
    print("\nNo human-vs-human games found, using trial count as expertise proxy")
    participant_stats['expertise_score'] = participant_stats['n_trials'] / participant_stats['n_trials'].max()

# ═══════════════════════════════════════════════════════
# Correlation Analysis: h ↔ Expertise
# ═══════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("Correlation Analysis: Planning Depth h ↔ Expertise")
print(f"{'='*80}")

# Remove NaN values
valid_data = participant_stats.dropna(subset=['h_estimated', 'expertise_score'])

if len(valid_data) > 3:
    # Pearson correlation
    r_pearson, p_pearson = pearsonr(valid_data['h_estimated'], valid_data['expertise_score'])
    
    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = spearmanr(valid_data['h_estimated'], valid_data['expertise_score'])
    
    print(f"\nPearson correlation:")
    print(f"  r = {r_pearson:.3f}, p = {p_pearson:.4f}")
    if p_pearson < 0.05:
        print(f"  ✓ Significant at p < 0.05")
    else:
        print(f"  ✗ Not significant")
    
    print(f"\nSpearman correlation:")
    print(f"  ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")
    if p_spearman < 0.05:
        print(f"  ✓ Significant at p < 0.05")
    else:
        print(f"  ✗ Not significant")
    
    # Split into novice vs expert
    expertise_median = valid_data['expertise_score'].median()
    novices = valid_data[valid_data['expertise_score'] < expertise_median]
    experts = valid_data[valid_data['expertise_score'] >= expertise_median]
    
    print(f"\nNovice vs Expert comparison:")
    print(f"  Novices (n={len(novices)}):")
    print(f"    Mean h: {novices['h_estimated'].mean():.2f} ± {novices['h_estimated'].std():.2f}")
    print(f"    Mean RT: {novices['rt_mean'].mean():.2f}s")
    
    print(f"  Experts (n={len(experts)}):")
    print(f"    Mean h: {experts['h_estimated'].mean():.2f} ± {experts['h_estimated'].std():.2f}")
    print(f"    Mean RT: {experts['rt_mean'].mean():.2f}s")
    
    # Statistical test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(experts['h_estimated'], novices['h_estimated'])
    
    print(f"\n  t-test (experts vs novices):")
    print(f"    t = {t_stat:.3f}, p = {p_value:.4f}")
    if p_value < 0.05:
        if t_stat > 0:
            print(f"    ✓ Experts have significantly higher h")
        else:
            print(f"    ✓ Novices have significantly higher h (unexpected!)")
    else:
        print(f"    ✗ No significant difference in h")
    
    # ═══════════════════════════════════════════════════════
    # Visualization
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("Creating Visualizations")
    print(f"{'='*80}")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Scatter h vs expertise
    ax = axes[0, 0]
    ax.scatter(valid_data['expertise_score'], valid_data['h_estimated'], alpha=0.6, s=50)
    ax.set_xlabel('Expertise Score', fontsize=12)
    ax.set_ylabel('Estimated Planning Depth (h)', fontsize=12)
    ax.set_title(f'Planning Depth vs Expertise\n(r={r_pearson:.3f}, p={p_pearson:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add regression line
    z = np.polyfit(valid_data['expertise_score'], valid_data['h_estimated'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(valid_data['expertise_score'].min(), valid_data['expertise_score'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()
    
    # Plot 2: h distribution by expertise group
    ax = axes[0, 1]
    data_to_plot = [novices['h_estimated'], experts['h_estimated']]
    labels = ['Novice', 'Expert']
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Estimated Planning Depth (h)', fontsize=12)
    ax.set_title(f'Planning Depth: Novice vs Expert\n(p={p_value:.4f})', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Response time vs h
    ax = axes[1, 0]
    ax.scatter(valid_data['rt_mean'], valid_data['h_estimated'], alpha=0.6, s=50, c=valid_data['expertise_score'], cmap='viridis')
    ax.set_xlabel('Mean Response Time (s)', fontsize=12)
    ax.set_ylabel('Estimated Planning Depth (h)', fontsize=12)
    ax.set_title('Response Time vs Planning Depth\n(colored by expertise)', 
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('Expertise Score', fontsize=10)
    
    # Plot 4: h distribution
    ax = axes[1, 1]
    ax.hist(valid_data['h_estimated'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    mean_h = valid_data['h_estimated'].mean()
    ax.axvline(mean_h, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_h:.2f}')
    ax.axvline(4, color='green', linestyle='--', linewidth=2, label='h=4 (expert level)')
    ax.set_xlabel('Estimated Planning Depth (h)', fontsize=12)
    ax.set_ylabel('Number of Participants', fontsize=12)
    ax.set_title('Distribution of Planning Depth Estimates', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('figures', exist_ok=True)
    output_path = 'figures/individual_h_estimation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()
    
    # Save results
    results_path = 'models/individual_h_estimates.csv'
    participant_stats.to_csv(results_path, index=False)
    print(f"✓ Saved individual estimates: {results_path}")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    print(f"\n1. Planning Depth Distribution:")
    print(f"   Mean h: {valid_data['h_estimated'].mean():.2f} ± {valid_data['h_estimated'].std():.2f}")
    print(f"   Range: [{valid_data['h_estimated'].min():.1f}, {valid_data['h_estimated'].max():.1f}]")
    
    print(f"\n2. Expertise ↔ Planning Depth Correlation:")
    if abs(r_pearson) > 0.3 and p_pearson < 0.05:
        print(f"   ✓ Strong correlation found (r={r_pearson:.3f}, p={p_pearson:.4f})")
    elif abs(r_pearson) > 0.1 and p_pearson < 0.05:
        print(f"   ✓ Moderate correlation found (r={r_pearson:.3f}, p={p_pearson:.4f})")
    else:
        print(f"   ⚠️  Weak or no significant correlation (r={r_pearson:.3f}, p={p_pearson:.4f})")
    
    print(f"\n3. Novice vs Expert Difference:")
    h_diff = experts['h_estimated'].mean() - novices['h_estimated'].mean()
    if p_value < 0.05:
        if h_diff > 0:
            print(f"   ✓ Experts plan deeper: +{h_diff:.2f} (p={p_value:.4f})")
            print(f"   → Supports van Opheusden et al. hypothesis")
        else:
            print(f"   ⚠️  Novices plan deeper: {h_diff:.2f} (p={p_value:.4f})")
            print(f"   → Contradicts van Opheusden et al. hypothesis")
    else:
        print(f"   ✗ No significant difference (p={p_value:.4f})")
    
    print(f"\n4. Comparison with AIRL Results:")
    print(f"   AIRL best performance: h=4")
    print(f"   Human expert mean h: {experts['h_estimated'].mean():.2f}")
    if abs(experts['h_estimated'].mean() - 4) < 1:
        print(f"   ✓ Human experts align with AIRL optimal h=4")
    else:
        print(f"   ⚠️  Mismatch between human expert h and AIRL optimal h")
    
    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")

else:
    print("\n⚠️  Insufficient data for correlation analysis")

