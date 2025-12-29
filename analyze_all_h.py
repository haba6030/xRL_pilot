"""
Cross-h analysis: Compare Option A performance across all planning depths
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Results from the comparison runs
results = {
    'h': [1, 2, 4, 8],
    'kl_divergence': [9.3621, 15.8159, 0.2389, 14.0141],
    'avg_traj_length': [2.1, 2.0, 6.7, 2.0],
    'expert_avg_length': 8.3
}

df = pd.DataFrame(results)

print("=" * 80)
print("Option A: Cross-Planning Depth Analysis")
print("=" * 80)
print(f"\nExpert average trajectory length: {results['expert_avg_length']:.1f} moves")
print("\nOption A Performance by Planning Depth (h):")
print(df.to_string(index=False))

# Find best h
best_h_idx = df['kl_divergence'].idxmin()
best_h = df.loc[best_h_idx, 'h']
best_kl = df.loc[best_h_idx, 'kl_divergence']

print(f"\n{'='*80}")
print(f"BEST PERFORMANCE: h={int(best_h)} (KL={best_kl:.4f})")
print(f"{'='*80}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: KL Divergence by h
ax = axes[0]
bars = ax.bar(df['h'], df['kl_divergence'], color=['lightcoral', 'salmon', 'lightgreen', 'coral'])
bars[best_h_idx].set_color('darkgreen')
ax.set_xlabel('Planning Depth (h)', fontsize=12)
ax.set_ylabel('KL(Expert || Generated)', fontsize=12)
ax.set_title('Expert Match Quality by Planning Depth\n(Lower is Better)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(df['h'])
ax.grid(axis='y', alpha=0.3)

# Annotate values
for i, (h, kl) in enumerate(zip(df['h'], df['kl_divergence'])):
    ax.text(h, kl + 0.5, f'{kl:.2f}', ha='center', va='bottom', fontsize=10)

# Plot 2: Trajectory Length Comparison
ax = axes[1]
x = np.arange(len(df['h']))
width = 0.35

bars1 = ax.bar(x - width/2, df['avg_traj_length'], width, 
               label='Option A Generated', color='skyblue')
bars2 = ax.bar(x + width/2, [results['expert_avg_length']] * len(df), width,
               label='Expert', color='coral', alpha=0.7)

ax.set_xlabel('Planning Depth (h)', fontsize=12)
ax.set_ylabel('Average Trajectory Length (moves)', fontsize=12)
ax.set_title('Trajectory Length: Generated vs Expert', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(df['h'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Annotate
for i, length in enumerate(df['avg_traj_length']):
    ax.text(i - width/2, length + 0.2, f'{length:.1f}', 
            ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('figures/option_a_cross_h_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: figures/option_a_cross_h_analysis.png")
plt.close()

# Key insights
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print(f"{'='*80}")
print(f"\n1. h=4 dramatically outperforms all other planning depths")
print(f"   - KL divergence: {df.loc[df['h']==4, 'kl_divergence'].values[0]:.4f} (vs {df.loc[df['h']!=4, 'kl_divergence'].mean():.2f} avg for others)")
print(f"   - Trajectory length: {df.loc[df['h']==4, 'avg_traj_length'].values[0]:.1f} moves (expert: {results['expert_avg_length']:.1f})")

print(f"\n2. h=1, 2, 8 all produce pathologically short games (~2 moves)")
print(f"   - Suggests these models learned trivial/degenerate strategies")
print(f"   - May need longer training or different hyperparameters")

print(f"\n3. Expert data suggests intermediate planning depth is optimal")
print(f"   - Matches van Opheusden et al. findings on human expertise")
print(f"   - Too shallow (h=1,2) or too deep (h=8) fails to match human behavior")

print(f"\n{'='*80}")
