"""
Compare Action Distributions from Multi-Step IK Policies

THE MOMENT OF TRUTH:
Does learning h-specific policies via multi-step IK create measurably different behaviors?

Success criteria: KL divergence > 0.1

Usage:
    python3 compare_ik_policy_distributions.py
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_actions(h, model='mlp', data_dir='data/multistep_ik_trajectories'):
    """Load actions for given h"""
    filepath = Path(data_dir) / f'actions_h{h}_{model}.pkl'
    with open(filepath, 'rb') as f:
        actions = pickle.load(f)
    return actions


def compute_distribution(actions, n_actions=36):
    """Compute empirical action distribution"""
    counts = Counter(actions)
    dist = np.zeros(n_actions)
    total = len(actions)

    for action, count in counts.items():
        dist[action] = count / total

    # Add epsilon and normalize
    dist = dist + 1e-10
    dist = dist / dist.sum()

    return dist, counts


def compute_divergence(p, q):
    """Compute KL and JS divergence"""
    kl_pq = np.sum(rel_entr(p, q))
    kl_qp = np.sum(rel_entr(q, p))
    js = jensenshannon(p, q, base=2) ** 2

    return kl_pq, kl_qp, js


def main():
    print("=" * 80)
    print("Multi-Step IK Policy Distribution Comparison")
    print("=" * 80)

    h_values = [1, 2, 3, 4]
    distributions = {}
    action_counts = {}
    n_actions_dict = {}

    # Load all distributions
    print("\n[1. Loading Generated Actions]")
    for h in h_values:
        actions = load_actions(h, model='mlp')
        dist, counts = compute_distribution(actions)

        entropy = -np.sum(dist * np.log2(dist + 1e-10))

        distributions[h] = dist
        action_counts[h] = counts
        n_actions_dict[h] = len(actions)

        print(f"\nh = {h}:")
        print(f"  Total actions: {len(actions)}")
        print(f"  Unique actions: {len(counts)}")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Top 5 actions: {counts.most_common(5)}")

    # Compute pairwise divergences
    print(f"\n[2. Pairwise Divergence Matrix]")
    n = len(h_values)
    kl_matrix = np.zeros((n, n))
    js_matrix = np.zeros((n, n))

    for i, h1 in enumerate(h_values):
        for j, h2 in enumerate(h_values):
            if i == j:
                kl_matrix[i, j] = 0.0
                js_matrix[i, j] = 0.0
            else:
                dist1 = distributions[h1]
                dist2 = distributions[h2]
                kl_pq, kl_qp, js = compute_divergence(dist1, dist2)
                kl_matrix[i, j] = kl_pq
                js_matrix[i, j] = js

    # Print matrices
    print("\nKL Divergence Matrix:")
    print("      ", end="")
    for h in h_values:
        print(f"h={h:1d}    ", end="")
    print()
    for i, h1 in enumerate(h_values):
        print(f"h={h1}: ", end="")
        for j in range(n):
            print(f"{kl_matrix[i,j]:.4f} ", end="")
        print()

    print("\nJS Divergence Matrix:")
    print("      ", end="")
    for h in h_values:
        print(f"h={h:1d}    ", end="")
    print()
    for i, h1 in enumerate(h_values):
        print(f"h={h1}: ", end="")
        for j in range(n):
            print(f"{js_matrix[i,j]:.4f} ", end="")
        print()

    # Adjacent contrasts
    print(f"\n[3. Adjacent Contrasts (h vs h+1)]")
    for i in range(n - 1):
        h1 = h_values[i]
        h2 = h_values[i + 1]

        if h2 == h1 + 1:
            kl = kl_matrix[i, i+1]
            js = js_matrix[i, i+1]
            status = "✅ PASS" if kl > 0.1 else "❌ FAIL"

            print(f"\nh={h1} vs h={h2}:")
            print(f"  KL divergence: {kl:.4f} {status}")
            print(f"  JS divergence: {js:.4f}")

    # Find best pair
    max_kl = 0
    best_pair = None

    for i in range(n):
        for j in range(i+1, n):
            kl = kl_matrix[i, j]
            if kl > max_kl:
                max_kl = kl
                best_pair = (h_values[i], h_values[j])

    print(f"\n[4. Best Pair]")
    print(f"\nHighest KL divergence:")
    print(f"  Pair: h={best_pair[0]} vs h={best_pair[1]}")
    print(f"  KL: {max_kl:.4f}")
    print(f"  Status: {'✅ PASS (proceed to AIRL)' if max_kl > 0.1 else '❌ FAIL (below threshold)'}")

    # FINAL DECISION
    print(f"\n{'='*80}")
    print("FINAL DECISION")
    print("=" * 80)

    if max_kl > 0.1:
        print(f"\n🎉 SUCCESS: Multi-step IK policy learning WORKS!")
        print(f"\n   Best pair: h={best_pair[0]} vs h={best_pair[1]}")
        print(f"   KL divergence: {max_kl:.4f} (threshold: 0.1)")
        print(f"   JS divergence: {js_matrix[h_values.index(best_pair[0]), h_values.index(best_pair[1])]:.4f}")
        print(f"\n   This confirms that:")
        print(f"   ✓ Multi-step IK objective creates h-dependent policies")
        print(f"   ✓ Planning depth h affects learned behavior")
        print(f"   ✓ No heuristics needed!")
        print(f"\nNext steps:")
        print(f"   → Step 0.3: Pilot AIRL with h={best_pair[0]} vs h={best_pair[1]}")
        print(f"   → Test IRL reward identifiability")
        print(f"   → Apply to pedestrian task")

    elif max_kl > 0.05:
        print(f"\n⚠️  PARTIAL SUCCESS: KL={max_kl:.4f} below threshold but shows promise")
        print(f"\nOptions:")
        print(f"   1. Try higher temperature (more exploration)")
        print(f"   2. Generate more trajectories (n>100)")
        print(f"   3. Try h=1 vs h=8 (extreme contrast)")
        print(f"   4. Proceed to AIRL anyway (may still work)")

    else:
        print(f"\n❌ FAILURE: KL={max_kl:.4f} still too low")
        print(f"\nProblem diagnosis:")
        print(f"   - Learned policies are too similar across h values")
        print(f"   - Model may be ignoring h embedding")
        print(f"   - Human data may not have enough h variation")
        print(f"\nOptions:")
        print(f"   1. Pivot to pedestrian task (simpler, more controlled)")
        print(f"   2. Use synthetic expert demonstrations with known h")
        print(f"   3. Try different architecture (deeper network, attention)")

    # Visualization
    visualize_comparison(distributions, kl_matrix, js_matrix, best_pair, max_kl, h_values, n_actions_dict)

    print("\n" + "=" * 80)


def visualize_comparison(distributions, kl_matrix, js_matrix, best_pair, max_kl, h_values, n_actions_dict):
    """Comprehensive visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1-4: Action distributions for each h
    for idx, h in enumerate(h_values):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        dist = distributions[h]
        entropy = -np.sum(dist * np.log2(dist + 1e-10))

        ax.bar(range(36), dist, alpha=0.7, color=f'C{idx}')
        ax.set_title(f'h={h} (H={entropy:.3f} bits, n={n_actions_dict[h]})',
                    fontweight='bold', fontsize=14)
        ax.set_xlabel('Action')
        ax.set_ylabel('Probability')
        ax.grid(True, alpha=0.3)

    # Plot 5: KL divergence heatmap
    ax = axes[0, 2]
    sns.heatmap(kl_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[f'h={h}' for h in h_values],
                yticklabels=[f'h={h}' for h in h_values],
                ax=ax, cbar_kws={'label': 'KL Divergence'})
    ax.set_title('KL Divergence Matrix', fontweight='bold', fontsize=14)

    # Plot 6: Summary text
    ax = axes[1, 2]
    color = 'green' if max_kl > 0.1 else 'orange' if max_kl > 0.05 else 'red'
    status = '✅ SUCCESS' if max_kl > 0.1 else '⚠️ PARTIAL' if max_kl > 0.05 else '❌ FAIL'

    summary = f"""
Multi-Step IK Policy Comparison
{'='*40}

Method: Learned policies via multi-step IK
Model: MLP (scikit-learn)
Trajectories: 100 per h

Best pair:
  h={best_pair[0]} vs h={best_pair[1]}
  KL = {max_kl:.4f}

Status: {status}

{'Proceed to AIRL!' if max_kl > 0.1 else 'Consider alternatives' if max_kl < 0.05 else 'Marginal - proceed with caution'}
"""

    ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
           verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    ax.axis('off')

    plt.tight_layout()

    output_path = Path('figures') / 'ik_policy_comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
