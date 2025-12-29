"""
Compare Action Distributions from Separate h-Specific Models

THE ULTIMATE TEST: Option B3 (Separate Encoders)

Does eliminating h-interference create measurably different behaviors?

Success criteria: KL divergence > 0.1

Usage:
    python3 compare_separate_h_distributions.py
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_actions(h, data_dir='data/separate_h_trajectories'):
    """Load actions for given h"""
    filepath = Path(data_dir) / f'actions_h{h}.pkl'
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
    print("Separate h-Specific Models Distribution Comparison")
    print("Option B3: Eliminate h-interference")
    print("=" * 80)

    # Load distributions
    print("\n[1. Loading Generated Actions]")

    actions_h1 = load_actions(1)
    actions_h4 = load_actions(4)

    dist_h1, counts_h1 = compute_distribution(actions_h1)
    dist_h4, counts_h4 = compute_distribution(actions_h4)

    entropy_h1 = -np.sum(dist_h1 * np.log2(dist_h1 + 1e-10))
    entropy_h4 = -np.sum(dist_h4 * np.log2(dist_h4 + 1e-10))

    print(f"\nh=1:")
    print(f"  Total actions: {len(actions_h1)}")
    print(f"  Unique actions: {len(counts_h1)}")
    print(f"  Entropy: {entropy_h1:.3f} bits")
    print(f"  Top 5 actions: {counts_h1.most_common(5)}")

    print(f"\nh=4:")
    print(f"  Total actions: {len(actions_h4)}")
    print(f"  Unique actions: {len(counts_h4)}")
    print(f"  Entropy: {entropy_h4:.3f} bits")
    print(f"  Top 5 actions: {counts_h4.most_common(5)}")

    # Compute divergences
    print(f"\n[2. Divergence Metrics]")

    kl_h1_h4, kl_h4_h1, js = compute_divergence(dist_h1, dist_h4)

    print(f"\nKL(h=1 || h=4): {kl_h1_h4:.4f}")
    print(f"KL(h=4 || h=1): {kl_h4_h1:.4f}")
    print(f"JS divergence:  {js:.4f}")

    # Comparison with previous methods
    print(f"\n[3. Comparison with Previous Methods]")

    comparisons = {
        'Heuristic (beta=1.0)': 0.0024,
        'Heuristic (beta=10.0)': 0.0126,
        'Multi-step IK (zero future)': 0.0319,
        'Multi-step IK (rollout)': 0.0399,
        'Separate h models (Option B3)': kl_h1_h4
    }

    print(f"\nKL Divergence (h=1 vs h=4) History:")
    for method, kl in comparisons.items():
        status = "✅ PASS" if kl > 0.1 else "❌ FAIL"
        improvement = kl / 0.0024
        print(f"  {method:40s}: {kl:.4f} ({improvement:5.1f}x) {status}")

    # FINAL DECISION
    print(f"\n{'='*80}")
    print("FINAL DECISION")
    print("=" * 80)

    threshold = 0.1

    if kl_h1_h4 > threshold:
        print(f"\n🎉🎉🎉 SUCCESS! 🎉🎉🎉")
        print(f"\nKL divergence: {kl_h1_h4:.4f} (threshold: {threshold})")
        print(f"JS divergence: {js:.4f}")
        print(f"\nOption B3 (Separate Encoders) WORKS!")
        print(f"\nKey findings:")
        print(f"  ✓ Eliminating h-interference creates distinct behaviors")
        print(f"  ✓ h=1 model (val acc {77.1}%) generates conservative play")
        print(f"  ✓ h=4 model (val acc {14.9}%) generates exploratory play")
        print(f"  ✓ Low h=4 accuracy actually HELPS create diversity!")
        print(f"\nNext steps:")
        print(f"  → Step 0.3: Pilot AIRL with h=1 vs h=4 policies")
        print(f"  → Test IRL reward identifiability")
        print(f"  → Apply to pedestrian task")
        print(f"  → Publish findings on planning-aware IRL")

    elif kl_h1_h4 > 0.05:
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"\nKL divergence: {kl_h1_h4:.4f}")
        print(f"Improvement: {kl_h1_h4 / 0.0024:.1f}x over baseline")
        print(f"Status: Below threshold but showing promise")
        print(f"\nOptions:")
        print(f"  1. Increase temperature (more exploration)")
        print(f"  2. Generate more episodes (better estimation)")
        print(f"  3. Proceed to AIRL anyway (may still work)")
        print(f"  4. Pivot to pedestrian task")

    else:
        print(f"\n❌ FAILURE")
        print(f"\nKL divergence: {kl_h1_h4:.4f} (threshold: {threshold})")
        print(f"\nEven separate models don't create enough difference.")
        print(f"\nPossible explanations:")
        print(f"  1. Four-in-a-row is fundamentally not sensitive to h")
        print(f"  2. Random rollouts add too much noise")
        print(f"  3. Human data doesn't contain h-variation")
        print(f"\nRecommended pivot:")
        print(f"  → Move to pedestrian task (simpler, more controlled)")
        print(f"  → Or try synthetic experts with known h")

    # Visualization
    visualize_comparison(dist_h1, dist_h4, kl_h1_h4, js,
                        counts_h1, counts_h4,
                        len(actions_h1), len(actions_h4),
                        entropy_h1, entropy_h4,
                        comparisons)

    print("\n" + "=" * 80)


def visualize_comparison(dist_h1, dist_h4, kl, js, counts_h1, counts_h4,
                        n_h1, n_h4, entropy_h1, entropy_h4, comparisons):
    """Comprehensive visualization"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Plot 1: h=1 distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(36), dist_h1, alpha=0.7, color='C0')
    ax1.set_title(f'h=1 (Separate Model)\nH={entropy_h1:.3f} bits, n={n_h1}',
                 fontweight='bold', fontsize=12)
    ax1.set_xlabel('Action')
    ax1.set_ylabel('Probability')
    ax1.grid(True, alpha=0.3)

    # Plot 2: h=4 distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(36), dist_h4, alpha=0.7, color='C3')
    ax2.set_title(f'h=4 (Separate Model)\nH={entropy_h4:.3f} bits, n={n_h4}',
                 fontweight='bold', fontsize=12)
    ax2.set_xlabel('Action')
    ax2.set_ylabel('Probability')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Side-by-side comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(36)
    width = 0.35
    ax3.bar(x - width/2, dist_h1, width, label='h=1', alpha=0.7, color='C0')
    ax3.bar(x + width/2, dist_h4, width, label='h=4', alpha=0.7, color='C3')
    ax3.set_title(f'Overlay (KL={kl:.4f})', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Action')
    ax3.set_ylabel('Probability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Improvement history
    ax4 = fig.add_subplot(gs[1, 0])
    methods = list(comparisons.keys())
    kls = list(comparisons.values())
    colors = ['red' if kl < 0.1 else 'green' for kl in kls]

    bars = ax4.barh(range(len(methods)), kls, color=colors, alpha=0.7)
    ax4.axvline(x=0.1, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax4.set_yticks(range(len(methods)))
    ax4.set_yticklabels([m[:25] for m in methods], fontsize=9)
    ax4.set_xlabel('KL Divergence')
    ax4.set_title('Method Comparison', fontweight='bold', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='x')

    # Plot 5: Top actions comparison
    ax5 = fig.add_subplot(gs[1, 1])
    top5_h1 = [a for a, _ in counts_h1.most_common(5)]
    top5_h4 = [a for a, _ in counts_h4.most_common(5)]

    overlap = len(set(top5_h1) & set(top5_h4))

    text = f"""
Top 5 Actions Comparison

h=1: {top5_h1}
h=4: {top5_h4}

Overlap: {overlap}/5 actions

Model Performance:
  h=1: 77.1% val acc
  h=4: 14.9% val acc

Divergence:
  KL: {kl:.4f}
  JS: {js:.4f}
"""
    ax5.text(0.1, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax5.axis('off')

    # Plot 6: Summary
    ax6 = fig.add_subplot(gs[1, 2])
    status = '✅ SUCCESS' if kl > 0.1 else '⚠️ PARTIAL' if kl > 0.05 else '❌ FAIL'
    color = 'green' if kl > 0.1 else 'orange' if kl > 0.05 else 'red'

    summary = f"""
Option B3: Separate Encoders
{'='*30}

Method: h-specific models
  - No h_onehot
  - No interference

Results:
  KL = {kl:.4f}
  JS = {js:.4f}

Improvement: {kl/0.0024:.1f}x

Status: {status}

{'PROCEED TO AIRL!' if kl > 0.1 else 'Consider alternatives' if kl < 0.05 else 'Marginal success'}
"""

    ax6.text(0.1, 0.5, summary, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    ax6.axis('off')

    plt.suptitle('Multi-Step IK: Separate h-Specific Models (h=1 vs h=4)',
                fontsize=14, fontweight='bold')

    output_path = Path('figures') / 'separate_h_comparison.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


if __name__ == '__main__':
    main()
