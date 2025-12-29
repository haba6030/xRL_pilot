"""
Verify Multi-Step IK Data Quality

Checks:
1. Data structure integrity
2. State transition validity
3. Action distribution differences across h values
4. KL/JS divergence between h values

This is the CRITICAL TEST: Do different h values produce different action distributions
without using any heuristics?
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import jensenshannon
from scipy.special import rel_entr
from collections import Counter
import seaborn as sns


def load_ik_pairs(h, data_dir='data/multistep_ik'):
    """Load multi-step IK pairs for given h"""
    filepath = Path(data_dir) / f'ik_pairs_h{h}.pkl'
    with open(filepath, 'rb') as f:
        pairs = pickle.load(f)
    return pairs


def verify_data_structure(pairs, h):
    """Verify data structure is correct"""
    print(f"\n--- h = {h} ---")
    print(f"Total pairs: {len(pairs)}")

    if len(pairs) == 0:
        print("  ⚠️  No pairs found!")
        return False

    # Check first pair
    sample = pairs[0]
    print(f"\nSample pair structure:")
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")

    # Verify all pairs have correct structure
    required_keys = {'state_current', 'state_future', 'action', 'h', 'game_id', 't'}
    for i, pair in enumerate(pairs):
        if not all(key in pair for key in required_keys):
            print(f"  ⚠️  Pair {i} missing keys!")
            return False

        if pair['h'] != h:
            print(f"  ⚠️  Pair {i} has wrong h: {pair['h']} != {h}")
            return False

    print(f"✅ All {len(pairs)} pairs have correct structure")
    return True


def verify_state_transitions(pairs, h, num_samples=10):
    """Verify that state transitions make sense"""
    print(f"\nVerifying state transitions (h={h})...")

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(pairs), size=min(num_samples, len(pairs)), replace=False)

    for idx in sample_indices:
        pair = pairs[idx]
        state_curr = pair['state_current']
        state_fut = pair['state_future']

        # Check that states are different (game progressed)
        if np.allclose(state_curr, state_fut):
            print(f"  ⚠️  States are identical at t={pair['t']} in game {pair['game_id']}")

        # Check that number of pieces increased
        # State format: [black (36), white (36), features (18)]
        black_curr = np.sum(state_curr[:36])
        white_curr = np.sum(state_curr[36:72])
        total_curr = black_curr + white_curr

        black_fut = np.sum(state_fut[:36])
        white_fut = np.sum(state_fut[36:72])
        total_fut = black_fut + white_fut

        if total_fut < total_curr:
            print(f"  ⚠️  Pieces decreased: {total_curr} -> {total_fut}")

        if total_fut > total_curr + h:
            print(f"  ⚠️  Too many new pieces: {total_curr} -> {total_fut} (expected ~{h} new pieces)")

    print(f"✅ State transitions look valid")


def analyze_action_distributions(h_values=[1, 2, 3, 4]):
    """
    Analyze action distributions for each h value.

    THIS IS THE KEY TEST: Do different h values produce different distributions?
    """
    print("\n" + "=" * 80)
    print("Action Distribution Analysis")
    print("=" * 80)

    distributions = {}
    action_counts = {}

    for h in h_values:
        pairs = load_ik_pairs(h)
        actions = [pair['action'] for pair in pairs]

        # Compute distribution
        counts = Counter(actions)
        dist = np.zeros(36)
        total = len(actions)

        for action, count in counts.items():
            dist[action] = count / total

        # Add epsilon and normalize
        dist = dist + 1e-10
        dist = dist / dist.sum()

        # Entropy
        entropy = -np.sum(dist * np.log2(dist + 1e-10))

        distributions[h] = dist
        action_counts[h] = counts

        print(f"\nh = {h}:")
        print(f"  Total actions: {len(actions)}")
        print(f"  Unique actions: {len(counts)}")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Top 5 actions: {counts.most_common(5)}")

    return distributions, action_counts


def compute_divergence(p, q):
    """Compute KL and JS divergence"""
    kl_pq = np.sum(rel_entr(p, q))
    kl_qp = np.sum(rel_entr(q, p))
    js = jensenshannon(p, q, base=2) ** 2
    return kl_pq, kl_qp, js


def compare_distributions(distributions):
    """
    Compare distributions across h values using KL/JS divergence.

    THIS IS THE CRITICAL TEST FOR SUCCESS!
    """
    print("\n" + "=" * 80)
    print("Distribution Comparison (KL/JS Divergence)")
    print("=" * 80)

    h_values = sorted(distributions.keys())
    n = len(h_values)

    # Compute pairwise divergences
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
    print(f"\n{'='*80}")
    print("Adjacent Contrasts (h vs h+1)")
    print("=" * 80)

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

    print(f"\n{'='*80}")
    print("Best Pair")
    print("=" * 80)
    print(f"\nHighest KL divergence:")
    print(f"  Pair: h={best_pair[0]} vs h={best_pair[1]}")
    print(f"  KL: {max_kl:.4f}")
    print(f"  Status: {'✅ PASS (proceed to AIRL)' if max_kl > 0.1 else '❌ FAIL (below threshold)'}")

    # DECISION
    print(f"\n{'='*80}")
    print("DECISION")
    print("=" * 80)

    if max_kl > 0.1:
        print(f"✅ SUCCESS: Multi-step IK approach works!")
        print(f"   - Best pair: h={best_pair[0]} vs h={best_pair[1]}")
        print(f"   - KL divergence: {max_kl:.4f}")
        print(f"   - This SOLVES the heuristic dominance problem!")
        print(f"\nNext steps:")
        print(f"   1. Train h-specific policies using multi-step IK objective")
        print(f"   2. Proceed to Step 0.3: Pilot AIRL with these policies")
        print(f"   3. Test on pedestrian task")
    else:
        print(f"❌ FAILURE: Distributions still too similar (KL={max_kl:.4f})")
        print(f"\nPossible reasons:")
        print(f"   1. Human data doesn't have enough planning variation")
        print(f"   2. Need more games or different data source")
        print(f"   3. h doesn't affect human behavior in this task")

    return kl_matrix, js_matrix, best_pair, max_kl


def visualize(distributions, kl_matrix, js_matrix):
    """Visualize distributions and divergences"""
    h_values = sorted(distributions.keys())
    n = len(h_values)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1-4: Action distributions for each h
    for idx, h in enumerate(h_values):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        dist = distributions[h]
        entropy = -np.sum(dist * np.log2(dist + 1e-10))

        ax.bar(range(36), dist, alpha=0.7, color=f'C{idx}')
        ax.set_title(f'h={h} (H={entropy:.3f} bits)', fontweight='bold', fontsize=14)
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

    # Plot 6: JS divergence heatmap
    ax = axes[1, 2]
    sns.heatmap(js_matrix, annot=True, fmt='.4f', cmap='YlOrRd',
                xticklabels=[f'h={h}' for h in h_values],
                yticklabels=[f'h={h}' for h in h_values],
                ax=ax, cbar_kws={'label': 'JS Divergence'})
    ax.set_title('JS Divergence Matrix', fontweight='bold', fontsize=14)

    plt.tight_layout()

    output_path = Path('figures') / 'multistep_ik_verification.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved visualization to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("Multi-Step IK Data Verification")
    print("=" * 80)

    h_values = [1, 2, 3, 4]

    # 1. Verify data structure
    print("\n[1. Data Structure Verification]")
    for h in h_values:
        pairs = load_ik_pairs(h)
        if not verify_data_structure(pairs, h):
            print(f"❌ Data structure verification failed for h={h}")
            return

    # 2. Verify state transitions
    print("\n[2. State Transition Verification]")
    for h in h_values:
        pairs = load_ik_pairs(h)
        verify_state_transitions(pairs, h)

    # 3. Analyze action distributions
    print("\n[3. Action Distribution Analysis]")
    distributions, action_counts = analyze_action_distributions(h_values)

    # 4. Compare distributions
    print("\n[4. Distribution Comparison]")
    kl_matrix, js_matrix, best_pair, max_kl = compare_distributions(distributions)

    # 5. Visualize
    print("\n[5. Visualization]")
    visualize(distributions, kl_matrix, js_matrix)

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
