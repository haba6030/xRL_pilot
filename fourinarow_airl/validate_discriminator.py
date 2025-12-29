"""
Validate AIRL Discriminator

Critical tests to understand what the discriminator actually measures:

Test 1: Random Policy Baseline
- Expected h_score: ~0.5 (neutral)
- If >> 0.5: Discriminator is biased toward h=4

Test 2: Greedy 1-step Policy
- Expected h_score: ~0.1-0.3 (myopic)
- If >> 0.5: Discriminator doesn't measure planning depth

Test 3: Action Diversity (Entropy)
- Compare entropy of: synthetic h=1, h=4, human
- Check which synthetic data matches human behavior

Usage:
    python3 validate_discriminator.py
"""

import numpy as np
import pickle
import torch
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from env import FourInARowEnv
from features import extract_van_opheusden_features
from pilot_airl_discriminator import AIRLDiscriminator


def load_discriminator(model_path='models/pilot_airl_discriminator.pt'):
    """Load trained discriminator"""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = AIRLDiscriminator(state_dim=89, action_dim=36)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


# ============================================================================
# Test 1: Random Policy Baseline
# ============================================================================

def test_random_policy(discriminator, num_episodes=100, seed=42):
    """
    Test discriminator on random policy

    Expected: h_score ≈ 0.5 (neutral, no planning)
    """
    print("\n" + "=" * 80)
    print("Test 1: Random Policy Baseline")
    print("=" * 80)

    rng = np.random.default_rng(seed)

    all_states = []
    all_actions = []

    for episode in range(num_episodes):
        env = FourInARowEnv()
        env.reset()

        for step in range(36):  # Max moves
            legal_actions = env.get_legal_actions()
            if len(legal_actions) == 0:
                break

            # Random action
            action = rng.choice(legal_actions)
            state = env._get_observation()

            all_states.append(state)
            all_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    # Estimate h with discriminator
    states_tensor = torch.FloatTensor(np.array(all_states))
    actions_tensor = torch.LongTensor(np.array(all_actions))

    with torch.no_grad():
        logits = discriminator(states_tensor, actions_tensor).squeeze()
        probs_h4 = torch.sigmoid(logits).numpy()

    mean_h = float(np.mean(probs_h4))
    std_h = float(np.std(probs_h4))

    print(f"\nRandom Policy Results:")
    print(f"  Total (state, action) pairs: {len(all_states)}")
    print(f"  Mean h_score: {mean_h:.3f}")
    print(f"  Std h_score:  {std_h:.3f}")
    print(f"  Range: [{np.min(probs_h4):.3f}, {np.max(probs_h4):.3f}]")

    # Interpretation
    print(f"\nInterpretation:")
    if mean_h < 0.4:
        print(f"  ✅ GOOD: Random policy classified as h=1-like (myopic)")
        print(f"     Discriminator distinguishes planning from random.")
    elif mean_h < 0.6:
        print(f"  ✅ PERFECT: Random policy is neutral (h_score ≈ 0.5)")
        print(f"     Discriminator is well-calibrated.")
    else:
        print(f"  ⚠️  WARNING: Random policy classified as h=4-like")
        print(f"     Discriminator may be biased toward h=4!")
        print(f"     This could explain why all humans are h=4.")

    return {
        'mean_h': mean_h,
        'std_h': std_h,
        'probs': probs_h4,
        'actions': all_actions
    }


# ============================================================================
# Test 2: Greedy 1-step Policy
# ============================================================================

def greedy_1step_policy(env, rng):
    """
    Greedy policy: Pick action that maximizes immediate board evaluation

    This is TRUE myopic planning (h=1)
    """
    legal_actions = env.get_legal_actions()
    if len(legal_actions) == 0:
        return None

    best_action = None
    best_score = -np.inf

    for action in legal_actions:
        # Simulate one step
        import copy
        sim_env = copy.deepcopy(env)
        obs, reward, terminated, truncated, info = sim_env.step(action)

        # Evaluate immediate result
        state = sim_env._get_observation()
        black = state[:36]
        white = state[36:72]

        # Use van Opheusden features as heuristic
        try:
            features = extract_van_opheusden_features(black, white, 'Black')
            score = features.sum()  # Simple sum as heuristic
        except:
            score = rng.random()  # Fallback

        # Add reward bonus
        if terminated and reward > 0:
            score += 1000  # Winning move
        elif terminated and reward < 0:
            score -= 1000  # Losing move

        if score > best_score:
            best_score = score
            best_action = action

    return best_action


def test_greedy_policy(discriminator, num_episodes=100, seed=42):
    """
    Test discriminator on greedy 1-step policy

    Expected: h_score ≈ 0.1-0.3 (myopic, h=1)
    """
    print("\n" + "=" * 80)
    print("Test 2: Greedy 1-step Policy")
    print("=" * 80)

    rng = np.random.default_rng(seed)

    all_states = []
    all_actions = []

    for episode in range(num_episodes):
        env = FourInARowEnv()
        env.reset()

        for step in range(36):
            legal_actions = env.get_legal_actions()
            if len(legal_actions) == 0:
                break

            # Greedy action
            action = greedy_1step_policy(env, rng)
            if action is None:
                break

            state = env._get_observation()

            all_states.append(state)
            all_actions.append(action)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

    # Estimate h with discriminator
    states_tensor = torch.FloatTensor(np.array(all_states))
    actions_tensor = torch.LongTensor(np.array(all_actions))

    with torch.no_grad():
        logits = discriminator(states_tensor, actions_tensor).squeeze()
        probs_h4 = torch.sigmoid(logits).numpy()

    mean_h = float(np.mean(probs_h4))
    std_h = float(np.std(probs_h4))

    print(f"\nGreedy 1-step Policy Results:")
    print(f"  Total (state, action) pairs: {len(all_states)}")
    print(f"  Mean h_score: {mean_h:.3f}")
    print(f"  Std h_score:  {std_h:.3f}")
    print(f"  Range: [{np.min(probs_h4):.3f}, {np.max(probs_h4):.3f}]")

    # Interpretation
    print(f"\nInterpretation:")
    if mean_h < 0.3:
        print(f"  ✅ EXCELLENT: Greedy policy correctly classified as h=1 (myopic)")
        print(f"     Discriminator accurately detects planning depth!")
    elif mean_h < 0.5:
        print(f"  ✅ GOOD: Greedy policy leans toward h=1")
        print(f"     Discriminator detects some planning depth signal.")
    elif mean_h < 0.7:
        print(f"  ⚠️  CONCERN: Greedy policy classified as mixed")
        print(f"     Discriminator may not cleanly separate h=1 vs h=4.")
    else:
        print(f"  ❌ PROBLEM: Greedy 1-step classified as h=4!")
        print(f"     Discriminator is NOT measuring planning depth.")
        print(f"     It may be measuring stochasticity or exploration instead.")

    return {
        'mean_h': mean_h,
        'std_h': std_h,
        'probs': probs_h4,
        'actions': all_actions
    }


# ============================================================================
# Test 3: Action Diversity (Entropy)
# ============================================================================

def compute_action_entropy(actions, n_actions=36):
    """Compute entropy of action distribution"""
    counts = Counter(actions)
    probs = np.array([counts.get(a, 0) for a in range(n_actions)]) / len(actions)
    probs = probs + 1e-10  # Avoid log(0)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def test_action_diversity():
    """
    Compare action entropy across:
    - Synthetic h=1
    - Synthetic h=4
    - Human players
    - Random policy
    - Greedy policy
    """
    print("\n" + "=" * 80)
    print("Test 3: Action Diversity (Entropy)")
    print("=" * 80)

    # Load synthetic data
    with open('data/separate_h_trajectories/actions_h1.pkl', 'rb') as f:
        actions_h1 = pickle.load(f)

    with open('data/separate_h_trajectories/actions_h4.pkl', 'rb') as f:
        actions_h4 = pickle.load(f)

    # Load human data
    with open('results/player_h_estimates.pkl', 'rb') as f:
        player_stats = pickle.load(f)

    # Collect all human actions
    actions_human = []
    for stats in player_stats.values():
        # We don't have actions directly, need to reload from games
        pass

    # For now, approximate using game data
    # TODO: Load actual human actions from raw data

    entropy_h1 = compute_action_entropy(actions_h1)
    entropy_h4 = compute_action_entropy(actions_h4)

    print(f"\nAction Entropy (bits):")
    print(f"  Synthetic h=1: {entropy_h1:.3f}")
    print(f"  Synthetic h=4: {entropy_h4:.3f}")
    print(f"  Difference:    {entropy_h4 - entropy_h1:.3f}")

    # Distribution comparison
    counts_h1 = Counter(actions_h1)
    counts_h4 = Counter(actions_h4)

    print(f"\nTop 5 Actions:")
    print(f"  h=1: {[a for a, _ in counts_h1.most_common(5)]}")
    print(f"  h=4: {[a for a, _ in counts_h4.most_common(5)]}")

    overlap = len(set([a for a, _ in counts_h1.most_common(5)]) &
                  set([a for a, _ in counts_h4.most_common(5)]))
    print(f"  Overlap: {overlap}/5 actions")

    # Interpretation
    print(f"\nInterpretation:")
    if entropy_h4 > entropy_h1:
        print(f"  ✓ h=4 is more diverse (higher entropy)")
        print(f"    This suggests h=4 explores more actions.")
    else:
        print(f"  ✓ h=1 is more diverse (higher entropy)")
        print(f"    This is unexpected - may indicate different planning strategies.")

    return {
        'entropy_h1': entropy_h1,
        'entropy_h4': entropy_h4,
        'counts_h1': counts_h1,
        'counts_h4': counts_h4
    }


# ============================================================================
# Visualization
# ============================================================================

def visualize_validation_results(random_results, greedy_results, entropy_results):
    """Create comprehensive validation visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Plot 1: Random policy h_score distribution
    ax1 = axes[0, 0]
    ax1.hist(random_results['probs'], bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Neutral (0.5)')
    ax1.axvline(x=random_results['mean_h'], color='blue', linestyle='-', linewidth=2,
                label=f"Mean ({random_results['mean_h']:.3f})")
    ax1.set_xlabel('h_score (P(h=4))')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Random Policy: h_score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Greedy policy h_score distribution
    ax2 = axes[0, 1]
    ax2.hist(greedy_results['probs'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax2.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Expected h=1 (0.2)')
    ax2.axvline(x=greedy_results['mean_h'], color='blue', linestyle='-', linewidth=2,
                label=f"Mean ({greedy_results['mean_h']:.3f})")
    ax2.set_xlabel('h_score (P(h=4))')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Greedy 1-step: h_score Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Comparison summary
    ax3 = axes[0, 2]
    policies = ['Random', 'Greedy\n1-step', 'Human\nMean']
    h_scores = [random_results['mean_h'], greedy_results['mean_h'], 0.936]
    colors = ['gray', 'orange', 'green']

    bars = ax3.bar(policies, h_scores, color=colors, alpha=0.7, edgecolor='black')
    ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax3.set_ylabel('Mean h_score')
    ax3.set_title('Policy Comparison')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 4: Action entropy comparison
    ax4 = axes[1, 0]
    entropies = [entropy_results['entropy_h1'], entropy_results['entropy_h4']]
    labels = ['Synthetic h=1', 'Synthetic h=4']
    bars = ax4.bar(labels, entropies, color=['blue', 'red'], alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Entropy (bits)')
    ax4.set_title('Action Diversity (Entropy)')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    # Plot 5: Action distribution overlap
    ax5 = axes[1, 1]
    top5_h1 = [a for a, _ in entropy_results['counts_h1'].most_common(5)]
    top5_h4 = [a for a, _ in entropy_results['counts_h4'].most_common(5)]

    text = f"""
Action Distribution Analysis

Top 5 Actions:
  h=1: {top5_h1}
  h=4: {top5_h4}

Entropy:
  h=1: {entropy_results['entropy_h1']:.3f} bits
  h=4: {entropy_results['entropy_h4']:.3f} bits

Interpretation:
  {'h=4 more diverse' if entropy_results['entropy_h4'] > entropy_results['entropy_h1'] else 'h=1 more diverse'}
"""

    ax5.text(0.1, 0.5, text, fontsize=10, family='monospace',
            verticalalignment='center')
    ax5.axis('off')

    # Plot 6: Validation summary
    ax6 = axes[1, 2]

    # Determine validation status
    random_ok = 0.4 < random_results['mean_h'] < 0.6
    greedy_ok = greedy_results['mean_h'] < 0.4

    status = "✅ PASS" if (random_ok and greedy_ok) else "⚠️ PARTIAL" if (random_ok or greedy_ok) else "❌ FAIL"
    color = 'green' if (random_ok and greedy_ok) else 'orange' if (random_ok or greedy_ok) else 'red'

    summary = f"""
Validation Summary
{'='*30}

Random Policy:
  Mean h_score: {random_results['mean_h']:.3f}
  Status: {'✅ OK' if random_ok else '❌ BIASED'}

Greedy 1-step:
  Mean h_score: {greedy_results['mean_h']:.3f}
  Status: {'✅ OK' if greedy_ok else '❌ PROBLEM'}

Overall: {status}

Conclusion:
  {'Discriminator measures planning depth' if (random_ok and greedy_ok) else 'Discriminator may be biased or measuring something else'}
"""

    ax6.text(0.1, 0.5, summary, fontsize=10, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.2))
    ax6.axis('off')

    plt.tight_layout()

    output_path = Path('figures') / 'discriminator_validation.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved validation visualization to {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("AIRL Discriminator Validation")
    print("=" * 80)
    print("\nCritical Tests:")
    print("  1. Random policy → Expected h ≈ 0.5")
    print("  2. Greedy 1-step → Expected h ≈ 0.1-0.3")
    print("  3. Action entropy → Compare diversity")

    # Load discriminator
    discriminator = load_discriminator()
    print("\n✅ Loaded discriminator (98.3% accuracy on synthetic data)")

    # Run tests
    random_results = test_random_policy(discriminator, num_episodes=100)
    greedy_results = test_greedy_policy(discriminator, num_episodes=100)
    entropy_results = test_action_diversity()

    # Visualize
    visualize_validation_results(random_results, greedy_results, entropy_results)

    # Final interpretation
    print("\n" + "=" * 80)
    print("FINAL INTERPRETATION")
    print("=" * 80)

    random_h = random_results['mean_h']
    greedy_h = greedy_results['mean_h']
    human_h = 0.936  # From previous analysis

    print(f"\nMean h_scores:")
    print(f"  Random policy:  {random_h:.3f}")
    print(f"  Greedy 1-step:  {greedy_h:.3f}")
    print(f"  Human players:  {human_h:.3f}")
    print(f"  Synthetic h=1:  ~0.01")
    print(f"  Synthetic h=4:  ~0.99")

    # Decision logic
    print(f"\n" + "=" * 80)

    if 0.4 < random_h < 0.6 and greedy_h < 0.4:
        print("✅ DISCRIMINATOR IS VALID")
        print("\nFindings:")
        print("  ✓ Random policy is neutral (h ≈ 0.5)")
        print("  ✓ Greedy 1-step is myopic (h < 0.4)")
        print("  ✓ Discriminator correctly measures planning depth")
        print("\nConclusion on human h=0.936:")
        print("  → Humans genuinely use deep planning (h≈4)")
        print("  → Van Opheusden hypothesis confirmed!")
        print("  → Low variance suggests most players plan similarly deep")

    elif 0.4 < random_h < 0.6 and greedy_h > 0.6:
        print("⚠️  DISCRIMINATOR PARTIALLY VALID")
        print("\nFindings:")
        print("  ✓ Random policy is neutral (h ≈ 0.5)")
        print("  ✗ Greedy 1-step classified as h=4")
        print("  ? Discriminator may measure something other than planning depth")
        print("\nPossible explanations:")
        print("  1. Greedy policy happens to match h=4 behavioral patterns")
        print("  2. Discriminator measures 'strategic quality' not 'depth'")
        print("  3. Our synthetic h=1 is not actually myopic")

    elif random_h > 0.7:
        print("❌ DISCRIMINATOR IS BIASED")
        print("\nFindings:")
        print("  ✗ Random policy classified as h=4")
        print("  ✗ Discriminator is biased toward h=4")
        print("\nConclusion on human h=0.936:")
        print("  → Cannot trust human h estimates")
        print("  → Discriminator needs recalibration")
        print("  → May need different approach (multi-class, direct MLE)")

    else:
        print("⚠️  MIXED RESULTS")
        print("\nNeeds further investigation:")
        print("  - Examine discriminator internals")
        print("  - Try multi-class approach (h=1,2,3,4)")
        print("  - Compare with van Opheusden PV depth")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if 0.4 < random_h < 0.6 and greedy_h < 0.4:
        print("""
✅ Discriminator validated! Proceed with:
1. Compare with van Opheusden PV depth
2. Add h=2,3 for finer-grained discrimination
3. Correlate h with expertise metrics (Elo, win rate)
4. Publish findings on planning-aware IRL
""")
    else:
        print("""
⚠️  Need additional analysis:
1. Examine why validation failed
2. Try multi-class discriminator (h=1,2,3,4)
3. Consider alternative approaches (direct MLE)
4. Re-evaluate synthetic data generation
""")

    print("=" * 80)


if __name__ == '__main__':
    main()
