"""
Validate h=4 Model Quality

Critical question: Is 14.9% accuracy meaningful planning, or just noise?

Tests:
1. Win rate: h=1 vs h=4 vs random
2. Action quality: Are h=4 actions better than random?
3. Consistency: Does h=4 model make consistent predictions?
4. Future prediction: Are predicted futures reasonable?

Usage:
    python3 validate_h4_model.py
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
import copy
sys.path.append('.')

from env import FourInARowEnv
from generate_trajectories_separate_h import SeparateHAgent


def play_game(agent1, agent2, rng, max_steps=36):
    """Play one game between two agents"""
    env = FourInARowEnv()
    env.reset()

    agents = [agent1, agent2]
    turn = 0

    for step in range(max_steps):
        current_agent = agents[turn % 2]

        action, info = current_agent.select_action(env, rng)
        if action is None:
            return 'draw', step

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated:
            winner = 'agent1' if turn % 2 == 0 else 'agent2'
            return winner, step

        if truncated:
            return 'draw', step

        turn += 1

    return 'draw', max_steps


def random_agent_select_action(env, rng):
    """Random baseline agent"""
    legal_actions = env.get_legal_actions()
    if len(legal_actions) == 0:
        return None, {}
    return rng.choice(legal_actions), {}


class RandomAgent:
    def select_action(self, env, rng):
        return random_agent_select_action(env, rng)


def test_win_rates(n_games=50):
    """Test 1: Win rates"""
    print("=" * 80)
    print("Test 1: Win Rates")
    print("=" * 80)

    # Load models
    agent_h1 = SeparateHAgent(
        Path('models/separate_h/model_h1.pkl'),
        h=1,
        temperature=1.0
    )

    agent_h4 = SeparateHAgent(
        Path('models/separate_h/model_h4.pkl'),
        h=4,
        temperature=1.0
    )

    agent_random = RandomAgent()

    rng = np.random.default_rng(42)

    # h=1 vs h=4
    print(f"\n[h=1 vs h=4] ({n_games} games)")
    results = {'agent1': 0, 'agent2': 0, 'draw': 0}

    for i in range(n_games):
        winner, steps = play_game(agent_h1, agent_h4, rng)
        results[winner] += 1

    print(f"  h=1 wins: {results['agent1']} ({results['agent1']/n_games*100:.1f}%)")
    print(f"  h=4 wins: {results['agent2']} ({results['agent2']/n_games*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/n_games*100:.1f}%)")

    # h=4 vs random
    print(f"\n[h=4 vs Random] ({n_games} games)")
    results = {'agent1': 0, 'agent2': 0, 'draw': 0}

    for i in range(n_games):
        winner, steps = play_game(agent_h4, agent_random, rng)
        results[winner] += 1

    print(f"  h=4 wins: {results['agent1']} ({results['agent1']/n_games*100:.1f}%)")
    print(f"  Random wins: {results['agent2']} ({results['agent2']/n_games*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/n_games*100:.1f}%)")

    # h=1 vs random (baseline)
    print(f"\n[h=1 vs Random] ({n_games} games)")
    results = {'agent1': 0, 'agent2': 0, 'draw': 0}

    for i in range(n_games):
        winner, steps = play_game(agent_h1, agent_random, rng)
        results[winner] += 1

    print(f"  h=1 wins: {results['agent1']} ({results['agent1']/n_games*100:.1f}%)")
    print(f"  Random wins: {results['agent2']} ({results['agent2']/n_games*100:.1f}%)")
    print(f"  Draws: {results['draw']} ({results['draw']/n_games*100:.1f}%)")


def test_action_quality():
    """Test 2: Are h=4 actions reasonable?"""
    print("\n" + "=" * 80)
    print("Test 2: Action Quality (Sanity Check)")
    print("=" * 80)

    # Load h=4 model
    model_path = Path('models/separate_h/model_h4.pkl')
    checkpoint = joblib.load(model_path)
    model = checkpoint['model']

    # Check if model learns ANYTHING meaningful
    # Test: Does it avoid obviously bad moves?

    print("\nChecking if h=4 model has learned basic patterns...")
    print("  (This is qualitative - just showing it's not random)")

    # Load some h=4 validation data
    data_path = Path('data/multistep_ik/ik_pairs_h4.pkl')
    with open(data_path, 'rb') as f:
        pairs = pickle.load(f)

    # Check prediction consistency
    rng = np.random.default_rng(42)
    sample_pairs = rng.choice(pairs, size=min(10, len(pairs)), replace=False)

    correct = 0
    for pair in sample_pairs:
        state_curr = pair['state_current']
        state_fut = pair['state_future']
        true_action = pair['action']

        features = np.concatenate([state_curr, state_fut]).reshape(1, -1)
        pred_action = model.predict(features)[0]

        if pred_action == true_action:
            correct += 1

    print(f"  Sample accuracy on validation: {correct}/{len(sample_pairs)}")
    print(f"  (Expected: ~14.9% = ~1.5/{len(sample_pairs)})")
    print(f"  Actual: {correct/len(sample_pairs)*100:.1f}%")


def test_prediction_variance():
    """Test 3: Prediction variance with different rollouts"""
    print("\n" + "=" * 80)
    print("Test 3: Prediction Consistency")
    print("=" * 80)

    agent_h4 = SeparateHAgent(
        Path('models/separate_h/model_h4.pkl'),
        h=4,
        temperature=1.0
    )

    # Sample a game state
    env = FourInARowEnv()
    env.reset()
    rng = np.random.default_rng(42)

    # Play a few random moves
    for _ in range(5):
        legal = env.get_legal_actions()
        if len(legal) == 0:
            break
        env.step(rng.choice(legal))

    # Run action selection multiple times
    print("\nRunning action selection 20 times from same state...")
    print("(Due to random rollouts, predictions may vary)")

    action_counts = {}
    for trial in range(20):
        action, info = agent_h4.select_action(env, rng)
        action_counts[action] = action_counts.get(action, 0) + 1

    print(f"\nAction distribution over 20 trials:")
    sorted_actions = sorted(action_counts.items(), key=lambda x: -x[1])
    for action, count in sorted_actions[:5]:
        print(f"  Action {action}: {count}/20 ({count/20*100:.0f}%)")

    # Entropy of this distribution
    probs = np.array([action_counts.get(a, 0) for a in range(36)]) / 20
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(action_counts))

    print(f"\nEntropy: {entropy:.2f} bits (max: {max_entropy:.2f})")
    print(f"Interpretation: {'High variance (close to random)' if entropy > max_entropy*0.8 else 'Low variance (consistent)'}")


def main():
    print("=" * 80)
    print("Validate h=4 Model Quality")
    print("Question: Is 14.9% accuracy meaningful?")
    print("=" * 80)

    # Test 1: Win rates
    test_win_rates(n_games=50)

    # Test 2: Action quality
    test_action_quality()

    # Test 3: Prediction variance
    test_prediction_variance()

    # Summary
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    print("""
If h=4 model is MEANINGFUL:
  ✓ h=4 beats random significantly
  ✓ h=4 shows some consistency in predictions
  ✓ Win rate h=4 < h=1 < random hierarchy makes sense

If h=4 model is just NOISE:
  ✗ h=4 ≈ random in win rate
  ✗ Predictions are completely inconsistent
  ✗ No pattern learned

Next steps depend on results:
  - If meaningful → Proceed to AIRL
  - If noise → Need to improve h=4 model first
""")


if __name__ == '__main__':
    main()
