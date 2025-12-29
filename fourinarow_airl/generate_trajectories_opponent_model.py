"""
Generate Trajectories with Opponent Model Rollout

Key difference from random rollout:
- Random rollout: simulates futures with uniform random policy
- Opponent rollout: simulates futures with LEARNED policy from human games

Expected result:
- E[h] between random rollout (2.87) and rollout-free (1.78)
- More realistic than random, but still has partial mismatch

Usage:
    python3 generate_trajectories_opponent_model.py
"""

import numpy as np
import pickle
import joblib
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from env import FourInARowEnv
from features import extract_van_opheusden_features


def train_opponent_policy(data_path='/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/xRL_pilot/opendata/raw_data.csv'):
    """
    Train opponent policy from human games

    Uses van Opheusden features (17-dim) to predict actions
    This captures human-like behavior better than random

    Returns:
        opponent_policy: trained LogisticRegression model
        accuracy: validation accuracy
    """
    print("\n" + "="*80)
    print("TRAINING OPPONENT POLICY FROM HUMAN GAMES")
    print("="*80)

    # Load human data
    df = pd.read_csv(data_path)
    df = df[df['experiment'] == 'human-vs-human'].copy()

    print(f"\nLoaded {len(df)} moves from human games")

    # Extract features and actions
    X = []
    y = []

    for idx, row in df.iterrows():
        # Parse board state
        black = np.array([int(c) for c in row['black_pieces']], dtype=np.float32)
        white = np.array([int(c) for c in row['white_pieces']], dtype=np.float32)

        # Extract van Opheusden features
        features = extract_van_opheusden_features(
            black,
            white,
            current_player=row['color']
        )

        # Concatenate: [black (36), white (36), features (17)] = 89-dim
        state = np.concatenate([black, white, features])

        X.append(state)
        y.append(row['move'])

    X = np.array(X)
    y = np.array(y)

    print(f"\nFeature shape: {X.shape}")
    print(f"Action shape: {y.shape}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain size: {len(X_train)}")
    print(f"Val size: {len(X_val)}")

    # Train logistic regression (same as h-specific models)
    print("\nTraining LogisticRegression opponent policy...")

    opponent_policy = LogisticRegression(
        max_iter=200,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    opponent_policy.fit(X_train, y_train)

    # Evaluate
    train_pred = opponent_policy.predict(X_train)
    val_pred = opponent_policy.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    print(f"\n{'='*80}")
    print("OPPONENT POLICY RESULTS")
    print(f"{'='*80}")
    print(f"Training accuracy: {train_acc:.3f}")
    print(f"Validation accuracy: {val_acc:.3f}")

    # Save opponent policy
    output_path = 'models/opponent_policy.pkl'
    Path('models').mkdir(exist_ok=True)
    joblib.dump(opponent_policy, output_path)
    print(f"\n✅ Saved opponent policy to {output_path}")

    return opponent_policy, val_acc


def generate_trajectory_with_opponent(env, h_model, h, opponent_policy, max_steps=50):
    """
    Generate one trajectory using h-specific model with opponent rollout

    Key difference from random rollout:
    - Uses learned opponent policy for simulation
    - More realistic futures than random

    Args:
        env: FourInARowEnv
        h_model: trained h-specific inverse model
        h: planning horizon
        opponent_policy: learned policy for opponent simulation
        max_steps: maximum episode length

    Returns:
        states: (T, 89) observation sequences
        actions: (T,) action sequences
    """
    obs = env.reset()
    states = [obs]
    actions = []

    for step in range(max_steps):
        if env.done:
            break

        # Get legal actions
        legal_actions = env.get_legal_actions()

        if len(legal_actions) == 0:
            break

        # Score each legal action using h-step rollout with OPPONENT policy
        action_scores = np.zeros(36)
        action_scores[:] = -np.inf

        for action in legal_actions:
            # Create copy of environment
            env_copy = env.deepcopy()

            # Apply action
            env_copy.step(action)

            # Simulate h-step rollout with OPPONENT policy
            rollout_states = [env_copy._get_observation()]

            for _ in range(h):
                if env_copy.done:
                    break

                # Use opponent policy to select action (not random!)
                rollout_obs = env_copy._get_observation()
                legal_rollout = env_copy.get_legal_actions()

                if len(legal_rollout) == 0:
                    break

                # Predict probabilities with opponent policy
                probs = opponent_policy.predict_proba(rollout_obs.reshape(1, -1))[0]

                # Filter to legal actions and renormalize
                legal_probs = np.zeros(36)
                legal_probs[legal_rollout] = probs[legal_rollout]
                legal_probs /= legal_probs.sum()

                # Sample action from opponent policy
                rollout_action = np.random.choice(36, p=legal_probs)

                env_copy.step(rollout_action)
                rollout_states.append(env_copy._get_observation())

            # Score: use h-specific model to predict action from (s_t, s_{t+h})
            if len(rollout_states) > h:
                state_current = obs
                state_future = rollout_states[h]

                # Concatenate states
                X = np.concatenate([state_current, state_future]).reshape(1, -1)

                # Get probability for this action
                action_prob = h_model.predict_proba(X)[0, action]
                action_scores[action] = action_prob
            else:
                # Rollout terminated early, use lower score
                action_scores[action] = 0.01

        # Select action with softmax over scores
        legal_scores = action_scores[legal_actions]

        # Avoid overflow in exp
        legal_scores = legal_scores - legal_scores.max()
        exp_scores = np.exp(legal_scores)
        probs = exp_scores / exp_scores.sum()

        # Sample action
        action_idx = np.random.choice(len(legal_actions), p=probs)
        action = legal_actions[action_idx]

        # Take action
        obs, reward, done, info = env.step(action)

        states.append(obs)
        actions.append(action)

    states = np.array(states[:-1])  # Remove last state
    actions = np.array(actions)

    return states, actions


def generate_all_trajectories(opponent_policy, h_values=[1, 2, 3, 4], n_episodes=100):
    """Generate trajectories for all h values using opponent model"""

    print("\n" + "="*80)
    print("GENERATING TRAJECTORIES WITH OPPONENT MODEL ROLLOUT")
    print("="*80)

    # Load h-specific models
    print("\nLoading h-specific models...")
    h_models = {}
    for h in h_values:
        model_path = f'models/separate_h/model_h{h}.pkl'
        data = joblib.load(model_path)
        h_models[h] = data['model']
        print(f"  Loaded model h={h} (val_acc={data['val_acc']:.3f})")

    # Generate trajectories for each h
    all_trajectories = {}

    for h in h_values:
        print(f"\n{'='*80}")
        print(f"Generating h={h} trajectories (opponent model rollout)")
        print(f"{'='*80}")

        trajectories = []
        total_actions = 0

        for episode in range(n_episodes):
            env = FourInARowEnv()

            states, actions = generate_trajectory_with_opponent(
                env,
                h_models[h],
                h,
                opponent_policy,
                max_steps=50
            )

            trajectories.append({
                'states': states,
                'actions': actions,
                'h': h,
                'episode': episode
            })

            total_actions += len(actions)

            if (episode + 1) % 20 == 0:
                avg_length = total_actions / (episode + 1)
                print(f"  Episode {episode+1}/{n_episodes} | Avg length: {avg_length:.1f}")

        all_trajectories[h] = trajectories

        print(f"\n✅ Generated {len(trajectories)} trajectories for h={h}")
        print(f"   Total actions: {total_actions}")
        print(f"   Avg episode length: {total_actions / len(trajectories):.1f}")

    # Save trajectories
    output_dir = Path('data/opponent_model_trajectories')
    output_dir.mkdir(exist_ok=True, parents=True)

    for h in h_values:
        output_path = output_dir / f'trajectories_h{h}.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(all_trajectories[h], f)
        print(f"\n✅ Saved h={h} trajectories to {output_path}")

    return all_trajectories


def main():
    """Main execution pipeline"""
    print("="*80)
    print("OPPONENT MODEL ROLLOUT TRAJECTORY GENERATION")
    print("="*80)

    # Step 1: Train opponent policy
    print("\n[1/2] Training opponent policy from human games...")
    opponent_policy, val_acc = train_opponent_policy()

    # Step 2: Generate trajectories
    print("\n[2/2] Generating trajectories with opponent model rollout...")
    all_trajectories = generate_all_trajectories(
        opponent_policy,
        h_values=[1, 2, 3, 4],
        n_episodes=100
    )

    print("\n" + "="*80)
    print("COMPLETED")
    print("="*80)
    print(f"\nOpponent policy accuracy: {val_acc:.3f}")
    print(f"Generated trajectories: {sum(len(t) for t in all_trajectories.values())}")
    print(f"\nNext steps:")
    print("  1. Train discriminator: python3 train_multiclass_discriminator_opponent.py")
    print("  2. Estimate human h: python3 estimate_player_h_opponent_model.py")


if __name__ == '__main__':
    main()
