"""
Generate Trajectories using Separate h-Specific Models

Uses h-specific models trained in train_separate_h_models.py

Key difference:
- model_h1 for h=1 trajectories
- model_h4 for h=4 trajectories
- Input: [state_current (89), state_future (89)] - NO h_onehot

Usage:
    python3 generate_trajectories_separate_h.py --h 1
    python3 generate_trajectories_separate_h.py --h 4
"""

import numpy as np
import pickle
from pathlib import Path
import argparse
import joblib
import sys
import copy
sys.path.append('.')

from env import FourInARowEnv


class SeparateHAgent:
    """Agent using h-specific model (no h_onehot)"""

    def __init__(self, model_path, h, temperature=1.0):
        """
        Args:
            model_path: Path to h-specific model
            h: Planning depth (for rollout simulation)
            temperature: Softmax temperature
        """
        checkpoint = joblib.load(model_path)
        self.model = checkpoint['model']
        self.h = h
        self.temperature = temperature

        print(f"Loaded h={h} specific model from {model_path}")
        print(f"  Train acc: {checkpoint['train_acc']:.3f}")
        print(f"  Val acc: {checkpoint['val_acc']:.3f}")
        print(f"Temperature: {temperature}")

    def rollout_future_state(self, env, action, rng):
        """Simulate h-step rollout"""
        sim_env = copy.deepcopy(env)
        sim_env.step(action)

        for step in range(self.h - 1):
            legal_actions = sim_env.get_legal_actions()
            if len(legal_actions) == 0:
                break
            sim_action = rng.choice(legal_actions)
            obs, reward, terminated, truncated, info = sim_env.step(sim_action)
            if terminated or truncated:
                break

        return sim_env._get_observation()

    def select_action(self, env, rng):
        """Select action using h-specific model with rollout"""
        legal_actions = env.get_legal_actions()

        if len(legal_actions) == 0:
            return None, {}

        if len(legal_actions) == 1:
            return legal_actions[0], {'strategy': 'forced'}

        current_state = env._get_observation()  # (89,)

        # Score each legal action
        action_scores = np.zeros(36)
        action_scores[:] = -np.inf

        for action in legal_actions:
            # Simulate h-step future
            future_state = self.rollout_future_state(env, action, rng)

            # Prepare features (NO h_onehot!)
            features = np.concatenate([current_state, future_state])  # (178,)
            features = features.reshape(1, -1)

            # Get action probability from h-specific model
            if hasattr(self.model, 'predict_proba'):
                probs_all = self.model.predict_proba(features)[0]  # (36,)
                score = probs_all[action]
            else:
                score = 1.0 / len(legal_actions)

            action_scores[action] = score

        # Softmax over legal actions
        legal_scores = action_scores[legal_actions]
        logits = np.log(legal_scores + 1e-10) / self.temperature
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Sample action
        action_idx = rng.choice(len(legal_actions), p=probs)
        selected_action = legal_actions[action_idx]

        info = {
            'legal_actions': legal_actions,
            'action_scores': action_scores,
            'probs': probs,
            'strategy': f'separate_h{self.h}_model'
        }

        return selected_action, info


def generate_episode(agent, rng, max_steps=36):
    """Generate one episode"""
    env = FourInARowEnv()
    env.reset()

    trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'terminated': False,
        'truncated': False,
        'length': 0
    }

    for step in range(max_steps):
        obs = env._get_observation()
        trajectory['observations'].append(obs)

        action, info = agent.select_action(env, rng)

        if action is None:
            break

        trajectory['actions'].append(action)

        obs, reward, terminated, truncated, info_env = env.step(action)
        trajectory['rewards'].append(reward)

        if terminated or truncated:
            trajectory['observations'].append(env._get_observation())
            trajectory['terminated'] = terminated
            trajectory['truncated'] = truncated
            break

    trajectory['length'] = len(trajectory['actions'])

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, required=True, choices=[1, 4],
                        help='Planning depth')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 80)
    print(f"Generate Trajectories with Separate h={args.h} Model")
    print("=" * 80)

    rng = np.random.default_rng(args.seed)

    # Load h-specific model
    model_path = Path('models/separate_h') / f'model_h{args.h}.pkl'

    agent = SeparateHAgent(
        model_path=model_path,
        h=args.h,
        temperature=args.temperature
    )

    # Generate episodes
    print(f"\nGenerating {args.num_episodes} episodes...")

    trajectories = []
    all_actions = []

    for ep in range(args.num_episodes):
        traj = generate_episode(agent, rng)
        trajectories.append(traj)
        all_actions.extend(traj['actions'])

        if (ep + 1) % 20 == 0:
            avg_length = np.mean([t['length'] for t in trajectories])
            print(f"  Episode {ep+1}/{args.num_episodes}, avg length: {avg_length:.1f}")

    # Statistics
    print(f"\n{'='*80}")
    print("Generation Statistics")
    print("=" * 80)

    lengths = [t['length'] for t in trajectories]
    print(f"\nEpisode lengths:")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Std: {np.std(lengths):.1f}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")

    print(f"\nTotal actions: {len(all_actions)}")
    print(f"Unique actions: {len(set(all_actions))}")

    # Save
    output_dir = Path('data/separate_h_trajectories')
    output_dir.mkdir(exist_ok=True, parents=True)

    traj_path = output_dir / f'trajectories_h{args.h}.pkl'
    actions_path = output_dir / f'actions_h{args.h}.pkl'

    with open(traj_path, 'wb') as f:
        pickle.dump(trajectories, f)

    with open(actions_path, 'wb') as f:
        pickle.dump(all_actions, f)

    print(f"\n✅ Saved {len(trajectories)} trajectories to {traj_path}")
    print(f"✅ Saved {len(all_actions)} actions to {actions_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == '__main__':
    main()
