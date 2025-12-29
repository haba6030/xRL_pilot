"""
Generate Trajectories using Multi-Step IK Policy

Key challenge: At test time, we don't know the future state!

Solution: Planning via rollout simulation
1. For each legal action, simulate h-step rollout (random or simple policy)
2. Use final state of each rollout as "predicted future state"
3. Policy scores each action based on (current_state, predicted_future, h)
4. Select action via softmax

This implements actual planning using the learned inverse model.

Usage:
    python3 generate_trajectories_with_ik_policy.py --h 1 --num_episodes 100
    python3 generate_trajectories_with_ik_policy.py --h 2 --num_episodes 100
    python3 generate_trajectories_with_ik_policy.py --h 3 --num_episodes 100
    python3 generate_trajectories_with_ik_policy.py --h 4 --num_episodes 100
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


class MultiStepIKAgent:
    """
    Agent that uses multi-step IK policy for action selection.

    Planning mechanism:
    1. Simulate h-step rollouts for each legal action
    2. Use learned policy to score (state_t, simulated_state_{t+h}, action)
    3. Select action via softmax over scores
    """

    def __init__(self, model_path, h, temperature=1.0, rollout_policy='random'):
        """
        Args:
            model_path: Path to trained sklearn model
            h: Planning depth
            temperature: Softmax temperature (higher = more exploration)
            rollout_policy: How to simulate future states ('random' or 'uniform')
        """
        # Load model
        checkpoint = joblib.load(model_path)
        self.model = checkpoint['model']
        self.model_type = checkpoint['model_type']

        self.h = h
        self.temperature = temperature
        self.rollout_policy = rollout_policy

        print(f"Loaded {self.model_type} model from {model_path}")
        print(f"Planning depth: h={h}")
        print(f"Temperature: {temperature}")

    def rollout_future_state(self, env, action, rng):
        """
        Simulate h-step rollout starting with given action.

        This is THE KEY: We need to predict what the state will be h steps ahead
        if we take this action now, to match the training data format.

        Args:
            env: FourInARowEnv (will be deep copied for simulation)
            action: Initial action to take
            rng: numpy random generator

        Returns:
            future_state: (89,) predicted state after h steps
        """
        # Deep copy environment for simulation (doesn't affect original)
        sim_env = copy.deepcopy(env)

        # Take initial action
        obs, reward, terminated, truncated, info = sim_env.step(action)

        # Simulate h-1 more steps with random policy
        for step in range(self.h - 1):
            if terminated or truncated:
                break

            # Select random action for simulation
            legal_actions = sim_env.get_legal_actions()
            if len(legal_actions) == 0:
                break

            if self.rollout_policy == 'random':
                sim_action = rng.choice(legal_actions)
            else:  # uniform (same as random for now)
                sim_action = rng.choice(legal_actions)

            obs, reward, terminated, truncated, info = sim_env.step(sim_action)

        # Get final state after h steps
        future_state = sim_env._get_observation()

        return future_state

    def select_action(self, env, rng):
        """
        Select action using multi-step IK policy with PROPER rollout simulation.

        KEY IDEA (Mhammedi 2023):
        - Model was trained on (state_t, state_{t+h}, action_t) tuples
        - At inference, we simulate h-step rollouts for each legal action
        - Model scores each action based on (current_state, simulated_future_state, h)
        - This creates genuine h-dependent planning behavior!

        Args:
            env: FourInARowEnv
            rng: numpy random generator

        Returns:
            action: int (0-35)
            info: dict with diagnostic information
        """
        legal_actions = env.get_legal_actions()

        if len(legal_actions) == 0:
            return None, {}

        if len(legal_actions) == 1:
            return legal_actions[0], {'strategy': 'forced'}

        # Get current state
        current_state = env._get_observation()  # (89,)

        # For each legal action, simulate h-step future and get model score
        action_scores = np.zeros(36)
        action_scores[:] = -np.inf

        h_onehot = np.zeros(4)
        h_onehot[self.h - 1] = 1.0

        for action in legal_actions:
            # Simulate h-step future starting with this action
            future_state = self.rollout_future_state(env, action, rng)  # (89,)

            # Prepare features for model
            # Features: [state_current (89), state_future (89), h_onehot (4)]
            features = np.concatenate([current_state, future_state, h_onehot])  # (182,)
            features = features.reshape(1, -1)  # (1, 182)

            # Get action probability from model
            if hasattr(self.model, 'predict_proba'):
                probs_all = self.model.predict_proba(features)[0]  # (36,)
                score = probs_all[action]  # Probability of this action given states
            else:
                # Fallback
                score = 1.0 / len(legal_actions)

            action_scores[action] = score

        # Apply temperature and softmax over legal actions
        legal_scores = action_scores[legal_actions]
        logits = np.log(legal_scores + 1e-10) / self.temperature
        logits = logits - np.max(logits)  # Numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)

        # Sample action
        action_idx = rng.choice(len(legal_actions), p=probs)
        selected_action = legal_actions[action_idx]

        info = {
            'legal_actions': legal_actions,
            'action_scores': action_scores,
            'probs': probs,
            'strategy': 'multistep_ik_with_rollout'
        }

        return selected_action, info


def generate_episode(agent, rng, max_steps=36):
    """Generate one episode using the agent"""

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
        # Get observation
        obs = env._get_observation()
        trajectory['observations'].append(obs)

        # Select action
        action, info = agent.select_action(env, rng)

        if action is None:
            # No legal actions
            break

        trajectory['actions'].append(action)

        # Step environment
        obs, reward, terminated, truncated, info_env = env.step(action)
        trajectory['rewards'].append(reward)

        if terminated or truncated:
            # Add final observation
            trajectory['observations'].append(env._get_observation())
            trajectory['terminated'] = terminated
            trajectory['truncated'] = truncated
            break

    trajectory['length'] = len(trajectory['actions'])

    return trajectory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', type=int, required=True, choices=[1, 2, 3, 4],
                        help='Planning depth')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of episodes to generate')
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'rf'],
                        help='Model type')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 80)
    print(f"Generate Trajectories with Multi-Step IK Policy (h={args.h})")
    print("=" * 80)

    # Set random seed
    rng = np.random.default_rng(args.seed)

    # Load model
    model_path = Path('models/multistep_ik') / f'policy_{args.model}.pkl'

    agent = MultiStepIKAgent(
        model_path=model_path,
        h=args.h,
        temperature=args.temperature,
        rollout_policy='random'
    )

    # Generate episodes
    print(f"\nGenerating {args.num_episodes} episodes...")

    trajectories = []
    all_actions = []

    for ep in range(args.num_episodes):
        traj = generate_episode(agent, rng)
        trajectories.append(traj)

        # Collect actions
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

    # Save trajectories
    output_dir = Path('data/multistep_ik_trajectories')
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / f'trajectories_h{args.h}_{args.model}.pkl'

    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)

    print(f"\n✅ Saved {len(trajectories)} trajectories to {output_path}")

    # Save actions for distribution analysis
    actions_path = output_dir / f'actions_h{args.h}_{args.model}.pkl'
    with open(actions_path, 'wb') as f:
        pickle.dump(all_actions, f)

    print(f"✅ Saved {len(all_actions)} actions to {actions_path}")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)

    print(f"\nNext: Run for all h values, then compare distributions")
    print(f"  python3 compare_ik_policy_distributions.py")


if __name__ == '__main__':
    main()
