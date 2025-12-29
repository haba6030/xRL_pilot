"""
Fixed Horizon Environment Wrapper for 4-in-a-row

Based on Pedestrian project's FixedHorizonEnvWrapper.
Solves variable horizon problem by padding episodes to fixed length.

Reference:
- Variable Horizon Environments Considered Harmful
  https://imitation.readthedocs.io/en/latest/main-concepts/variable_horizon.html
- Pedestrian project: analysis/irl/util.py
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FixedHorizonWrapper(gym.Wrapper):
    """
    Fixed horizon wrapper for 4-in-a-row environment

    Problem:
        - h=1 episodes: avg 17 turns
        - h=4 episodes: avg 26 turns
        → Discriminator can cheat by learning episode length instead of behavior

    Solution:
        - Pad all episodes to max_episode_length (36 = board size)
        - Use absorbing state after termination
        - Add absorbing indicator to observation (90-dim = 89 + 1)

    Key features:
        - Terminal episodes continue with absorbing state transitions
        - Absorbing state: [0, 0, ..., 0, 1] (last dim = 1)
        - Absorbing actions: no-op (action 0 or any legal action)
        - Absorbing rewards: 0

    Usage:
        env = FourInARowEnv()
        env = FixedHorizonWrapper(env, max_episode_length=36)

        # Now all episodes have exactly 36 steps
        # Observation: (90,) instead of (89,)
    """

    def __init__(self, env, max_episode_length: int = 36):
        """
        Initialize fixed horizon wrapper

        Args:
            env: FourInARowEnv instance
            max_episode_length: Maximum episode length (default: 36 for 6x6 board)
        """
        super().__init__(env)

        self.max_episode_length = int(max_episode_length)
        self.cur_step = 0
        self.is_absorbing = False

        # Modify observation space to include absorbing indicator
        assert isinstance(env.observation_space, spaces.Box)
        low = env.observation_space.low
        high = env.observation_space.high
        obs_dim = len(low)

        assert obs_dim == 89, f"Expected obs_dim=89, got {obs_dim}"

        # Add absorbing indicator dimension (0=ongoing, 1=absorbing)
        self.observation_space = spaces.Box(
            low=np.concatenate([low, [0.0]], axis=0),
            high=np.concatenate([high, [1.0]], axis=0),
            dtype=env.observation_space.dtype
        )

        self._absorbing_obs = self._make_absorbing_obs()

    def reset(self, **kwargs):
        """Reset environment and counters"""
        self.cur_step = 0
        self.is_absorbing = False

        obs, info = self.env.reset(**kwargs)
        obs_with_flag = self._add_absorbing_flag(obs, absorbing=False)

        return obs_with_flag, info

    def step(self, action):
        """
        Take step in environment

        If game already ended, return absorbing state transition.
        Episode truncates after max_episode_length steps.
        """
        self.cur_step += 1

        if not self.is_absorbing:
            # Normal transition
            obs, rew, terminated, truncated, info = self.env.step(action)
            obs_with_flag = self._add_absorbing_flag(obs, absorbing=False)

            # Check if game ended
            if terminated or truncated:
                self.is_absorbing = True
        else:
            # Absorbing transition (game already ended)
            obs_with_flag = self._absorbing_obs
            rew = 0.0
            terminated = False
            truncated = False
            info = {}

        # Truncate episode after max_episode_length steps
        truncated = self.cur_step >= self.max_episode_length

        return obs_with_flag, rew, False, truncated, info

    def _add_absorbing_flag(self, obs: np.ndarray, absorbing: bool) -> np.ndarray:
        """
        Add absorbing indicator to observation

        Args:
            obs: (89,) observation
            absorbing: True if absorbing state

        Returns:
            obs_with_flag: (90,) observation with absorbing indicator
        """
        flag = 1.0 if absorbing else 0.0
        return np.concatenate([obs, [flag]], axis=0).astype(np.float32)

    def _make_absorbing_obs(self) -> np.ndarray:
        """
        Create absorbing state observation

        Returns:
            absorbing_obs: (90,) array with last dim = 1
        """
        absorbing_obs = np.zeros(90, dtype=np.float32)
        absorbing_obs[-1] = 1.0  # Absorbing indicator
        return absorbing_obs


def create_fixed_horizon_trajectory(
    trajectory_dict,
    max_episode_length: int = 36
):
    """
    Convert variable-length trajectory to fixed-length trajectory

    This is for converting existing trajectories that were collected
    WITHOUT FixedHorizonWrapper.

    Args:
        trajectory_dict: Dict with keys:
            - 'observations': (T+1, 89) numpy array
            - 'actions': (T,) numpy array
        max_episode_length: Target episode length

    Returns:
        fixed_traj: Dict with keys:
            - 'observations': (max_episode_length+1, 90) numpy array
            - 'actions': (max_episode_length,) numpy array
            - 'terminal': True

    Example:
        # Original trajectory (17 steps)
        traj = {
            'observations': np.array(...),  # (18, 89)
            'actions': np.array(...),       # (17,)
        }

        # Fixed trajectory (36 steps)
        fixed = create_fixed_horizon_trajectory(traj, max_episode_length=36)
        # fixed['observations']: (37, 90)
        # fixed['actions']: (36,)
        # Steps 0-16: original data
        # Steps 17-35: absorbing state padding
    """
    obs = trajectory_dict['observations']  # (T+1, 89)
    acts = trajectory_dict['actions']      # (T,)

    T = len(acts)
    assert len(obs) == T + 1, f"Observations should be T+1={T+1}, got {len(obs)}"
    assert T <= max_episode_length, \
        f"Episode length {T} exceeds max {max_episode_length}"

    # Add absorbing flag to all observations (flag=0 for ongoing states)
    obs_with_flag = []
    for o in obs:
        obs_with_flag.append(np.concatenate([o, [0.0]], axis=0))
    obs_with_flag = np.array(obs_with_flag)  # (T+1, 90)

    # Padding with absorbing states
    pad_length = max_episode_length - T

    if pad_length > 0:
        # Create absorbing observation
        absorbing_obs = np.zeros(90, dtype=np.float32)
        absorbing_obs[-1] = 1.0

        # Pad observations
        obs_padding = np.tile(absorbing_obs, (pad_length, 1))  # (pad_length, 90)
        obs_with_flag = np.concatenate([obs_with_flag, obs_padding], axis=0)

        # Pad actions (arbitrary, will be ignored in absorbing state)
        acts_padding = np.zeros(pad_length, dtype=np.int64)
        acts = np.concatenate([acts, acts_padding], axis=0)

    assert obs_with_flag.shape == (max_episode_length + 1, 90)
    assert acts.shape == (max_episode_length,)

    return {
        'observations': obs_with_flag,
        'actions': acts,
        'terminal': True
    }


# Testing functions
def test_fixed_horizon_wrapper():
    """Test fixed horizon wrapper"""
    print("=" * 80)
    print("Testing Fixed Horizon Wrapper")
    print("=" * 80)

    from env import FourInARowEnv

    # Test 1: Wrapper initialization
    print("\nTest 1: Wrapper initialization")
    env = FourInARowEnv()
    wrapped_env = FixedHorizonWrapper(env, max_episode_length=36)

    print(f"  Original obs space: {env.observation_space.shape}")
    print(f"  Wrapped obs space: {wrapped_env.observation_space.shape}")
    assert wrapped_env.observation_space.shape == (90,), "Obs space should be 90-dim"
    print("  ✓ Passed")

    # Test 2: Episode with early termination
    print("\nTest 2: Episode with early termination")
    obs, info = wrapped_env.reset(seed=42)
    assert obs.shape == (90,), f"Obs shape should be (90,), got {obs.shape}"
    assert obs[-1] == 0.0, "Initial obs should not be absorbing"

    step_count = 0
    done = False
    absorbing_count = 0

    while not done:
        action = wrapped_env.action_space.sample()
        obs, rew, terminated, truncated, info = wrapped_env.step(action)
        step_count += 1
        done = terminated or truncated

        if obs[-1] == 1.0:
            absorbing_count += 1

    print(f"  Episode length: {step_count}")
    print(f"  Absorbing transitions: {absorbing_count}")
    assert step_count == 36, f"Episode should be exactly 36 steps, got {step_count}"
    print("  ✓ Passed")

    # Test 3: Trajectory conversion
    print("\nTest 3: Trajectory conversion")
    traj = {
        'observations': np.random.rand(18, 89).astype(np.float32),
        'actions': np.random.randint(0, 36, size=17)
    }

    fixed_traj = create_fixed_horizon_trajectory(traj, max_episode_length=36)

    print(f"  Original: obs={traj['observations'].shape}, acts={traj['actions'].shape}")
    print(f"  Fixed: obs={fixed_traj['observations'].shape}, acts={fixed_traj['actions'].shape}")

    assert fixed_traj['observations'].shape == (37, 90)
    assert fixed_traj['actions'].shape == (36,)
    assert np.all(fixed_traj['observations'][:18, -1] == 0.0), "First 18 obs should not be absorbing"
    assert np.all(fixed_traj['observations'][18:, -1] == 1.0), "Padding obs should be absorbing"
    print("  ✓ Passed")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_fixed_horizon_wrapper()
