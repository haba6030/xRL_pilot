"""
Depth-Limited Planning Policy for 4-in-a-row

Implements planning depth h as a POLICY-level constraint (not reward-level).
Follows PLANNING_DEPTH_PRINCIPLES.md.

This is a baseline implementation that works WITHOUT C++ BFS.
For full BFS integration, see bfs_wrapper.py.
"""

import numpy as np
import copy
from typing import Optional, Tuple
from dataclasses import dataclass

try:
    from .env import FourInARowEnv
    from .features import extract_van_opheusden_features
    from .bfs_wrapper import BFSParameters
except ImportError:
    from env import FourInARowEnv
    from features import extract_van_opheusden_features
    from bfs_wrapper import BFSParameters


@dataclass
class PlanningResult:
    """Result of depth-limited planning"""
    action_values: np.ndarray  # Q(s, a) for each action
    action_probs: np.ndarray   # π(a|s) after softmax + lapse
    best_action: int           # argmax action
    nodes_expanded: int        # Number of states evaluated


class SimpleHeuristicEvaluator:
    """
    Simple heuristic state evaluator based on Van Opheusden features

    Uses linear combination of features to estimate state value.
    This is a BASELINE - full BFS uses more sophisticated evaluation.
    """

    def __init__(self, params: Optional[BFSParameters] = None):
        """
        Initialize evaluator

        Args:
            params: BFS parameters (if available)
                   If None, uses default weights
        """
        self.params = params

        # Default feature weights (if no params provided)
        # These approximate the Van Opheusden weights
        if params is None:
            self.center_weight = 0.8
            self.connected_2_weight = 1.0
            self.unconnected_2_weight = 0.5
            self.three_weight = 5.0
            self.four_weight = 100.0
        else:
            self.center_weight = params.center_weight
            self.connected_2_weight = params.connected_2_weight
            self.unconnected_2_weight = params.unconnected_2_weight
            self.three_weight = params.three_in_a_row_weight
            self.four_weight = params.four_in_a_row_weight

    def evaluate(self, state: np.ndarray) -> float:
        """
        Evaluate state value using heuristic features

        Args:
            state: 89-dim state (72 board + 17 features)

        Returns:
            value: Estimated state value
        """
        # Extract features (positions 72-88)
        features = state[72:89]

        # Linear combination of features
        value = 0.0

        # Center control
        value += self.center_weight * features[0]

        # Connected 2-in-a-row (4 orientations)
        value += self.connected_2_weight * features[1:5].sum()

        # Unconnected 2-in-a-row (4 orientations)
        value += self.unconnected_2_weight * features[5:9].sum()

        # 3-in-a-row (4 orientations)
        value += self.three_weight * features[9:13].sum()

        # 4-in-a-row (4 orientations) - WINNING!
        value += self.four_weight * features[13:17].sum()

        return value


class DepthLimitedPolicy:
    """
    Planning policy with fixed depth h

    CRITICAL: This implements depth as a POLICY constraint, NOT a reward parameter.
    See PLANNING_DEPTH_PRINCIPLES.md for theoretical justification.

    Planning algorithm:
    1. For each legal action a:
       - Clone environment
       - Simulate h steps ahead using lookahead
       - Evaluate terminal state
       - Backpropagate value to Q(s, a)
    2. Select action via softmax(Q) + lapse
    """

    def __init__(
        self,
        h: int,
        params: Optional[BFSParameters] = None,
        beta: float = 1.0,
        lapse_rate: float = 0.1,
        discount: float = 0.99
    ):
        """
        Initialize depth-limited policy

        Args:
            h: Planning depth (number of lookahead steps)
               THIS IS THE KEY PARAMETER - it constrains planning, not reward
            params: BFS parameters for heuristic evaluation
            beta: Inverse temperature for softmax
            lapse_rate: Random action probability
            discount: Discount factor γ for future rewards
        """
        self.h = h  # ← Depth lives HERE (in policy, not discriminator)
        self.params = params
        self.beta = beta
        self.lapse_rate = lapse_rate if params is None else params.lapse_rate
        self.discount = discount

        self.evaluator = SimpleHeuristicEvaluator(params)
        self.nodes_expanded = 0

    def plan(self, env: FourInARowEnv) -> PlanningResult:
        """
        Plan h steps ahead and compute action values

        Args:
            env: Current environment state

        Returns:
            PlanningResult with Q-values and action probabilities
        """
        self.nodes_expanded = 0
        legal_actions = env.get_legal_actions()

        if len(legal_actions) == 0:
            # No legal actions (shouldn't happen in normal game)
            return PlanningResult(
                action_values=np.zeros(36),
                action_probs=np.zeros(36),
                best_action=-1,
                nodes_expanded=0
            )

        # Compute Q-value for each legal action
        Q = np.full(36, -np.inf)  # Initialize with -inf for illegal actions

        for action in legal_actions:
            Q[action] = self._evaluate_action(env, action)

        # Convert to probabilities via softmax
        action_probs = self._softmax_with_lapse(Q, legal_actions)

        # Best action
        best_action = legal_actions[np.argmax(Q[legal_actions])]

        return PlanningResult(
            action_values=Q,
            action_probs=action_probs,
            best_action=best_action,
            nodes_expanded=self.nodes_expanded
        )

    def _evaluate_action(self, env: FourInARowEnv, action: int) -> float:
        """
        Evaluate a single action by h-step lookahead

        Args:
            env: Current environment
            action: Action to evaluate

        Returns:
            value: Estimated Q(s, a)
        """
        # Clone environment
        env_sim = copy.deepcopy(env)

        # Take action
        obs, reward, terminated, truncated, info = env_sim.step(action)
        self.nodes_expanded += 1

        # Immediate reward
        cumulative_value = reward

        # If game ended, return immediate reward
        if terminated or truncated:
            return cumulative_value

        # Otherwise, simulate h-1 more steps using greedy policy
        current_discount = self.discount

        for step in range(self.h - 1):
            # Get legal actions
            legal_actions_sim = env_sim.get_legal_actions()

            if len(legal_actions_sim) == 0:
                break

            # Greedy action selection (simple heuristic)
            action_values_sim = []
            for a in legal_actions_sim:
                env_test = copy.deepcopy(env_sim)
                obs_test, _, _, _, _ = env_test.step(a)
                value = self.evaluator.evaluate(obs_test)
                action_values_sim.append(value)
                self.nodes_expanded += 1

            # Select best action
            best_idx = np.argmax(action_values_sim)
            next_action = legal_actions_sim[best_idx]

            # Execute action
            obs, reward, terminated, truncated, info = env_sim.step(next_action)
            cumulative_value += current_discount * reward
            current_discount *= self.discount

            if terminated or truncated:
                break

        # Add terminal state value estimate
        if not (terminated or truncated):
            terminal_value = self.evaluator.evaluate(obs)
            cumulative_value += current_discount * terminal_value

        return cumulative_value

    def _softmax_with_lapse(
        self,
        Q: np.ndarray,
        legal_actions: np.ndarray
    ) -> np.ndarray:
        """
        Convert Q-values to action probabilities with softmax + lapse

        Args:
            Q: Action values (36-dim, -inf for illegal actions)
            legal_actions: Indices of legal actions

        Returns:
            probs: Action probabilities (36-dim, sums to 1)
        """
        # Softmax over legal actions only
        Q_legal = Q[legal_actions]
        Q_legal = Q_legal - Q_legal.max()  # Numerical stability

        exp_Q = np.exp(self.beta * Q_legal)
        softmax_probs = exp_Q / exp_Q.sum()

        # Map back to 36-dim
        policy_probs = np.zeros(36)
        policy_probs[legal_actions] = softmax_probs

        # Add lapse (uniform random)
        uniform_probs = np.zeros(36)
        uniform_probs[legal_actions] = 1.0 / len(legal_actions)

        # Mix
        probs = (1 - self.lapse_rate) * policy_probs + self.lapse_rate * uniform_probs

        # Renormalize (shouldn't be necessary, but for safety)
        probs = probs / probs.sum()

        return probs

    def select_action(
        self,
        env: FourInARowEnv,
        rng: Optional[np.random.Generator] = None
    ) -> Tuple[int, PlanningResult]:
        """
        Select action by planning h steps ahead

        Args:
            env: Current environment
            rng: Random number generator

        Returns:
            action: Selected action
            result: Planning result (for debugging/analysis)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Plan
        result = self.plan(env)

        # Sample action
        action = rng.choice(36, p=result.action_probs)

        return action, result


def test_depth_limited_policy():
    """Test depth-limited policy"""
    print("=" * 80)
    print("Testing Depth-Limited Planning Policy")
    print("=" * 80)

    # Test different depths
    depths = [1, 2, 4, 8]

    for h in depths:
        print(f"\n[Test h={h}]")

        # Create environment
        env = FourInARowEnv()
        obs, info = env.reset(seed=42)

        # Create policy
        policy = DepthLimitedPolicy(h=h, beta=1.0, lapse_rate=0.1)

        # Plan
        result = policy.plan(env)

        print(f"  Legal actions: {env.get_legal_actions()[:5]}... ({len(env.get_legal_actions())} total)")
        print(f"  Best action: {result.best_action}")
        print(f"  Best Q-value: {result.action_values[result.best_action]:.3f}")
        print(f"  Action prob: {result.action_probs[result.best_action]:.3f}")
        print(f"  Nodes expanded: {result.nodes_expanded}")

    # Test episode rollout
    print(f"\n[Test Episode Rollout with h=4]")
    env = FourInARowEnv()
    obs, info = env.reset(seed=123)
    policy = DepthLimitedPolicy(h=4, beta=2.0, lapse_rate=0.05)

    rng = np.random.default_rng(123)
    episode_length = 0
    total_nodes = 0

    for step in range(36):  # Max game length
        action, result = policy.select_action(env, rng)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_length += 1
        total_nodes += result.nodes_expanded

        if terminated or truncated:
            print(f"  Game ended after {episode_length} moves")
            print(f"  Final reward: {reward}")
            print(f"  Total nodes expanded: {total_nodes}")
            print(f"  Avg nodes per move: {total_nodes / episode_length:.1f}")
            break

    # Compare depths on same position
    print(f"\n[Test Depth Comparison on Fixed Position]")
    env = FourInARowEnv()
    env.reset(seed=42)

    # Make a few moves to create interesting position
    env.step(17)  # Black
    env.step(18)  # White
    env.step(23)  # Black

    print(f"  Board after 3 moves:")
    print(env.render())

    for h in [1, 2, 4, 8]:
        policy = DepthLimitedPolicy(h=h, beta=1.0, lapse_rate=0.0)  # No lapse for comparison
        result = policy.plan(env)

        legal_actions = env.get_legal_actions()
        top3_actions = legal_actions[np.argsort(-result.action_values[legal_actions])[:3]]

        print(f"\n  h={h}:")
        print(f"    Best action: {result.best_action}")
        print(f"    Top 3 actions: {top3_actions}")
        print(f"    Top 3 Q-values: {result.action_values[top3_actions]}")
        print(f"    Nodes expanded: {result.nodes_expanded}")

    print("\n" + "=" * 80)
    print("Depth-Limited Policy Test: ✅ COMPLETE")
    print("=" * 80)
    print("\nKey observations:")
    print("  1. Deeper planning (higher h) expands more nodes")
    print("  2. Different h values may prefer different actions")
    print("  3. Action selection is stochastic (softmax + lapse)")
    print("  4. Policy works without C++ BFS (baseline implementation)")
    print("\nReady for AIRL integration.")


if __name__ == '__main__':
    test_depth_limited_policy()
