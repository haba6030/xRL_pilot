"""
BFS Policy Wrapper
Simplified Python wrapper for van Opheusden BFS policy

For full BFS distillation, this would need to interface with C++ code.
For now, provides a baseline random policy and parameter loading utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BFSParameters:
    """Parameters for BFS policy from van Opheusden model"""
    # Search parameters
    pruning_threshold: float
    stopping_probability: float

    # Cognitive parameters
    lapse_rate: float
    feature_drop_rate: float
    active_scaling_constant: float

    # Feature weights (17 weights)
    center_weight: float
    connected_2_weight: float
    unconnected_2_weight: float
    three_in_a_row_weight: float
    four_in_a_row_weight: float

    # Metadata
    participant_id: int
    log_likelihood: float

    # Planning depth (if available)
    planning_depth: Optional[int] = None

    @staticmethod
    def from_model_fits(model_fits_df: pd.DataFrame, participant_id: int) -> 'BFSParameters':
        """
        Load BFS parameters for a participant from model_fits CSV

        Args:
            model_fits_df: DataFrame from opendata/model_fits_main_model.csv
            participant_id: Participant ID to load

        Returns:
            BFSParameters object
        """
        # Filter to participant
        participant_data = model_fits_df[
            model_fits_df['participant'] == participant_id
        ].iloc[0]  # Take first row (they should all be similar)

        params = BFSParameters(
            pruning_threshold=float(participant_data['pruning threshold']),
            stopping_probability=float(participant_data['stopping probability']),
            lapse_rate=float(participant_data['lapse rate']),
            feature_drop_rate=float(participant_data['feature drop rate']),
            active_scaling_constant=float(participant_data['active scaling constant']),
            center_weight=float(participant_data['center weight']),
            connected_2_weight=float(participant_data['connected 2-in-a-row weight']),
            unconnected_2_weight=float(participant_data['unconnected 2-in-a-row weight']),
            three_in_a_row_weight=float(participant_data['3-in-a-row weight']),
            four_in_a_row_weight=float(participant_data['4-in-a-row weight']),
            participant_id=int(participant_id),
            log_likelihood=float(participant_data['log-likelihood'])
        )

        return params


class BFSPolicy:
    """
    BFS Policy Wrapper

    For full implementation, this would call C++ BFS code.
    Currently provides baseline functionality for testing.
    """

    def __init__(
        self,
        params: Optional[BFSParameters] = None,
        planning_depth: Optional[int] = None,
        beta: float = 1.0
    ):
        """
        Initialize BFS policy

        Args:
            params: BFS parameters (if None, uses default)
            planning_depth: Fixed planning depth h (if None, uses default BFS)
            beta: Inverse temperature for action selection
        """
        self.params = params
        self.planning_depth = planning_depth
        self.beta = beta

    def get_action_probabilities(
        self,
        state: np.ndarray,
        legal_actions: np.ndarray
    ) -> np.ndarray:
        """
        Get action probabilities for a state

        Args:
            state: 89-dim state vector
            legal_actions: Array of legal action indices

        Returns:
            probs: Probability distribution over actions (36-dim, sums to 1)
        """
        # TODO: Call C++ BFS to get Q-values
        # For now, use uniform random over legal actions with small lapse

        lapse_rate = self.params.lapse_rate if self.params else 0.1

        # Start with uniform over legal actions
        probs = np.zeros(36)
        if len(legal_actions) > 0:
            probs[legal_actions] = 1.0 / len(legal_actions)

        # Apply lapse rate (mixture with uniform)
        lapse_probs = np.ones(36) / 36.0
        probs = (1 - lapse_rate) * probs + lapse_rate * lapse_probs

        # Renormalize
        probs = probs / probs.sum()

        return probs

    def sample_action(
        self,
        state: np.ndarray,
        legal_actions: np.ndarray,
        rng: Optional[np.random.Generator] = None
    ) -> int:
        """
        Sample an action from the policy

        Args:
            state: 89-dim state vector
            legal_actions: Array of legal action indices
            rng: Random number generator

        Returns:
            action: Sampled action index
        """
        if rng is None:
            rng = np.random.default_rng()

        probs = self.get_action_probabilities(state, legal_actions)
        action = rng.choice(36, p=probs)

        return action


def load_all_participant_parameters(
    model_fits_path: str = 'opendata/model_fits_main_model.csv'
) -> Dict[int, BFSParameters]:
    """
    Load BFS parameters for all participants

    Args:
        model_fits_path: Path to model fits CSV

    Returns:
        params_dict: Dict mapping participant_id -> BFSParameters
    """
    model_fits = pd.read_csv(model_fits_path)

    params_dict = {}
    for participant_id in model_fits['participant'].unique():
        try:
            params = BFSParameters.from_model_fits(model_fits, participant_id)
            params_dict[participant_id] = params
        except Exception as e:
            print(f"Warning: Failed to load parameters for participant {participant_id}: {e}")

    return params_dict


def test_bfs_wrapper():
    """Test BFS wrapper"""
    print("=" * 80)
    print("Testing BFS Policy Wrapper")
    print("=" * 80)

    # Load parameters
    import os
    if os.path.exists('opendata/model_fits_main_model.csv'):
        model_fits_path = 'opendata/model_fits_main_model.csv'
    elif os.path.exists('../opendata/model_fits_main_model.csv'):
        model_fits_path = '../opendata/model_fits_main_model.csv'
    else:
        print("ERROR: Cannot find model_fits_main_model.csv")
        return

    print(f"\nLoading parameters from {model_fits_path}...")
    params_dict = load_all_participant_parameters(model_fits_path)
    print(f"Loaded parameters for {len(params_dict)} participants")

    # Show first participant's parameters
    participant_id = list(params_dict.keys())[0]
    params = params_dict[participant_id]

    print(f"\nParticipant {participant_id} parameters:")
    print(f"  Pruning threshold: {params.pruning_threshold:.4f}")
    print(f"  Stopping probability: {params.stopping_probability:.4f}")
    print(f"  Lapse rate: {params.lapse_rate:.4f}")
    print(f"  Feature drop rate: {params.feature_drop_rate:.4f}")
    print(f"  Center weight: {params.center_weight:.4f}")
    print(f"  Log-likelihood: {params.log_likelihood:.4f}")

    # Create policy
    policy = BFSPolicy(params=params, beta=1.0)

    # Test action sampling
    print(f"\nTesting action sampling...")
    state = np.random.rand(89).astype(np.float32)
    legal_actions = np.array([0, 1, 2, 5, 10, 17], dtype=np.int64)

    print(f"Legal actions: {legal_actions}")

    probs = policy.get_action_probabilities(state, legal_actions)
    print(f"Action probabilities (legal only):")
    for action in legal_actions:
        print(f"  Action {action}: {probs[action]:.4f}")

    # Sample a few actions
    print(f"\nSampling 10 actions:")
    rng = np.random.default_rng(42)
    samples = [policy.sample_action(state, legal_actions, rng) for _ in range(10)]
    print(f"  Sampled: {samples}")


if __name__ == '__main__':
    test_bfs_wrapper()
