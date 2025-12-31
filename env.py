"""
FourInARowEnv: Gymnasium environment for 4-in-a-row game
State: 89-dim (72 board + 17 Van Opheusden features)
Action: Discrete(36) - board positions 0-35
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any


class FourInARowEnv(gym.Env):
    """
    4-in-a-row game environment (6x6 board)

    State representation:
        - black_pieces: 36-dim binary vector (positions 0-35)
        - white_pieces: 36-dim binary vector (positions 0-35)
        - van_opheusden_features: 17-dim feature vector
        Total: 89-dim

    Action space:
        - Discrete(36): place piece at position 0-35

    Rewards:
        - +1 for win
        - -1 for loss
        - 0 for draw
        - 0 for ongoing
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.board_size = 6
        self.num_positions = self.board_size * self.board_size  # 36

        # Observation space: 72 (board) + 17 (features) = 89
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(89,),
            dtype=np.float32
        )

        # Action space: 36 positions
        self.action_space = spaces.Discrete(self.num_positions)

        self.render_mode = render_mode

        # Initialize state
        self.black_pieces = np.zeros(self.num_positions, dtype=np.float32)
        self.white_pieces = np.zeros(self.num_positions, dtype=np.float32)
        self.current_player = 0  # 0 = Black, 1 = White
        self.move_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)

        self.black_pieces = np.zeros(self.num_positions, dtype=np.float32)
        self.white_pieces = np.zeros(self.num_positions, dtype=np.float32)
        self.current_player = 0
        self.move_count = 0

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment

        Args:
            action: Position to place piece (0-35)

        Returns:
            observation: 89-dim state
            reward: +1 (win), -1 (loss), 0 (ongoing/draw)
            terminated: True if game ended
            truncated: Always False (no time limit)
            info: Additional information
        """
        # Validate action
        if not self._is_valid_action(action):
            # Invalid move: return negative reward and terminate
            obs = self._get_observation()
            info = self._get_info()
            info['invalid_move'] = True
            return obs, -1.0, True, False, info

        # Apply action
        if self.current_player == 0:  # Black
            self.black_pieces[action] = 1.0
        else:  # White
            self.white_pieces[action] = 1.0

        self.move_count += 1

        # Check win/draw
        terminated = False
        reward = 0.0

        if self._check_win(action):
            terminated = True
            reward = 1.0  # Current player wins
        elif self._is_board_full():
            terminated = True
            reward = 0.0  # Draw

        # Switch player
        self.current_player = 1 - self.current_player

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def _is_valid_action(self, action: int) -> bool:
        """Check if action is legal (position is empty)"""
        if action < 0 or action >= self.num_positions:
            return False
        return (self.black_pieces[action] == 0 and
                self.white_pieces[action] == 0)

    def _check_win(self, last_move: int) -> bool:
        """
        Check if the last move resulted in 4-in-a-row

        Checks horizontal, vertical, and diagonal lines through last_move
        """
        row = last_move // self.board_size
        col = last_move % self.board_size

        # Get current player's pieces
        pieces = (self.black_pieces if self.current_player == 0
                  else self.white_pieces)

        # Convert to 2D for easier checking
        board = pieces.reshape(self.board_size, self.board_size)

        # Check horizontal
        if self._check_line(board, row, col, 0, 1):
            return True

        # Check vertical
        if self._check_line(board, row, col, 1, 0):
            return True

        # Check diagonal (top-left to bottom-right)
        if self._check_line(board, row, col, 1, 1):
            return True

        # Check anti-diagonal (top-right to bottom-left)
        if self._check_line(board, row, col, 1, -1):
            return True

        return False

    def _check_line(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        dr: int,
        dc: int
    ) -> bool:
        """
        Check for 4-in-a-row along a line direction

        Args:
            board: 6x6 board
            row, col: Starting position
            dr, dc: Direction vector (e.g., (0,1) for horizontal)
        """
        count = 1  # Count the placed piece

        # Check positive direction
        r, c = row + dr, col + dc
        while (0 <= r < self.board_size and
               0 <= c < self.board_size and
               board[r, c] == 1):
            count += 1
            r += dr
            c += dc

        # Check negative direction
        r, c = row - dr, col - dc
        while (0 <= r < self.board_size and
               0 <= c < self.board_size and
               board[r, c] == 1):
            count += 1
            r -= dr
            c -= dc

        return count >= 4

    def _is_board_full(self) -> bool:
        """Check if board is completely filled"""
        return self.move_count >= self.num_positions

    def _get_observation(self) -> np.ndarray:
        """
        Construct 89-dim observation:
        - Positions 0-35: black_pieces
        - Positions 36-71: white_pieces
        - Positions 72-88: Van Opheusden features
        """
        try:
            from .features import extract_van_opheusden_features
        except ImportError:
            from features import extract_van_opheusden_features

        features = extract_van_opheusden_features(
            self.black_pieces,
            self.white_pieces,
            self.current_player
        )

        obs = np.concatenate([
            self.black_pieces,      # 36-dim
            self.white_pieces,      # 36-dim
            features                # 17-dim
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Return additional information"""
        return {
            'current_player': self.current_player,
            'move_count': self.move_count,
            'legal_actions': self.get_legal_actions()
        }

    def get_legal_actions(self) -> np.ndarray:
        """Return array of legal action indices"""
        occupied = self.black_pieces + self.white_pieces
        legal_mask = (occupied == 0)
        return np.where(legal_mask)[0]

    def render(self):
        """Render the board state"""
        if self.render_mode == 'ansi' or self.render_mode == 'human':
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Return ASCII representation of board"""
        board = np.zeros((self.board_size, self.board_size), dtype=str)
        board[:] = '.'

        # Place black pieces
        for i in range(self.num_positions):
            if self.black_pieces[i] == 1:
                row = i // self.board_size
                col = i % self.board_size
                board[row, col] = 'X'

        # Place white pieces
        for i in range(self.num_positions):
            if self.white_pieces[i] == 1:
                row = i // self.board_size
                col = i % self.board_size
                board[row, col] = 'O'

        output = "\n"
        output += "  0 1 2 3 4 5\n"
        for i, row in enumerate(board):
            output += f"{i} {' '.join(row)}\n"
        output += f"Current player: {'Black (X)' if self.current_player == 0 else 'White (O)'}\n"

        if self.render_mode == 'human':
            print(output)

        return output


if __name__ == '__main__':
    # Quick test
    env = FourInARowEnv(render_mode='ansi')
    obs, info = env.reset()

    print("Initial state:")
    print(env.render())
    print(f"Observation shape: {obs.shape}")
    print(f"Legal actions: {info['legal_actions']}")

    # Play a few moves
    moves = [17, 18, 23, 24, 29, 30]  # Should create vertical win for Black
    for move in moves:
        obs, reward, terminated, truncated, info = env.step(move)
        print(f"\nAfter move {move}:")
        print(env.render())
        print(f"Reward: {reward}, Terminated: {terminated}")

        if terminated:
            break
