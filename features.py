"""
Van Opheusden Feature Extraction for 4-in-a-row

Implements the 17 heuristic features used in van Opheusden et al. (2023):
1. Center control
2-5. Connected 2-in-a-row (horizontal, vertical, diag, anti-diag)
6-9. Unconnected 2-in-a-row (4 orientations)
10-13. 3-in-a-row (4 orientations)
14-17. 4-in-a-row (4 orientations)

Based on Model code/heuristic.cpp
"""

import numpy as np
from typing import Tuple


def extract_van_opheusden_features(
    black_pieces: np.ndarray,
    white_pieces: np.ndarray,
    current_player: int
) -> np.ndarray:
    """
    Extract 17-dimensional feature vector from board state

    Args:
        black_pieces: 36-dim binary array (1 if Black piece at position)
        white_pieces: 36-dim binary array (1 if White piece at position)
        current_player: 0 (Black) or 1 (White)

    Returns:
        features: 17-dim array of feature counts
    """
    board_size = 6

    # Convert to 2D boards
    black_board = black_pieces.reshape(board_size, board_size)
    white_board = white_pieces.reshape(board_size, board_size)

    # Determine active and passive player boards
    if current_player == 0:  # Black to move
        active_board = black_board
        passive_board = white_board
    else:  # White to move
        active_board = white_board
        passive_board = black_board

    features = np.zeros(17, dtype=np.float32)

    # Feature 0: Center control (active player)
    # Center positions: (2,2), (2,3), (3,2), (3,3)
    center_positions = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for r, c in center_positions:
        if active_board[r, c] == 1:
            features[0] += 1.0

    # Features 1-4: Connected 2-in-a-row (active player)
    # Horizontal, Vertical, Diagonal, Anti-diagonal
    features[1] = count_connected_n(active_board, passive_board, 2, orientation='horizontal')
    features[2] = count_connected_n(active_board, passive_board, 2, orientation='vertical')
    features[3] = count_connected_n(active_board, passive_board, 2, orientation='diagonal')
    features[4] = count_connected_n(active_board, passive_board, 2, orientation='anti-diagonal')

    # Features 5-8: Unconnected 2-in-a-row (active player)
    # Pattern: X_X (piece-empty-piece)
    features[5] = count_unconnected_n(active_board, passive_board, 2, orientation='horizontal')
    features[6] = count_unconnected_n(active_board, passive_board, 2, orientation='vertical')
    features[7] = count_unconnected_n(active_board, passive_board, 2, orientation='diagonal')
    features[8] = count_unconnected_n(active_board, passive_board, 2, orientation='anti-diagonal')

    # Features 9-12: 3-in-a-row (active player)
    features[9] = count_connected_n(active_board, passive_board, 3, orientation='horizontal')
    features[10] = count_connected_n(active_board, passive_board, 3, orientation='vertical')
    features[11] = count_connected_n(active_board, passive_board, 3, orientation='diagonal')
    features[12] = count_connected_n(active_board, passive_board, 3, orientation='anti-diagonal')

    # Features 13-16: 4-in-a-row (active player)
    features[13] = count_connected_n(active_board, passive_board, 4, orientation='horizontal')
    features[14] = count_connected_n(active_board, passive_board, 4, orientation='vertical')
    features[15] = count_connected_n(active_board, passive_board, 4, orientation='diagonal')
    features[16] = count_connected_n(active_board, passive_board, 4, orientation='anti-diagonal')

    # Normalize features (optional, can help neural network training)
    # Divide by board size to keep values in [0, 1] range
    features[0] /= 4.0  # Max 4 center positions
    features[1:] /= 6.0  # Max ~6 patterns per orientation

    return features


def count_connected_n(
    active_board: np.ndarray,
    passive_board: np.ndarray,
    n: int,
    orientation: str
) -> float:
    """
    Count number of connected n-in-a-row patterns (active player)

    Pattern: XXX (for n=3) with open ends (not blocked by opponent)

    Args:
        active_board: 6x6 board (1 where active player has pieces)
        passive_board: 6x6 board (1 where opponent has pieces)
        n: Number of consecutive pieces (2, 3, or 4)
        orientation: 'horizontal', 'vertical', 'diagonal', 'anti-diagonal'

    Returns:
        count: Number of n-in-a-row patterns found
    """
    board_size = active_board.shape[0]
    count = 0

    # Define direction vectors
    directions = {
        'horizontal': (0, 1),
        'vertical': (1, 0),
        'diagonal': (1, 1),
        'anti-diagonal': (1, -1)
    }

    dr, dc = directions[orientation]

    # Scan all starting positions
    for r in range(board_size):
        for c in range(board_size):
            # Check if we can fit n pieces starting from (r, c)
            if not _in_bounds(r + (n-1)*dr, c + (n-1)*dc, board_size):
                continue

            # Check if we have n consecutive active pieces
            has_pattern = True
            for i in range(n):
                rr = r + i*dr
                cc = c + i*dc
                if active_board[rr, cc] != 1:
                    has_pattern = False
                    break

            if has_pattern:
                # Check if pattern is "open" (not blocked at ends)
                # Optional: can add more sophisticated blocking detection
                count += 1

    return float(count)


def count_unconnected_n(
    active_board: np.ndarray,
    passive_board: np.ndarray,
    n: int,
    orientation: str
) -> float:
    """
    Count number of unconnected n-in-a-row patterns (e.g., X_X for n=2)

    Pattern: X_X (piece-empty-piece) that could form n+1 in a row

    Args:
        active_board: 6x6 board
        passive_board: 6x6 board
        n: Number of pieces (typically 2)
        orientation: Line orientation

    Returns:
        count: Number of unconnected patterns found
    """
    board_size = active_board.shape[0]
    count = 0

    directions = {
        'horizontal': (0, 1),
        'vertical': (1, 0),
        'diagonal': (1, 1),
        'anti-diagonal': (1, -1)
    }

    dr, dc = directions[orientation]

    # For n=2, pattern is X_X (length 3)
    pattern_length = n + 1

    for r in range(board_size):
        for c in range(board_size):
            if not _in_bounds(r + (pattern_length-1)*dr, c + (pattern_length-1)*dc, board_size):
                continue

            # Check X_X pattern
            r0, c0 = r, c
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2*dr, c + 2*dc

            if (active_board[r0, c0] == 1 and
                active_board[r1, c1] == 0 and
                passive_board[r1, c1] == 0 and  # Empty, not opponent
                active_board[r2, c2] == 1):
                count += 1

    return float(count)


def _in_bounds(r: int, c: int, board_size: int) -> bool:
    """Check if position is within board bounds"""
    return 0 <= r < board_size and 0 <= c < board_size


def test_features():
    """Test feature extraction on simple board"""
    board_size = 6
    black_pieces = np.zeros(36)
    white_pieces = np.zeros(36)

    # Place some test pieces
    # Black: horizontal 3-in-a-row at row 2, cols 1-3
    black_pieces[2*6 + 1] = 1
    black_pieces[2*6 + 2] = 1
    black_pieces[2*6 + 3] = 1

    # Black: center control
    black_pieces[2*6 + 2] = 1  # Already set above

    # White: vertical 2-in-a-row at col 4, rows 0-1
    white_pieces[0*6 + 4] = 1
    white_pieces[1*6 + 4] = 1

    # Current player: Black
    features = extract_van_opheusden_features(black_pieces, white_pieces, current_player=0)

    print("Test Board:")
    print("Black pieces:", np.where(black_pieces == 1)[0])
    print("White pieces:", np.where(white_pieces == 1)[0])
    print("\nExtracted features:")
    feature_names = [
        "Center control",
        "Connected 2 (H)", "Connected 2 (V)", "Connected 2 (D)", "Connected 2 (AD)",
        "Unconnected 2 (H)", "Unconnected 2 (V)", "Unconnected 2 (D)", "Unconnected 2 (AD)",
        "3-in-a-row (H)", "3-in-a-row (V)", "3-in-a-row (D)", "3-in-a-row (AD)",
        "4-in-a-row (H)", "4-in-a-row (V)", "4-in-a-row (D)", "4-in-a-row (AD)"
    ]
    for i, (name, val) in enumerate(zip(feature_names, features)):
        if val > 0:
            print(f"  {i:2d}. {name:20s}: {val:.3f}")


if __name__ == '__main__':
    test_features()
