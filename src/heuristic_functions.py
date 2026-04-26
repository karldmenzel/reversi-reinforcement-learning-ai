import os
import numpy as np
from reversi import reversi
from utils import get_legal_moves, WEIGHT_MATRIX, CENTER_BONUS

# During adaptive training, load perturbed WEIGHT_MATRIX if env var is set
_cycle_wm = os.environ.get('CYCLE_WEIGHT_MATRIX_PATH', '')
if _cycle_wm and os.path.exists(_cycle_wm):
    WEIGHT_MATRIX = np.load(_cycle_wm)

def heuristic_nic_2(board, player):
    """
    Evaluates the board from the perspective of `player`.

    Combines four factors:
      1. Positional weights    - rewards corners/edges, penalizes risky squares
      2. Center control bonus  - early-game incentive to hold the center 4x4;
                                 fades to zero once ~36 pieces are on the board
      3. Piece difference      - having more pieces is good (especially late-game)
      4. Mobility              - having more moves available is strategically valuable
    """
    opponent = -player
    net_mask = (board == player).astype(float) - (board == opponent).astype(float)

    # Base positional score from corner/edge weight matrix
    positional_score = float(np.sum(net_mask * WEIGHT_MATRIX))

    player_count = int(np.sum(board == player))
    opponent_count = int(np.sum(board == opponent))
    total_pieces = player_count + opponent_count

    # Early-game center control bonus: linearly decreases from full strength at
    # game start (4 pieces) to zero at 36 pieces. Holding the center 4x4 forces
    # the opponent onto the outer ring, setting up edge and corner steals later.
    center_scale = max(0.0, (36 - total_pieces) / 32.0)
    if center_scale > 0:
        positional_score += center_scale * float(np.sum(net_mask * CENTER_BONUS))

    # Piece difference
    if total_pieces > 0:
        piece_score = 100.0 * (player_count - opponent_count) / total_pieces
    else:
        piece_score = 0.0

    # Mobility: number of legal moves available to each side
    game = reversi()
    game.board = board.copy()
    player_moves = len(get_legal_moves(game, player))
    game.board = board.copy()
    opponent_moves = len(get_legal_moves(game, opponent))
    if player_moves + opponent_moves != 0:
        mobility_score = 100.0 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
    else:
        mobility_score = 0.0

    # Weighted combination
    if total_pieces > 52:
        # Late game: piece count matters more
        return positional_score + 3.0 * piece_score + mobility_score
    else:
        # Early/mid game: position and mobility matter more
        return 3.0 * positional_score + piece_score + 1.5 * mobility_score


def heuristic_nic(board, player):
    """
    Evaluates the board from the perspective of `player`.

    Combines four factors:
      1. Positional weights    - rewards corners/edges, penalizes risky squares
      2. Center control bonus  - early-game incentive to hold the center 4x4;
                                 fades to zero once ~36 pieces are on the board
      3. Piece difference      - having more pieces is good (especially late-game)
      4. Mobility              - having more moves available is strategically valuable
    """
    opponent = -player
    net_mask = (board == player).astype(float) - (board == opponent).astype(float)

    # Base positional score from corner/edge weight matrix
    positional_score = float(np.sum(net_mask * WEIGHT_MATRIX))

    player_count = int(np.sum(board == player))
    opponent_count = int(np.sum(board == opponent))
    total_pieces = player_count + opponent_count

    # Early-game center control bonus: linearly decreases from full strength at
    # game start (4 pieces) to zero at 36 pieces. Holding the center 4x4 forces
    # the opponent onto the outer ring, setting up edge and corner steals later.
    center_scale = max(0.0, (36 - total_pieces) / 32.0)
    if center_scale > 0:
        positional_score += center_scale * float(np.sum(net_mask * CENTER_BONUS))

    # Piece difference
    if total_pieces > 0:
        piece_score = 100.0 * (player_count - opponent_count) / total_pieces
    else:
        piece_score = 0.0

    # Mobility: number of legal moves available to each side
    game = reversi()
    game.board = board.copy()
    player_moves = len(get_legal_moves(game, player))
    game.board = board.copy()
    opponent_moves = len(get_legal_moves(game, opponent))
    if player_moves + opponent_moves != 0:
        mobility_score = 100.0 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
    else:
        mobility_score = 0.0

    # Weighted combination
    if total_pieces > 52:
        # Late game: piece count matters more
        return positional_score + 3.0 * piece_score + mobility_score
    else:
        # Early/mid game: position and mobility matter more
        return 2.0 * positional_score + piece_score + 2.0 * mobility_score