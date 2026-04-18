import numpy as np

def calculate_final_score(board):
    black_tiles = 0
    white_tiles = 0
    for row in board:
        for tile in row:
            if tile == -1:
                black_tiles += 1
            elif tile == 1:
                white_tiles += 1
    print("white score", white_tiles, "black score", black_tiles)

    return white_tiles, black_tiles

def get_legal_moves(game, piece):
    """Return list of (x, y) legal moves for the given piece."""
    moves = []
    for i in range(8):
        for j in range(8):
            if game.step(i, j, piece, False) > 0:
                moves.append((i, j))
    return moves

def apply_move(board, game, x, y, piece):
    """Apply a move on a copy of the board and return the new board state."""
    new_board = board.copy()
    game.board = new_board
    game.step(x, y, piece, True)
    return new_board

# Positional weight matrix for the heuristic evaluation.
# Corners are highly valuable, edges are good, positions adjacent
# to corners (X-squares and C-squares) are dangerous. This can be played around with. Probably makes a big difference
WEIGHT_MATRIX = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [  10,  -2,   5,   1,   1,   5,  -2,   10],
    [   5,  -2,   1,   0,   0,   1,  -2,    5],
    [   5,  -2,   1,   0,   0,   1,  -2,    5],
    [  10,  -2,   5,   1,   1,   5,  -2,   10],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
])

# Early-game bonus for controlling the center 4x4 square (rows/cols 2-5).
# Holding the center forces the opponent to play on the outer ring, making
# edge and corner captures easier in the mid-to-late game.
# This bonus fades linearly to zero once 36 pieces are on the board.
CENTER_BONUS = np.zeros((8, 8))
CENTER_BONUS[2:6, 2:6] = np.array([
    [ 8, 12, 12,  8],
    [12, 20, 20, 12],
    [12, 20, 20, 12],
    [ 8, 12, 12,  8],
])

