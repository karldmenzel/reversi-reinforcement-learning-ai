import numpy as np

# ── Zobrist Hashing ──────────────────────────────────────────────────────────
# Pre-computed random 64-bit integers for incremental board hashing.
# Index 0 = white (piece == 1), index 1 = black (piece == -1).
_rng = np.random.RandomState(seed=0xDEADBEEF)
ZOBRIST_TABLE = _rng.randint(0, 2**63, size=(8, 8, 2), dtype=np.int64)
ZOBRIST_TURN = int(_rng.randint(0, 2**63, dtype=np.int64))

def _piece_index(piece):
    """Map piece value (1 or -1) to Zobrist table index (0 or 1)."""
    return 0 if piece == 1 else 1

def zobrist_hash(board):
    """Compute the full Zobrist hash of a board from scratch."""
    h = 0
    for r in range(8):
        for c in range(8):
            p = board[r, c]
            if p != 0:
                h ^= int(ZOBRIST_TABLE[r, c, _piece_index(p)])
    return h


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

def apply_move_with_hash(board, game, x, y, piece, z_hash):
    """Apply a move and return (new_board, updated_zobrist_hash).

    Incrementally updates the hash by XOR-ing out flipped pieces and
    XOR-ing in the new ones, plus toggling the side-to-move bit.
    This is O(flips) instead of O(64).
    """
    new_board = board.copy()
    old_board = board  # reference to original for diff

    game.board = new_board
    game.step(x, y, piece, True)

    # XOR in the newly placed piece
    h = z_hash ^ int(ZOBRIST_TABLE[x, y, _piece_index(piece)])

    # XOR out old values and XOR in new values for every flipped cell
    opponent = -piece
    opp_idx = _piece_index(opponent)
    piece_idx = _piece_index(piece)
    for r in range(8):
        for c in range(8):
            if old_board[r, c] == opponent and new_board[r, c] == piece:
                h ^= int(ZOBRIST_TABLE[r, c, opp_idx]) ^ int(ZOBRIST_TABLE[r, c, piece_idx])

    # Toggle side-to-move
    h ^= ZOBRIST_TURN

    return new_board, h

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

