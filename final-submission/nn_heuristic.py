import numpy as np
import os

# Positional weight matrix (same as utils.py)
WEIGHT_MATRIX = np.array([
    [ 100, -20,  10,   5,   5,  10, -20,  100],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [  10,  -2,   5,   1,   1,   5,  -2,   10],
    [   5,  -2,   1,   0,   0,   1,  -2,    5],
    [   5,  -2,   1,   0,   0,   1,  -2,    5],
    [  10,  -2,   5,   1,   1,   5,  -2,   10],
    [ -20, -50,  -2,  -2,  -2,  -2, -50,  -20],
    [ 100, -20,  10,   5,   5,  10, -20,  100],
], dtype=np.float32)

# Corner positions
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]

# X-squares (diagonal-adjacent to corners)
X_SQUARES = [(1, 1), (1, 6), (6, 1), (6, 6)]

# Edge mask (all border cells)
EDGE_MASK = np.zeros((8, 8), dtype=np.float32)
EDGE_MASK[0, :] = 1
EDGE_MASK[7, :] = 1
EDGE_MASK[:, 0] = 1
EDGE_MASK[:, 7] = 1

# Flatten the weight matrix once
WEIGHT_MATRIX_FLAT = WEIGHT_MATRIX.flatten().astype(np.float32)

# Normalization constant for weight features
WEIGHT_NORM = np.max(np.abs(WEIGHT_MATRIX_FLAT))


def extract_features(board, player):
    """Extract 139 features from a board state for a given player.

    Features:
        0-63:   net_mask flattened ((board == player) - (board == opponent))
        64-127: net_mask * WEIGHT_MATRIX flattened (normalized)
        128:    player piece count (normalized)
        129:    opponent piece count (normalized)
        130:    piece differential (normalized)
        131:    game phase (total_pieces / 64)
        132:    player corner count (/ 4)
        133:    opponent corner count (/ 4)
        134:    player X-square count (/ 4)
        135:    opponent X-square count (/ 4)
        136:    player edge count (/ 28)
        137:    opponent edge count (/ 28)
        138:    frontier ratio
    """
    opponent = -player
    player_mask = (board == player)
    opponent_mask = (board == opponent)
    net_mask = (player_mask.astype(np.float32) - opponent_mask.astype(np.float32))

    # 0-63: net_mask flattened
    net_flat = net_mask.flatten()

    # 64-127: positional encoding
    weighted_flat = net_flat * WEIGHT_MATRIX_FLAT / WEIGHT_NORM

    # Scalar features
    player_count = float(np.sum(player_mask))
    opponent_count = float(np.sum(opponent_mask))
    total_pieces = player_count + opponent_count

    # Piece counts normalized by 32 (max per side)
    f_player_count = player_count / 32.0
    f_opponent_count = opponent_count / 32.0
    f_piece_diff = (player_count - opponent_count) / max(total_pieces, 1.0)
    f_game_phase = total_pieces / 64.0

    # Corner control
    p_corners = sum(1 for r, c in CORNERS if board[r, c] == player)
    o_corners = sum(1 for r, c in CORNERS if board[r, c] == opponent)
    f_p_corners = p_corners / 4.0
    f_o_corners = o_corners / 4.0

    # X-square danger
    p_xsq = sum(1 for r, c in X_SQUARES if board[r, c] == player)
    o_xsq = sum(1 for r, c in X_SQUARES if board[r, c] == opponent)
    f_p_xsq = p_xsq / 4.0
    f_o_xsq = o_xsq / 4.0

    # Edge presence
    p_edges = float(np.sum(player_mask * EDGE_MASK))
    o_edges = float(np.sum(opponent_mask * EDGE_MASK))
    f_p_edges = p_edges / 28.0
    f_o_edges = o_edges / 28.0

    # Frontier ratio: pieces adjacent to empty squares
    empty = (board == 0)
    # Shift in all 8 directions and OR to find empty neighbors
    has_empty_neighbor = np.zeros((8, 8), dtype=bool)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(empty, dr, axis=0), dc, axis=1)
            # Zero out wrapped edges
            if dr == -1:
                shifted[7, :] = False
            elif dr == 1:
                shifted[0, :] = False
            if dc == -1:
                shifted[:, 7] = False
            elif dc == 1:
                shifted[:, 0] = False
            has_empty_neighbor |= shifted

    p_frontier = float(np.sum(player_mask & has_empty_neighbor))
    o_frontier = float(np.sum(opponent_mask & has_empty_neighbor))
    total_frontier = p_frontier + o_frontier
    f_frontier = (p_frontier - o_frontier) / max(total_frontier, 1.0)

    scalars = np.array([
        f_player_count, f_opponent_count, f_piece_diff, f_game_phase,
        f_p_corners, f_o_corners, f_p_xsq, f_o_xsq,
        f_p_edges, f_o_edges, f_frontier,
    ], dtype=np.float32)

    return np.concatenate([net_flat, weighted_flat, scalars])


class NNHeuristic:
    """Neural network heuristic using pure numpy inference.

    Architecture: 139 -> 1024 -> 512 -> 256 -> 1 (~800K parameters)

    Loads weights from a .npz file and evaluates board positions
    with the same signature as hand-crafted heuristics:
        heuristic(board, player) -> float
    """

    def __init__(self, weights_path):
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"NN weights not found at {weights_path}. "
                "Run training/train_nn.py first."
            )
        data = np.load(weights_path)
        self.w1 = data['w1'].astype(np.float32)  # (139, 1024)
        self.b1 = data['b1'].astype(np.float32)  # (1024,)
        self.w2 = data['w2'].astype(np.float32)  # (1024, 512)
        self.b2 = data['b2'].astype(np.float32)  # (512,)
        self.w3 = data['w3'].astype(np.float32)  # (512, 256)
        self.b3 = data['b3'].astype(np.float32)  # (256,)
        self.w4 = data['w4'].astype(np.float32)  # (256, 1)
        self.b4 = data['b4'].astype(np.float32)  # (1,)

    def __call__(self, board, player):
        x = extract_features(board, player)

        # Hidden layer 1: ReLU  (139 -> 1024)
        x = x @ self.w1 + self.b1
        x = np.maximum(x, 0)

        # Hidden layer 2: ReLU  (1024 -> 512)
        x = x @ self.w2 + self.b2
        x = np.maximum(x, 0)

        # Hidden layer 3: ReLU  (512 -> 256)
        x = x @ self.w3 + self.b3
        x = np.maximum(x, 0)

        # Output layer: tanh scaled to [-1000, 1000]  (256 -> 1)
        x = x @ self.w4 + self.b4
        return float(np.tanh(x[0]) * 1000.0)
