import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reversi import reversi
from utils import get_legal_moves

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

# Edge runs from corners for stable disc computation
# Each entry: (corner, list of edge directions to scan)
_CORNER_EDGES = [
    ((0, 0), [(0, 1), (1, 0)]),   # top-left  -> right, down
    ((0, 7), [(0, -1), (1, 0)]),  # top-right -> left, down
    ((7, 0), [(0, 1), (-1, 0)]),  # bot-left  -> right, up
    ((7, 7), [(0, -1), (-1, 0)]), # bot-right -> left, up
]


def _count_stable_discs(board, player):
    """Count stable discs using corner-anchored edge runs.

    A disc is stable if it's part of a contiguous run of same-color discs
    along an edge starting from an owned corner. This is a simplified but
    effective approximation that captures the most important stable discs.
    """
    stable = np.zeros((8, 8), dtype=bool)

    for (cr, cc), directions in _CORNER_EDGES:
        if board[cr, cc] != player:
            continue
        # Mark corner as stable
        stable[cr, cc] = True
        # Scan along each edge from this corner
        for dr, dc in directions:
            r, c = cr + dr, cc + dc
            while 0 <= r < 8 and 0 <= c < 8 and board[r, c] == player:
                stable[r, c] = True
                r += dr
                c += dc

    return int(np.sum(stable))


def extract_features(board, player, game=None):
    """Extract 144 features from a board state for a given player.

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
        139:    player mobility (/ 20)
        140:    opponent mobility (/ 20)
        141:    mobility ratio
        142:    player stable discs (/ 64)
        143:    opponent stable discs (/ 64)
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

    # Mobility features
    if game is not None:
        p_moves = len(get_legal_moves(game, player))
        o_moves = len(get_legal_moves(game, opponent))
        f_p_mobility = p_moves / 20.0
        f_o_mobility = o_moves / 20.0
        f_mobility_ratio = (p_moves - o_moves) / max(p_moves + o_moves, 1)
    else:
        f_p_mobility = 0.0
        f_o_mobility = 0.0
        f_mobility_ratio = 0.0

    # Stable disc features
    p_stable = _count_stable_discs(board, player)
    o_stable = _count_stable_discs(board, opponent)
    f_p_stable = p_stable / 64.0
    f_o_stable = o_stable / 64.0

    scalars = np.array([
        f_player_count, f_opponent_count, f_piece_diff, f_game_phase,
        f_p_corners, f_o_corners, f_p_xsq, f_o_xsq,
        f_p_edges, f_o_edges, f_frontier,
        f_p_mobility, f_o_mobility, f_mobility_ratio,
        f_p_stable, f_o_stable,
    ], dtype=np.float32)

    return np.concatenate([net_flat, weighted_flat, scalars])


class NNHeuristic:
    """Neural network heuristic using pure numpy inference.

    Architecture: 144 -> 512 -> 256 -> 128 -> 1 (~170K parameters)
    ReLU activations, no BatchNorm (simpler, avoids train/eval mismatch).
    Backward-compatible with older BN weights via _has_bn flag.

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
        self._input_dim = data['w1'].shape[0]  # detect old (139) vs new (144) weights
        self.w1 = data['w1'].astype(np.float32)
        self.b1 = data['b1'].astype(np.float32)
        self.w2 = data['w2'].astype(np.float32)
        self.b2 = data['b2'].astype(np.float32)
        self.w3 = data['w3'].astype(np.float32)
        self.b3 = data['b3'].astype(np.float32)
        self.w4 = data['w4'].astype(np.float32)
        self.b4 = data['b4'].astype(np.float32)

        # Batch norm parameters (optional for backward compat with old weights)
        self._has_bn = 'bn1_mean' in data
        if self._has_bn:
            self.bn1_mean = data['bn1_mean'].astype(np.float32)
            self.bn1_var = data['bn1_var'].astype(np.float32)
            self.bn1_weight = data['bn1_weight'].astype(np.float32)
            self.bn1_bias = data['bn1_bias'].astype(np.float32)
            self.bn2_mean = data['bn2_mean'].astype(np.float32)
            self.bn2_var = data['bn2_var'].astype(np.float32)
            self.bn2_weight = data['bn2_weight'].astype(np.float32)
            self.bn2_bias = data['bn2_bias'].astype(np.float32)
            self.bn3_mean = data['bn3_mean'].astype(np.float32)
            self.bn3_var = data['bn3_var'].astype(np.float32)
            self.bn3_weight = data['bn3_weight'].astype(np.float32)
            self.bn3_bias = data['bn3_bias'].astype(np.float32)

    def _bn_forward(self, x, mean, var, weight, bias):
        """Batch norm in eval mode: element-wise affine transform."""
        return weight * (x - mean) / np.sqrt(var + 1e-5) + bias

    def __call__(self, board, player):
        # Create a temporary game for mobility computation
        game = reversi()
        game.board = board.copy()
        x = extract_features(board, player, game=game)

        # Truncate features for backward compat with old 139-dim weights
        if self._input_dim < len(x):
            x = x[:self._input_dim]

        # Hidden layer 1: Linear -> BN -> ReLU
        x = x @ self.w1 + self.b1
        if self._has_bn:
            x = self._bn_forward(x, self.bn1_mean, self.bn1_var,
                                 self.bn1_weight, self.bn1_bias)
        x = np.maximum(x, 0)

        # Hidden layer 2: Linear -> BN -> ReLU
        x = x @ self.w2 + self.b2
        if self._has_bn:
            x = self._bn_forward(x, self.bn2_mean, self.bn2_var,
                                 self.bn2_weight, self.bn2_bias)
        x = np.maximum(x, 0)

        # Hidden layer 3: Linear -> BN -> ReLU
        x = x @ self.w3 + self.b3
        if self._has_bn:
            x = self._bn_forward(x, self.bn3_mean, self.bn3_var,
                                 self.bn3_weight, self.bn3_bias)
        x = np.maximum(x, 0)

        # Output layer: tanh scaled to [-1000, 1000]
        x = x @ self.w4 + self.b4
        return float(np.tanh(x[0]) * 1000.0)
