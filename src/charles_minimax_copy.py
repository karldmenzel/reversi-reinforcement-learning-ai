# Zijie Zhang, Sep.24/2023
# Optimized with: Zobrist TT, PV move ordering, killer moves, incremental hashing

# ── Heuristic selection ───────────────────────────────────────────────────────
# Set this to any function with the signature: heuristic(board, player) -> float
# Tries to load the NN heuristic; falls back to the classic hand-crafted one
# if the weights file is missing or empty.
import os
import pickle
import socket
import time

import numpy as np

from heuristic_functions import heuristic_nic
from nn_heuristic import NNHeuristic
from reversi import reversi
from utils import (
    # WEIGHT_MATRIX,
    ZOBRIST_TURN,
    apply_move_with_hash,
    get_legal_moves,
    zobrist_hash,
)

_weights_path = os.path.join(
    os.path.dirname(__file__), "", "weights", "heuristic_best_v4.npz"
)

WEIGHT_MATRIX = np.array(
    [
        [1000, -20, 100, 50, 50, 100, -20, 1000],
        [-20, -20, 2, 2, 2, 2, -20, -20],
        [100, 2, 50, 10, 10, 50, 2, 100],
        [50, 2, 10, 0, 0, 10, 2, 50],
        [50, 2, 10, 0, 0, 10, 2, 50],
        [100, 2, 50, 10, 10, 50, 2, 100],
        [-20, -20, 2, 2, 2, 2, -20, -20],
        [1000, -20, 100, 50, 50, 100, -20, 1000],
    ]
)

try:
    CHOSEN_HEURISTIC = NNHeuristic(_weights_path)
except (FileNotFoundError, EOFError):
    print(
        f"[WARNING] NN weights not found or empty at {_weights_path}, "
        "falling back to classic heuristic."
    )
    CHOSEN_HEURISTIC = heuristic_nic
# ─────────────────────────────────────────────────────────────────────────────

TIME_LIMIT = 4.90  # seconds per move
MAX_DEPTH = 15  # hard cap;

# ── Transposition table flags ────────────────────────────────────────────────
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2
depth_tot = 0


class TimeUp(Exception):
    """Raised inside minimax when the deadline has been exceeded."""

    pass


def evaluate_final_board(board, player):
    player_count = np.sum(board == player)
    opponent_count = np.sum(board == -player)
    if player_count > opponent_count:
        return 10000 + player_count - opponent_count, None
    elif opponent_count > player_count:
        return -10000 - opponent_count + player_count, None
    else:
        return 0, None


def _order_moves(legal_moves, tt_move, killer_moves_at_depth):
    """Sort moves by: TT move first, then killer moves, then WEIGHT_MATRIX."""

    def sort_key(m):
        if tt_move is not None and m[0] == tt_move[0] and m[1] == tt_move[1]:
            return (0, 0)
        if killer_moves_at_depth is not None:
            for km in killer_moves_at_depth:
                if km is not None and m[0] == km[0] and m[1] == km[1]:
                    return (1, 0)
        return (2, -WEIGHT_MATRIX[m[0], m[1]])

    return sorted(legal_moves, key=sort_key)


def minimax(
    board,
    game,
    depth,
    alpha,
    beta,
    maximizing_player,
    player,
    deadline,
    heuristic,
    z_hash,
    tt,
    killer_moves,
    pv_move,
):
    """
    Minimax search with alpha-beta pruning, transposition table, killer moves,
    PV move ordering, and a hard time deadline.

    Args:
        board:             current board state (np.ndarray)
        game:              reversi instance used for move simulation
        depth:             remaining search depth
        alpha:             best score the maximizer can guarantee
        beta:              best score the minimizer can guarantee
        maximizing_player: True if current layer maximizes
        player:            the piece value (1 or -1) of the original caller
        deadline:          time.time() value after which search must stop
        heuristic:         heuristic function that determines the best move
        z_hash:            current Zobrist hash of the board
        tt:                transposition table dict
        killer_moves:      killer_moves[depth] = [move1, move2]
        pv_move:           principal variation move hint (from previous ID depth)

    Returns:
        (score, best_move) tuple
    """
    if time.time() >= deadline:
        raise TimeUp()

    orig_alpha = alpha
    orig_beta = beta

    # ── TT probe ─────────────────────────────────────────────────────────
    tt_entry = tt.get(z_hash)
    tt_move = None
    if tt_entry is not None and tt_entry[0] >= depth:
        tt_depth, flag, tt_score, tt_best = tt_entry
        tt_move = tt_best
        if flag == TT_EXACT:
            return tt_score, tt_best
        if flag == TT_LOWERBOUND:
            alpha = max(alpha, tt_score)
        elif flag == TT_UPPERBOUND:
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score, tt_best
    elif tt_entry is not None:
        # Shallower entry — still use its best move for ordering
        tt_move = tt_entry[3]

    current_piece = player if maximizing_player else -player

    game.board = board.copy()
    legal_moves = get_legal_moves(game, current_piece)

    # ── Escape conditions: max depth or no moves for either side ─────────
    if depth == 0 or len(legal_moves) == 0:
        if len(legal_moves) == 0:
            game.board = board.copy()
            opponent_moves = get_legal_moves(game, -current_piece)

            if len(opponent_moves) == 0:
                # Game is over — use definitive piece-count score (BUG FIX: was missing return)
                return evaluate_final_board(board, player)
            else:
                # Pass turn to opponent (don't decrement depth — no move was played)
                opp_hash = z_hash ^ ZOBRIST_TURN
                return minimax(
                    board,
                    game,
                    depth,
                    alpha,
                    beta,
                    not maximizing_player,
                    player,
                    deadline,
                    heuristic,
                    opp_hash,
                    tt,
                    killer_moves,
                    None,
                )

        score = heuristic(board, player)
        return score, None

    # ── Move ordering ────────────────────────────────────────────────────
    # Priority: PV move (root only) > TT move > killer moves > WEIGHT_MATRIX
    hint_move = pv_move if pv_move is not None else tt_move
    km_at_depth = killer_moves.get(depth)
    ordered_moves = _order_moves(legal_moves, hint_move, km_at_depth)

    if maximizing_player:
        max_eval = float("-inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            new_board, new_hash = apply_move_with_hash(
                board, game, move[0], move[1], current_piece, z_hash
            )
            eval_score, _ = minimax(
                new_board,
                game,
                depth - 1,
                alpha,
                beta,
                False,
                player,
                deadline,
                heuristic,
                new_hash,
                tt,
                killer_moves,
                None,
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                _store_killer(killer_moves, depth, move)
                break

        # ── TT store ─────────────────────────────────────────────────────
        if max_eval >= beta:
            flag = TT_LOWERBOUND
        elif max_eval <= orig_alpha:
            flag = TT_UPPERBOUND
        else:
            flag = TT_EXACT
        tt[z_hash] = (depth, flag, max_eval, best_move)

        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = ordered_moves[0]

        for move in ordered_moves:
            new_board, new_hash = apply_move_with_hash(
                board, game, move[0], move[1], current_piece, z_hash
            )
            eval_score, _ = minimax(
                new_board,
                game,
                depth - 1,
                alpha,
                beta,
                True,
                player,
                deadline,
                heuristic,
                new_hash,
                tt,
                killer_moves,
                None,
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                _store_killer(killer_moves, depth, move)
                break

        # ── TT store ─────────────────────────────────────────────────────
        if min_eval <= alpha:
            flag = TT_UPPERBOUND
        elif min_eval >= orig_beta:
            flag = TT_LOWERBOUND
        else:
            flag = TT_EXACT
        tt[z_hash] = (depth, flag, min_eval, best_move)

        return min_eval, best_move


def _store_killer(killer_moves, depth, move):
    if depth not in killer_moves:
        killer_moves[depth] = [move, None]
    else:
        slots = killer_moves[depth]
        # Don't store duplicates
        if slots[0] is not None and slots[0][0] == move[0] and slots[0][1] == move[1]:
            return
        slots[1] = slots[0]
        slots[0] = move


def get_best_move(board, game, player, heuristic):
    global depth_tot
    deadline = time.time() + TIME_LIMIT
    best_move = get_legal_moves(game, player)[0]  # safe fallback

    z_hash = zobrist_hash(board)
    tt = {}  # fresh transposition table per move decision
    prev_best_move = None
    c_Depth = 0

    for depth in range(1, MAX_DEPTH + 1):
        killer_moves = {}  # fresh killer table per ID depth
        try:
            _, move = minimax(
                board,
                game,
                depth,
                float("-inf"),
                float("inf"),
                True,
                player,
                deadline,
                heuristic,
                z_hash,
                tt,
                killer_moves,
                prev_best_move,
            )
            best_move = move  # only update on a fully completed search
            prev_best_move = move  # PV hint for next depth
            c_Depth += 1
        except TimeUp:
            break

    depth_tot += c_Depth
    return best_move


def choose_move(turn, board, game) -> list:
    # A copy of the board allows the algo to mutate the board freely without effecting the actual game board.
    search_game = reversi()
    search_game.board = board.copy()

    legal_moves = get_legal_moves(search_game, turn)

    if len(legal_moves) == 0:
        return [-1, -1]

    x, y = get_best_move(board, search_game, turn, CHOSEN_HEURISTIC)
    return [x, y]


def main():
    global depth_tot
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))
    game = reversi()
    number_of_moves = 0
    while True:
        # Receive play request from the server
        # turn : 1 --> you are playing as white | -1 --> you are playing as black
        # board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        # Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        # Debug info
        # print(turn)
        # print(board)

        # Find best move via iterative-deepening minimax (4-second limit)
        game.board = board.copy()
        legal_moves = get_legal_moves(game, turn)

        if len(legal_moves) == 0:
            x, y = -1, -1
        else:
            best_move = get_best_move(board, game, turn, CHOSEN_HEURISTIC)
            number_of_moves += 1
            x, y = best_move
            print(f"Best move: ({x}, {y})")

        # Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x, y]))
    print(f"Average depth: {depth_tot / number_of_moves}")


if __name__ == "__main__":
    main()
