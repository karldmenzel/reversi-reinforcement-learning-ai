# Group 1, March 16, 2026

import pickle
import socket
import time
from random import choice

import numpy as np

from reversi import reversi


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
WEIGHT_MATRIX = np.array(
    [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [5, -2, 1, 0, 0, 1, -2, 5],
        [10, -2, 5, 1, 1, 5, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100],
    ]
)

# Early-game bonus for controlling the center 4x4 square (rows/cols 2-5).
# Holding the center forces the opponent to play on the outer ring, making
# edge and corner captures easier in the mid-to-late game.
# This bonus fades linearly to zero once 36 pieces are on the board.
CENTER_BONUS = np.zeros((8, 8))
CENTER_BONUS[2:6, 2:6] = np.array(
    [
        [8, 12, 12, 8],
        [12, 20, 20, 12],
        [12, 20, 20, 12],
        [8, 12, 12, 8],
    ]
)


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
        mobility_score = (
            100.0 * (player_moves - opponent_moves) / (player_moves + opponent_moves)
        )
    else:
        mobility_score = 0.0

    # Weighted combination
    if total_pieces > 52:
        # Late game: piece count matters more
        return positional_score + 3.0 * piece_score + mobility_score
    else:
        # Early/mid game: position and mobility matter more
        return 2.0 * positional_score + piece_score + 2.0 * mobility_score


# ── Heuristic selection ───────────────────────────────────────────────────────
# Set this to any function with the signature: heuristic(board, player) -> float
CHOSEN_HEURISTIC = heuristic_nic
# ─────────────────────────────────────────────────────────────────────────────

TIME_LIMIT = 4.0  # seconds per move
MAX_DEPTH = 12  # hard cap; iterative deepening rarely reaches this


class TimeUp(Exception):
    """Raised inside minimax when the deadline has been exceeded."""

    pass


def minimax(
    board, game, depth, alpha, beta, maximizing_player, player, deadline, heuristic
):
    """
    Minimax search with alpha-beta pruning and a hard time deadline.

    Raises TimeUp if the deadline is exceeded so the caller can fall back to
    the best move found at the previous completed depth.

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

    Returns:
        (score, best_move) tuple
    """
    if time.time() >= deadline:
        raise TimeUp()

    current_piece = player if maximizing_player else -player

    game.board = board.copy()
    legal_moves = get_legal_moves(game, current_piece)

    # Order moves by weight (best squares first) so alpha-beta
    # encounters tighter bounds earlier and prunes more branches. (eliminating unnecessary searches)
    legal_moves.sort(key=lambda m: WEIGHT_MATRIX[m[0], m[1]], reverse=True)

    # Escape conditions: max depth or no moves for either side
    if depth == 0 or len(legal_moves) == 0:
        # if no available moves it's the opponents "turn" and evaluate their options
        if len(legal_moves) == 0:
            game.board = board.copy()
            opponent_moves = get_legal_moves(game, -current_piece)

            # if they have no moves left. The game is over (not sure if this is necessary)
            if len(opponent_moves) == 0:
                # Game is over - evaluate by final piece count
                player_count = np.sum(board == player)
                opponent_count = np.sum(board == -player)
                if player_count > opponent_count:
                    return 10000 + player_count - opponent_count, None
                elif opponent_count > player_count:
                    return -10000 - opponent_count + player_count, None
                else:
                    return 0, None
            else:
                # if opponent has moves, recursively call this function as the minimizing player
                return minimax(
                    board,
                    game,
                    depth - 1,
                    alpha,
                    beta,
                    not maximizing_player,
                    player,
                    deadline,
                    heuristic,
                )

        # determine the best move by calculating the score through the heuristic
        return heuristic(board, player), None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = legal_moves[0]  # obtain the best base move in the sorted array
        # randomize best move
        best_move_list = []

        # for every move taken, you apply the "best move" to the board and evaluate the potential score through recursively calling minimax function
        # if the current move in the iteration has a greater score that's the new "best move"
        for move in legal_moves:
            new_board = apply_move(board, game, move[0], move[1], current_piece)
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
            )

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                if (
                    len(best_move_list) != 0
                    and max_eval > max(best_move_list, key=lambda x: x[0])[0]
                ):
                    best_move_list = []
                    best_move_list.append((max_eval, best_move))
                else:
                    best_move_list.append((max_eval, best_move))

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        best_move = choice(best_move_list)[1]
        return max_eval, best_move
    else:  # essentially doing the same as above but looking for the "worst" score (the score that is most detrimental to us)
        min_eval = float("inf")
        best_move = legal_moves[0]

        for move in legal_moves:
            new_board = apply_move(board, game, move[0], move[1], current_piece)
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
            )

            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move

            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff

        return min_eval, best_move


def get_best_move(board, game, player, heuristic):
    deadline = time.time() + TIME_LIMIT
    best_move = get_legal_moves(game, player)[0]  # safe fallback
    for depth in range(1, MAX_DEPTH + 1):
        # try to find best move in given time-limit if time limit is reached. throw exception. return the best move so far
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
            )
            best_move = move  # only update on a fully completed search
            # print(f"  depth {depth} -> {best_move}")
        except TimeUp:
            # print(f"  time up at depth {depth}, using depth {depth - 1} result")
            break

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
    game_socket = socket.socket()
    game_socket.connect(("127.0.0.1", 33333))
    game = reversi()

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
        print(turn)
        print(board)

        # Find best move via iterative-deepening minimax (4-second limit)
        game.board = board.copy()
        legal_moves = get_legal_moves(game, turn)

        if len(legal_moves) == 0:
            x, y = -1, -1
        else:
            best_move = get_best_move(board, game, turn, CHOSEN_HEURISTIC)
            x, y = best_move
            print(f"Best move: ({x}, {y})")

        # Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x, y]))


if __name__ == "__main__":
    main()
