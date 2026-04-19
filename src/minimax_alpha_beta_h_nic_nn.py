#Zijie Zhang, Sep.24/2023

import time
import socket, pickle
import numpy as np
from reversi import reversi

from utils import get_legal_moves, apply_move, WEIGHT_MATRIX
from heuristic_functions import heuristic_nic

# ── Heuristic selection ───────────────────────────────────────────────────────
# Set this to any function with the signature: heuristic(board, player) -> float
# Option 1: Classic hand-crafted heuristic
# CHOSEN_HEURISTIC = heuristic_nic
#
# Option 2: Neural network heuristic (requires trained weights)
import os
_weights_path = os.path.join(os.path.dirname(__file__), '', 'weights', 'heuristic_v1.npz')

from nn_heuristic import NNHeuristic
CHOSEN_HEURISTIC = NNHeuristic(_weights_path)
# ─────────────────────────────────────────────────────────────────────────────

TIME_LIMIT = 4.75   # seconds per move
MAX_DEPTH   = 12   # hard cap; iterative deepening rarely reaches this


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

def minimax(board, game, depth, alpha, beta, maximizing_player, player, deadline, heuristic):
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
                evaluate_final_board(board, player)
            else:
                # if opponent has moves, recursively call this function as the minimizing player
                return minimax(board, game, depth - 1, alpha, beta,
                               not maximizing_player, player, deadline, heuristic)

        # determine the best move by calculating the score through the heuristic
        return heuristic(board, player), None

    if maximizing_player:
        max_eval = float('-inf')
        best_move = legal_moves[0] #obtain the best base move in the sorted array

        # for every move taken, you apply the "best move" to the board and evaluate the potential score through recursively calling minimax function
        # if the current move in the iteration has a greater score that's the new "best move"
        for move in legal_moves:
            new_board = apply_move(board, game, move[0], move[1], current_piece)
            eval_score, _ = minimax(new_board, game, depth - 1, alpha, beta,
                                    False, player, deadline, heuristic)

            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move

            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff

        return max_eval, best_move
    else: # essentially doing the same as above but looking for the "worst" score (the score that is most detrimental to us)
        min_eval = float('inf')
        best_move = legal_moves[0]

        for move in legal_moves:
            new_board = apply_move(board, game, move[0], move[1], current_piece)
            eval_score, _ = minimax(new_board, game, depth - 1, alpha, beta,
                                    True, player, deadline, heuristic)

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
            _, move = minimax(board, game, depth,
                              float('-inf'), float('inf'),
                              True, player, deadline, heuristic)
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
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    while True:

        #Receive play request from the server
        #turn : 1 --> you are playing as white | -1 --> you are playing as black
        #board : 8*8 numpy array
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        #Turn = 0 indicates game ended
        if turn == 0:
            game_socket.close()
            return

        #Debug info
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

        #Send your move to the server. Send (x,y) = (-1,-1) to tell the server you have no hand to play
        game_socket.send(pickle.dumps([x, y]))


if __name__ == '__main__':
    main()
