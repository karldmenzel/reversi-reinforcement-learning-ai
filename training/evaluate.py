"""Tournament evaluation: NN heuristic vs heuristic_nic.

Runs head-to-head matches using the auto-server pattern.
Each matchup plays games as both colors for fairness.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reversi import reversi
from utils import get_legal_moves
from heuristic_functions import heuristic_nic
from nn_heuristic import NNHeuristic
from minimax_alpha_beta_h_nic import get_best_move

# ── Configuration ──────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'heuristic_v1.npz')
NUM_GAMES = 50  # games per color assignment (total = 2 * NUM_GAMES)
# ───────────────────────────────────────────────────────────────────────────────


def make_choose_move(heuristic):
    """Create a choose_move function compatible with AutoGameServer."""
    def choose_move(turn, board, game):
        search_game = reversi()
        search_game.board = board.copy()
        legal_moves = get_legal_moves(search_game, turn)
        if len(legal_moves) == 0:
            return [-1, -1]
        x, y = get_best_move(board, search_game, turn, heuristic)
        return [x, y]
    return choose_move


def play_game(player1_fn, player2_fn):
    """Play a single game. Returns +1 white wins, -1 black wins, 0 draw."""
    game = reversi()
    turn = 1
    consecutive_passes = 0

    while True:
        current_fn = player1_fn if turn == 1 else player2_fn
        move = current_fn(turn, game.board.copy(), game)
        x, y = move

        if x == -1 and y == -1:
            consecutive_passes += 1
        else:
            result = game.step(x, y, turn)
            if result >= 0:
                consecutive_passes = 0
            else:
                consecutive_passes += 1

        if consecutive_passes >= 2:
            break
        turn *= -1

    white = game.white_count
    black = game.black_count
    if white > black:
        return 1
    elif black > white:
        return -1
    return 0


def main():
    print(f"Loading NN weights from {WEIGHTS_PATH}")
    nn_heuristic = NNHeuristic(WEIGHTS_PATH)

    nn_move = make_choose_move(nn_heuristic)
    classic_move = make_choose_move(heuristic_nic)

    nn_wins = 0
    classic_wins = 0
    draws = 0

    total_games = NUM_GAMES * 2
    start_time = time.time()

    # NN as White
    print(f"\n--- NN (White) vs Classic (Black): {NUM_GAMES} games ---")
    for i in range(NUM_GAMES):
        result = play_game(nn_move, classic_move)
        if result == 1:
            nn_wins += 1
        elif result == -1:
            classic_wins += 1
        else:
            draws += 1
        print(f"  Game {i + 1}/{NUM_GAMES}: "
              f"{'NN wins' if result == 1 else 'Classic wins' if result == -1 else 'Draw'} | "
              f"Running: NN {nn_wins} - Classic {classic_wins} - Draws {draws}")

    # NN as Black
    print(f"\n--- Classic (White) vs NN (Black): {NUM_GAMES} games ---")
    for i in range(NUM_GAMES):
        result = play_game(classic_move, nn_move)
        if result == -1:
            nn_wins += 1
        elif result == 1:
            classic_wins += 1
        else:
            draws += 1
        print(f"  Game {i + 1}/{NUM_GAMES}: "
              f"{'NN wins' if result == -1 else 'Classic wins' if result == 1 else 'Draw'} | "
              f"Running: NN {nn_wins} - Classic {classic_wins} - Draws {draws}")

    elapsed = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"RESULTS ({total_games} games, {elapsed:.0f}s)")
    print(f"{'=' * 50}")
    print(f"  NN Heuristic:  {nn_wins} wins ({100 * nn_wins / total_games:.1f}%)")
    print(f"  Classic:       {classic_wins} wins ({100 * classic_wins / total_games:.1f}%)")
    print(f"  Draws:         {draws} ({100 * draws / total_games:.1f}%)")
    print(f"{'=' * 50}")

    if nn_wins > classic_wins:
        print("NN heuristic is Wins.")
    elif classic_wins > nn_wins:
        print("Classic heuristic Wins. :(")
    else:
        print("It's a tie!")


if __name__ == '__main__':
    main()
