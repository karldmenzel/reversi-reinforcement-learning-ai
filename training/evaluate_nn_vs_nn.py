"""Head-to-head evaluation: NN v4 vs NN v5.

Compares minimax_alpha_beta_h_nic_nn (v4 weights) against
minimax_alpha_beta_h_nic_nn_copy (v5 weights).
Each matchup plays games as both colors for fairness.
Uses multiprocessing to parallelize independent games.
"""

import os
import sys
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from reversi import reversi

# ── Configuration ──────────────────────────────────────────────────────────────
NUM_GAMES = 5  # games per color assignment (total = 2 * NUM_GAMES)
NUM_WORKERS = max(1, cpu_count() - 1)
PLAYER_A_NAME = "NN_v4"
PLAYER_B_NAME = "NN_v5"
# ───────────────────────────────────────────────────────────────────────────────


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
        return 1, white, black
    elif black > white:
        return -1, white, black
    return 0, white, black


def _play_eval_game(args):
    """Worker: play one evaluation game. Returns (a_is_white, result, white, black)."""
    game_idx, a_is_white = args
    # Import inside worker — module-level closures aren't picklable
    from minimax_alpha_beta_h_nic_nn import choose_move as v4_choose_move
    from minimax_alpha_beta_h_nic_nn_copy import choose_move as v5_choose_move

    if a_is_white:
        result, white, black = play_game(v4_choose_move, v5_choose_move)
    else:
        result, white, black = play_game(v5_choose_move, v4_choose_move)
    return a_is_white, result, white, black


def main():
    print(f"Player A: {PLAYER_A_NAME} (heuristic_best_v4.npz)")
    print(f"Player B: {PLAYER_B_NAME} (heuristic_best_v5.npz)")
    print(f"Using {NUM_WORKERS} worker processes")

    a_wins = 0
    b_wins = 0
    draws = 0

    total_games = NUM_GAMES * 2
    start_time = time.time()

    # Build work items: half as white, half as black
    work_items = [(i, True) for i in range(NUM_GAMES)] + [
        (i, False) for i in range(NUM_GAMES)
    ]

    print(f"\n--- Running {total_games} games ({NUM_GAMES} per color) ---\n")

    with Pool(processes=NUM_WORKERS) as pool:
        for completed, (a_is_white, result, white, black) in enumerate(
            pool.imap_unordered(_play_eval_game, work_items), 1
        ):
            if a_is_white:
                if result == 1:
                    a_wins += 1
                    outcome = f"{PLAYER_A_NAME} wins"
                elif result == -1:
                    b_wins += 1
                    outcome = f"{PLAYER_B_NAME} wins"
                else:
                    draws += 1
                    outcome = "Draw"
            else:
                if result == -1:
                    a_wins += 1
                    outcome = f"{PLAYER_A_NAME} wins"
                elif result == 1:
                    b_wins += 1
                    outcome = f"{PLAYER_B_NAME} wins"
                else:
                    draws += 1
                    outcome = "Draw"

            color = "White" if a_is_white else "Black"
            print(
                f"  Game {completed}/{total_games} ({PLAYER_A_NAME}={color}): "
                f"{outcome} ({white}-{black}) | "
                f"Running: {PLAYER_A_NAME} {a_wins} - {PLAYER_B_NAME} {b_wins} - Draws {draws}"
            )

    elapsed = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"RESULTS ({total_games} games, {elapsed:.0f}s)")
    print(f"{'=' * 50}")
    print(f"  {PLAYER_A_NAME}:  {a_wins} wins ({100 * a_wins / total_games:.1f}%)")
    print(f"  {PLAYER_B_NAME}:  {b_wins} wins ({100 * b_wins / total_games:.1f}%)")
    print(f"  Draws:    {draws} ({100 * draws / total_games:.1f}%)")
    print(f"{'=' * 50}")

    if a_wins > b_wins:
        print(f"{PLAYER_A_NAME} wins the tournament!")
    elif b_wins > a_wins:
        print(f"{PLAYER_B_NAME} wins the tournament!")
    else:
        print("It's a tie!")


if __name__ == "__main__":
    main()
