"""Iterative training pipeline: generate data -> train -> evaluate -> repeat.

Runs at least MIN_CYCLES full iterations. After each training round, the new
model is evaluated against the previous best. If it wins, it becomes the new
best and is used for the next round of data generation.
"""

import sys
import os
import shutil
import time
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reversi import reversi
from utils import get_legal_moves
from nn_heuristic import NNHeuristic
from heuristic_functions import heuristic_nic
from minimax_alpha_beta_h_nic_nn import get_best_move

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_CYCLES = 5
EVAL_GAMES = 25          # games per color (total = 2 * EVAL_GAMES)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights')
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, 'heuristic_v1.npz')
CANDIDATE_WEIGHTS = os.path.join(WEIGHTS_DIR, 'heuristic_candidate.npz')
NUM_WORKERS = max(1, cpu_count() - 1)
# ──────────────────────────────────────────────────────────────────────────────


def make_choose_move(heuristic_fn):
    """Create a choose_move function for evaluation games."""
    def choose_move(turn, board, game):
        search_game = reversi()
        search_game.board = board.copy()
        legal_moves = get_legal_moves(search_game, turn)
        if len(legal_moves) == 0:
            return [-1, -1]
        x, y = get_best_move(board, search_game, turn, heuristic_fn)
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


def _play_eval_game(args):
    """Worker: play one evaluation game. Returns (candidate_is_white, result)."""
    game_idx, candidate_is_white, candidate_weights, baseline_weights = args
    # Load heuristics inside worker (closures aren't picklable)
    candidate_fn = NNHeuristic(candidate_weights)
    baseline_fn = NNHeuristic(baseline_weights) if baseline_weights != '__CH__' else heuristic_nic
    candidate_move = make_choose_move(candidate_fn)
    baseline_move = make_choose_move(baseline_fn)

    if candidate_is_white:
        result = play_game(candidate_move, baseline_move)
    else:
        result = play_game(baseline_move, candidate_move)
    return candidate_is_white, result


def evaluate(candidate_weights, baseline_weights, num_games):
    """Run a parallel head-to-head tournament. Returns (candidate_wins, baseline_wins, draws).

    Args:
        candidate_weights: path to candidate .npz weights
        baseline_weights: path to baseline .npz weights, or None for classic heuristic
        num_games: games per color assignment
    """
    c_wins = 0
    b_wins = 0
    draws = 0
    total_games = num_games * 2

    work_items = [(i, True, candidate_weights, baseline_weights) for i in range(num_games)] + \
                 [(i, False, candidate_weights, baseline_weights) for i in range(num_games)]

    with Pool(processes=NUM_WORKERS) as pool:
        for completed, (candidate_is_white, result) in enumerate(
            pool.imap_unordered(_play_eval_game, work_items), 1
        ):
            if candidate_is_white:
                if result == 1:
                    c_wins += 1
                elif result == -1:
                    b_wins += 1
                else:
                    draws += 1
                outcome = 'Cand wins' if result == 1 else 'Base wins' if result == -1 else 'Draw'
            else:
                if result == -1:
                    c_wins += 1
                elif result == 1:
                    b_wins += 1
                else:
                    draws += 1
                outcome = 'Cand wins' if result == -1 else 'Base wins' if result == 1 else 'Draw'

            color = "CandW" if candidate_is_white else "CandB"
            print(f"  [{color}] Game {completed}/{total_games}: "
                  f"{outcome} | Candidate {c_wins} - Baseline {b_wins} - Draws {draws}")

    return c_wins, b_wins, draws


def run_generate_data():
    """Run data generation as a subprocess so it picks up the latest weights."""
    import subprocess
    script = os.path.join(os.path.dirname(__file__), 'generate_data.py')
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__))
    if result.returncode != 0:
        raise RuntimeError("Data generation failed")


def run_training(output_path):
    """Run training as a subprocess, outputting to a specific weights file."""
    import subprocess
    script = os.path.join(os.path.dirname(__file__), 'train_nn.py')
    env = os.environ.copy()
    env['NN_OUTPUT_PATH'] = output_path
    result = subprocess.run([sys.executable, script], cwd=os.path.dirname(__file__), env=env)
    if result.returncode != 0:
        raise RuntimeError("Training failed")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("=" * 60)
    print("ITERATIVE TRAINING PIPELINE")
    print(f"  Cycles: {MIN_CYCLES}")
    print(f"  Eval games per color: {EVAL_GAMES} (total: {EVAL_GAMES * 2})")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 60)

    for cycle in range(1, MIN_CYCLES + 1):
        cycle_start = time.time()
        print(f"\n{'#' * 60}")
        print(f"# CYCLE {cycle}/{MIN_CYCLES}")
        print(f"{'#' * 60}")

        # ── Step 1: Generate data ─────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 1/3: Generating self-play data...")
        run_generate_data()

        # ── Step 2: Train ─────────────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 2/3: Training NN...")
        run_training(CANDIDATE_WEIGHTS)

        # ── Step 3: Evaluate ──────────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 3/3: Evaluating candidate vs baseline...")

        baseline_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else '__CH__'
        baseline_name = "previous best NN" if baseline_weights != '__CH__' else "classic heuristic"
        baseline_is_ch = (baseline_weights == '__CH__')

        print(f"  Candidate: {CANDIDATE_WEIGHTS}")
        print(f"  Baseline:  {baseline_name}")

        # ── Gate 1: candidate vs baseline ────────────────────────────
        print(f"\n  [Gate 1] Candidate vs {baseline_name}")
        c_wins, b_wins, draws = evaluate(CANDIDATE_WEIGHTS, baseline_weights, EVAL_GAMES)
        total = c_wins + b_wins + draws

        print(f"\n  Gate 1 Results: Candidate {c_wins}/{total} "
              f"({100*c_wins/total:.1f}%) | "
              f"Baseline {b_wins}/{total} ({100*b_wins/total:.1f}%) | "
              f"Draws {draws}")

        passed_gate1 = c_wins >= b_wins

        # ── Gate 2: candidate vs CH (skip if baseline is already CH) ─
        if baseline_is_ch:
            passed_gate2 = True
            print(f"\n  [Gate 2] Skipped (baseline is already CH)")
        else:
            print(f"\n  [Gate 2] Candidate vs classic heuristic (CH)")
            ch_c_wins, ch_b_wins, ch_draws = evaluate(
                CANDIDATE_WEIGHTS, '__CH__', EVAL_GAMES)
            ch_total = ch_c_wins + ch_b_wins + ch_draws

            print(f"\n  Gate 2 Results: Candidate {ch_c_wins}/{ch_total} "
                  f"({100*ch_c_wins/ch_total:.1f}%) | "
                  f"CH {ch_b_wins}/{ch_total} ({100*ch_b_wins/ch_total:.1f}%) | "
                  f"Draws {ch_draws}")
            passed_gate2 = ch_c_wins >= ch_b_wins

        # Promote candidate only if it passes both gates
        promoted = passed_gate1 and passed_gate2
        if promoted:
            shutil.copy2(CANDIDATE_WEIGHTS, BEST_WEIGHTS)
            print(f"  >> Candidate PROMOTED to best weights")
        else:
            reasons = []
            if not passed_gate1:
                reasons.append("lost to baseline")
            if not passed_gate2:
                reasons.append("lost to CH")
            print(f"  >> Candidate rejected ({', '.join(reasons)})")

        # Clean up candidate file
        if os.path.exists(CANDIDATE_WEIGHTS):
            os.remove(CANDIDATE_WEIGHTS)

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"  Best weights: {BEST_WEIGHTS}")
    print(f"={'=' * 59}")


if __name__ == '__main__':
    main()
