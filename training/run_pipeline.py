"""Iterative training pipeline: generate data -> train -> evaluate -> repeat.

Runs at least MIN_CYCLES full iterations. After each training round, the new
model is evaluated against the previous best. If it wins, it becomes the new
best and is used for the next round of data generation.
"""

import json
import os
import shutil
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from cycle_utils import (
    CYCLE_CONFIG_PATH,
    DEFAULTS,
    TRAIN_METRICS_PATH,
    load_cycle_config,
    load_train_metrics,
)

from heuristic_functions import heuristic_nic
from minimax_alpha_beta_h_nic_nn import get_best_move
from nn_heuristic import NNHeuristic
from reversi import reversi
from utils import WEIGHT_MATRIX, get_legal_moves

# ── Configuration ─────────────────────────────────────────────────────────────
MIN_CYCLES = 5
EVAL_GAMES = 50  # games per color (total = 2 * EVAL_GAMES)
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "weights")
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "heuristic_v1.npz")
CANDIDATE_WEIGHTS = os.path.join(WEIGHTS_DIR, "heuristic_candidate.npz")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TRAINING_DATA_PATH = os.path.join(DATA_DIR, "training_data.npz")
MAX_ACCUMULATED_SAMPLES = 7_000_000  # keep ~2 cycles of augmented data
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
    baseline_fn = (
        NNHeuristic(baseline_weights) if baseline_weights != "__CH__" else heuristic_nic
    )
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

    work_items = [
        (i, True, candidate_weights, baseline_weights) for i in range(num_games)
    ] + [(i, False, candidate_weights, baseline_weights) for i in range(num_games)]

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
                outcome = (
                    "Cand wins"
                    if result == 1
                    else "Base wins"
                    if result == -1
                    else "Draw"
                )
            else:
                if result == -1:
                    c_wins += 1
                elif result == 1:
                    b_wins += 1
                else:
                    draws += 1
                outcome = (
                    "Cand wins"
                    if result == -1
                    else "Base wins"
                    if result == 1
                    else "Draw"
                )

            color = "CandW" if candidate_is_white else "CandB"
            print(
                f"  [{color}] Game {completed}/{total_games}: "
                f"{outcome} | Candidate {c_wins} - Baseline {b_wins} - Draws {draws}"
            )

    return c_wins, b_wins, draws


def merge_training_data(new_data_path, accumulated_path, max_samples):
    """Merge new training data with accumulated data from previous cycles.

    Keeps the most recent samples up to max_samples. New data is appended
    at the end so oldest samples get trimmed first.
    """
    new_data = np.load(new_data_path)
    new_features = new_data["features"]
    new_outcomes = new_data["outcomes"]
    new_hscores = new_data["hscores"]

    if os.path.exists(accumulated_path) and accumulated_path != new_data_path:
        old_data = np.load(accumulated_path)
        old_features = old_data["features"]
        old_outcomes = old_data["outcomes"]
        old_hscores = old_data["hscores"]

        features = np.concatenate([old_features, new_features], axis=0)
        outcomes = np.concatenate([old_outcomes, new_outcomes], axis=0)
        hscores = np.concatenate([old_hscores, new_hscores], axis=0)
    else:
        features = new_features
        outcomes = new_outcomes
        hscores = new_hscores

    # Trim oldest samples if over cap
    if len(features) > max_samples:
        features = features[-max_samples:]
        outcomes = outcomes[-max_samples:]
        hscores = hscores[-max_samples:]

    np.savez_compressed(
        new_data_path,
        features=features,
        outcomes=outcomes,
        hscores=hscores,
    )
    print(f"  Accumulated training data: {len(features)} samples (cap: {max_samples})")


CYCLE_WEIGHT_MATRIX_PATH = os.path.join(
    os.path.dirname(__file__), "cycle_weight_matrix.npy"
)


def perturb_weight_matrix(base, strength, rng):
    """Apply multiplicative noise to WEIGHT_MATRIX, preserving signs and zeros."""
    noise = 1.0 + rng.uniform(-strength, strength, size=base.shape)
    perturbed = base * noise
    # Zeros stay zero (center squares)
    perturbed[base == 0] = 0.0
    # Ensure signs never flip
    sign_flip = np.sign(perturbed) != np.sign(base)
    perturbed[sign_flip] = base[sign_flip] * 0.1
    return perturbed


def compute_next_config(cycle, config, metrics, c_wins, b_wins, draws):
    """Compute the next cycle's config based on eval results and training metrics.

    Returns a new config dict with adapted hyperparameters.
    """
    new = dict(config)
    total = c_wins + b_wins + draws
    win_rate = c_wins / total if total > 0 else 0.5

    # ── Adapt based on win rate ───────────────────────────────────────────
    if win_rate > 0.65:
        # Winning easily — push NN harder, less CH guidance
        new["nn_ratio"] = min(0.75, new["nn_ratio"] + 0.05)
        new["ch_ratio"] = max(0.10, new["ch_ratio"] - 0.05)
        new["epochs"] = max(80, new["epochs"] - 10)
        new["perturbation_strength"] = min(0.25, new["perturbation_strength"] + 0.05)
    elif win_rate < 0.45:
        # Losing — more CH guidance, more training
        new["ch_ratio"] = min(0.40, new["ch_ratio"] + 0.05)
        new["nn_ratio"] = max(0.25, new["nn_ratio"] - 0.05)
        new["epochs"] = min(250, new["epochs"] + 20)
        new["heuristic_weight_start"] = min(0.95, new["heuristic_weight_start"] + 0.05)
        new["perturbation_strength"] = max(0.0, new["perturbation_strength"] - 0.05)
    else:
        # Stable range — mild perturbation increase for diversity
        new["perturbation_strength"] = min(0.25, new["perturbation_strength"] + 0.02)

    # ── Adapt based on training metrics ───────────────────────────────────
    if metrics:
        epochs_run = metrics.get("epochs_run", new["epochs"])
        stopped_early = metrics.get("stopped_early", False)

        if stopped_early and epochs_run < 50:
            # Converged very fast — reduce LR for finer tuning
            new["lr"] = max(0.0001, new["lr"] * 0.8)
        elif not stopped_early:
            # Ran full epochs without converging — give more time
            new["epochs"] = min(250, new["epochs"] + 20)

    # ── Rebalance cross_ratio as remainder ────────────────────────────────
    new["cross_ratio"] = round(1.0 - new["nn_ratio"] - new["ch_ratio"], 2)
    new["cross_ratio"] = max(0.05, new["cross_ratio"])

    # ── Clamp all values ──────────────────────────────────────────────────
    new["nn_ratio"] = round(max(0.25, min(0.75, new["nn_ratio"])), 2)
    new["ch_ratio"] = round(max(0.10, min(0.40, new["ch_ratio"])), 2)
    new["epochs"] = int(max(80, min(250, new["epochs"])))
    new["lr"] = round(max(0.0001, min(0.001, new["lr"])), 6)
    new["perturbation_strength"] = round(
        max(0.0, min(0.25, new["perturbation_strength"])), 2
    )
    new["epsilon"] = round(max(0.04, min(0.15, new["epsilon"])), 3)
    new["heuristic_weight_start"] = round(
        max(0.5, min(0.95, new["heuristic_weight_start"])), 2
    )
    new["heuristic_weight_floor"] = round(
        max(0.3, min(0.7, new["heuristic_weight_floor"])), 2
    )

    return new


def write_cycle_config(config):
    """Write the cycle config to JSON for subprocesses to read."""
    with open(CYCLE_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def print_config_diff(old_config, new_config):
    """Print a summary of what changed between cycles."""
    changes = []
    for key in sorted(new_config):
        old_val = old_config.get(key)
        new_val = new_config.get(key)
        if old_val != new_val:
            changes.append(f"    {key}: {old_val} -> {new_val}")
    if changes:
        print("  Config changes for next cycle:")
        for line in changes:
            print(line)
    else:
        print("  Config unchanged for next cycle.")


def run_generate_data(env_extras=None):
    """Run data generation as a subprocess so it picks up the latest weights."""
    import subprocess

    script = os.path.join(os.path.dirname(__file__), "generate_data.py")
    env = os.environ.copy()
    if env_extras:
        env.update(env_extras)
    result = subprocess.run(
        [sys.executable, script], cwd=os.path.dirname(__file__), env=env
    )
    if result.returncode != 0:
        raise RuntimeError("Data generation failed")


def run_training(output_path, warmstart_path=None):
    """Run training as a subprocess, outputting to a specific weights file."""
    import subprocess

    script = os.path.join(os.path.dirname(__file__), "train_nn.py")
    env = os.environ.copy()
    env["NN_OUTPUT_PATH"] = output_path
    if warmstart_path and os.path.exists(warmstart_path):
        env["NN_WARMSTART_PATH"] = warmstart_path
    result = subprocess.run(
        [sys.executable, script], cwd=os.path.dirname(__file__), env=env
    )
    if result.returncode != 0:
        raise RuntimeError("Training failed")


def main():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("=" * 60)
    print("ADAPTIVE ITERATIVE TRAINING PIPELINE")
    print(f"  Cycles: {MIN_CYCLES}")
    print(f"  Eval games per color: {EVAL_GAMES} (total: {EVAL_GAMES * 2})")
    print(f"  Workers: {NUM_WORKERS}")
    print("=" * 60)

    # Initialize adaptive config with defaults
    current_config = dict(DEFAULTS)
    rng = np.random.default_rng(42)

    for cycle in range(1, MIN_CYCLES + 1):
        cycle_start = time.time()
        print(f"\n{'#' * 60}")
        print(f"# CYCLE {cycle}/{MIN_CYCLES}")
        print(f"{'#' * 60}")

        # ── Write adaptive config for subprocesses ────────────────────
        write_cycle_config(current_config)
        print(f"\n  Cycle config: {json.dumps(current_config, indent=2)}")

        # ── Write perturbed weight matrix (if perturbation > 0) ───────
        env_extras = {}
        strength = current_config.get("perturbation_strength", 0.0)
        if strength > 0:
            perturbed = perturb_weight_matrix(WEIGHT_MATRIX, strength, rng)
            np.save(CYCLE_WEIGHT_MATRIX_PATH, perturbed)
            env_extras["CYCLE_WEIGHT_MATRIX_PATH"] = CYCLE_WEIGHT_MATRIX_PATH
            print(f"  Perturbed WEIGHT_MATRIX (strength={strength:.2f})")
        else:
            # Clean up any stale perturbed matrix
            if os.path.exists(CYCLE_WEIGHT_MATRIX_PATH):
                os.remove(CYCLE_WEIGHT_MATRIX_PATH)

        # ── Step 1: Generate data ─────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 1/4: Generating self-play data...")
        accumulated_path = os.path.join(DATA_DIR, "accumulated_data.npz")
        if os.path.exists(TRAINING_DATA_PATH) and cycle > 1:
            shutil.copy2(TRAINING_DATA_PATH, accumulated_path)
        run_generate_data(env_extras=env_extras)

        # ── Step 1.5: Merge with accumulated data ─────────────────────
        print(f"\n[Cycle {cycle}] Step 2/4: Merging training data...")
        merge_training_data(
            TRAINING_DATA_PATH, accumulated_path, MAX_ACCUMULATED_SAMPLES
        )

        # ── Step 2: Train ─────────────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 3/4: Training NN...")
        warmstart = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) and os.path.getsize(BEST_WEIGHTS) > 0 else None
        run_training(CANDIDATE_WEIGHTS, warmstart_path=warmstart)

        # ── Read training metrics ─────────────────────────────────────
        metrics = load_train_metrics()
        if metrics:
            print(
                f"  Training metrics: epochs_run={metrics.get('epochs_run')}, "
                f"best_val_loss={metrics.get('best_val_loss', 0):.4f}, "
                f"stopped_early={metrics.get('stopped_early')}"
            )

        # ── Step 3: Evaluate ──────────────────────────────────────────
        print(f"\n[Cycle {cycle}] Step 4/4: Evaluating candidate vs baseline...")

        baseline_weights = (
            BEST_WEIGHTS
            if (os.path.exists(BEST_WEIGHTS) and os.path.getsize(BEST_WEIGHTS) > 0)
            else "__CH__"
        )
        baseline_name = (
            "previous best NN" if baseline_weights != "__CH__" else "classic heuristic"
        )
        baseline_is_ch = baseline_weights == "__CH__"

        print(f"  Candidate: {CANDIDATE_WEIGHTS}")
        print(f"  Baseline:  {baseline_name}")

        # ── Gate 1: candidate vs baseline ────────────────────────────
        print(f"\n  [Gate 1] Candidate vs {baseline_name}")
        c_wins, b_wins, draws = evaluate(
            CANDIDATE_WEIGHTS, baseline_weights, EVAL_GAMES
        )
        total = c_wins + b_wins + draws

        print(
            f"\n  Gate 1 Results: Candidate {c_wins}/{total} "
            f"({100 * c_wins / total:.1f}%) | "
            f"Baseline {b_wins}/{total} ({100 * b_wins / total:.1f}%) | "
            f"Draws {draws}"
        )

        passed_gate1 = c_wins >= b_wins

        # ── Gate 2: candidate vs CH (skip if baseline is already CH) ─
        if baseline_is_ch:
            passed_gate2 = True
            print(f"\n  [Gate 2] Skipped (baseline is already CH)")
        else:
            print(f"\n  [Gate 2] Candidate vs classic heuristic (CH)")
            ch_c_wins, ch_b_wins, ch_draws = evaluate(
                CANDIDATE_WEIGHTS, "__CH__", EVAL_GAMES
            )
            ch_total = ch_c_wins + ch_b_wins + ch_draws

            print(
                f"\n  Gate 2 Results: Candidate {ch_c_wins}/{ch_total} "
                f"({100 * ch_c_wins / ch_total:.1f}%) | "
                f"CH {ch_b_wins}/{ch_total} ({100 * ch_b_wins / ch_total:.1f}%) | "
                f"Draws {ch_draws}"
            )
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

        # ── Compute next cycle's adaptive config ──────────────────────
        old_config = dict(current_config)
        current_config = compute_next_config(
            cycle, current_config, metrics, c_wins, b_wins, draws
        )
        print_config_diff(old_config, current_config)

        elapsed = time.time() - cycle_start
        print(f"\n  Cycle {cycle} completed in {elapsed:.0f}s")

    # ── Cleanup adaptive artifacts ────────────────────────────────────
    for path in [CYCLE_CONFIG_PATH, TRAIN_METRICS_PATH, CYCLE_WEIGHT_MATRIX_PATH]:
        if os.path.exists(path):
            os.remove(path)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"  Best weights: {BEST_WEIGHTS}")
    print(f"={'=' * 59}")


if __name__ == "__main__":
    main()
