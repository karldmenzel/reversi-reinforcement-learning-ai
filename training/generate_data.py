

import sys
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Add src to path so we can import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from reversi import reversi
from utils import get_legal_moves, apply_move
from heuristic_functions import heuristic_nic
from minimax_alpha_beta_h_nic_nn import minimax, TimeUp
from nn_heuristic import NNHeuristic, extract_features

# ── NN Heuristic Setup ───────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '../src', 'weights', 'heuristic_v1.npz')
if os.path.exists(WEIGHTS_PATH):
    _nn_heuristic = NNHeuristic(WEIGHTS_PATH)
    HEURISTIC_FN = _nn_heuristic
    print(f"Using NN heuristic from {WEIGHTS_PATH}")
else:
    HEURISTIC_FN = heuristic_nic
    print(f"NN weights not found at {WEIGHTS_PATH}, falling back to heuristic_nic")


# ── Configuration ──────────────────────────────────────────────────────────────
NUM_GAMES = 10000
SEARCH_DEPTH = 3              # base depth per move during data generation
DEPTH_JITTER = 1              # depth varies in [SEARCH_DEPTH - jitter, + jitter]
TIME_PER_MOVE = 2.0           # seconds per move
RANDOM_OPENING_MOVES = 6      # first N moves of each game are fully random
EPSILON = 0.15                # probability of picking a random move mid-game
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAVE_EVERY = 500              # save checkpoint every N games
NUM_WORKERS = max(1, cpu_count() - 1)  # leave 1 core free
SEED = None                   # set to int for reproducibility, None for random
# ───────────────────────────────────────────────────────────────────────────────


def play_game(depth, heuristic_in, rng):
    """Play a single self-play game with randomized moves for diversity.

    Returns:
        positions: list of (board_copy, current_player, heuristic_score)
        outcome:   +1 if white wins, -1 if black wins, 0 draw
    """
    game = reversi()
    positions = []
    turn = 1  # white starts
    consecutive_passes = 0
    move_number = 0

    while True:
        board = game.board.copy()
        search_game = reversi()
        search_game.board = board.copy()

        legal_moves = get_legal_moves(search_game, turn)

        if len(legal_moves) == 0:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
            turn *= -1
            continue

        consecutive_passes = 0

        # Record position and heuristic score from current player's perspective
        h_score = heuristic_in(board, turn)
        positions.append((board.copy(), turn, h_score))

        # Decide whether to use a random move or minimax
        use_random = (move_number < RANDOM_OPENING_MOVES or
                      rng.random() < EPSILON)

        if use_random:
            idx = rng.integers(len(legal_moves))
            best_move = legal_moves[idx]
        else:
            # Jitter the search depth for variety
            jittered_depth = max(1, depth + rng.integers(-DEPTH_JITTER,
                                                          DEPTH_JITTER + 1))

            search_game.board = board.copy()
            deadline = time.time() + TIME_PER_MOVE

            best_move = legal_moves[0]
            try:
                for d in range(1, jittered_depth + 1):
                    _, move = minimax(board, search_game, d,
                                      float('-inf'), float('inf'),
                                      True, turn, deadline, heuristic_in)
                    best_move = move
            except TimeUp:
                pass

        # Apply move
        result = game.step(best_move[0], best_move[1], turn)
        if result < 0:
            consecutive_passes += 1
            if consecutive_passes >= 2:
                break
        else:
            move_number += 1
            turn *= -1

    # Determine outcome
    white = int(np.sum(game.board == 1))
    black = int(np.sum(game.board == -1))
    if white > black:
        outcome = 1
    elif black > white:
        outcome = -1
    else:
        outcome = 0

    return positions, outcome


def augment_d4(board):
    """Apply D4 symmetry group to a board, yielding 8 augmented boards."""
    augmented = []
    b = board
    for _ in range(4):
        augmented.append(b.copy())
        augmented.append(np.fliplr(b).copy())
        b = np.rot90(b)
    return augmented


def _process_game_result(args):
    """Worker function: play one game and return extracted features.

    Each worker gets its own RNG seeded uniquely to avoid duplicate games.
    Returns (features_list, outcomes_list, hscores_list).
    """
    game_idx, worker_seed = args
    rng = np.random.default_rng(worker_seed)

    positions, outcome = play_game(SEARCH_DEPTH, HEURISTIC_FN, rng)

    features = []
    outcomes = []
    hscores = []

    for board, player, h_score in positions:
        player_outcome = outcome * player

        aug_boards = augment_d4(board)
        for aug_board in aug_boards:
            feat = extract_features(aug_board, player)
            features.append(feat)
            outcomes.append(player_outcome)
            hscores.append(np.clip(h_score / 1000.0, -1.0, 1.0))

    return features, outcomes, hscores


def _save(features, outcomes, hscores, path):
    np.savez_compressed(
        path,
        features=np.array(features, dtype=np.float32),
        outcomes=np.array(outcomes, dtype=np.float32),
        hscores=np.array(hscores, dtype=np.float32),
    )
    print(f"  Saved checkpoint: {path} ({len(features)} samples)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate unique seeds for each game up front
    master_rng = np.random.default_rng(SEED)
    game_seeds = master_rng.integers(0, 2**63, size=NUM_GAMES)

    all_features = []
    all_outcomes = []
    all_hscores = []

    start_time = time.time()

    print(f"Generating {NUM_GAMES} self-play games across {NUM_WORKERS} workers")
    print(f"  Search depth: {SEARCH_DEPTH} +/- {DEPTH_JITTER}")
    print(f"  Random opening moves: {RANDOM_OPENING_MOVES}")
    print(f"  Epsilon (random move prob): {EPSILON}")
    print()

    # Process games in chunks so we can report progress and save checkpoints
    CHUNK_SIZE = min(NUM_WORKERS * 4, SAVE_EVERY)
    games_done = 0

    work_items = [(i, int(game_seeds[i])) for i in range(NUM_GAMES)]

    with Pool(processes=NUM_WORKERS) as pool:
        # imap_unordered gives results as they finish, not in submission order
        for features, outcomes, hscores in pool.imap_unordered(
                _process_game_result, work_items, chunksize=CHUNK_SIZE):
            all_features.extend(features)
            all_outcomes.extend(outcomes)
            all_hscores.extend(hscores)
            games_done += 1

            if games_done % 10 == 0:
                elapsed = time.time() - start_time
                rate = games_done / elapsed
                eta = (NUM_GAMES - games_done) / rate
                print(f"Game {games_done}/{NUM_GAMES} | "
                      f"Positions: {len(all_features)} | "
                      f"Rate: {rate:.1f} games/s | "
                      f"ETA: {eta:.0f}s")

            if games_done % SAVE_EVERY == 0:
                _save(all_features, all_outcomes, all_hscores,
                      os.path.join(OUTPUT_DIR, f'checkpoint_{games_done}.npz'))

    # Final save
    out_path = os.path.join(OUTPUT_DIR, 'training_data.npz')
    _save(all_features, all_outcomes, all_hscores, out_path)

    elapsed = time.time() - start_time
    print(f"\nDone! {NUM_GAMES} games, {len(all_features)} positions "
          f"(with augmentation) in {elapsed:.0f}s")
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
