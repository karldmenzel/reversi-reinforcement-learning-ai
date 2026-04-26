


import sys
import os
import time
import numpy as np
from multiprocessing import Pool, cpu_count

# Add src to path so we can import game modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# Add training dir so cycle_utils is importable
sys.path.insert(0, os.path.dirname(__file__))

from reversi import reversi
from utils import get_legal_moves, apply_move, zobrist_hash, ZOBRIST_TURN
from heuristic_functions import heuristic_nic
from minimax_alpha_beta_h_nic_nn import minimax, TimeUp
from nn_heuristic import NNHeuristic, extract_features

# ── NN Heuristic Setup ───────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '../src', 'weights', 'heuristic_v1.npz')
NN_HEURISTIC_FN = None   # initialized per-worker via _init_worker
CH_HEURISTIC_FN = None   # always heuristic_nic

# ── Game mode constants & ratios ─────────────────────────────────────────────
MODE_NN_SELF = 'nn_self'    # NN vs NN
MODE_CH_SELF = 'ch_self'    # CH vs CH
MODE_CROSS   = 'cross'      # NN vs CH (alternating colors)

NN_RATIO   = 0.5   # fraction of games: NN self-play
CH_RATIO   = 0.2   # fraction of games: CH self-play
CROSS_RATIO = 0.3  # fraction of games: cross-play


def _init_worker():
    """Initialize both heuristics in each worker process."""
    global NN_HEURISTIC_FN, CH_HEURISTIC_FN
    CH_HEURISTIC_FN = heuristic_nic
    if os.path.exists(WEIGHTS_PATH):
        try:
            NN_HEURISTIC_FN = NNHeuristic(WEIGHTS_PATH)
        except (EOFError, Exception):
            NN_HEURISTIC_FN = None  # weights file exists but is empty/corrupt
    else:
        NN_HEURISTIC_FN = None  # no NN available yet


# ── Configuration ──────────────────────────────────────────────────────────────
NUM_GAMES = 10000
SEARCH_DEPTH = 4              # base depth per move during data generation
DEPTH_JITTER = 1              # depth varies in [SEARCH_DEPTH - jitter, + jitter]
TIME_PER_MOVE = 3.0           # seconds per move
RANDOM_OPENING_MOVES = 4      # first N moves of each game are fully random
EPSILON = 0.08               # probability of picking a random move mid-game
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAVE_EVERY = 500              # save checkpoint every N games
NUM_WORKERS = max(1, cpu_count() - 1)  # leave 1 core free
SEED = None                   # set to int for reproducibility, None for random
# ───────────────────────────────────────────────────────────────────────────────


def play_game(depth, white_heuristic, black_heuristic, hscore_heuristic, rng):
    """Play a game where each color can use a different heuristic.

    Args:
        depth: base search depth
        white_heuristic: heuristic function for white player
        black_heuristic: heuristic function for black player
        hscore_heuristic: heuristic used for recording h_score labels
        rng: numpy random generator

    Returns:
        positions: list of (board_copy, current_player, heuristic_score, move_index)
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

        # Pick the heuristic for the current player
        current_heuristic = white_heuristic if turn == 1 else black_heuristic

        # Record position; use hscore_heuristic for the label
        h_score = hscore_heuristic(board, turn)
        positions.append((board.copy(), turn, h_score, move_number))

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

            z_hash = zobrist_hash(board)
            tt = {}
            prev_best = None
            best_move = legal_moves[0]
            try:
                for d in range(1, jittered_depth + 1):
                    killer_moves = {}
                    _, move = minimax(board, search_game, d,
                                      float('-inf'), float('inf'),
                                      True, turn, deadline, current_heuristic,
                                      z_hash, tt, killer_moves, prev_best)
                    best_move = move
                    prev_best = move
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
    game_idx, worker_seed, game_mode = args
    rng = np.random.default_rng(worker_seed)

    # Determine heuristics for each color based on game mode
    nn_fn = NN_HEURISTIC_FN
    ch_fn = CH_HEURISTIC_FN

    if game_mode == MODE_NN_SELF and nn_fn is not None:
        # Fix: always use CH for h_score labels to avoid circular bias
        white_h, black_h, hscore_h = nn_fn, nn_fn, ch_fn
    elif game_mode == MODE_CROSS and nn_fn is not None:
        # Alternate who plays white: even games NN=white, odd games NN=black
        if game_idx % 2 == 0:
            white_h, black_h = nn_fn, ch_fn
        else:
            white_h, black_h = ch_fn, nn_fn
        # Always use CH for h_score labels to avoid circular bias
        hscore_h = ch_fn
    else:
        # CH self-play (or fallback when no NN exists)
        white_h, black_h, hscore_h = ch_fn, ch_fn, ch_fn

    positions, outcome = play_game(SEARCH_DEPTH, white_h, black_h, hscore_h, rng)

    features = []
    outcomes = []
    hscores = []

    total_moves = max(len(positions), 1)

    for board, player, h_score, move_index in positions:
        player_outcome = outcome * player

        # Temporal discounting: early positions get 50% signal, late get 100%
        discount = 0.5 + 0.5 * (move_index / total_moves)
        discounted_outcome = player_outcome * discount

        # Create a temporary game for mobility features
        temp_game = reversi()
        temp_game.board = board.copy()

        aug_boards = augment_d4(board)
        for aug_board in aug_boards:
            # For augmented boards, create a game for mobility
            aug_game = reversi()
            aug_game.board = aug_board.copy()
            feat = extract_features(aug_board, player, game=aug_game)
            features.append(feat)
            outcomes.append(discounted_outcome)
            hscores.append(float(np.tanh(h_score / 800.0)))

    return features, outcomes, hscores


def _save(features, outcomes, hscores, path):
    np.savez_compressed(
        path,
        features=np.array(features, dtype=np.float32),
        outcomes=np.array(outcomes, dtype=np.float32),
        hscores=np.array(hscores, dtype=np.float32),
    )
    print(f"  Saved checkpoint: {path} ({len(features)} samples)")


def _assign_game_modes(num_games, nn_available):
    """Pre-assign a game mode to each game index based on ratio config."""
    if not nn_available:
        # No NN weights yet — all games are CH self-play
        return [MODE_CH_SELF] * num_games

    n_nn = int(num_games * NN_RATIO)
    n_ch = int(num_games * CH_RATIO)
    n_cross = num_games - n_nn - n_ch  # remainder goes to cross-play
    modes = ([MODE_NN_SELF] * n_nn +
             [MODE_CH_SELF] * n_ch +
             [MODE_CROSS] * n_cross)
    # Shuffle so different modes are interleaved across workers
    np.random.default_rng(42).shuffle(modes)
    return modes


def main():
    global NUM_GAMES, NN_RATIO, CH_RATIO, CROSS_RATIO, EPSILON, RANDOM_OPENING_MOVES

    # Override defaults from adaptive cycle config (if present)
    from cycle_utils import load_cycle_config
    cfg = load_cycle_config()
    NUM_GAMES = int(cfg['num_games'])
    NN_RATIO = float(cfg['nn_ratio'])
    CH_RATIO = float(cfg['ch_ratio'])
    CROSS_RATIO = float(cfg['cross_ratio'])
    EPSILON = float(cfg['epsilon'])
    RANDOM_OPENING_MOVES = int(cfg['random_opening_moves'])

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate unique seeds for each game up front
    master_rng = np.random.default_rng(SEED)
    game_seeds = master_rng.integers(0, 2**63, size=NUM_GAMES)

    nn_available = os.path.exists(WEIGHTS_PATH) and os.path.getsize(WEIGHTS_PATH) > 0
    game_modes = _assign_game_modes(NUM_GAMES, nn_available)

    all_features = []
    all_outcomes = []
    all_hscores = []

    start_time = time.time()

    # Count modes for logging
    from collections import Counter
    mode_counts = Counter(game_modes)
    print(f"Generating {NUM_GAMES} games across {NUM_WORKERS} workers")
    print(f"  NN available: {nn_available}")
    print(f"  Game mix: NN self-play={mode_counts.get(MODE_NN_SELF, 0)}, "
          f"CH self-play={mode_counts.get(MODE_CH_SELF, 0)}, "
          f"Cross-play={mode_counts.get(MODE_CROSS, 0)}")
    print(f"  Search depth: {SEARCH_DEPTH} +/- {DEPTH_JITTER}")
    print(f"  Random opening moves: {RANDOM_OPENING_MOVES}")
    print(f"  Epsilon (random move prob): {EPSILON}")
    print()

    # Process games in chunks so we can report progress and save checkpoints
    CHUNK_SIZE = min(NUM_WORKERS * 4, SAVE_EVERY)
    games_done = 0

    work_items = [(i, int(game_seeds[i]), game_modes[i]) for i in range(NUM_GAMES)]

    with Pool(processes=NUM_WORKERS, initializer=_init_worker) as pool:
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
