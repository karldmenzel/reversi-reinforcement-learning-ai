# Data Generation

## Overview

`generate_data.py` produces training data by playing thousands of Reversi games using minimax search. To prevent mode collapse in the iterative training pipeline, games are split across three modes that ensure the NN sees diverse strategies.

## Game Modes

| Mode | Ratio | White Player | Black Player | h_score Source | Purpose |
|------|-------|-------------|-------------|----------------|---------|
| NN self-play | 50% | NN | NN | NN | Learn from the NN's own strategy |
| CH self-play | 20% | CH | CH | CH | Inject strategic diversity from domain knowledge |
| Cross-play | 30% | NN or CH | CH or NN | CH | Learn to exploit and defend against the CH |

- **Cross-play** alternates which side plays NN (even-indexed games: NN=white, odd: NN=black)
- **h_score labels** in cross-play always come from the CH to avoid circular bias — if the NN scored its own positions, errors would compound across training cycles
- When **no NN weights exist** (cycle 0 / first run), all games default to 100% CH self-play

## How It Works

1. **Game modes are pre-assigned** to each game index based on the configured ratios, then shuffled for even distribution across workers.

2. **Each worker process** initializes both the NN heuristic (if weights exist) and the CH heuristic (`heuristic_nic`).

3. **For each game**, the worker selects the appropriate heuristic pair based on the assigned mode, then plays the game using minimax with the following diversity mechanisms:
   - **Random openings:** First 6 moves are fully random
   - **Epsilon-greedy:** 25% of mid-game moves are random
   - **Depth jitter:** Search depth varies by +/-1 each move

4. **Every board position** is recorded with the current player, game outcome, and h_score from the designated heuristic.

5. **D4 symmetry augmentation** expands each position 8x (4 rotations x 2 reflections).

## Usage

```bash
python training/generate_data.py
```

Output: `training/data/training_data.npz`

Checkpoints are saved every 500 games to `training/data/checkpoint_*.npz`.

## Configuration

Edit the top of `training/generate_data.py`:

### Game Mode Ratios

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NN_RATIO` | 0.5 | Fraction of games: NN self-play |
| `CH_RATIO` | 0.2 | Fraction of games: CH self-play |
| `CROSS_RATIO` | 0.3 | Fraction of games: NN-vs-CH cross-play |

These must sum to 1.0. Adjust to shift the balance — e.g., increase `CROSS_RATIO` if the NN is drifting away from beating the CH, or increase `NN_RATIO` if you want faster self-improvement.

### Game Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_GAMES` | 10000 | Total number of games to generate |
| `SEARCH_DEPTH` | 3 | Base minimax depth per move |
| `DEPTH_JITTER` | 1 | Depth varies in [depth - jitter, depth + jitter] |
| `RANDOM_OPENING_MOVES` | 6 | First N moves are fully random |
| `EPSILON` | 0.25 | Probability of a random move after the opening |
| `TIME_PER_MOVE` | 1.0 | Time limit per move (seconds) |
| `SAVE_EVERY` | 500 | Checkpoint frequency (games) |
| `NUM_WORKERS` | cpu_count - 1 | Parallel worker processes |
| `SEED` | None | RNG seed (None = random, int = reproducible) |

## Output Format

`training_data.npz` contains three arrays:

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| `features` | (N, 139) | float32 | Board features: 64 piece positions + 64 positional weights + 11 scalars |
| `outcomes` | (N,) | float32 | Game result from current player's perspective (+1 win, -1 loss, 0 draw) |
| `hscores` | (N,) | float32 | Heuristic score normalized to [-1, 1] |

With 10,000 games and 8x augmentation, expect ~4-5M training samples.

## Example Log Output

```
Generating 10000 games across 7 workers
  NN available: True
  Game mix: NN self-play=5000, CH self-play=2000, Cross-play=3000
  Search depth: 3 +/- 1
  Random opening moves: 6
  Epsilon (random move prob): 0.25

Game 10/10000 | Positions: 4320 | Rate: 1.2 games/s | ETA: 8325s
...
```

When no NN weights exist:
```
Generating 10000 games across 7 workers
  NN available: False
  Game mix: NN self-play=0, CH self-play=10000, Cross-play=0
  ...
```
