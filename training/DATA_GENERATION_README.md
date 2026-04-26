# Data Generation

## Overview

`generate_data.py` produces training data by playing thousands of Reversi games using minimax search. To prevent mode collapse in the iterative training pipeline, games are split across three modes that ensure the NN sees diverse strategies. When run via the adaptive pipeline, all key parameters are overridden by `cycle_config.json`.

## Game Modes

| Mode | Default Ratio | White Player | Black Player | h_score Source | Purpose |
|------|---------------|-------------|-------------|----------------|---------|
| NN self-play | 50% | NN | NN | CH | Learn from the NN's own strategy |
| CH self-play | 20% | CH | CH | CH | Inject strategic diversity from domain knowledge |
| Cross-play | 30% | NN or CH | CH or NN | CH | Learn to exploit and defend against the CH |

- **Cross-play** alternates which side plays NN (even-indexed games: NN=white, odd: NN=black)
- **h_score labels** always come from the CH (even in NN self-play) to avoid circular bias — if the NN scored its own positions, errors would compound across training cycles
- When **no NN weights exist** (cycle 0 / first run), all games default to 100% CH self-play

## How It Works

1. **Adaptive config is loaded** from `cycle_config.json` (if present). This overrides `NUM_GAMES`, `NN_RATIO`, `CH_RATIO`, `CROSS_RATIO`, `EPSILON`, and `RANDOM_OPENING_MOVES`. If no config file exists, hardcoded defaults are used.

2. **Game modes are pre-assigned** to each game index based on the configured ratios, then shuffled for even distribution across workers.

3. **Each worker process** initializes both the NN heuristic (if weights exist) and the CH heuristic (`heuristic_nic`). If the `CYCLE_WEIGHT_MATRIX_PATH` env var is set, `heuristic_nic` uses the perturbed positional weights instead of the canonical ones.

4. **For each game**, the worker selects the appropriate heuristic pair based on the assigned mode, then plays the game using minimax with the following diversity mechanisms:
   - **Random openings:** First N moves are fully random (default: 4)
   - **Epsilon-greedy:** Configurable % of mid-game moves are random (default: 8%)
   - **Depth jitter:** Search depth varies by +/-1 each move

5. **Every board position** is recorded with the current player, game outcome (temporally discounted), and h_score from the CH heuristic.

6. **D4 symmetry augmentation** expands each position 8x (4 rotations x 2 reflections).

## Usage

```bash
python training/generate_data.py
```

Output: `training/data/training_data.npz`

Checkpoints are saved every 500 games to `training/data/checkpoint_*.npz`.

## Configuration

### Adaptive Config (Pipeline Mode)

When run via `run_pipeline.py`, the following are overridden by `cycle_config.json`:

| Parameter | Default | Adaptive Range | Description |
|-----------|---------|----------------|-------------|
| `NUM_GAMES` | 10,000 | — | Total games to generate |
| `NN_RATIO` | 0.5 | 0.25 - 0.75 | Fraction of NN self-play games |
| `CH_RATIO` | 0.2 | 0.10 - 0.40 | Fraction of CH self-play games |
| `CROSS_RATIO` | 0.3 | 0.05+ | Remainder after NN + CH ratios |
| `EPSILON` | 0.08 | 0.04 - 0.15 | Random move probability |
| `RANDOM_OPENING_MOVES` | 4 | — | First N moves are random |

### Static Parameters

These are only changed by editing `generate_data.py` directly:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SEARCH_DEPTH` | 4 | Base minimax depth per move |
| `DEPTH_JITTER` | 1 | Depth varies in [depth - jitter, depth + jitter] |
| `TIME_PER_MOVE` | 3.0 | Time limit per move (seconds) |
| `SAVE_EVERY` | 500 | Checkpoint frequency (games) |
| `NUM_WORKERS` | cpu_count - 1 | Parallel worker processes |
| `SEED` | None | RNG seed (None = random, int = reproducible) |

### WEIGHT_MATRIX Perturbation

When the `CYCLE_WEIGHT_MATRIX_PATH` environment variable is set (by `run_pipeline.py`), the CH heuristic loads a perturbed WEIGHT_MATRIX from that path instead of the canonical one. This creates diverse positional scoring across cycles while preserving strategic structure (corners positive, X-squares negative, center zero).

The perturbation is:
- **Multiplicative** — noise proportional to cell magnitude (a corner at 100 might become 85-115)
- **Sign-preserving** — positive cells stay positive, negative stay negative
- **Zero-preserving** — center squares remain zero
- **Scoped to data generation** — production play always uses the canonical WEIGHT_MATRIX

## Output Format

`training_data.npz` contains three arrays:

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| `features` | (N, 144) | float32 | Board features: 64 piece positions + 64 positional weights + 16 scalars |
| `outcomes` | (N,) | float32 | Game result from current player's perspective, temporally discounted |
| `hscores` | (N,) | float32 | CH heuristic score normalized to [-1, 1] via tanh(h/800) |

With 10,000 games and 8x augmentation, expect ~4M training samples per cycle.

## Example Log Output

```
Generating 10000 games across 7 workers
  NN available: True
  Game mix: NN self-play=5000, CH self-play=2000, Cross-play=3000
  Search depth: 4 +/- 1
  Random opening moves: 4
  Epsilon (random move prob): 0.08

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
