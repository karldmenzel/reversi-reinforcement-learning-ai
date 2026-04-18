# Training the Neural Network Heuristic

## Prerequisites

- Python 3.10+
- numpy (already installed)
- PyTorch (needed only for training, not inference)

```bash
pip install torch
```

## Pipeline Overview

```
Step 1: Generate Data  -->  Step 2: Train Network  -->  Step 3: Evaluate  -->  Step 4: Play
   ~2-5 hours                  ~1-3 hours                 ~1-2 hours            Automatic
```

## Step 1: Generate Self-Play Data

Runs 10,000 self-play games using `heuristic_nic` with minimax. Records every board position with its heuristic score and game outcome, then augments with D4 symmetry (8x multiplier).

Since minimax with a fixed heuristic is deterministic, three randomness sources ensure every game is unique:
- **Random openings:** the first 6 moves are chosen randomly to create diverse starting positions
- **Epsilon-greedy:** 15% of mid-game moves are random instead of minimax-optimal
- **Depth jitter:** search depth varies by +/-1 each move

```bash
cd src
python ../training/generate_data.py
```

Output: `training/data/training_data.npz`

**Tunable parameters** (edit top of `training/generate_data.py`):

| Parameter              | Default | Description                                    |
|------------------------|---------|------------------------------------------------|
| `NUM_GAMES`            | 10000   | Number of self-play games                      |
| `SEARCH_DEPTH`         | 3       | Base minimax depth per move                    |
| `DEPTH_JITTER`         | 1       | Depth varies in [depth - jitter, depth + jitter] |
| `RANDOM_OPENING_MOVES` | 6       | First N moves are fully random                 |
| `EPSILON`              | 0.15    | Probability of a random move after the opening |
| `TIME_PER_MOVE`        | 2.0     | Time limit per move (seconds)                  |
| `SAVE_EVERY`           | 500     | Checkpoint frequency (games)                   |
| `SEED`                 | None    | RNG seed (None = random, int = reproducible)   |

## Step 2: Train the Network

Trains a 3-layer MLP on the generated data using a dual loss function. Automatically exports weights for numpy inference.

```bash
cd src
python ../training/train_nn.py
```

Output: `weights/heuristic_v1.npz`

**Tunable parameters** (edit top of `training/train_nn.py`):

| Parameter              | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `BATCH_SIZE`           | 512     | Training batch size                      |
| `EPOCHS`               | 80      | Number of training epochs                |
| `LR`                   | 0.001   | Learning rate                            |
| `VALIDATION_SPLIT`     | 0.1     | Fraction held out for validation         |
| `OUTCOME_WEIGHT`       | 0.7     | Weight for game-outcome loss             |
| `HEURISTIC_WEIGHT_START` | 0.3   | Initial weight for heuristic-bootstrap loss (decays to 0) |

## Step 3: Evaluate Against Classic Heuristic

Plays 50 head-to-head games (25 per color assignment) between the NN heuristic and `heuristic_current_best`.

```bash
cd src
python ../training/evaluate.py
```

**Target:** >60% win rate for the NN heuristic.

**Tunable parameters** (edit top of `training/evaluate.py`):

| Parameter    | Default | Description                          |
|--------------|---------|--------------------------------------|
| `NUM_GAMES`  | 25      | Games per color assignment (total = 2x) |

## Step 4: Play

Once `weights/heuristic_v1.npz` exists, the minimax bot automatically uses it. No code changes needed.

To switch back to the classic heuristic, delete or rename the weights file, or edit `src/minimax_alpha_beta_h_nic.py` and set:

```python
CHOSEN_HEURISTIC = whatever_heuristic_you_want
```

## Retraining

To retrain with more data or different parameters:

1. Delete or rename `training/data/training_data.npz`
2. Adjust parameters in `training/generate_data.py`
3. Re-run Steps 1-3

Checkpoints are saved every 500 games in `training/data/checkpoint_*.npz` so you can resume from partial runs by renaming a checkpoint to `training_data.npz`.
