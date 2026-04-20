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

## Step 1: Generate Training Data

Runs 10,000 games using minimax with mixed game modes to prevent mode collapse:

| Mode | Ratio | Description |
|------|-------|-------------|
| NN self-play | 50% | NN plays itself (when NN weights exist) |
| CH self-play | 20% | Classical heuristic plays itself |
| Cross-play | 30% | NN vs classical heuristic (alternating colors) |

When no NN weights exist (first run), all games default to CH self-play.

Records every board position with its heuristic score and game outcome, then augments with D4 symmetry (8x multiplier). Cross-play games always use the CH for h_score labels to avoid circular bias.

Three randomness sources ensure every game is unique:
- **Random openings:** the first 6 moves are chosen randomly to create diverse starting positions
- **Epsilon-greedy:** 25% of mid-game moves are random instead of minimax-optimal
- **Depth jitter:** search depth varies by +/-1 each move

```bash
python training/generate_data.py
```

Output: `training/data/training_data.npz`

See [DATA_GENERATION_README.md](DATA_GENERATION_README.md) for full details and all tunable parameters.

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
| `HEURISTIC_WEIGHT_START` | 0.3   | Initial weight for heuristic-bootstrap loss (decays to floor) |
| `HEURISTIC_WEIGHT_FLOOR` | 0.05  | Minimum heuristic weight (regularizer, never decays below this) |

## Step 3: Evaluate Against Classic Heuristic

Plays 50 head-to-head games (25 per color assignment) between the NN heuristic and `heuristic_current_best`.

When run via the pipeline (`run_pipeline.py`), evaluation uses a dual gate:
- **Gate 1:** Candidate vs previous best NN (must win to prove improvement)
- **Gate 2:** Candidate vs CH (must win to prevent mode collapse drift)

The candidate is only promoted if it passes both gates.

```bash
python training/evaluate.py
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
