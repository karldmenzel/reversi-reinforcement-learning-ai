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
┌─────────────────────────────────────────────────────────────────────┐
│                     ADAPTIVE TRAINING PIPELINE                      │
│                                                                     │
│  For each cycle:                                                    │
│                                                                     │
│  1. Write cycle_config.json ──► adaptive hyperparameters            │
│  2. Write cycle_weight_matrix.npy ──► perturbed positional weights  │
│  3. Generate data ──► train ──► evaluate                            │
│  4. Read train_metrics.json ──► compute next cycle's config         │
│                                                                     │
│  Repeats for MIN_CYCLES (default: 5)                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Step 1: Generate Training Data

Runs 10,000 games using minimax with mixed game modes to prevent mode collapse. Game mode ratios adapt between cycles based on evaluation results:

| Mode | Default Ratio | Description |
|------|---------------|-------------|
| NN self-play | 50% | NN plays itself (when NN weights exist) |
| CH self-play | 20% | Classical heuristic plays itself |
| Cross-play | 30% | NN vs classical heuristic (alternating colors) |

When no NN weights exist (first run), all games default to CH self-play.

Records every board position with its heuristic score and game outcome, then augments with D4 symmetry (8x multiplier). Cross-play games always use the CH for h_score labels to avoid circular bias. With 10,000 games, expect ~4M augmented samples per cycle.

Three randomness sources ensure every game is unique:
- **Random openings:** the first 4 moves are chosen randomly to create diverse starting positions
- **Epsilon-greedy:** 8% of mid-game moves are random instead of minimax-optimal
- **Depth jitter:** search depth varies by +/-1 each move

When run via the pipeline, a **perturbed WEIGHT_MATRIX** can be injected via environment variable to create diverse positional scoring across cycles. The perturbation is multiplicative (proportional to cell magnitude), preserves signs, and keeps zeros at zero. This only affects CH heuristic scoring during data generation — the NN feature extraction uses its own fixed copy.

```bash
python training/generate_data.py
```

Output: `training/data/training_data.npz`

See [DATA_GENERATION_README.md](DATA_GENERATION_README.md) for full details and all tunable parameters.

## Step 2: Train the Network

Trains a 4-layer MLP (144->512->256->128->1, ~238K params) on the generated data using a dual loss function. Automatically exports weights for numpy inference.

```bash
python training/train_nn.py
```

Output: `src/weights/heuristic_v1.npz` (or path set via `NN_OUTPUT_PATH` env var)

### Dual Loss Function

The network trains on two targets simultaneously with a decaying heuristic weight schedule:

```
L = (1 - h_weight) * MSE(pred, game_outcome) + h_weight * MSE(pred, heuristic_score)
```

- `h_weight` starts at 0.9 (near-pure CH distillation) and decays linearly to a floor of 0.5
- This means outcome weight grows from 0.1 to 0.5 over training
- The high floor keeps a strong CH signal as a regularizer throughout training

### Default Parameters

| Parameter              | Default | Description                              |
|------------------------|---------|------------------------------------------|
| `BATCH_SIZE`           | 512     | Training batch size                      |
| `EPOCHS`               | 150     | Number of training epochs                |
| `LR`                   | 0.0005  | Learning rate                            |
| `WEIGHT_DECAY`         | 1e-4    | Adam weight decay                        |
| `VALIDATION_SPLIT`     | 0.1     | Fraction held out for validation         |
| `GRAD_CLIP_NORM`       | 1.0     | Gradient clipping norm                   |
| `EARLY_STOPPING_PATIENCE` | 20   | Epochs without improvement before stopping |
| `HEURISTIC_WEIGHT_START` | 0.9   | Initial weight for heuristic loss        |
| `HEURISTIC_WEIGHT_FLOOR` | 0.5   | Minimum heuristic weight (never decays below) |

When run via the pipeline, `EPOCHS`, `LR`, `HEURISTIC_WEIGHT_START`, and `HEURISTIC_WEIGHT_FLOOR` are overridden by `cycle_config.json`. After training, metrics (`best_val_loss`, `final_train_loss`, `epochs_run`, `stopped_early`) are written to `train_metrics.json` for the pipeline to read.

## Step 3: Evaluate Against Baseline

Plays 50 head-to-head games per color (100 total) between the candidate and the baseline.

When run via the pipeline (`run_pipeline.py`), evaluation uses a dual gate:
- **Gate 1:** Candidate vs previous best NN (must win to prove improvement)
- **Gate 2:** Candidate vs CH (must win to prevent mode collapse drift)

The candidate is only promoted if it passes both gates.

```bash
python training/evaluate.py
```

**Target:** >60% win rate for the NN heuristic.

## Step 4: Play

Once `src/weights/heuristic_v1.npz` exists, the minimax bot automatically uses it. No code changes needed.

## Adaptive Training Cycles

The pipeline adapts hyperparameters between cycles based on evaluation results and training metrics:

| Condition | Action |
|-----------|--------|
| Win rate > 65% | Increase NN ratio, decrease CH ratio, reduce epochs, increase perturbation |
| Win rate < 45% | Increase CH ratio, decrease NN ratio, increase epochs, increase h_weight_start, reduce perturbation |
| Win rate 45-65% | Stable — mild perturbation increase for diversity |
| Early stop < 50 epochs | Reduce LR by 20% for finer tuning |
| No early stop (ran full) | Increase epochs to allow more convergence time |

### Parameter Bounds

| Parameter | Min | Max |
|-----------|-----|-----|
| `nn_ratio` | 0.25 | 0.75 |
| `ch_ratio` | 0.10 | 0.40 |
| `epochs` | 80 | 250 |
| `lr` | 0.0001 | 0.001 |
| `perturbation_strength` | 0.0 | 0.25 |
| `epsilon` | 0.04 | 0.15 |
| `heuristic_weight_start` | 0.5 | 0.95 |
| `heuristic_weight_floor` | 0.3 | 0.7 |

### Data Accumulation

Training data is accumulated across cycles up to a cap of 7M samples (~2 full cycles of augmented data). Oldest samples are trimmed first when the cap is exceeded. This gives the model exposure to positions from different training stages.

## Retraining

To retrain with more data or different parameters:

1. Delete or rename `training/data/training_data.npz`
2. Adjust parameters in `training/generate_data.py` (or let the pipeline adapt them)
3. Re-run the pipeline or individual steps

Checkpoints are saved every 500 games in `training/data/checkpoint_*.npz` so you can resume from partial runs by renaming a checkpoint to `training_data.npz`.
