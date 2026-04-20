# Training Pipeline

## Overview

The NN heuristic is trained iteratively through mixed self-play and cross-play. Each cycle generates diverse training data using a mix of NN self-play, classical heuristic (CH) self-play, and NN-vs-CH cross-play. A dual evaluation gate prevents mode collapse by requiring the candidate to beat both the previous best NN and the classical heuristic.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       run_pipeline.py                            │
│                     (orchestrates cycles)                        │
│                                                                  │
│   ┌──────────────┐    ┌───────────┐    ┌─────────────────────┐   │
│   │ generate     │───>│  train    │───>│     evaluate         │   │
│   │ _data.py     │    │  _nn.py   │    │                     │   │
│   │              │    │           │    │  Gate 1: candidate   │   │
│   │ 50% NN self  │    │ h_weight  │    │    vs baseline      │   │
│   │ 20% CH self  │    │ floor=0.05│    │  Gate 2: candidate   │   │
│   │ 30% cross    │    │           │    │    vs CH            │   │
│   └──────────────┘    └───────────┘    └─────────────────────┘   │
│        ^                                        │                │
│        │            ┌──────────────────┐        │                │
│        │            │    promote?      │<───────┘                │
│        │            │ pass both gates  │                         │
│        │            └────────┬─────────┘                         │
│        │                     │ yes: candidate -> best weights    │
│        └─────────────────────┘ no:  keep previous best          │
│                                                                  │
│                       repeats for N cycles                       │
└──────────────────────────────────────────────────────────────────┘
```

## Anti-Mode-Collapse Design

Pure NN self-play causes mode collapse: the NN develops a narrow strategy that beats itself but is brittle against the classical heuristic (CH). Three mechanisms prevent this:

1. **Mixed data generation** — Every cycle generates data from three game types (50% NN self-play, 20% CH self-play, 30% NN-vs-CH cross-play), ensuring the NN sees diverse strategies and learns to exploit/defend against the CH.

2. **Dual evaluation gate** — After training, the candidate must beat both the previous best NN *and* the CH. This prevents the NN from drifting away from basic positional competence.

3. **Heuristic weight floor** — The heuristic loss weight decays from 0.3 but never drops below 0.05, keeping the NN anchored to CH-like position evaluations as a regularizer.

## Iterative Training Process

### Cycle 1 (Bootstrap — no NN weights exist)
- **Data generation:** 100% CH self-play (no NN available yet)
- **Evaluation:** Candidate vs CH only (Gate 2 skipped since baseline is already CH)

### Cycle 2+ (Mixed self-improvement)
- **Data generation:** 50% NN self-play, 20% CH self-play, 30% cross-play
- **Evaluation:** Candidate must pass both Gate 1 (vs previous best NN) and Gate 2 (vs CH)
- Cross-play h_score labels always come from the CH to avoid circular bias

## Quick Start

### Automated Pipeline (Recommended)

```bash
python training/run_pipeline.py
```

Runs 5 full cycles of: mixed data generation -> training -> dual-gate evaluation. Each cycle uses the previous best model (or CH on the first run) and attempts to improve on it.

### Manual Steps

Each step can also be run independently:

#### 1. Generate mixed training data
```bash
python training/generate_data.py
```
- Plays 10,000 games using minimax search with mixed game modes
- If NN weights exist: 50% NN self-play, 20% CH self-play, 30% cross-play
- If no NN weights: 100% CH self-play (bootstrap)
- Diversity via random openings (first 6 moves), epsilon-greedy exploration (25%), and depth jitter (+/- 1)
- Data is augmented 8x using D4 board symmetry (4 rotations x 2 reflections)
- Outputs to `training/data/training_data.npz`

See [DATA_GENERATION_README.md](DATA_GENERATION_README.md) for full details.

#### 2. Train the NN
```bash
python training/train_nn.py
```
- Network: `Input(139) -> Dense(1024, ReLU) -> Dense(512, ReLU) -> Dense(256, ReLU) -> Dense(1, Tanh)` (~800K params)
- Dual loss with decaying heuristic weight:
  - `(1 - h_weight) * MSE(pred, game_outcome)` — learn what actually wins
  - `h_weight * MSE(pred, heuristic_score)` — bootstrap from existing knowledge
  - `h_weight` starts at 0.3 and decays linearly to a floor of 0.05
- Exports best weights (by validation loss) to `weights/heuristic_v1.npz`
- Supports `NN_OUTPUT_PATH` env var to write to a custom path (used by the pipeline)

#### 3. Evaluate
```bash
python training/evaluate.py
```
- Plays NN heuristic vs classical heuristic head-to-head
- Each side plays as both colors for fairness

## File Structure

```
training/
├── run_pipeline.py      # Automated iterative pipeline (dual-gate eval)
├── generate_data.py     # Mixed data generation (NN/CH/cross-play)
├── train_nn.py          # PyTorch training script (h_weight floor)
├── evaluate.py          # Standalone NN vs classical evaluation
├── data/
│   └── training_data.npz    # Generated training data
│
weights/
├── heuristic_v1.npz         # Current best weights (used at inference)
│
src/
├── nn_heuristic.py          # Feature extraction + numpy inference
├── minimax_alpha_beta_h_nic.py  # Minimax search (loads heuristic at import)
└── heuristic_functions.py   # Classical hand-crafted heuristic
```

## Training Data Format

Each sample in `training_data.npz` contains:
- `features` (139 floats): board encoding — 64 piece positions, 64 positionally-weighted positions, 11 scalar features (piece counts, corners, edges, frontier ratio, game phase)
- `outcomes` (float): game result from current player's perspective (+1 win, -1 loss, 0 draw)
- `hscores` (float): heuristic score at that position, normalized to [-1, 1]. In cross-play games, this always comes from the CH to avoid circular bias.

## Pipeline Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `MIN_CYCLES` | `run_pipeline.py` | 5 | Number of full training iterations |
| `EVAL_GAMES` | `run_pipeline.py` | 25 | Games per color in evaluation (50 total) |
| `NUM_GAMES` | `generate_data.py` | 10,000 | Games per cycle |
| `NN_RATIO` | `generate_data.py` | 0.5 | Fraction of NN self-play games |
| `CH_RATIO` | `generate_data.py` | 0.2 | Fraction of CH self-play games |
| `CROSS_RATIO` | `generate_data.py` | 0.3 | Fraction of NN-vs-CH cross-play games |
| `SEARCH_DEPTH` | `generate_data.py` | 3 | Base minimax depth during data generation |
| `EPOCHS` | `train_nn.py` | 80 | Training epochs per cycle |
| `BATCH_SIZE` | `train_nn.py` | 512 | Training batch size |
| `HEURISTIC_WEIGHT_FLOOR` | `train_nn.py` | 0.05 | Minimum heuristic loss weight |
