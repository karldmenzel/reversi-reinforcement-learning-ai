# Training Pipeline

## Overview

The NN heuristic is trained iteratively through self-play. Each generation produces stronger training data by using the previous generation's NN as the evaluation function. The minimax search algorithm remains unchanged — only the leaf-node evaluation (heuristic) is learned.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                           │
│                  (orchestrates cycles)                       │
│                                                             │
│   ┌───────────┐    ┌───────────┐    ┌───────────────────┐   │
│   │ generate   │───>│  train    │───>│     evaluate      │   │
│   │ _data.py   │    │  _nn.py   │    │  candidate vs     │   │
│   │            │    │           │    │  current best     │   │
│   └───────────┘    └───────────┘    └───────────────────┘   │
│        ^                                     │              │
│        │            ┌────────────┐           │              │
│        │            │  promote?  │<──────────┘              │
│        │            │ wins >= ?  │                           │
│        │            └─────┬──────┘                           │
│        │                  │ yes: candidate -> best weights   │
│        └──────────────────┘ no:  keep previous best         │
│                                                             │
│                    repeats for N cycles                      │
└─────────────────────────────────────────────────────────────┘
```

## Iterative Training Process

### Generation 0 (Bootstrap)
- **Heuristic used:** `heuristic_nic` (hand-crafted classical heuristic)
- **Purpose:** Generate initial training data using domain knowledge (positional weights, corner control, mobility, frontier discs)
- **Dual loss:** The NN learns from both game outcomes (who won) and the classical heuristic scores, with heuristic weight decaying to 0 over training

### Generation 1+ (Self-Improvement)
- **Heuristic used:** `NNHeuristic` (trained NN from previous generation)
- **Purpose:** Generate higher-quality training data using the NN's learned evaluation
- **Gating:** A new model only replaces the previous best if it wins the evaluation tournament — preventing regression

## Quick Start

### Automated Pipeline (Recommended)

```bash
python training/run_pipeline.py
```

Runs 5 full cycles of: data generation -> training -> evaluation. Each cycle takes the output of the previous best model (or the classical heuristic on the first run) and attempts to improve on it.

### Manual Steps

Each step can also be run independently:

#### 1. Generate self-play data
```bash
python training/generate_data.py
```
- Plays 10,000 self-play games using minimax search
- Automatically uses NN heuristic if `weights/heuristic_v1.npz` exists, otherwise falls back to the classical heuristic
- Introduces diversity via random openings (first 6 moves), epsilon-greedy exploration (15%), and depth jitter (+/- 1)
- Data is augmented 8x using D4 board symmetry (4 rotations x 2 reflections)
- Outputs to `training/data/training_data.npz`

#### 2. Train the NN
```bash
python training/train_nn.py
```
- Network: `Input(139) -> Dense(1024, ReLU) -> Dense(512, ReLU) -> Dense(256, ReLU) -> Dense(1, Tanh)` (~800K params)
- Dual loss with decaying heuristic weight:
  - `(1 - h_weight) * MSE(pred, game_outcome)` — learn what actually wins
  - `h_weight * MSE(pred, heuristic_score)` — bootstrap from existing knowledge
  - `h_weight` starts at 0.3 and decays linearly to 0 over training
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
├── run_pipeline.py      # Automated iterative pipeline
├── generate_data.py     # Self-play data generation
├── train_nn.py          # PyTorch training script
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
- `hscores` (float): heuristic score at that position, normalized to [-1, 1]

## Pipeline Configuration

| Parameter | Location | Default | Description |
|-----------|----------|---------|-------------|
| `MIN_CYCLES` | `run_pipeline.py` | 5 | Number of full training iterations |
| `EVAL_GAMES` | `run_pipeline.py` | 25 | Games per color in evaluation (50 total) |
| `NUM_GAMES` | `generate_data.py` | 10,000 | Self-play games per cycle |
| `SEARCH_DEPTH` | `generate_data.py` | 3 | Base minimax depth during data generation |
| `EPOCHS` | `train_nn.py` | 80 | Training epochs per cycle |
| `BATCH_SIZE` | `train_nn.py` | 512 | Training batch size |
