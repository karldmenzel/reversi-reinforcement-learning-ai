# Training Pipeline

## Overview

The NN heuristic is trained iteratively through mixed self-play and cross-play with **adaptive hyperparameter tuning**. Each cycle generates diverse training data using a mix of NN self-play, classical heuristic (CH) self-play, and NN-vs-CH cross-play. A dual evaluation gate prevents mode collapse by requiring the candidate to beat both the previous best NN and the classical heuristic. Between cycles, the pipeline adjusts game ratios, training parameters, and positional weight perturbation based on evaluation results and training metrics.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          run_pipeline.py                                 │
│                     (orchestrates adaptive cycles)                       │
│                                                                          │
│   ┌─────────────────┐                                                    │
│   │ cycle_config.json│◄── compute_next_config() adapts hyperparams       │
│   │ + perturbed WM   │    based on eval results & training metrics       │
│   └────────┬────────┘                                                    │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────┐    ┌───────────┐    ┌─────────────────────┐          │
│   │ generate     │───>│  train    │───>│     evaluate         │          │
│   │ _data.py     │    │  _nn.py   │    │                     │          │
│   │              │    │           │    │  Gate 1: candidate   │          │
│   │ Reads config │    │ Reads cfg │    │    vs baseline      │          │
│   │ for ratios,  │    │ for lr,   │    │  Gate 2: candidate   │          │
│   │ epsilon, etc │    │ epochs,   │    │    vs CH            │          │
│   │              │    │ h_weight  │    │                     │          │
│   │ Reads WM env │    │           │    └─────────┬───────────┘          │
│   │ for perturbed│    │ Writes    │              │                      │
│   │ positional   │    │ train_    │              │                      │
│   │ weights      │    │ metrics   │              │                      │
│   └──────────────┘    │ .json     │              │                      │
│                       └───────────┘              │                      │
│        ^                                         │                      │
│        │            ┌──────────────────┐         │                      │
│        │            │    promote?      │◄────────┘                      │
│        │            │ pass both gates  │                                │
│        │            └────────┬─────────┘                                │
│        │                     │ yes: candidate -> best weights           │
│        └─────────────────────┘ no:  keep previous best                 │
│                                                                          │
│              ┌──────────────────────────────┐                            │
│              │ compute_next_config()        │                            │
│              │ adapts: ratios, epochs, lr,  │                            │
│              │ h_weight, perturbation, etc  │                            │
│              └──────────────────────────────┘                            │
│                                                                          │
│                       repeats for N cycles                               │
└──────────────────────────────────────────────────────────────────────────┘
```

## Adaptive Config System

The pipeline uses a JSON-based config system (`cycle_config.json`) to communicate adaptive hyperparameters to subprocesses:

### Config Flow

1. **Before each cycle:** `run_pipeline.py` writes `cycle_config.json` with the current hyperparameters
2. **Data generation:** `generate_data.py` reads the config and overrides its globals (ratios, epsilon, num_games, random_opening_moves)
3. **Training:** `train_nn.py` reads the config and overrides its globals (epochs, lr, h_weight_start, h_weight_floor)
4. **After training:** `train_nn.py` writes `train_metrics.json` with results (best_val_loss, epochs_run, stopped_early)
5. **After evaluation:** `run_pipeline.py` reads the metrics and eval results, computes the next cycle's config

### Adaptive Rules

| Condition | Action |
|-----------|--------|
| Win rate > 65% | Push NN harder: +nn_ratio, -ch_ratio, -epochs, +perturbation |
| Win rate < 45% | More guidance: +ch_ratio, -nn_ratio, +epochs, +h_weight_start, -perturbation |
| Win rate 45-65% | Stable: mild perturbation increase for diversity |
| Early stop < 50 epochs | Converged fast: reduce LR by 20% |
| No early stop | Needs more time: increase epochs |

### Backward Compatibility

When no `cycle_config.json` exists (e.g., running scripts standalone), all parameters fall back to their hardcoded defaults via `cycle_utils.py`. The adaptive system is completely opt-in.

## WEIGHT_MATRIX Perturbation

Between cycles, the pipeline can perturb the positional WEIGHT_MATRIX used by the CH heuristic during data generation. This creates diverse training data while preserving strategic structure.

**Design:**
- **Multiplicative noise** proportional to cell magnitude (corners get +-15 at strength=0.15, X-squares get +-7.5)
- **Signs always preserved** — corners stay positive, X-squares stay negative
- **Zeros stay zero** — center squares are unperturbed
- **Only affects CH scoring** during data generation (via `CYCLE_WEIGHT_MATRIX_PATH` env var)
- **NN feature extraction** uses its own fixed WEIGHT_MATRIX copy — unaffected
- **Production play** is unaffected — the env var is only set during pipeline data gen

## Anti-Mode-Collapse Design

Pure NN self-play causes mode collapse: the NN develops a narrow strategy that beats itself but is brittle against the classical heuristic (CH). Four mechanisms prevent this:

1. **Mixed data generation** — Every cycle generates data from three game types (default 50% NN self-play, 20% CH self-play, 30% NN-vs-CH cross-play), ensuring the NN sees diverse strategies.

2. **Dual evaluation gate** — After training, the candidate must beat both the previous best NN *and* the CH. This prevents the NN from drifting away from basic positional competence.

3. **Heuristic weight floor** — The heuristic loss weight decays from 0.9 but never drops below 0.5, keeping the NN anchored to CH-like position evaluations.

4. **WEIGHT_MATRIX perturbation** — Positional weight randomization between cycles prevents the NN from overfitting to a single positional weighting scheme.

## Iterative Training Process

### Cycle 1 (Bootstrap — no NN weights exist)
- **Data generation:** 100% CH self-play (no NN available yet)
- **Evaluation:** Candidate vs CH only (Gate 2 skipped since baseline is already CH)

### Cycle 2+ (Adaptive mixed self-improvement)
- **Data generation:** Ratios adapted from previous cycle's results
- **Evaluation:** Candidate must pass both Gate 1 (vs previous best NN) and Gate 2 (vs CH)
- Cross-play h_score labels always come from the CH to avoid circular bias
- Config adjusts automatically based on how well the candidate performed

## Data Accumulation

Training data is accumulated across cycles up to a cap of **7M samples** (~2 full cycles of augmented data). Oldest samples are trimmed first when the cap is exceeded. This ensures the model trains on a diverse mix of positions from different training stages rather than discarding data.

## Quick Start

### Automated Pipeline (Recommended)

```bash
python training/run_pipeline.py
```

Runs 5 full adaptive cycles of: config write -> mixed data generation -> training -> dual-gate evaluation -> adaptive config update. Each cycle uses the previous best model (or CH on the first run) and attempts to improve on it.

### Manual Steps

Each step can also be run independently (using default parameters):

#### 1. Generate mixed training data
```bash
python training/generate_data.py
```
- Plays 10,000 games using minimax search with mixed game modes
- If NN weights exist: 50% NN self-play, 20% CH self-play, 30% cross-play
- If no NN weights: 100% CH self-play (bootstrap)
- Diversity via random openings (first 4 moves), epsilon-greedy exploration (8%), and depth jitter (+/- 1)
- Data is augmented 8x using D4 board symmetry (4 rotations x 2 reflections)
- Produces ~4M augmented samples per cycle
- Outputs to `training/data/training_data.npz`

See [DATA_GENERATION_README.md](DATA_GENERATION_README.md) for full details.

#### 2. Train the NN
```bash
python training/train_nn.py
```
- Network: `Input(144) -> Dense(512, ReLU) -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(1, Tanh)` (~238K params)
- Dual loss with decaying heuristic weight:
  - `(1 - h_weight) * MSE(pred, game_outcome)` — learn what actually wins
  - `h_weight * MSE(pred, heuristic_score)` — bootstrap from existing knowledge
  - `h_weight` starts at 0.9 and decays linearly to a floor of 0.5
- Exports best weights (by validation loss) to `src/weights/heuristic_v1.npz`
- Supports `NN_OUTPUT_PATH` env var to write to a custom path (used by the pipeline)
- Writes `train_metrics.json` with training results for the adaptive pipeline

#### 3. Evaluate
```bash
python training/evaluate.py
```
- Plays NN heuristic vs classical heuristic head-to-head
- Each side plays as both colors for fairness

## File Structure

```
training/
├── run_pipeline.py          # Adaptive iterative pipeline (dual-gate eval)
├── cycle_utils.py           # Shared config loading (DEFAULTS, load_cycle_config)
├── generate_data.py         # Mixed data generation (reads cycle_config.json)
├── train_nn.py              # PyTorch training (reads config, writes metrics)
├── evaluate.py              # Standalone NN vs classical evaluation
├── cycle_config.json        # [generated] adaptive config per cycle
├── train_metrics.json       # [generated] training results per cycle
├── cycle_weight_matrix.npy  # [generated] perturbed positional weights
├── data/
│   └── training_data.npz    # Generated training data

src/weights/
├── heuristic_v1.npz         # Current best weights (used at inference)

src/
├── nn_heuristic.py          # Feature extraction (144 features) + numpy inference
├── heuristic_functions.py   # Classical heuristic (loads perturbed WM via env var)
├── minimax_alpha_beta_h_nic_nn.py  # Minimax search (auto-detects NN weights)
└── utils.py                 # Canonical WEIGHT_MATRIX (unchanged)
```

## Training Data Format

Each sample in `training_data.npz` contains:
- `features` (144 floats): board encoding — 64 piece positions, 64 positionally-weighted positions, 16 scalar features (piece counts, corners, edges, frontier ratio, game phase, mobility, stable discs)
- `outcomes` (float): game result from current player's perspective (+1 win, -1 loss, 0 draw), temporally discounted (early positions get 50% signal, late get 100%)
- `hscores` (float): heuristic score at that position, normalized to [-1, 1] via tanh(h/800). In cross-play and NN self-play games, this always comes from the CH to avoid circular bias.

## Pipeline Configuration

| Parameter | Location | Default | Adaptive? | Description |
|-----------|----------|---------|-----------|-------------|
| `MIN_CYCLES` | `run_pipeline.py` | 5 | No | Number of full training iterations |
| `EVAL_GAMES` | `run_pipeline.py` | 50 | No | Games per color in evaluation (100 total) |
| `MAX_ACCUMULATED_SAMPLES` | `run_pipeline.py` | 7,000,000 | No | Max samples kept across cycles |
| `NUM_GAMES` | `generate_data.py` | 10,000 | Yes | Games per cycle |
| `NN_RATIO` | `generate_data.py` | 0.5 | Yes | Fraction of NN self-play games |
| `CH_RATIO` | `generate_data.py` | 0.2 | Yes | Fraction of CH self-play games |
| `CROSS_RATIO` | `generate_data.py` | 0.3 | Yes | Fraction of cross-play games |
| `EPSILON` | `generate_data.py` | 0.08 | Yes | Random move probability |
| `RANDOM_OPENING_MOVES` | `generate_data.py` | 4 | Yes | Random opening moves |
| `SEARCH_DEPTH` | `generate_data.py` | 4 | No | Base minimax depth |
| `EPOCHS` | `train_nn.py` | 150 | Yes | Training epochs per cycle |
| `LR` | `train_nn.py` | 0.0005 | Yes | Learning rate |
| `BATCH_SIZE` | `train_nn.py` | 512 | No | Training batch size |
| `HEURISTIC_WEIGHT_START` | `train_nn.py` | 0.9 | Yes | Initial heuristic loss weight |
| `HEURISTIC_WEIGHT_FLOOR` | `train_nn.py` | 0.5 | Yes | Minimum heuristic loss weight |
| `perturbation_strength` | `run_pipeline.py` | 0.0 | Yes | WEIGHT_MATRIX perturbation |
