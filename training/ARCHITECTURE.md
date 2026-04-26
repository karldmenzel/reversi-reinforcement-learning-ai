# Architecture: Neural Network Heuristic for Reversi

## System Overview

The NN heuristic replaces the hand-crafted board evaluation function inside the existing minimax alpha-beta search. The minimax framework stays unchanged — only the leaf-node scoring function is swapped.

```
                    ┌─────────────────────────┐
                    │   Minimax Alpha-Beta     │
                    │   (iterative deepening)  │
                    │                          │
                    │   Unchanged — searches   │
                    │   the game tree, pruning  │
                    │   with alpha-beta bounds  │
                    └───────────┬──────────────┘
                                │
                         leaf node reached
                                │
                    ┌───────────▼──────────────┐
                    │   Heuristic Function      │
                    │                          │
                    │   OLD: heuristic_nic()    │
                    │     - positional weights  │
                    │     - center control      │
                    │     - piece difference    │
                    │     - mobility (slow)     │
                    │                          │
                    │   NEW: NNHeuristic()      │
                    │     - 144 features        │
                    │     - 4-layer MLP          │
                    │     - ~0.04ms per call    │
                    │     - no get_legal_moves  │
                    └──────────────────────────┘
```

## Neural Network Architecture

```
 Input Layer (144 features)
       │
       ▼
 ┌──────────────┐
 │  Dense(512)  │──── ReLU ──── Dropout(0.15)
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  Dense(256)  │──── ReLU ──── Dropout(0.10)
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  Dense(128)  │──── ReLU
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │  Dense(1)    │──── Tanh × 1000
 └──────┬───────┘
        ▼
   Score ∈ [-1000, 1000]

 ~238,000 parameters (under 1M budget)
 Inference: 4 matrix multiplies + ReLU clamps + tanh
```

### Why MLP Over CNN

The board is 8x8 (64 cells). At this size, a small MLP with engineered spatial features captures patterns as well as a CNN, but inference is significantly faster — just four matrix multiplies in numpy with no convolution overhead.

## Feature Engineering (144 Features)

```
Feature Index   Count   Description
─────────────   ─────   ───────────────────────────────────────────
  0 - 63         64     Net mask: (board == player) - (board == opponent)
                        Encodes piece placement as +1 / -1 / 0

 64 - 127        64     Net mask × WEIGHT_MATRIX (normalized)
                        Encodes positional value of each piece

128              1      Player piece count (/ 32)
129              1      Opponent piece count (/ 32)
130              1      Piece differential (/ total)
131              1      Game phase (total_pieces / 64)
132              1      Player corners held (/ 4)
133              1      Opponent corners held (/ 4)
134              1      Player X-squares occupied (/ 4)
135              1      Opponent X-squares occupied (/ 4)
136              1      Player edge pieces (/ 28)
137              1      Opponent edge pieces (/ 28)
138              1      Frontier ratio (who has more exposed pieces)
139              1      Player mobility (/ 20)
140              1      Opponent mobility (/ 20)
141              1      Mobility ratio
142              1      Player stable discs (/ 64)
143              1      Opponent stable discs (/ 64)
                ───
                144     Total
```

**Key design choice:** Features 64-127 (WEIGHT_MATRIX features) use a fixed copy in `nn_heuristic.py`. During adaptive training, the CH heuristic may use a perturbed WEIGHT_MATRIX, but the NN feature space stays fixed so weights transfer between cycles.

## Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA GENERATION                              │
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ cycle_config  │──── adaptive ratios, epsilon, etc                │
│  │ .json         │                                                  │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────┐     ┌───────────────┐     ┌──────────────────────┐   │
│  │ reversi  │────▶│  Mixed Play   │────▶│  Raw Positions       │   │
│  │ engine   │     │  NN/CH/Cross  │     │  (board, player,     │   │
│  │          │     │  (depth-4     │     │   h_score, outcome)  │   │
│  └──────────┘     │   minimax)    │     └──────────┬───────────┘   │
│                   │               │                │               │
│  ┌──────────┐     │  Optionally   │                │               │
│  │perturbed │────▶│  uses         │                │               │
│  │WEIGHT_   │     │  perturbed WM │                │               │
│  │MATRIX    │     └───────────────┘                │               │
│  └──────────┘                                      ▼               │
│                                          ┌─────────────────────┐   │
│                                          │  D4 Symmetry        │   │
│                                          │  Augmentation       │   │
│                                          │                     │   │
│                                          │  4 rotations        │   │
│                                          │  × 2 reflections    │   │
│                                          │  = 8x data          │   │
│                                          └──────────┬──────────┘   │
│                                                     │              │
│                                                     ▼              │
│                                          ┌─────────────────────┐   │
│                                          │  training_data.npz  │   │
│                                          │  ~4M samples/cycle  │   │
│                                          └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### D4 Symmetry Augmentation

A Reversi board has 8 symmetries (the dihedral group D4). Every position is strategically equivalent under any 90-degree rotation or reflection. Augmenting with all 8 transforms gives 8x the training data for free.

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TRAINING                                   │
│                                                                     │
│  ┌─────────────────┐   ┌──────────────┐                            │
│  │ training_data   │   │ cycle_config │──── epochs, lr, h_weight   │
│  │     .npz        │   │    .json     │                            │
│  └────────┬────────┘   └──────┬───────┘                            │
│           │                   │                                     │
│           ▼                   ▼                                     │
│  ┌─────────────────┐     ┌──────────────────────────────────────┐  │
│  │  PyTorch        │     │  Dual Loss Function                  │  │
│  │  DataLoader     │────▶│                                      │  │
│  │  (batch=512)    │     │  L = w_o × MSE(pred, outcome)        │  │
│  └─────────────────┘     │    + w_h × MSE(pred, heuristic)      │  │
│                          │                                      │  │
│                          │  w_h starts at 0.9, decays to 0.5    │  │
│                          │  (values adapt between cycles)        │  │
│                          └──────────────┬───────────────────────┘  │
│                                         │                          │
│                                         ▼                          │
│                          ┌──────────────────────────────────────┐  │
│                          │  Adam Optimizer                      │  │
│                          │  lr=0.0005 (adaptive), cosine decay  │  │
│                          │  weight_decay=1e-4, grad_clip=1.0    │  │
│                          └──────────────┬───────────────────────┘  │
│                                         │                          │
│                                         ▼                          │
│                          ┌──────────────────────────────────────┐  │
│                          │  Best model (by val loss)            │  │
│                          │  exported to weights/heuristic_v1.npz│  │
│                          │                                      │  │
│                          │  Writes train_metrics.json:          │  │
│                          │  - best_val_loss, epochs_run         │  │
│                          │  - final_train_loss, stopped_early   │  │
│                          └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Dual Loss Rationale

| Loss Component | Purpose |
|---|---|
| `MSE(pred, game_outcome)` | Teaches the NN what board states actually lead to wins. Primary signal. |
| `MSE(pred, heuristic_score)` | Bootstraps from `heuristic_nic`'s existing knowledge. Decays but maintains a strong floor (0.5) to keep the NN anchored to sound positional play. |

The heuristic weight decay schedule (default):

```
w_h  0.9 ┤████
         │    ████
         │        ████
         │            ████
   0.5   ┤                ████████████████
         └──────────────────────────────▶ epoch
         0                             150
```

## Inference Pipeline (Game Time)

```
  Board State + Player
        │
        ▼
  ┌──────────────────┐
  │ extract_features  │    144 floats
  │                   │    (all numpy, no game engine calls)
  └────────┬─────────┘
           │
           ▼
  ┌──────────────────┐
  │ x = features     │
  │ x = x @ W1 + b1  │    (144 × 512)
  │ x = ReLU(x)      │
  │ x = x @ W2 + b2  │    (512 × 256)
  │ x = ReLU(x)      │
  │ x = x @ W3 + b3  │    (256 × 128)
  │ x = ReLU(x)      │
  │ x = x @ W4 + b4  │    (128 × 1)
  │ x = tanh(x)×1000 │
  └────────┬─────────┘
           │
           ▼
     Score ∈ [-1000, 1000]
```

**No PyTorch at game time.** Inference uses only numpy — four matrix multiplies, three ReLU clamps, one tanh. This keeps dependencies minimal and performance fast.

## File Map

```
reversi-reinforcement-learning-ai/
├── src/
│   ├── reversi.py                       # Game engine (unchanged)
│   ├── utils.py                         # Canonical WEIGHT_MATRIX (unchanged)
│   ├── heuristic_functions.py           # CH heuristic (loads perturbed WM via env var)
│   ├── nn_heuristic.py                  # NN inference: 144 features + numpy MLP
│   ├── minimax_alpha_beta_h_nic_nn.py   # Minimax search (auto-detects NN weights)
│   └── weights/
│       └── heuristic_v1.npz             # Trained weights (created by train_nn.py)
├── training/
│   ├── run_pipeline.py                  # Adaptive iterative pipeline
│   ├── cycle_utils.py                   # Shared config loading (DEFAULTS, loaders)
│   ├── generate_data.py                 # Mixed data generation (reads cycle_config)
│   ├── train_nn.py                      # PyTorch training (reads config, writes metrics)
│   ├── evaluate.py                      # Tournament evaluation
│   ├── cycle_config.json                # [generated] adaptive params per cycle
│   ├── train_metrics.json               # [generated] training results per cycle
│   └── cycle_weight_matrix.npy          # [generated] perturbed positional weights
```

## Performance Comparison

| Metric | Classic Heuristic | NN Heuristic |
|---|---|---|
| Heuristic call time | ~0.5-1ms | ~0.04ms |
| Bottleneck | `get_legal_moves()` (128 direction checks) | Matrix multiply (144x512) |
| Evaluations per 4s turn | ~4K-8K | ~100K |
| Effective search depth | 5-7 | 7-9 |
| Evaluation basis | 4 hand-tuned weights | Learned from 10K+ game outcomes |
| Parameters | 0 (hardcoded) | ~238K (learned) |

## Opponent-Denial Strategy

The NN captures opponent denial at three levels:

1. **Feature level:** Explicit opponent corner count, X-square danger, edge features, stable discs, and mobility let the network reason about opponent threats directly.

2. **Training level:** Game outcome loss naturally teaches that denying opponent advantages leads to wins — the NN sees positions where opponent corners were allowed and correlates them with losses.

3. **Search level:** Faster heuristic = deeper minimax search = more opponent response layers explored. The min layers in minimax explicitly model "what's the best my opponent can do?"
