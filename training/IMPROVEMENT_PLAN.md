# Plan: Improve Reversi NN Training Pipeline

## Context

The NN heuristic isn't improving across pipeline cycles. After analyzing the full architecture (data generation, features, network, training loop, evaluation), I identified several root causes ranked by impact.

## Root Cause Analysis

### 1. Missing Mobility Feature (CRITICAL)
The classic heuristic weights mobility at **2x** in early/mid game — it's the most important Reversi signal. But `extract_features()` deliberately excludes it "for speed." This is a false economy: mobility computation is ~500 ops vs ~800K ops for the NN matrix multiplies (<0.1% overhead). The NN literally cannot learn good Reversi without knowing how many moves each side has.

### 2. Circular h_score Bias in NN Self-Play
In `MODE_NN_SELF` (50% of games!), `hscore_h = nn_fn` — the NN's own evaluation is used as the training label. This is circular: the NN learns to match its own output, reinforcing whatever biases it already has.

### 3. No Temporal Discounting on Outcomes
Every position in a game gets the same outcome label (+1/-1/0). A random opening move at turn 3 gets the same credit as a decisive endgame move at turn 55. This adds massive noise to the training signal.

### 4. No Data Accumulation Across Cycles
Each pipeline cycle generates 10K games, trains on them, then the next cycle generates a fresh 10K and discards the old. The model never sees a large, diverse dataset.

### 5. Network Lacks Regularization
No batch normalization, no dropout — only weight decay (1e-4). The 800K-parameter network can easily overfit to the noisy labels.

### 6. Heuristic Weight Floor Caps Improvement
`HEURISTIC_WEIGHT_FLOOR = 0.05` permanently anchors the NN to the classic heuristic. The NN can never fully diverge from CH, even when CH is wrong.

---

## Changes (ordered by file)

### A. Feature Engineering — `src/nn_heuristic.py`

**Add 5 new features** (139 → 144 input dim):

| Index | Feature | Description |
|-------|---------|-------------|
| 139 | Player mobility | `player_legal_moves / 20.0` |
| 140 | Opponent mobility | `opponent_legal_moves / 20.0` |
| 141 | Mobility ratio | `(p_moves - o_moves) / max(p_moves + o_moves, 1)` |
| 142 | Player stable discs | Count of player's unflippable discs / 64 |
| 143 | Opponent stable discs | Count of opponent's unflippable discs / 64 |

**Stable disc computation:** Flood-fill from each owned corner along edges. A disc is stable if it's on a fully-filled edge anchored by an owned corner, or adjacent to other stable discs along all axes. Use a simplified version: count discs on complete edge runs from owned corners (cheap, captures the most important cases).

**Mobility computation:** Add optional `game` parameter to `extract_features(board, player, game=None)`. When provided, compute exact mobility via `get_legal_moves`. When `None`, set mobility features to 0 (backward-compatible). Update callers in `generate_data.py` and `NNHeuristic.__call__` to pass the game object.

**Parameter impact:** 5 extra inputs × 1024 first-layer neurons = +5,120 params. Total ~805K, well under 1M limit.

### B. Data Generation — `training/generate_data.py`

1. **Fix circular h_score bias:** Change `MODE_NN_SELF` to always use `ch_fn` for `hscore_h` (same as cross-play already does).

```python
# Line 176: Change from:
white_h, black_h, hscore_h = nn_fn, nn_fn, nn_fn
# To:
white_h, black_h, hscore_h = nn_fn, nn_fn, ch_fn
```

2. **Add temporal discounting:** Store move index alongside each position. Weight outcomes by proximity to game end:

```python
discount = 0.5 + 0.5 * (move_index / total_moves)  # ranges 0.5 to 1.0
discounted_outcome = outcome * discount
```

This means early positions get 50% of the outcome signal, late positions get 100%.

3. **Lower epsilon from 0.15 to 0.08:** Less random noise in training data while still maintaining exploration diversity.

4. **Accumulate data across cycles:** Append new data to existing `training_data.npz` instead of overwriting. Keep last 3 cycles (~30K games, ~240K augmented samples). Newer data replaces oldest when the cap is exceeded.

### C. Network Architecture — `training/train_nn.py`

Update `ReversiNet` to add batch normalization and dropout:

```python
class ReversiNet(nn.Module):
    def __init__(self, input_dim=144):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )
```

**Numpy inference update** (`nn_heuristic.py`): Add batch norm parameters (running_mean, running_var, gamma, beta) to the numpy forward pass. BN in eval mode is just: `x = gamma * (x - mean) / sqrt(var + eps) + beta` — a simple element-wise operation.

### D. Training Loop — `training/train_nn.py`

1. **Remove heuristic weight floor:** Change `HEURISTIC_WEIGHT_FLOOR` from 0.05 to 0.0. Let the heuristic loss decay fully to zero so the NN can surpass the CH.

2. **Add gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

3. **Update `export_weights`** to also export batch norm parameters (running_mean, running_var, weight, bias for each BN layer).

4. **Update input_dim default** from 139 to 144.

### E. Numpy Inference — `src/nn_heuristic.py`

1. **Update `NNHeuristic.__init__`** to load BN params (`bn1_mean, bn1_var, bn1_weight, bn1_bias`, etc.)
2. **Update `NNHeuristic.__call__`** to apply BN in inference mode between each linear + ReLU
3. **Update `extract_features`** signature: add optional `game` and `player_int` params for mobility computation. When `game` is None, create a temporary one.
4. **Update input dim** in docstring and anywhere 139 is referenced.

### F. Pipeline — `training/run_pipeline.py`

1. **Data accumulation logic:** After generating new data, merge with previous `training_data.npz` if it exists. Keep most recent ~240K samples (3 cycles × 10K games × 8 augmentations). Trim oldest samples when over cap.

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/nn_heuristic.py` | Add 5 features, update NNHeuristic for BN, add mobility helper |
| `training/train_nn.py` | Add BatchNorm + Dropout, remove h_weight floor, add grad clipping, update export |
| `training/generate_data.py` | Fix circular bias, add temporal discounting, lower epsilon, accumulate data |
| `training/run_pipeline.py` | Add data merging logic between cycles |

## Verification

1. Run `python -c "from nn_heuristic import extract_features; import numpy as np; print(extract_features(np.zeros((8,8)), 1).shape)"` — should print `(144,)`
2. Run `python training/generate_data.py` with `NUM_GAMES=100` — verify no crashes, check output shapes
3. Run `python training/train_nn.py` — verify training completes with new architecture
4. Run `python training/evaluate.py` — verify evaluation works with new weights
5. Run one full pipeline cycle (`python training/run_pipeline.py` with `MIN_CYCLES=1`) — verify end-to-end

## What This Does NOT Change

- The minimax/alpha-beta search (`minimax_alpha_beta_h_nic_nn.py`) — no changes needed
- The classic heuristic functions — untouched
- The game engine (`reversi.py`) — untouched
- The evaluation structure (dual-gate) — untouched
- Layer sizes (1024/512/256) — kept the same
