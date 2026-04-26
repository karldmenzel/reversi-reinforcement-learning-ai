import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cycle_utils import load_cycle_config, TRAIN_METRICS_PATH

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'training_data.npz')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'src', 'weights')
OUTPUT_PATH = os.environ.get('NN_OUTPUT_PATH',
                             os.path.join(WEIGHTS_DIR, 'heuristic_v1.npz'))

BATCH_SIZE = 512
EPOCHS = 80
LR = 0.001
WEIGHT_DECAY = 1e-4
VALIDATION_SPLIT = 0.1

# Dual loss weights — outcome-dominant (Charles style)
HEURISTIC_WEIGHT_START = 0.3   # start with 70% outcome, 30% heuristic
HEURISTIC_WEIGHT_FLOOR = 0.05  # decay heuristic to 5%
# ───────────────────────────────────────────────────────────────────────────────


class ReversiNet(nn.Module):
    def __init__(self, input_dim=139):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def export_weights(model, path):
    """Export PyTorch model weights to numpy .npz for inference.

    Architecture (no Dropout): Linear → ReLU layers.
    Layer mapping:
      net.0 = Linear(input, 1024)
      net.1 = ReLU
      net.2 = Linear(1024, 512)
      net.3 = ReLU
      net.4 = Linear(512, 256)
      net.5 = ReLU
      net.6 = Linear(256, 1)
      net.7 = Tanh
    """
    state = model.state_dict()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.savez(
        path,
        w1=state['net.0.weight'].cpu().numpy().T,
        b1=state['net.0.bias'].cpu().numpy(),
        w2=state['net.2.weight'].cpu().numpy().T,
        b2=state['net.2.bias'].cpu().numpy(),
        w3=state['net.4.weight'].cpu().numpy().T,
        b3=state['net.4.bias'].cpu().numpy(),
        w4=state['net.6.weight'].cpu().numpy().T,
        b4=state['net.6.bias'].cpu().numpy(),
    )
    print(f"Exported weights to {path}")


def main():
    global EPOCHS, LR, HEURISTIC_WEIGHT_START, HEURISTIC_WEIGHT_FLOOR

    # Override defaults from adaptive cycle config (if present)
    cfg = load_cycle_config()
    EPOCHS = int(cfg['epochs'])
    LR = float(cfg['lr'])
    HEURISTIC_WEIGHT_START = float(cfg['heuristic_weight_start'])
    HEURISTIC_WEIGHT_FLOOR = float(cfg['heuristic_weight_floor'])

    # Load data
    print(f"Loading data from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    features = data['features']
    outcomes = data['outcomes']
    hscores = data['hscores']
    print(f"Loaded {len(features)} samples, feature dim: {features.shape[1]}")

    # Train/val split
    n = len(features)
    indices = np.random.permutation(n)
    val_size = int(n * VALIDATION_SPLIT)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create datasets
    def make_loader(idx, shuffle=True):
        ds = TensorDataset(
            torch.from_numpy(features[idx]).float(),
            torch.from_numpy(outcomes[idx]).float(),
            torch.from_numpy(hscores[idx]).float(),
        )
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    train_loader = make_loader(train_idx, shuffle=True)
    val_loader = make_loader(val_idx, shuffle=False)

    # Model
    model = ReversiNet(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\nTraining for {EPOCHS} epochs...")
    print(f"  Train samples: {len(train_idx)}")
    print(f"  Val samples:   {len(val_idx)}")
    print(f"  Parameters:    {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        # Heuristic weight decays linearly but never below floor
        h_weight = max(HEURISTIC_WEIGHT_FLOOR,
                       HEURISTIC_WEIGHT_START * (1.0 - epoch / EPOCHS))
        o_weight = 1.0 - h_weight

        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0
        for feat_batch, out_batch, hs_batch in train_loader:
            feat_batch = feat_batch.to(device)
            out_batch = out_batch.to(device)
            hs_batch = hs_batch.to(device)

            pred = model(feat_batch)
            loss = (o_weight * nn.functional.mse_loss(pred, out_batch) +
                    h_weight * nn.functional.mse_loss(pred, hs_batch))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= n_batches

        # Validation — outcome loss only (matches Charles)
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for feat_batch, out_batch, hs_batch in val_loader:
                feat_batch = feat_batch.to(device)
                out_batch = out_batch.to(device)
                pred = model(feat_batch)
                loss = nn.functional.mse_loss(pred, out_batch)
                val_loss += loss.item()
                val_batches += 1
        val_loss /= val_batches

        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:3d}/{EPOCHS} | "
              f"train: {train_loss:.4f} | "
              f"val_o: {val_loss:.4f} | "
              f"h_w: {h_weight:.3f} | "
              f"lr: {lr:.6f}")

        # Save best model based on outcome validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            export_weights(model, OUTPUT_PATH)
            print(f"  -> New best val_loss: {best_val_loss:.4f}")

    epochs_run = EPOCHS
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Weights saved to {OUTPUT_PATH}")

    # Write metrics for the adaptive pipeline to read
    metrics = {
        'best_val_loss': float(best_val_loss),
        'epochs_run': epochs_run,
    }
    with open(TRAIN_METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote training metrics to {TRAIN_METRICS_PATH}")


if __name__ == '__main__':
    main()
