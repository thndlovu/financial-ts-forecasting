"""
train.py
--------
Training loop for the ETF LSTM forecaster.
Trains on GPU if available, falls back to CPU.
Saves best model weights and training curves.

Usage:
    python models/train.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.lstm_model import ETFForecaster

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR    = os.path.dirname(__file__)
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR   = os.path.join(RESULTS_DIR, "metrics")

EPOCHS        = 100
BATCH_SIZE    = 32
LR            = 1e-3
PATIENCE      = 15       # early stopping patience
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------
def load_split(seq_dir, split):
    X = np.load(os.path.join(seq_dir, f"X_{split}.npy")).astype(np.float32)
    y = np.load(os.path.join(seq_dir, f"y_{split}.npy")).astype(np.float32)
    return torch.tensor(X), torch.tensor(y)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    seq_dir = os.path.join(PROCESSED_DIR, "sequences")
    X_train, y_train = load_split(seq_dir, "train")
    X_val,   y_val   = load_split(seq_dir, "val")
    X_test,  y_test  = load_split(seq_dir, "test")

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model = ETFForecaster(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=y_train.shape[1],
        dropout=DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}\n")

    optimizer  = torch.optim.Adam(model.parameters(), lr=LR)
    criterion  = nn.MSELoss()
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=7, factor=0.5, verbose=True
    )

    best_val_loss = float("inf")
    patience_ctr  = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        train_loss = epoch_loss / len(X_train)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred      = model(X_batch)
                val_loss += criterion(pred, y_batch).item() * len(X_batch)
        val_loss /= len(X_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  "
                  f"train_loss: {train_loss:.6f}  "
                  f"val_loss:   {val_loss:.6f}")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr  = 0
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, "lstm_best.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {PATIENCE} epochs)")
                break

    print(f"\nBest val loss: {best_val_loss:.6f}")

    # --- Save training curves ---
    curves = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
    curves.index.name = "epoch"
    curves.to_csv(os.path.join(METRICS_DIR, "training_curves.csv"))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(curves["train_loss"], label="Train loss", color="#378ADD", linewidth=1.4)
    ax.plot(curves["val_loss"],   label="Val loss",   color="#D85A30", linewidth=1.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("LSTM training vs validation loss")
    ax.legend()
    ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "training_curves.png"), dpi=150)
    print(f"Training curves saved -> results/figures/training_curves.png")

    # --- Test set predictions ---
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm_best.pt"),
                                     map_location=device))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()

    np.save(os.path.join(METRICS_DIR, "lstm_test_predictions.npy"), y_pred)
    np.save(os.path.join(METRICS_DIR, "lstm_test_actuals.npy"),     y_test.numpy())
    print(f"Test predictions saved -> results/metrics/lstm_test_predictions.npy")
    print("\nTraining complete.")


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    train()