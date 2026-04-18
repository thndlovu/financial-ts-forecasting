"""
evaluate.py
-----------
Loads saved LSTM test predictions and compares against the baseline.
Produces a side-by-side metrics table and predicted vs actual plots.

Usage:
    python models/evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
METRICS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
TICKERS       = ["SPY", "QQQ", "GLD", "VEQT"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
COLORS = {"SPY": "#378ADD", "QQQ": "#7F77DD", "GLD": "#EF9F27", "VEQT": "#1D9E75"}


def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def compute_metrics(y_true, y_pred, ticker):
    return {
        "Ticker": ticker,
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Directional Accuracy": directional_accuracy(y_true, y_pred),
    }


if __name__ == "__main__":
    # Load predictions
    y_pred   = np.load(os.path.join(METRICS_DIR, "lstm_test_predictions.npy"))
    y_actual = np.load(os.path.join(METRICS_DIR, "lstm_test_actuals.npy"))

    seq_dir    = os.path.join(PROCESSED_DIR, "sequences")
    test_dates = pd.to_datetime(
        pd.read_csv(os.path.join(seq_dir, "dates_test.csv")).squeeze()
    )

    # Load baseline metrics for comparison
    baseline_df = pd.read_csv(
        os.path.join(METRICS_DIR, "baseline_metrics.csv"), index_col=0
    )

    # Compute LSTM metrics
    lstm_metrics = []
    for i, ticker in enumerate(TICKERS):
        lstm_metrics.append(
            compute_metrics(y_actual[:, i], y_pred[:, i], ticker)
        )

    lstm_df = pd.DataFrame(lstm_metrics).set_index("Ticker")

    # Comparison table
    print("\n=== Baseline vs LSTM — Test Set Metrics ===\n")
    for ticker in TICKERS:
        print(f"{ticker}:")
        print(f"  {'':25s} {'Baseline':>12}  {'LSTM':>12}")
        print(f"  {'MAE':25s} {baseline_df.loc[ticker,'MAE']:>12.6f}  {lstm_df.loc[ticker,'MAE']:>12.6f}")
        print(f"  {'RMSE':25s} {baseline_df.loc[ticker,'RMSE']:>12.6f}  {lstm_df.loc[ticker,'RMSE']:>12.6f}")
        print(f"  {'Directional Accuracy':25s} {baseline_df.loc[ticker,'Directional Accuracy']:>12.4f}  {lstm_df.loc[ticker,'Directional Accuracy']:>12.4f}")
        print()

    # Save combined metrics
    combined = pd.concat(
        [baseline_df.add_prefix("Baseline_"), lstm_df.add_prefix("LSTM_")],
        axis=1
    )
    combined.to_csv(os.path.join(METRICS_DIR, "comparison_metrics.csv"))
    print(f"Comparison table saved -> results/metrics/comparison_metrics.csv")

    # Plot predicted vs actual for each ETF
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, ticker in enumerate(TICKERS):
        ax = axes[i]
        ax.plot(test_dates, y_actual[:, i], label="Actual",
                color=COLORS[ticker], linewidth=1.0, alpha=0.8)
        ax.plot(test_dates, y_pred[:, i],   label="LSTM predicted",
                color="#888780", linewidth=1.0, linestyle="--")
        da = lstm_df.loc[ticker, "Directional Accuracy"]
        ax.set_title(f"{ticker}  (dir. acc: {da:.3f})", fontsize=11)
        ax.set_ylabel("Daily return")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.3, color="#cccccc")

    fig.suptitle("LSTM predicted vs actual returns — test period", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "lstm_predictions.png"), dpi=150)
    print(f"Plot saved -> results/figures/lstm_predictions.png")
    print("\nEvaluation complete.")