"""
baseline.py
-----------
Historical mean return estimator for each ETF.
Predicts next-day return as the trailing 60-day mean.
Evaluates on the test set and saves results.

Usage:
    python models/baseline.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR   = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR   = os.path.join(RESULTS_DIR, "metrics")
LOOKBACK      = 60   # trailing days for mean estimate
TICKERS       = ["SPY", "QQQ", "GLD", "VEQT"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def directional_accuracy(y_true, y_pred):
    """Fraction of predictions where the sign matches actual."""
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def evaluate(y_true, y_pred, ticker):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    da   = directional_accuracy(y_true, y_pred)
    return {"Ticker": ticker, "MAE": mae, "RMSE": rmse, "Directional Accuracy": da}


if __name__ == "__main__":
    # Load returns and test dates
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "returns.csv"),
        index_col=0, parse_dates=True
    )

    seq_dir    = os.path.join(PROCESSED_DIR, "sequences")
    test_dates = pd.read_csv(
        os.path.join(seq_dir, "dates_test.csv"), header=0
    ).squeeze()
    test_dates = pd.to_datetime(test_dates)

    y_test = np.load(os.path.join(seq_dir, "y_test.npy"))

    print(f"Test period: {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()}")
    print(f"Test samples: {len(test_dates)}\n")

    all_metrics = []
    all_preds   = {}

    for i, ticker in enumerate(TICKERS):
        preds = []
        for date in test_dates:
            # Use all returns strictly before this date as history
            history = returns[ticker].loc[:date].iloc[:-1]
            trailing = history.iloc[-LOOKBACK:]
            preds.append(trailing.mean())

        preds    = np.array(preds)
        actuals  = y_test[:, i]

        metrics = evaluate(actuals, preds, ticker)
        all_metrics.append(metrics)
        all_preds[ticker] = preds

        print(f"{ticker}:")
        print(f"  MAE:                  {metrics['MAE']:.6f}")
        print(f"  RMSE:                 {metrics['RMSE']:.6f}")
        print(f"  Directional Accuracy: {metrics['Directional Accuracy']:.4f}")

    # Save metrics
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_df = pd.DataFrame(all_metrics).set_index("Ticker")
    metrics_df.to_csv(os.path.join(METRICS_DIR, "baseline_metrics.csv"))
    print(f"\nMetrics saved -> results/metrics/baseline_metrics.csv")

    # Save predictions
    preds_df = pd.DataFrame(all_preds, index=test_dates)
    preds_df.to_csv(os.path.join(METRICS_DIR, "baseline_predictions.csv"))
    print(f"Predictions saved -> results/metrics/baseline_predictions.csv")

    # Plot: predicted vs actual for each ETF
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()
    COLORS = {"SPY": "#378ADD", "QQQ": "#7F77DD", "GLD": "#EF9F27", "VEQT": "#1D9E75"}

    for i, ticker in enumerate(TICKERS):
        ax      = axes[i]
        actuals = y_test[:, i]
        preds   = all_preds[ticker]

        ax.plot(test_dates, actuals, label="Actual",    color=COLORS[ticker], linewidth=1.0, alpha=0.8)
        ax.plot(test_dates, preds,   label="Predicted", color="#888780",       linewidth=1.0, linestyle="--")
        ax.set_title(f"{ticker} — baseline predicted vs actual", fontsize=11)
        ax.set_ylabel("Daily return")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.3, color="#cccccc")

    fig.suptitle("Baseline model (60-day trailing mean) — test period", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "baseline_predictions.png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved -> results/figures/baseline_predictions.png")

    print("\nBaseline complete.")
    print("\nSummary:")
    print(metrics_df.round(6).to_string())