"""
optimize.py
-----------
Rolling mean-variance portfolio optimization comparing:
  A) Historical mean expected returns (traditional)
  B) LSTM-adjusted expected returns (our approach)
     mu_lstm = mu_hist + alpha * lstm_signal
     where lstm_signal = normalized LSTM predictions
  C) SPY buy-and-hold benchmark

Uses PyPortfolioOpt EfficientFrontier with max-Sharpe objective.

Usage:
    pip install PyPortfolioOpt
    python portfolio/optimize.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
METRICS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")
FIGURES_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
LOOKBACK      = 252
TICKERS       = ["SPY", "QQQ", "GLD", "VEQT"]
RISK_FREE     = 0.05
ALPHA         = 0.3      # blending factor: how much weight to give LSTM signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

COLORS = {
    "LSTM Portfolio":       "#7F77DD",
    "Historical Portfolio": "#D85A30",
    "SPY Buy-and-Hold":     "#378ADD",
}


def max_sharpe_weights(mu, cov):
    """Max-Sharpe optimization. Falls back to equal weights on failure."""
    try:
        ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
        ef.max_sharpe(risk_free_rate=RISK_FREE / 252)
        w  = ef.clean_weights()
        return np.array([w[t] for t in TICKERS])
    except Exception:
        return np.ones(len(TICKERS)) / len(TICKERS)


def run_backtest(returns, lstm_preds, test_dates):
    hist_returns, lstm_returns, spy_returns = [], [], []
    hist_weights_log, lstm_weights_log      = [], []

    for day_idx, date in enumerate(test_dates):
        history = returns.loc[:date].iloc[-(LOOKBACK + 1):-1]

        if len(history) < 60:
            w_hist = np.ones(len(TICKERS)) / len(TICKERS)
            w_lstm = w_hist.copy()
        else:
            cov = pd.DataFrame(
                np.cov(history.values.T) * 252,
                index=TICKERS, columns=TICKERS
                )

            # --- Portfolio A: pure historical mean ---
            mu_hist = pd.Series(history.mean() * 252, index=TICKERS)
            w_hist  = max_sharpe_weights(mu_hist, cov)

            # --- Portfolio B: historical mean + LSTM signal blend ---
            # Normalize LSTM predictions to have same scale as mu_hist
            lstm_raw    = pd.Series(lstm_preds[day_idx], index=TICKERS)
            lstm_signal = (lstm_raw - lstm_raw.mean()) / (lstm_raw.std() + 1e-8)
            # Scale signal to match magnitude of historical returns
            hist_scale  = mu_hist.std()
            mu_lstm     = mu_hist + ALPHA * lstm_signal * hist_scale
            w_lstm      = max_sharpe_weights(mu_lstm, cov)

        hist_weights_log.append(w_hist)
        lstm_weights_log.append(w_lstm)

        actual = returns.loc[date][TICKERS].values
        hist_returns.append(np.dot(w_hist, actual))
        lstm_returns.append(np.dot(w_lstm, actual))
        spy_returns.append(returns.loc[date]["SPY"])

    return (
        np.array(hist_returns),
        np.array(lstm_returns),
        np.array(spy_returns),
        np.array(hist_weights_log),
        np.array(lstm_weights_log),
    )


def performance_metrics(daily_returns, label):
    cum      = (1 + daily_returns).cumprod()
    total    = cum[-1] - 1
    ann_ret  = (1 + total) ** (252 / len(daily_returns)) - 1
    ann_vol  = daily_returns.std() * np.sqrt(252)
    sharpe   = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0
    drawdown = (cum / np.maximum.accumulate(cum) - 1).min()

    print(f"{label}:")
    print(f"  Total return:      {total:+.2%}")
    print(f"  Annualised return: {ann_ret:+.2%}")
    print(f"  Annualised vol:    {ann_vol:.2%}")
    print(f"  Sharpe ratio:      {sharpe:.3f}")
    print(f"  Max drawdown:      {drawdown:.2%}")
    print()

    return {
        "Portfolio":       label,
        "Total Return":    total,
        "Ann. Return":     ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio":    sharpe,
        "Max Drawdown":    drawdown,
    }


def plot_cumulative(test_dates, hist_ret, lstm_ret, spy_ret):
    hist_cum = (1 + hist_ret).cumprod()
    lstm_cum = (1 + lstm_ret).cumprod()
    spy_cum  = (1 + spy_ret).cumprod()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(test_dates, lstm_cum, label="LSTM-adjusted portfolio",
            color=COLORS["LSTM Portfolio"], linewidth=1.8)
    ax.plot(test_dates, hist_cum, label="Historical mean portfolio",
            color=COLORS["Historical Portfolio"], linewidth=1.8)
    ax.plot(test_dates, spy_cum,  label="SPY buy-and-hold",
            color=COLORS["SPY Buy-and-Hold"], linewidth=1.8, linestyle="--")

    ax.set_title("Portfolio cumulative return — test period (Feb–Dec 2024)", fontsize=13)
    ax.set_ylabel("Portfolio value (starting at 1.0)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.set_xlim(test_dates.iloc[0], test_dates.iloc[-1])
    ax.legend(fontsize=10)
    ax.grid(axis="y", linewidth=0.3, color="#cccccc")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "portfolio_cumulative_returns.png"), dpi=150)
    print("Saved -> results/figures/portfolio_cumulative_returns.png")


def plot_weights(test_dates, hist_weights, lstm_weights):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    titles = ["Historical mean portfolio weights", "LSTM-adjusted portfolio weights"]

    for ax, weights, title in zip(axes, [hist_weights, lstm_weights], titles):
        for i, ticker in enumerate(TICKERS):
            ax.plot(test_dates, weights[:, i], label=ticker, linewidth=1.2)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Portfolio weight")
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.legend(fontsize=9)
        ax.grid(axis="y", linewidth=0.3, color="#cccccc")

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "portfolio_weights.png"), dpi=150)
    print("Saved -> results/figures/portfolio_weights.png")


if __name__ == "__main__":
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "returns.csv"),
        index_col=0, parse_dates=True
    )
    seq_dir    = os.path.join(PROCESSED_DIR, "sequences")
    test_dates = pd.to_datetime(
        pd.read_csv(os.path.join(seq_dir, "dates_test.csv")).squeeze()
    )
    lstm_preds = np.load(os.path.join(METRICS_DIR, "lstm_test_predictions.npy"))

    print(f"Test period: {test_dates.iloc[0].date()} to {test_dates.iloc[-1].date()}")
    print(f"Running backtest over {len(test_dates)} trading days...\n")

    hist_ret, lstm_ret, spy_ret, hist_w, lstm_w = run_backtest(
        returns, lstm_preds, test_dates
    )

    print("=== Portfolio Performance — Test Period ===\n")
    metrics = []
    metrics.append(performance_metrics(lstm_ret, "LSTM-adjusted Portfolio"))
    metrics.append(performance_metrics(hist_ret, "Historical Portfolio"))
    metrics.append(performance_metrics(spy_ret,  "SPY Buy-and-Hold"))

    metrics_df = pd.DataFrame(metrics).set_index("Portfolio")
    metrics_df.to_csv(os.path.join(METRICS_DIR, "portfolio_metrics.csv"))
    print(f"Metrics saved -> results/metrics/portfolio_metrics.csv")

    daily_df = pd.DataFrame({
        "LSTM-adjusted Portfolio": lstm_ret,
        "Historical Portfolio":    hist_ret,
        "SPY Buy-and-Hold":        spy_ret,
    }, index=test_dates)
    daily_df.to_csv(os.path.join(METRICS_DIR, "portfolio_daily_returns.csv"))
    print(f"Daily returns saved -> results/metrics/portfolio_daily_returns.csv")

    print("\nGenerating plots...")
    plot_cumulative(test_dates, hist_ret, lstm_ret, spy_ret)
    plot_weights(test_dates, hist_w, lstm_w)
    print("\nChunk 5 complete.")
