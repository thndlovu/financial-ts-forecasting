"""
engineer.py
-----------
Computes technical indicators for each ETF, builds 20-day sequences,
performs time-ordered train/val/test split, normalises features,
and saves everything to data/processed/.

Usage:
    python features/engineer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import ta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
FEATURES_DIR  = os.path.dirname(__file__)
SEQUENCE_LEN  = 20       # 20 trading days (~1 month) as model input
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# test = remaining 15%

TICKERS = ["SPY", "QQQ", "GLD", "VEQT"]


# ------------------------------------------------------------------
# 1. Feature computation
# ------------------------------------------------------------------

def compute_features(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators for all ETFs and combine into one wide DataFrame.
    Each column is named: TICKER_feature (e.g. SPY_rsi, QQQ_macd)
    """
    all_features = []

    for ticker in TICKERS:
        close  = prices[ticker]
        ret    = returns[ticker]
        df     = pd.DataFrame(index=prices.index)

        # --- Raw return & log return ---
        df[f"{ticker}_return"]     = ret
        df[f"{ticker}_log_return"] = np.log1p(ret)

        # --- RSI (14-day) ---
        df[f"{ticker}_rsi"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()

        # --- MACD line (12/26 EMA difference) ---
        macd = ta.trend.MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df[f"{ticker}_macd"]        = macd.macd()
        df[f"{ticker}_macd_signal"] = macd.macd_signal()

        # --- Exponential moving averages ---
        df[f"{ticker}_ema20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
        df[f"{ticker}_ema50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()

        # --- EMA ratio (price relative to trend) ---
        df[f"{ticker}_ema20_ratio"] = close / df[f"{ticker}_ema20"]
        df[f"{ticker}_ema50_ratio"] = close / df[f"{ticker}_ema50"]

        # --- Rolling 20-day volatility (annualised) ---
        df[f"{ticker}_vol20"] = ret.rolling(20).std() * np.sqrt(252)

        all_features.append(df)

    features = pd.concat(all_features, axis=1)

    # Drop rows with NaN (from indicator warm-up periods — first ~50 rows)
    features.dropna(inplace=True)

    return features


# ------------------------------------------------------------------
# 2. Target computation
# ------------------------------------------------------------------

def compute_targets(returns: pd.DataFrame, features_index: pd.Index) -> pd.DataFrame:
    """
    Target = next-day return for each ETF.
    Shift returns by -1 so that on day t, target = return on day t+1.
    Align to features index, then drop last row (no next-day return available).
    """
    targets = returns.shift(-1)
    targets = targets.loc[features_index]
    targets.columns = [f"{t}_target" for t in TICKERS]
    return targets


# ------------------------------------------------------------------
# 3. Sequence builder
# ------------------------------------------------------------------

def build_sequences(features: pd.DataFrame, targets: pd.DataFrame, seq_len: int):
    """
    Build overlapping sequences of length seq_len.
    X shape: (n_samples, seq_len, n_features)
    y shape: (n_samples, n_targets)
    dates:   the date of the prediction (day after the sequence ends)
    """
    feat_arr   = features.values
    target_arr = targets.values
    dates      = features.index

    X, y, pred_dates = [], [], []

    for i in range(seq_len, len(feat_arr)):
        # Check no NaN in this window or target
        window = feat_arr[i - seq_len:i]
        tgt    = target_arr[i]
        if np.isnan(window).any() or np.isnan(tgt).any():
            continue
        X.append(window)
        y.append(tgt)
        pred_dates.append(dates[i])

    return np.array(X), np.array(y), pd.DatetimeIndex(pred_dates)


# ------------------------------------------------------------------
# 4. Time-ordered split
# ------------------------------------------------------------------

def time_split(X, y, dates, train_frac, val_frac):
    n       = len(X)
    t_end   = int(n * train_frac)
    v_end   = int(n * (train_frac + val_frac))

    splits = {
        "train": (X[:t_end],    y[:t_end],    dates[:t_end]),
        "val":   (X[t_end:v_end], y[t_end:v_end], dates[t_end:v_end]),
        "test":  (X[v_end:],    y[v_end:],    dates[v_end:]),
    }

    print(f"\nSplit sizes:")
    for name, (Xs, ys, ds) in splits.items():
        print(f"  {name:5s}: {len(Xs):4d} samples  "
              f"({ds[0].date()} to {ds[-1].date()})")

    return splits


# ------------------------------------------------------------------
# 5. Normalisation — fit on train only
# ------------------------------------------------------------------

def normalise(splits, features_dir):
    """
    Fit StandardScaler on train features only.
    Apply to val and test. Save scaler to disk.
    """
    X_train, y_train, d_train = splits["train"]

    # Reshape to 2D for scaler: (samples * seq_len, features)
    n_samples, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    scaler_path = os.path.join(features_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved -> {scaler_path}")

    normalised = {}
    for split_name, (X, y, d) in splits.items():
        n, s, f    = X.shape
        X_scaled   = scaler.transform(X.reshape(-1, f)).reshape(n, s, f)
        normalised[split_name] = (X_scaled, y, d)

    return normalised


# ------------------------------------------------------------------
# 6. Save
# ------------------------------------------------------------------

def save_splits(normalised, processed_dir, feature_names, target_names):
    out = os.path.join(processed_dir, "sequences")
    os.makedirs(out, exist_ok=True)

    for split_name, (X, y, dates) in normalised.items():
        np.save(os.path.join(out, f"X_{split_name}.npy"), X)
        np.save(os.path.join(out, f"y_{split_name}.npy"), y)
        pd.Series(dates).to_csv(
            os.path.join(out, f"dates_{split_name}.csv"), index=False
        )
        print(f"  Saved X_{split_name}.npy  {X.shape}  "
              f"y_{split_name}.npy  {y.shape}")

    # Save feature and target names for reference
    pd.Series(feature_names).to_csv(
        os.path.join(out, "feature_names.csv"), index=False
    )
    pd.Series(target_names).to_csv(
        os.path.join(out, "target_names.csv"), index=False
    )
    print(f"\nAll sequences saved -> {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading processed data...")
    returns = pd.read_csv(
        os.path.join(PROCESSED_DIR, "returns.csv"),
        index_col=0, parse_dates=True
    )
    prices = pd.read_csv(
        os.path.join(PROCESSED_DIR, "prices.csv"),
        index_col=0, parse_dates=True
    )

    print("Computing features...")
    features = compute_features(prices, returns)
    print(f"  Features shape: {features.shape}")
    print(f"  Feature columns ({len(features.columns)}): {list(features.columns)}")

    print("\nComputing targets...")
    targets = compute_targets(returns, features.index)

    # Drop last row — no next-day return available
    features = features.iloc[:-1]
    targets  = targets.iloc[:-1]
    print(f"  Targets shape: {targets.shape}")

    print("\nBuilding sequences...")
    X, y, dates = build_sequences(features, targets, SEQUENCE_LEN)
    print(f"  X shape: {X.shape}  (samples, seq_len, features)")
    print(f"  y shape: {y.shape}  (samples, n_etfs)")

    print("\nSplitting (time-ordered, no shuffling)...")
    splits = time_split(X, y, dates, TRAIN_FRAC, VAL_FRAC)

    print("\nNormalising (scaler fit on train only)...")
    normalised = normalise(splits, FEATURES_DIR)

    print("\nSaving sequences to data/processed/sequences/...")
    feature_names = list(features.columns)
    target_names  = list(targets.columns)
    save_splits(normalised, PROCESSED_DIR, feature_names, target_names)
