"""
preprocess.py
-------------
Loads raw OHLCV CSVs, computes daily returns, aligns all ETFs on a common
date index, and saves a single combined returns DataFrame to data/processed/.
"""

import os
import pandas as pd
from fetch_data import load_raw

TICKERS       = ["SPY", "QQQ", "GLD", "VEQT.TO"]
RAW_DIR       = os.path.join(os.path.dirname(__file__), "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed")


def build_combined(tickers, raw_dir):
    series = {}
    prices = {}

    for ticker in tickers:
        df = load_raw(ticker, raw_dir)
        clean = ticker.replace(".TO", "")

        print(f"\n--- {ticker} ---")
        print(f"  shape: {df.shape}")
        print(f"  index dtype: {df.index.dtype}")
        print(f"  head:\n{df.head(3)}")

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Strip timezone if present — mixed-tz causes outer join to produce all-NaN
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        series[clean] = df["Close"].pct_change()
        prices[clean] = df["Close"]

    returns_df = pd.DataFrame(series)
    prices_df  = pd.DataFrame(prices)

    print(f"\nBefore cleaning — shape: {returns_df.shape}")
    print(f"NaN counts per column:\n{returns_df.isna().sum()}")

    returns_df.dropna(how="all", inplace=True)
    returns_df.ffill(inplace=True)
    prices_df.ffill(inplace=True)
    returns_df.dropna(inplace=True)
    prices_df.dropna(inplace=True)

    # Drop first row — always NaN from pct_change
    returns_df = returns_df.iloc[1:]
    prices_df  = prices_df.loc[returns_df.index]

    return returns_df, prices_df


def report(returns_df):
    print(f"\nFinal shape:  {returns_df.shape}")
    print(f"Date range:   {returns_df.index[0].date()} to {returns_df.index[-1].date()}")
    print(f"Trading days: {len(returns_df)}")
    print("\nMissing values per ETF:")
    print(returns_df.isna().sum())
    print("\nDescriptive stats (daily returns):")
    print(returns_df.describe().round(5))


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    returns_df, prices_df = build_combined(TICKERS, RAW_DIR)

    if returns_df.empty:
        print("\nERROR: returns_df is empty after processing.")
        print("Paste the debug output above and share it — will fix immediately.")
        raise SystemExit(1)

    report(returns_df)

    returns_path = os.path.join(PROCESSED_DIR, "returns.csv")
    prices_path  = os.path.join(PROCESSED_DIR, "prices.csv")

    returns_df.to_csv(returns_path)
    prices_df.to_csv(prices_path)

    print(f"\nSaved: {returns_path}")
    print(f"Saved: {prices_path}")