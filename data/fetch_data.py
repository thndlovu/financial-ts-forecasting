"""
fetch_data.py
-------------
Downloads daily OHLCV data using yfinance with individual ticker downloads
and a longer delay to avoid rate limiting.

Usage:
    pip install yfinance --upgrade
    python data/fetch_data.py
"""

import os
import time
import yfinance as yf
import pandas as pd

TICKERS = ["SPY", "QQQ", "GLD", "VEQT.TO"]
START   = "2015-01-01"
END     = "2024-12-31"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def fetch_one(ticker, start, end):
    t = yf.Ticker(ticker)
    df = t.history(start=start, end=end, auto_adjust=True)

    if df.empty:
        return df

    # Strip timezone
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only OHLCV
    cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[cols]
    df.dropna(how="all", inplace=True)
    df.sort_index(inplace=True)
    df.index.name = "Date"

    return df


def fetch_and_save(tickers, start, end, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    for i, ticker in enumerate(tickers):
        safe_name = ticker.replace(".", "_")
        out_path  = os.path.join(out_dir, f"{safe_name}.csv")

        print(f"Downloading {ticker} ...", end=" ", flush=True)
        try:
            df = fetch_one(ticker, start, end)
            if df.empty:
                print("FAILED: empty response — still rate limited, wait longer.")
            else:
                df.to_csv(out_path)
                print(f"{len(df)} rows -> {out_path}")
        except Exception as e:
            print(f"FAILED: {e}")

        if i < len(tickers) - 1:
            print("  pausing 5s...")
            time.sleep(5)


def load_raw(ticker, raw_dir=RAW_DIR):
    """Helper used by downstream scripts to load a single ETF's raw CSV."""
    safe_name = ticker.replace(".", "_")
    path = os.path.join(raw_dir, f"{safe_name}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df


if __name__ == "__main__":
    fetch_and_save(TICKERS, START, END, RAW_DIR)
    print("\nAll done. Check data/raw/ for CSVs.")
