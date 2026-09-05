# # === R2-NOSLEEP applied ===
#!/usr/bin/env python3
"""
01_fetch_yahoo.py — Fetch Yahoo Finance price & options data.
(MODIFIED: cache-first from local CSV seed, UTF-8 stdout, retry + backoff)
Output: data_cache/yahoo_*.parquet
"""
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import argparse, os, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))
import yfinance as yf
import pandas as pd
import numpy as np
from config import YF_TICKERS, CACHE_DIR, START_DATE, END_DATE, RANDOM_SEED

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Cache-first helper
def _try_load_cache(ticker_name):
    name = ticker_name.lstrip("^").replace("=F", "").lower()
    cache_path = CACHE_DIR / f"yahoo_{name}.parquet"
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if isinstance(df, pd.DataFrame) and len(df) > 100:
                import os as __os; __os.environ["_R2_LAST_WAS_CACHE_HIT"]="1"
                print(f"  [cache] {ticker_name} <- {cache_path.name} ({len(df)} rows)")
                return df
        except Exception as e:
            print(f"  [cache-warn] {cache_path.name}: {e}")
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)
    print("=" * 60)
    print("01_fetch_yahoo.py — Fetch Yahoo Finance data (cache-first)")
    print("=" * 60)

    # Price data
    print("\n[1/2] Fetching price data...")
    close_dfs = {}
    for i, (name, ticker) in enumerate(YF_TICKERS.items()):
        if i > 0:
            print(f"  [sleep 30s before next ticker to avoid rate limit]")
            import os as __os2
            if __os2.environ.pop("_R2_LAST_WAS_CACHE_HIT", None) == "1":
                print("  [R2-nosleep] cache hit, skipping 30s rate-limit sleep")
            else:
                time.sleep(30)

        print(f"  Fetching {name} ({ticker})...")

        df = _try_load_cache(ticker)
        if df is not None:
            path = CACHE_DIR / f"yahoo_{name.lower()}.parquet"
            df.to_parquet(path)
            print(f"  OK {name} -> {path}")
            if 'Close' in df.columns:
                close_col = df['Close']
                if isinstance(close_col, pd.DataFrame):
                    close_col = close_col.iloc[:, 0]
                close_dfs[name] = close_col
            continue

        for attempt in range(3):
            try:
                df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=False)
                if df is not None and len(df) > 0:
                    break
                print(f"    [warn] empty dataframe on attempt {attempt+1}")
            except Exception as e:
                wait = 60 * (2 ** attempt)
                print(f"    [retry] attempt {attempt+1} failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    print(f"    [retry] sleeping {wait}s before retry")
                    time.sleep(wait)
                else:
                    print(f"    [fatal] giving up on {ticker} after 3 attempts")
                    raise

        print(f"    -> {len(df)} rows, {df.columns.tolist()}")
        path = CACHE_DIR / f"yahoo_{name.lower()}.parquet"
        df.to_parquet(path)
        print(f"  OK {name} -> {path}")

        if 'Close' in df.columns:
            close_col = df['Close']
            if isinstance(close_col, pd.DataFrame):
                close_col = close_col.iloc[:, 0]
            close_dfs[name] = close_col

    closes = pd.concat(close_dfs, axis=1)
    closes.columns = [k for k in close_dfs.keys()]
    closes.index = pd.to_datetime(closes.index)
    closes.to_parquet(CACHE_DIR / "yahoo_closes.parquet")
    print(f"\n  OK Combined closes -> {CACHE_DIR / 'yahoo_closes.parquet'} ({closes.shape})")

    # Options data
    print("\n[2/2] Fetching options data...")
    for name in ["SPY", "KRE", "JPM"]:
        print(f"  Fetching options for {name}...")
        try:
            t = yf.Ticker(name)
            exps = t.options
            if not exps:
                print(f"  WARNING: {name}: no options data")
                continue
            chain = t.option_chain(exps[0])
            df = chain.calls[["strike", "lastPrice", "impliedVolatility", "volume", "openInterest"]].copy()
            df["expiration"] = exps[0]
            df.to_parquet(CACHE_DIR / f"options_{name.lower()}.parquet")
            print(f"  OK {name}: {len(df)} contracts, exp={exps[0]}")
        except Exception as e:
            print(f"  WARNING: {name}: {e}")

    # ── Compute Amihud illiquidity panel and save to data/derived/ ──────
    print("\n[3/3] Computing Amihud illiquidity panel...")
    derived_dir = CACHE_DIR.parent.parent / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    equity_tickers = ["KRE", "JPM", "BAC", "C", "GS", "MS", "WFC", "KBE", "SPY", "XLF"]
    amihud_parts = {}
    for ticker in equity_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=START_DATE, end=END_DATE, auto_adjust=True)
            if hist.empty:
                print(f"  WARNING: {ticker}: no history")
                continue
            hist.index = pd.to_datetime(hist.index).tz_localize(None)
            ret = hist["Close"].pct_change().abs()
            vol = hist["Volume"] * hist["Close"]  # dollar volume
            amihud = (ret / vol.replace(0, float("nan"))) * 1e6  # scale
            amihud_parts[ticker] = amihud.rename(ticker)
            # Save individual ticker file
            ind_df = amihud.to_frame("amihud_illiq")
            ind_df.index.name = "Date"
            ind_df.to_parquet(derived_dir / f"amihud_{ticker}_daily.parquet")
        except Exception as e:
            print(f"  WARNING: amihud {ticker}: {e}")

    if amihud_parts:
        panel = pd.concat(amihud_parts, axis=1)
        panel.index.name = "Date"
        panel_path = derived_dir / "amihud_panel_daily.parquet"
        # Also save long-format (required by 10_table2_qst_stats.py)
        panel_long = panel.stack().reset_index()
        panel_long.columns = ["date", "ticker", "amihud_illiq"]
        panel_long["date"] = pd.to_datetime(panel_long["date"])
        panel_long = panel_long.dropna(subset=["amihud_illiq"])
        # 20-day rolling mean
        panel_long = panel_long.sort_values(["ticker", "date"])
        panel_long["amihud_20d_ma"] = (
            panel_long.groupby("ticker")["amihud_illiq"]
            .transform(lambda x: x.rolling(20, min_periods=5).mean())
        )
        panel_long.to_parquet(panel_path, index=False)
        print(f"  OK Amihud panel (long) -> {panel_path} ({panel_long.shape})")
    else:
        print("  WARNING: No amihud data computed")

    print(f"\nOK 01_fetch_yahoo.py done")
    return 0

if __name__ == "__main__":
    sys.exit(main())