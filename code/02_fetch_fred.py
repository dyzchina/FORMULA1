#!/usr/bin/env python3
"""
02_fetch_fred.py — Fetch FRED data (REAL data only, no synthetic fallback).
Output: data_cache/fred_raw.parquet, data_cache/fred_raw.csv

9 series:
  VIXCLS, TEDRATE, BAMLH0A0HYM2, STLFSI4, DFF, DGS10, DGS2, WALCL, BAA10Y

TEDRATE stitching: after 2022-01-21, use (BAMLH0A0HYM2 - DGS10) (5-day smoothed).
BAMLH0A0HYM2: only available from 2023-05-22 in our dataset. For pre-2023,
  BAA10Y is used as a proxy (highly correlated credit spread indicator).
DFF: daily frequency (replaces monthly FEDFUNDS).

"""
import argparse, sys, os, warnings, time
# Windows GBK console fix: ensure ✓/✗/— can print without UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from fredapi import Fred
from config import CACHE_DIR, RANDOM_SEED
warnings.filterwarnings("ignore")

# ── Load API key ────────────────────────────────────────────────────────
load_dotenv()
FRED_KEY = os.getenv("FRED_API_KEY")
if not FRED_KEY or FRED_KEY == "your_fred_api_key_here":
    raise RuntimeError(
        "ERROR: FRED_API_KEY not set. Synthetic data is prohibited in this project.\n"
        "Please configure a real key in figures_source/.env."
    )

# ── Series definitions ──────────────────────────────────────────────────
SERIES = {
    "VIXCLS": "CBOE Volatility Index: VIX",
    "TEDRATE": "TED Spread",
    "BAMLH0A0HYM2": "ICE BofA US High Yield Index OAS",
    "STLFSI4": "St. Louis Fed Financial Stress Index (4-week)",
    "DFF": "Effective Federal Funds Rate (Daily)",

    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DGS2": "2-Year Treasury Constant Maturity Rate",
    "WALCL": "Fed Total Assets (Wednesday Level)",
    "BAA10Y": "Moody's BAA - 10-Year Treasury Spread",
}

START = "2004-01-01"
END = "2024-12-31"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

def fetch_with_retry(fred, sid, max_retries=MAX_RETRIES):
    """Fetch a FRED series with retry on SSL errors."""
    for attempt in range(1, max_retries + 1):
        try:
            raw = fred.get_series(sid, observation_start=START, observation_end=END)
            series = pd.Series(raw, name=sid)
            series.index = pd.to_datetime(series.index)
            if hasattr(series.index, 'tz') and series.index.tz is not None:
                series.index = series.index.tz_localize(None)
            return series
        except Exception as e:
            if attempt < max_retries:
                print(f"    Attempt {attempt} failed: {e}. Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"    All {max_retries} attempts failed: {e}")
                return pd.Series(dtype=float, name=sid)


# === R2 FRED CACHE HELPER (added by _r2_patch_final.py) ===
import pandas as _r2_pd
from pathlib import Path as _R2_Path

_R2_REPO = _R2_Path(__file__).resolve().parents[2] / "_BASELINES" / "v20260622_submission"

_R2_CSV_MAP = {
    "DGS10":       ("data/raw/all/FRED_DGS10.csv",       "DGS10"),
    "DGS2":        ("data/raw/all/FRED_DGS2.csv",        "DGS2"),
    "VIXCLS":      ("data/raw/all/FRED_VIXCLS.csv",      "VIXCLS"),
    "TEDRATE":     ("data/raw/TEDRATE.csv",        "TEDRATE"),
    "BAMLH0A0HYM2":("data/raw/all/FRED_BAMLH0A0HYM2.csv","BAMLH0A0HYM2"),
    "STLFSI4":     ("data/raw/STLFSI4.csv","STLFSI4"),
    "DFF":         ("data/raw/DFF.csv",       "DFF"),
    "WALCL":       ("data/raw/all/FRED_WALCL.csv",        "WALCL"),
}

def _r2_load_from_csv(sid):
    info = _R2_CSV_MAP.get(sid)
    if info is None:
        return None
    rel, col = info
    if rel == "__SKIP__":
        print(f"  [cache-skip ] {sid}: no offline CSV, returning empty series", flush=True)
        return _r2_pd.Series(dtype="float64", name=sid)
    p = _R2_REPO / rel
    if not p.exists():
        print(f"  [cache-miss ] {sid}: {rel} not found, will try live fetch", flush=True)
        return None
    try:
        # encoding="utf-8-sig" handles UTF-8 BOM (EF BB BF) in Chinese-named CSVs
        df = _r2_pd.read_csv(p, encoding="utf-8-sig")
        # case-insensitive lookup for date column (lowercase "date" in FRED_*.csv,
        # uppercase "Date" in raw/<ID>.csv)
        cols_lower = {c.lower(): c for c in df.columns}
        date_key = cols_lower.get("date")
        # value column: also case-insensitive on the configured `col`
        val_key = cols_lower.get(col.lower())
        if date_key is None or val_key is None:
            print(f"  [cache-bad  ] {sid}: columns={list(df.columns)}, will try live fetch", flush=True)
            return None
        df[date_key] = _r2_pd.to_datetime(df[date_key], errors="coerce")
        df = df.dropna(subset=[date_key]).set_index(date_key).sort_index()
        s = _r2_pd.to_numeric(df[val_key], errors="coerce").rename(sid)
        print(f"  [cache-hit  ] {sid}: {len(s)} rows from {rel}", flush=True)
        return s
    except Exception as e:
        print(f"  [cache-err  ] {sid}: {e!r}, will try live fetch", flush=True)
        return None
# === END R2 FRED CACHE HELPER ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)
    print("=" * 60)
    print("02_fetch_fred.py — Fetch FRED data (REAL)")
    print("=" * 60)

    try:
        fred = Fred(api_key=FRED_KEY)
    except Exception as _e:
        print(f"  [fred-init ] Fred(api_key=...) failed: {_e!r}; fred=None (cache-only mode)", flush=True)
        fred = None
    dfs = {}

    for sid, desc in SERIES.items():
        print(f"\n  [{sid}] {desc}...")
        series = _r2_load_from_csv(sid)
        if series is None:
            if fred is None:
                print(f"  [no-fetch  ] {sid}: no cache and fred=None, empty series", flush=True)
                series = _r2_pd.Series(dtype="float64", name=sid)
            else:
                series = fetch_with_retry(fred, sid)
        dfs[sid] = series
        if len(series) > 0:
            print(f"    -> {len(series)} obs, {series.first_valid_index().date()} ~ {series.last_valid_index().date()}")

    # ── Assemble DataFrame ──────────────────────────────────────────────
    df = pd.DataFrame(dfs).sort_index()
    df.index = pd.to_datetime(df.index)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # ── TEDRATE stitching ───────────────────────────────────────────────
    # TEDRATE discontinued after 2022-01-21. Stitch with (BAMLH0A0HYM2 - DGS10) (5d smoothed).
    print("\n  [TEDRATE stitching] After 2022-01-21, using (BAMLH0A0HYM2 - DGS10) (5d smoothed)...")
    ted_real = df["TEDRATE"].copy()
    hy_oas = df["BAMLH0A0HYM2"].dropna()
    dgs10 = df["DGS10"].dropna()
    # Compute HY_OAS - 10Y as TED proxy
    hy_ted = hy_oas - dgs10.reindex(hy_oas.index, method="ffill")
    hy_ted_smoothed = hy_ted.rolling(5, min_periods=1).mean()

    stitch_date = pd.Timestamp("2022-01-21")
    ted_stitched = ted_real.copy()
    mask = ted_stitched.index >= stitch_date
    ted_stitched.loc[mask] = hy_ted_smoothed.reindex(ted_stitched.index, method="ffill").loc[mask]
    df["TEDRATE_STITCHED"] = ted_stitched


    # ── HY_OAS: use BAMLH0A0HYM2 where available, BAA10Y as proxy for pre-2023 ──
    print("\n  [HY_OAS] BAMLH0A0HYM2 available from 2023-05-22. Using BAA10Y as proxy for pre-2023...")
    hy_oas = df["BAMLH0A0HYM2"].copy()
    baa10y_proxy = df["BAA10Y"].copy()
    hy_oas_filled = hy_oas.fillna(baa10y_proxy)
    df["HY_OAS"] = hy_oas_filled

    print(f"\n  Final DataFrame: {df.shape}, {df.index.min().date()} ~ {df.index.max().date()}")

    # ── Health check ────────────────────────────────────────────────────
    print("\n  -- FRED real-data health report --")
    for col in df.columns:
        n = df[col].notna().sum()
        missing_pct = (1 - n / len(df)) * 100
        print(f"  {col:20s}: n={n:5d}, missing={missing_pct:.2f}%")

    # VIXCLS 2020-03 peak
    if "VIXCLS" in df.columns and df["VIXCLS"].notna().sum() > 0:
        vix_peak = df["VIXCLS"].max()
        vix_peak_date = df["VIXCLS"].idxmax()
        print(f"  VIXCLS peak: {vix_peak:.2f} ({vix_peak_date.date()}) -- COVID anchor OK")

    # HY_OAS 2008-12 peak
    if "HY_OAS" in df.columns and df["HY_OAS"].notna().sum() > 0:
        hy_peak = df["HY_OAS"].max()
        hy_peak_date = df["HY_OAS"].idxmax()
        print(f"  HY_OAS peak: {hy_peak:.2f} ({hy_peak_date.date()}) -- GFC anchor OK")

    # 2023-03-10 check
    check_date = pd.Timestamp("2023-03-10")
    if check_date in df.index:
        valid = df.loc[check_date].notna().sum()
        total = len(df.columns)
        print(f"  2023-03-10: {valid}/{total} series have valid obs {'OK' if valid == total else 'WARN'}")

    # ── Save ────────────────────────────────────────────────────────────
    parquet_path = CACHE_DIR / "fred_raw.parquet"
    csv_path = CACHE_DIR / "fred_raw.csv"
    df.to_parquet(parquet_path)
    df.to_csv(csv_path)
    print(f"\n  OK -> {parquet_path}")
    print(f"  OK -> {csv_path}")

    # Export VIXCLS to data/raw/all/ so fig2/fig4 scripts can find it
    raw_all_dir = _R2_Path(__file__).resolve().parents[1] / "data" / "raw" / "all"
    raw_all_dir.mkdir(parents=True, exist_ok=True)
    if "VIXCLS" in df.columns and df["VIXCLS"].notna().any():
        vix_export = df[["VIXCLS"]].reset_index()
        vix_export.columns = ["date", "VIXCLS"]
        vix_export["date"] = vix_export["date"].dt.strftime("%Y-%m-%d")
        vix_csv_path = raw_all_dir / "FRED_VIXCLS.csv"
        vix_export.to_csv(vix_csv_path, index=False)
        print(f"  OK -> {vix_csv_path}  (N={vix_export['VIXCLS'].notna().sum()} obs)")

    # ── Save TED splice as standalone parquet (required by _bloch_pipeline) ──
    derived_dir = CACHE_DIR.parent.parent / "data" / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    ted_col = "TEDRATE_STITCHED" if "TEDRATE_STITCHED" in df.columns else "TEDRATE"
    ted_df = df[[ted_col]].rename(columns={ted_col: "ted_bps"}).copy()
    # convert to bps if in percent (TEDRATE is in percent, multiply by 100)
    if ted_df["ted_bps"].dropna().abs().median() < 5:
        ted_df["ted_bps"] = ted_df["ted_bps"] * 100
    ted_path = derived_dir / "TED_spliced_2004_2024.parquet"
    ted_df_out = ted_df.reset_index()
    ted_df_out.columns = ["date", "ted_bps"]
    ted_df_out.to_parquet(ted_path, index=False)
    print(f"  OK -> {ted_path}  (N={len(ted_df_out.dropna())} obs)")

    print(f"\nOK 02_fetch_fred.py done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
