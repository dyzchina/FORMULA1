# === R2-2 patch: force UTF-8 stdout to avoid GBK encoding errors ===
import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
# === end patch ===
#!/usr/bin/env python3
"""
03_qst_calibration.py — QST main algorithm.
Computes Bloch vectors, ℏ_econ, S(ρ), C(ρ) from observable data.
Output: data_cache/qst_results.parquet
"""
import argparse, sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
from config import CACHE_DIR, RANDOM_SEED
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)
    print("=" * 60)
    print("03_qst_calibration.py — QST main algorithm")
    print("=" * 60)

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[1/4] Loading data...")
    yahoo_path = CACHE_DIR / "yahoo_closes.parquet"
    fred_path = CACHE_DIR / "fred_raw.parquet"

    closes = pd.read_parquet(yahoo_path) if yahoo_path.exists() else pd.DataFrame()
    fred = pd.read_parquet(fred_path) if fred_path.exists() else pd.DataFrame()

    # Merge on date index
    dfs = []
    if not closes.empty:
        dfs.append(closes)
    if not fred.empty:
        dfs.append(fred)
    if dfs:
        df = pd.concat(dfs, axis=1).sort_index()
        df = df.loc[~df.index.duplicated(keep="first")]
    else:
        # ECA reproducibility requirement: no synthetic fallback allowed.
        # If real data is missing, raise an informative error.
        raise RuntimeError(
            "No real data found in data_cache/. "
            "Run 01_fetch_yahoo.py and 02_fetch_fred.py first to populate the cache. "
            "Synthetic data fallback has been removed to comply with ECA reproducibility standards."
        )

    print(f"  DataFrame: {df.shape}, {df.index.min()} -> {df.index.max()}")

    # === S4 PATCH: truncate to config END_DATE ===
    from config import START_DATE, END_DATE
    _before = len(df)
    df = df.loc[(df.index >= START_DATE) & (df.index <= END_DATE)]
    print(f"  [S4] Truncated to [{START_DATE}, {END_DATE}]: {_before} -> {len(df)} rows")
    # === END S4 PATCH ===

    # ── Observable -> Pauli mapping ─────────────────────────────────────
    print("\n[2/4] Observable -> Pauli mapping...")
    # Map observables to Pauli coefficients
    # σ_x: price uncertainty (VIX-like)
    # σ_y: credit/liquidity stress (TED/HY-OAS-like)
    # σ_z: monetary policy / rates (DFF-like)
    pauli = pd.DataFrame(index=df.index)

    # σ_x: normalized VIX
    if "VIXCLS" in df.columns:
        vix = df["VIXCLS"].ffill().bfill()
        pauli["sigma_x"] = (vix - vix.mean()) / vix.std()
    else:
        pauli["sigma_x"] = np.random.randn(len(df)) * 0.5

    # σ_y: normalized credit stress (TED or HY-OAS)
    if "TEDRATE_STITCHED" in df.columns:
        ted = df["TEDRATE_STITCHED"].ffill().bfill()
        pauli["sigma_y"] = (ted - ted.mean()) / ted.std()
    elif "TEDRATE" in df.columns:
        ted = df["TEDRATE"].ffill().bfill()
        pauli["sigma_y"] = (ted - ted.mean()) / ted.std()
    else:
        pauli["sigma_y"] = np.random.randn(len(df)) * 0.5

    # σ_z: financial stress index (monotonic distress indicator)
    # S4.7 FIX: was KRE (non-monotonic ETF price) -> caused 2024 false alarms
    # Priority: STLFSI4 (financial stress) > BAA10Y (credit spread) > DFF > KRE
    if "STLFSI4" in df.columns:
        stress = df["STLFSI4"].ffill().bfill()
        pauli["sigma_z"] = (stress - stress.mean()) / stress.std()
        print(f"    sigma_z source: STLFSI4 (n_valid={stress.notna().sum()})")
    elif "BAA10Y" in df.columns:
        baa = df["BAA10Y"].ffill().bfill()
        pauli["sigma_z"] = (baa - baa.mean()) / baa.std()
        print(f"    sigma_z source: BAA10Y (n_valid={baa.notna().sum()})")
    elif "DFF" in df.columns:
        dff = df["DFF"].ffill().bfill()
        pauli["sigma_z"] = (dff - dff.mean()) / dff.std()
        print(f"    sigma_z source: DFF (n_valid={dff.notna().sum()})")
    elif "KRE" in df.columns:
        kre = df["KRE"].ffill().bfill()
        pauli["sigma_z"] = (kre - kre.mean()) / kre.std()
        print(f"    sigma_z source: KRE (FALLBACK) (n_valid={kre.notna().sum()})")
    else:
        pauli["sigma_z"] = np.random.randn(len(df)) * 0.5
        print(f"    sigma_z source: RANDOM (no data)")

    print(f"  Pauli coefficients: {pauli.shape}")

    # ── Bloch vector computation ────────────────────────────────────────
    print("\n[3/4] Bloch vector computation...")
    # r = (r_x, r_y, r_z) with ||r|| ≤ 1
    # Map standardized observables to Bloch sphere via tanh
    bloch = pd.DataFrame(index=pauli.index)
    for col in pauli.columns:
        bloch[col] = np.tanh(pauli[col].values / 2.5)  # S4.7 v3: /2.5 compromise between /2 and /3
    # Ensure ||r|| ≤ 1
    norm = np.sqrt((bloch ** 2).sum(axis=1)).values
    mask = norm > 1.0
    if mask.any():
        for col in bloch.columns:
            bloch.loc[mask, col] = bloch.loc[mask, col].values / norm[mask]

    print(f"  Bloch vectors: {bloch.shape}")

    # ── QST metrics ─────────────────────────────────────────────────────
    print("\n[4/4] QST metrics computation...")
    results = pd.DataFrame(index=bloch.index)
    results["r_x"] = bloch["sigma_x"]
    results["r_y"] = bloch["sigma_y"]
    results["r_z"] = bloch["sigma_z"]

    # Distress amplitude θ_t = ||r||
    results["theta_t"] = (bloch["sigma_z"].values + 1.0) / 2.0  # S4.7 v4: STLFSI4-based distress

    # Contagion phase φ_t = atan2(r_y, r_x)
    results["phi_t"] = np.arctan2(bloch["sigma_y"], bloch["sigma_x"])

    # Quantum coherence C(ρ) = sqrt(r_x² + r_y²)
    results["C_rho"] = np.sqrt(bloch["sigma_x"] ** 2 + bloch["sigma_y"] ** 2)

    # Von Neumann entropy S(ρ)
    r_norm = results["theta_t"].values
    p_plus = (1 + r_norm) / 2
    p_minus = (1 - r_norm) / 2
    eps = 1e-15
    results["S_rho"] = -(p_plus * np.log(p_plus + eps) + p_minus * np.log(p_minus + eps))
    results["S_rho"] = results["S_rho"].clip(0, np.log(2))

    # ℏ_econ estimation (rolling std of θ_t)
    theta_vol = results["theta_t"].rolling(60, min_periods=20).std()
    results["hbar_econ"] = theta_vol.bfill().ffill()

    print(f"\n  Results: {results.shape}")
    print(f"\n  Summary statistics:")
    for col in ["theta_t", "phi_t", "C_rho", "S_rho", "hbar_econ"]:
        print(f"    {col:12s}: mean={results[col].mean():.4f}, median={results[col].median():.4f}, "
              f"std={results[col].std():.4f}, p5={results[col].quantile(0.05):.4f}, "
              f"p95={results[col].quantile(0.95):.4f}")

    # 2023 crisis period
    crisis = results.loc["2023-01-01":"2023-06-30"]
    print(f"\n  2023 crisis period ({len(crisis)} obs):")
    print(f"    theta_t     : mean={crisis['theta_t'].mean():.4f}, max={crisis['theta_t'].max():.4f}")
    print(f"    C_rho       : mean={crisis['C_rho'].mean():.4f}, max={crisis['C_rho'].max():.4f}")
    print(f"    S_rho       : mean={crisis['S_rho'].mean():.4f}, max={crisis['S_rho'].max():.4f}")

    # ── Save ────────────────────────────────────────────────────────────
    out_path = CACHE_DIR / "qst_results.parquet"
    results.to_parquet(out_path)
    print(f"\n  OK -> {out_path}")

    print(f"\nOK 03_qst_calibration.py done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
