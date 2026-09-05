#!/usr/bin/env python3
"""
Shared Bloch trajectory pipeline for Fig 1, Fig 2, Fig 3, Fig 4.

Loads L1 macro inputs, standardizes, maps to Pauli observables, applies
smooth Bloch normalization, returns per-day (r_x, r_y, r_z, |r|, S(ρ), C(ρ)).

Deterministic. Input paths and column names hardcoded to L1 sources
documented in _recon_20260703/data_provenance_v3.md.

Axis v2 r_z convention:
  r_z = 0.5 * (EFFR_z - GSPC_ret_z)
  so that r_z > 0 corresponds to monetary tightening + equity sell-off (crisis).
"""
import numpy as np
import pandas as pd
from pathlib import Path


def load_l1_inputs(root: Path, window_start: str, window_end: str) -> pd.DataFrame:
    """
    Load VIX, STLFSI4, HY-OAS, EFFR, KRE returns, GSPC returns, TED for
    the given window. Returns a pandas DataFrame indexed by daily
    business dates with these seven columns:
      vix, stlfsi4, hy_oas, effr, kre_ret, gspc_ret, ted
    """
    DATA_BAK = root / "raw"
    DERIVED = root / "derived"

    # ── VIX ──
    vix = pd.read_csv(DATA_BAK / "VIXCLS.csv", parse_dates=["Date"])
    vix = vix.rename(columns={"Date": "date", "VIXCLS": "vix"}).set_index("date")
    vix.index = pd.to_datetime(vix.index)

    # ── STLFSI4 (weekly, forward-fill to daily) ──
    stl = pd.read_csv(DATA_BAK / "STLFSI4.csv", parse_dates=["Date"])
    stl = stl.rename(columns={"Date": "date", "STLFSI4": "stlfsi4"}).set_index("date")
    stl.index = pd.to_datetime(stl.index)

    # ── HY-OAS (credit stress) ──
    hy = pd.read_csv(DATA_BAK / "BAMLH0A0HYM2.csv", parse_dates=["Date"])
    hy = hy.rename(columns={"Date": "date", "BAMLH0A0HYM2": "hy_oas"}).set_index("date")
    hy.index = pd.to_datetime(hy.index)

    # ── EFFR (DFF) ──
    dff = pd.read_csv(DATA_BAK / "DFF.csv", parse_dates=["Date"])
    dff = dff.rename(columns={"Date": "date", "DFF": "effr"}).set_index("date")
    dff.index = pd.to_datetime(dff.index)

    # ── KRE (log returns) ──
    kre = pd.read_csv(DATA_BAK / "KRE.csv", parse_dates=["Date"])
    kre = kre.set_index("Date")
    kre.index = pd.to_datetime(kre.index)
    kre_ret = np.log(kre["Close"] / kre["Close"].shift(1)).rename("kre_ret")

    # ── GSPC (log returns) ──
    gspc = pd.read_csv(DATA_BAK / "GSPC.csv", parse_dates=["Date"])
    gspc = gspc.set_index("Date")
    gspc.index = pd.to_datetime(gspc.index)
    gspc_ret = np.log(gspc["Close"] / gspc["Close"].shift(1)).rename("gspc_ret")

    # ── TED spread (spliced parquet) ──
    ted = pd.read_parquet(DERIVED / "TED_spliced_2004_2024.parquet")
    ted = ted.set_index("date")
    ted.index = pd.to_datetime(ted.index)
    ted = ted["ted_bps"].rename("ted")

    # ── Merge all on date index ──
    def _ensure_df(x):
        return x.to_frame() if isinstance(x, pd.Series) else x
    dfs = [_ensure_df(v) for v in [vix, stl, hy, dff, kre_ret, gspc_ret, ted]]
    merged = dfs[0].join(dfs[1:], how="outer")

    # Filter to window
    merged = merged.loc[window_start:window_end].copy()

    # Forward-fill STLFSI4 (weekly → daily)
    merged["stlfsi4"] = merged["stlfsi4"].ffill()

    # HY-OAS: before first available date, set to 0 (no credit stress signal)
    merged["hy_oas"] = merged["hy_oas"].ffill().fillna(0.0)

    # Drop rows where ALL macro series are NaN
    merged = merged.dropna(subset=["vix", "effr", "ted"], how="all")

    # Drop non-trading days: rows where BOTH kre_ret and gspc_ret are NaN
    # (weekends/holidays — financial data not available)
    merged = merged.dropna(subset=["kre_ret", "gspc_ret"], how="all")

    return merged


def standardize_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each column over the window. For HY-OAS, only z-score the
    non-zero (available) portion; leave pre-availability dates at 0.0.
    Returns a DataFrame with same columns as input, z-scored.
    """
    cols_to_std = ["vix", "stlfsi4", "effr", "kre_ret", "gspc_ret", "ted"]
    df_std = df[cols_to_std].copy()
    for col in cols_to_std:
        mu = df_std[col].mean()
        sigma = df_std[col].std()
        if sigma > 1e-15:
            df_std[col] = (df_std[col] - mu) / sigma
        else:
            df_std[col] = 0.0

    # HY-OAS: z-score only on non-zero (available) portion
    hy_oas = df["hy_oas"].copy()
    non_zero_mask = hy_oas > 0
    hy_oas_z = pd.Series(0.0, index=hy_oas.index)
    if non_zero_mask.any():
        mu_nz = hy_oas[non_zero_mask].mean()
        std_nz = hy_oas[non_zero_mask].std()
        if std_nz > 1e-15:
            hy_oas_z[non_zero_mask] = (hy_oas[non_zero_mask] - mu_nz) / std_nz
    df_std["hy_oas"] = hy_oas_z

    return df_std


def map_to_pauli(df_std: pd.DataFrame) -> pd.DataFrame:
    """
    Compute raw (r_x, r_y, r_z) from standardized inputs.

    Axis v2 convention:
      r_x_raw = mean(VIX_z, STLFSI4_z, TED_z)       — market stress
      r_y_raw = HY_OAS_z                             — credit stress (0 pre-availability)
      r_z_raw = 0.5 * (EFFR_z - GSPC_ret_z)          — rates/equity crisis axis
        (positive = tightening + sell-off)

    Returns DataFrame with columns r_x_raw, r_y_raw, r_z_raw.
    """
    result = pd.DataFrame(index=df_std.index)
    result["r_x_raw"] = df_std[["vix", "stlfsi4", "ted"]].mean(axis=1)
    result["r_y_raw"] = df_std["hy_oas"]
    result["r_z_raw"] = 0.5 * (df_std["effr"] - df_std["gspc_ret"])
    return result


def smooth_bloch(r_raw: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Map unbounded R^3 vector to open Bloch ball ||r|| < 1 via tanh(||r||/scale) * unit.
    Preserves direction.
    """
    norm_raw = np.linalg.norm(r_raw)
    if norm_raw < 1e-12:
        return r_raw.copy()
    u = r_raw / norm_raw
    return u * np.tanh(norm_raw / scale)


def compute_bloch_series(root: Path, window_start: str, window_end: str,
                         scale: float = 2.0) -> pd.DataFrame:
    """
    Full pipeline: L1 inputs → standardized → Pauli mapping → smooth normalization.

    Returns DataFrame indexed by date with columns:
      r_x, r_y, r_z, r_norm, entropy, coherence,
      r_x_raw, r_y_raw, r_z_raw
    """
    df = load_l1_inputs(root, window_start, window_end)
    df_std = standardize_inputs(df)
    df_raw = map_to_pauli(df_std)

    out = df_raw.copy()

    # Apply smooth normalization per row
    raw_cols = ["r_x_raw", "r_y_raw", "r_z_raw"]
    smooth_vals = df_raw[raw_cols].apply(
        lambda row: smooth_bloch(row.values, scale=scale),
        axis=1, result_type="expand"
    )
    smooth_vals.columns = ["r_x", "r_y", "r_z"]
    out = pd.concat([out, smooth_vals], axis=1)

    # Norm
    out["r_norm"] = np.sqrt(out["r_x"]**2 + out["r_y"]**2 + out["r_z"]**2)

    # Von Neumann entropy: for a qubit with Bloch norm r, eigenvalues are (1±r)/2
    lam1 = (1 + out["r_norm"]) / 2
    lam2 = (1 - out["r_norm"]) / 2

    def safe_h(l):
        return np.where(l > 1e-15, -l * np.log(l), 0.0)
    out["entropy"] = safe_h(lam1) + safe_h(lam2)

    # Coherence: l1-norm of off-diagonal = sqrt(r_x² + r_y²) for single qubit
    out["coherence"] = np.sqrt(out["r_x"]**2 + out["r_y"]**2)

    return out
