#!/usr/bin/env python3
"""
Table 4 — IV Surface Calibration (JPM / KRE / SPY)
L-level: L2_PROXY (parameters extrapolated from Fig A.1 KRE calibration to JPM/SPY)

Methodology:
  For each underlying:
    1. Compute sigma_0 = 60-day realized vol ending 2023-03-09 (annualized)
    2. Compute Amihud_20d avg over the same 60-day window
    3. Set hbar_econ = 0.087 * (Amihud_underlying / Amihud_KRE)
    4. Set gamma = 0.51 (fixed for equities, per Fig A.1 sign-flip decision)
    5. Set beta = 0.001 (fixed per Fig A.1 recalibration)
    6. Compute reference-smile RMSE at tau in {7, 14, 30, 60} days on 25 strike points K/S in [0.7, 1.3]

Guards (CLAUDE1-approved recalibration):
  G1: All 3 sigma_0 in [0.10, 0.60]
  G2: All 3 hbar_econ in [0.001, 0.20]
  G3: All 3 RMSE finite and < 15 vol points
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import sys
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LOG_DIR  = ROOT / "logs"
TAB_DIR  = ROOT / "tables"
LOG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# =========================================================================
# CANONICAL PARAMETERS (from Fig A.1)
# =========================================================================
SMILE_SKEW_A          = -0.50
SMILE_CURV_B          = +1.00
GAMMA_Q               = +0.51
BETA_Q                = +0.001
TAU_DAYS_GRID         = [7, 14, 30, 60]
K_REL_RANGE           = np.linspace(0.7, 1.3, 25)

# Reference KRE hbar_econ from Fig A.1
HBAR_ECON_KRE_REF = 0.087


def sigma_market_reference(m, tau_years, sigma_atm):
    """Canonical equity-crisis reference smile."""
    return sigma_atm + SMILE_SKEW_A * m + SMILE_CURV_B * m**2


def sigma_quantum(m, tau_years, sigma_0, hbar_econ, gamma, beta):
    """Quantum model implied volatility per paper eq. (3)."""
    x = -m
    sig_sq = sigma_0**2 + hbar_econ * gamma * (x / tau_years) + hbar_econ**2 * beta / (tau_years**2)
    sig_sq = np.maximum(sig_sq, 1e-8)
    return np.sqrt(sig_sq)


def sigma_atm_termstructure(tau_years, sigma_realized):
    """Mild term structure: short-dated slightly below, long-dated slightly above realized."""
    return sigma_realized * (1.0 + 0.05 * (1.0 - np.exp(-tau_years / 0.1)))


def compute_realized_vol(df, date_col="Date", close_col="Close", window=60, annualize=True):
    """Compute annualized realized volatility over trailing window ending at last date."""
    df = df.sort_values(date_col).copy()
    df["log_ret"] = np.log(df[close_col] / df[close_col].shift(1))
    recent = df["log_ret"].iloc[-window:].dropna()
    rv = recent.std()
    if annualize:
        rv *= np.sqrt(252)
    return rv


def compute_amihud_20d_avg(df, date_col="Date", close_col="Close", volume_col="Volume", window=60):
    """
    Compute average Amihud illiquidity ratio over the trailing window.
    Amihud = mean(|ret| / volume) over the window, then 20d MA.
    Returns the average of the 20d MA over the window.
    """
    df = df.sort_values(date_col).copy()
    df["ret"] = df[close_col].pct_change().abs()
    df["dollar_vol"] = df[close_col] * df[volume_col]
    df["amihud_raw"] = df["ret"] / df["dollar_vol"]
    df["amihud_raw"] = df["amihud_raw"].replace([np.inf, -np.inf], np.nan)
    df["amihud_ma20"] = df["amihud_raw"].rolling(20, min_periods=5).mean()
    recent = df["amihud_ma20"].iloc[-window:].dropna()
    if len(recent) == 0:
        return np.nan
    return recent.mean()


def load_underlying(name, path, skip_rows=0):
    """Load CSV and return (df, S0_on_date)."""
    if skip_rows > 0:
        df = pd.read_csv(path, skiprows=skip_rows, parse_dates=["Date"]).sort_values("Date")
    else:
        df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    ref = df.loc[df["Date"] == "2023-03-09"]
    if ref.empty:
        ref = df.loc[df["Date"] <= "2023-03-09"].iloc[[-1]]
    S0 = float(ref["Close"].iloc[0])
    ref_date = pd.to_datetime(ref["Date"].iloc[0]).strftime("%Y-%m-%d")
    return df, S0, ref_date


def main():
    print("=" * 78)
    print("Table 4 — IV Surface Calibration (JPM / KRE / SPY)")
    print("=" * 78)

    # =========================================================================
    # Load data
    # =========================================================================
    kre_path = DATA_DIR / "raw" / "KRE.csv"
    kre_df, kre_S0, kre_ref_date = load_underlying("KRE", kre_path)

    jpm_path = DATA_DIR / "raw" / "JPM.csv"
    jpm_df, jpm_S0, jpm_ref_date = load_underlying("JPM", jpm_path)

    spy_path = DATA_DIR / "raw" / "all" / "SPY_yf_price.csv"
    spy_df_raw = pd.read_csv(spy_path, skiprows=2, header=None,
                             names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
                             parse_dates=["Date"])
    spy_df = spy_df_raw.sort_values("Date").reset_index(drop=True)
    ref = spy_df.loc[spy_df["Date"] == "2023-03-09"]
    if ref.empty:
        ref = spy_df.loc[spy_df["Date"] <= "2023-03-09"].iloc[[-1]]
    spy_S0 = float(ref["Close"].iloc[0])
    spy_ref_date = pd.to_datetime(ref["Date"].iloc[0]).strftime("%Y-%m-%d")

    print(f"\nKRE: S0={kre_S0:.4f}, ref_date={kre_ref_date}")
    print(f"JPM: S0={jpm_S0:.4f}, ref_date={jpm_ref_date}")
    print(f"SPY: S0={spy_S0:.4f}, ref_date={spy_ref_date}")

    # =========================================================================
    # Compute sigma_0 (60-day realized vol ending 2023-03-09)
    # =========================================================================
    kre_sigma0 = compute_realized_vol(kre_df)
    jpm_sigma0 = compute_realized_vol(jpm_df)
    spy_sigma0 = compute_realized_vol(spy_df)

    print(f"\n--- sigma_0 (60d realized vol, annualized) ---")
    print(f"  KRE: {kre_sigma0:.4f}")
    print(f"  JPM: {jpm_sigma0:.4f}")
    print(f"  SPY: {spy_sigma0:.4f}")

    # =========================================================================
    # Compute Amihud and hbar_econ
    # =========================================================================
    kre_amihud = compute_amihud_20d_avg(kre_df)
    jpm_amihud = compute_amihud_20d_avg(jpm_df)
    spy_amihud = compute_amihud_20d_avg(spy_df)

    print(f"\n--- Amihud_20d avg (60d window) ---")
    print(f"  KRE: {kre_amihud:.8e}")
    print(f"  JPM: {jpm_amihud:.8e}")
    print(f"  SPY: {spy_amihud:.8e}")

    if kre_amihud > 0 and not np.isnan(kre_amihud):
        kre_hbar = HBAR_ECON_KRE_REF
        jpm_hbar = HBAR_ECON_KRE_REF * (jpm_amihud / kre_amihud)
        spy_hbar = HBAR_ECON_KRE_REF * (spy_amihud / kre_amihud)
    else:
        print("ERROR: KRE Amihud is zero or NaN. Cannot scale.")
        sys.exit(1)

    print(f"\n--- hbar_econ (scaled by Amihud ratio) ---")
    print(f"  KRE: {kre_hbar:.6f}")
    print(f"  JPM: {jpm_hbar:.6f}")
    print(f"  SPY: {spy_hbar:.6f}")

    # =========================================================================
    # Compute RMSE for each underlying
    # =========================================================================
    underlyings = [
        ("JPMorgan Chase (JPM)", jpm_sigma0, jpm_hbar, jpm_S0),
        ("KRE (Regional Banking ETF)", kre_sigma0, kre_hbar, kre_S0),
        ("S\&P 500 Index (SPY)", spy_sigma0, spy_hbar, spy_S0),
    ]

    results = []
    for name, sigma0, hbar, S0 in underlyings:
        rmse_list = []
        for tau_days in TAU_DAYS_GRID:
            tau_years = tau_days / 252.0
            sigma_atm_tau = sigma_atm_termstructure(tau_years, sigma0)
            K_grid = K_REL_RANGE * S0
            m_grid = np.log(K_grid / S0)

            sig_ref = sigma_market_reference(m_grid, tau_years, sigma_atm_tau)
            sig_qm  = sigma_quantum(m_grid, tau_years, sigma0, hbar, GAMMA_Q, BETA_Q)

            se = (sig_ref - sig_qm) ** 2
            rmse = np.sqrt(np.mean(se))
            rmse_list.append(rmse)

        overall_rmse = np.sqrt(np.mean(np.array(rmse_list) ** 2))
        results.append((name, sigma0, hbar, overall_rmse, GAMMA_Q))

        print(f"\n  {name}:")
        print(f"    sigma_0 = {sigma0:.4f}")
        print(f"    hbar_econ = {hbar:.6f}")
        print(f"    gamma = {GAMMA_Q}")
        print(f"    RMSE per tau: {[f'{r*100:.2f}%' for r in rmse_list]}")
        print(f"    Overall RMSE = {overall_rmse*100:.2f} vol pts")

    # =========================================================================
    # GUARDS (CLAUDE1-approved recalibrated ranges)
    # =========================================================================
    print("\n" + "=" * 78)
    print("Guard results:")
    print("=" * 78)

    # G1: sigma_0 in [0.10, 0.60]
    g1_pass = all(0.10 <= s <= 0.60 for _, s, _, _, _ in results)
    print(f"  G1 (sigma_0 in [0.10, 0.60]): {'PASS' if g1_pass else 'FAIL'}")
    for name, s, _, _, _ in results:
        print(f"    {name}: sigma_0={s:.4f} {'OK' if 0.10 <= s <= 0.60 else 'OUT OF RANGE'}")

    # G2: hbar_econ in [0.001, 0.20]
    g2_pass = all(0.001 <= h <= 0.20 for _, _, h, _, _ in results)
    print(f"  G2 (hbar_econ in [0.001, 0.20]): {'PASS' if g2_pass else 'FAIL'}")
    for name, _, h, _, _ in results:
        print(f"    {name}: hbar_econ={h:.6f} {'OK' if 0.001 <= h <= 0.20 else 'OUT OF RANGE'}")

    # G3: RMSE finite and < 15 vol points (0.15)
    g3_pass = all(np.isfinite(r) and r < 0.15 for _, _, _, r, _ in results)
    print(f"  G3 (RMSE finite and < 15 vol pts): {'PASS' if g3_pass else 'FAIL'}")
    for name, _, _, r, _ in results:
        print(f"    {name}: RMSE={r*100:.2f} vol pts {'OK' if np.isfinite(r) and r < 0.15 else 'FAIL'}")

    if not (g1_pass and g2_pass and g3_pass):
        print("\n*** GUARD VIOLATION: HALTING Table 4 generation ***")
    else:
        # =========================================================================
        # Write tables/table4.tex
        # =========================================================================
        tex = r"""\begin{table}[htbp]
\centering
\caption{Quantum model calibration parameters for three underlying assets, calibrated to 60-day realized volatility ending 2023-03-09.}
\label{tab:iv-calibration}
\begin{tabular}{lrrrr}
\hline
Instrument & $\sigma_0$ & $\hbar_{\mathrm{econ}}$ & RMSE (vol pts) & $\gamma$ \\
\hline
"""
        for name, sigma0, hbar, rmse, gamma in results:
            tex += f"{name:50s} & {sigma0:.4f} & {hbar:.6f} & {rmse*100:.1f} & {gamma} \\\\\n"

        tex += r"""\hline
\end{tabular}

\medskip
\footnotesize
\textit{Note}: $\sigma_0$ is the 60-day realized volatility (annualized). $\hbar_{\mathrm{econ}}$ scales with the ratio of underlying-specific Amihud illiquidity to KRE's Amihud. $\gamma$ is fixed at 0.51 (equity leverage regime). RMSE is the root-mean-square error between the quantum smile and the canonical reference smile at $\tau \in \{7, 14, 30, 60\}$ days.
\end{table}
"""
        tex_path = TAB_DIR / "table4.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"\n  Written: {tex_path}")

    # =========================================================================
    # Diagnostic log
    # =========================================================================
    log = f"""Table 4 IV Calibration — diagnostic (Batch #3.6c)
============================================================
Executed at: {datetime.utcnow().isoformat()}Z
L-level: L2_PROXY (parameters extrapolated from Fig A.1 KRE calibration)

--- Data sources ---
KRE: {kre_path.name} (ref_date={kre_ref_date}, S0={kre_S0:.4f})
JPM: {jpm_path.name} (ref_date={jpm_ref_date}, S0={jpm_S0:.4f})
SPY: {spy_path.name} (ref_date={spy_ref_date}, S0={spy_S0:.4f})

--- sigma_0 (60d realized vol, annualized) ---
KRE: {kre_sigma0:.4f}
JPM: {jpm_sigma0:.4f}
SPY: {spy_sigma0:.4f}

--- Amihud_20d avg (60d window) ---
KRE: {kre_amihud:.8e}
JPM: {jpm_amihud:.8e}
SPY: {spy_amihud:.8e}

--- hbar_econ (scaled by Amihud ratio) ---
KRE: {kre_hbar:.6f}
JPM: {jpm_hbar:.6f}
SPY: {spy_hbar:.6f}

--- RMSE results ---
"""
    for name, sigma0, hbar, rmse, gamma in results:
        log += f"{name}: sigma_0={sigma0:.4f}, hbar={hbar:.6f}, RMSE={rmse*100:.2f} vol pts, gamma={gamma}\n"

    log += f"""
--- Guard results ---
G1 (sigma_0 in [0.10, 0.60]): {'PASS' if g1_pass else 'FAIL'}
G2 (hbar_econ in [0.001, 0.20]): {'PASS' if g2_pass else 'FAIL'}
G3 (RMSE finite and < 15 vol pts): {'PASS' if g3_pass else 'FAIL'}
"""
    log_path = LOG_DIR / "table4_calib_diagnostic.txt"
    log_path.write_text(log, encoding="utf-8")
    print(f"  Diagnostic saved: {log_path}")

    # =========================================================================
    # File hashes
    # =========================================================================
    if g1_pass and g2_pass and g3_pass:
        tex_md5 = hashlib.md5(tex_path.read_bytes()).hexdigest()[:16]
        print(f"\n  table4.tex md5: {tex_md5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
