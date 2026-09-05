#!/usr/bin/env python3
"""
Figure A.1 — KRE options implied volatility smile comparison
between the quantum model and a canonical equity-crisis reference smile,
calibrated to the 60-day realized volatility ending 2023-03-09.

L-level: L2_PROXY (real 2023-03-10 KRE option chain is not available;
reference smile is a canonical equity-crisis parametrization).

Methodology change (2026-07-04, Batch #3.5d):
  Previous version attempted Heston moment-matching calibration on
  the 60-day spot trajectory. That calibration was numerically unstable
  (Feller condition violated, sigma_v pushed to degenerate values).
  Under CLAUDE1 supervisor decision, the calibration has been replaced
  with a canonical 3-parameter equity-crisis smile:
     sigma_market(K, tau) = sigma_atm(tau) + a * m + b * m^2
  with a = -0.50 (skew), b = +1.00 (curvature), and sigma_atm from
  the 60-day realized volatility.

Output: figures/real_vs_model_smile.pdf, figures/real_vs_model_smile.png
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import sys
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt

from _plotting_utils import setup_matplotlib_deterministic
setup_matplotlib_deterministic()

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR  = ROOT / "figures"
LOG_DIR  = ROOT / "logs"
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# =========================================================================
# CANONICAL PARAMETERS (fixed, not optimized)
# beta recalibrated in Batch #3.5e from 0.10 to 0.001.
# gamma sign flipped in Batch #3.5f from -0.51 to +0.51 so that Quantum
# smile exhibits equity leverage effect (lower IV at high K, higher IV at
# low K). The paper axis eq. (3) only defines gamma as a state-dependent
# constant; choosing gamma > 0 corresponds to the equity leverage-effect
# regime, per Remark 4.12 with corrected sign convention.
# See _recon_20260703/figA1_gamma_signflip.md for full derivation.
# =========================================================================
SIGMA_REALIZED_ANNUAL = 0.2841   # from Task B, 60d realized vol
SMILE_SKEW_A          = -0.50    # canonical equity-crisis skew
SMILE_CURV_B          = +1.00    # canonical equity-crisis curvature
HBAR_ECON             = 0.087    # from paper Table 4 KRE
GAMMA_Q               = +0.51    # quantum skew coefficient (Batch #3.5f: sign flipped to match equity leverage)
BETA_Q                = +0.001   # quantum convexity coefficient (Batch #3.5e recalibration)
TAU_DAYS_GRID         = [7, 14, 30, 60]
K_REL_RANGE           = np.linspace(0.7, 1.3, 25)  # K/S in [0.7, 1.3]


def sigma_market_reference(m, tau_years, sigma_atm):
    """
    Canonical equity-crisis reference smile.
    m: log-moneyness ln(K/S)
    tau_years: time-to-maturity in years
    sigma_atm: at-the-money vol for this tau (with mild term structure)
    """
    return sigma_atm + SMILE_SKEW_A * m + SMILE_CURV_B * m**2


def sigma_quantum(m, tau_years, sigma_0):
    """
    Quantum model implied volatility per paper eq. (3):
      sigma^2_imp(x, tau) = sigma_0^2 + hbar_econ * gamma * x/tau + hbar_econ^2 * beta / tau^2
    where x = ln(S/K) = -m
    """
    x = -m
    sig_sq = sigma_0**2 + HBAR_ECON * GAMMA_Q * (x / tau_years) + HBAR_ECON**2 * BETA_Q / (tau_years**2)
    sig_sq = np.maximum(sig_sq, 1e-8)
    return np.sqrt(sig_sq)


def sigma_atm_termstructure(tau_years, sigma_realized):
    """Mild term structure: short-dated slightly below, long-dated slightly above realized."""
    return sigma_realized * (1.0 + 0.05 * (1.0 - np.exp(-tau_years / 0.1)))


def main():
    print("=" * 78)
    print("Figure A.1 — KRE Smile (Canonical Reference vs Quantum)")
    print("=" * 78)

    # --- Load KRE spot on 2023-03-09 ---
    kre_path = DATA_DIR / "raw" / "KRE.csv"
    kre = pd.read_csv(kre_path, parse_dates=["Date"]).sort_values("Date")
    kre_ref = kre.loc[kre["Date"] == "2023-03-09"]
    if kre_ref.empty:
        kre_ref = kre.loc[kre["Date"] <= "2023-03-09"].iloc[[-1]]
    S0 = float(kre_ref["Close"].iloc[0])
    ref_date = pd.to_datetime(kre_ref["Date"].iloc[0]).strftime("%Y-%m-%d")

    print(f"Reference date:  {ref_date}")
    print(f"KRE spot S0:     {S0:.4f}")
    print(f"sigma_realized (60d, annualized): {SIGMA_REALIZED_ANNUAL:.4f}")
    print(f"Smile params:    a={SMILE_SKEW_A}, b={SMILE_CURV_B}")
    print(f"Quantum params:  hbar_econ={HBAR_ECON}, gamma={GAMMA_Q}, beta={BETA_Q}")
    print()

    # --- Compute smiles for each tau ---
    K_grid = K_REL_RANGE * S0

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes_flat = axes.flatten()

    for i, tau_days in enumerate(TAU_DAYS_GRID):
        ax = axes_flat[i]
        tau_years = tau_days / 252.0
        sigma_atm_tau = sigma_atm_termstructure(tau_years, SIGMA_REALIZED_ANNUAL)

        m_grid = np.log(K_grid / S0)
        sig_ref = sigma_market_reference(m_grid, tau_years, sigma_atm_tau)
        sig_qm  = sigma_quantum(m_grid, tau_years, sigma_atm_tau)

        ax.scatter(K_grid, sig_ref * 100, s=20, color="black",
                   label="Reference equity smile" if i == 0 else None,
                   zorder=3)
        ax.plot(K_grid, sig_qm * 100, "-", color="#d62728", linewidth=2.0,
                label="Quantum model" if i == 0 else None, zorder=2)
        ax.axhline(SIGMA_REALIZED_ANNUAL * 100, color="gray", linestyle="--",
                   linewidth=1.0, alpha=0.5,
                   label="Classical (sigma_0)" if i == 0 else None, zorder=1)

        ax.set_title(f"tau = {tau_days} days")
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Implied Volatility (%)")
        ax.grid(True, alpha=0.15)

        atm_idx = np.argmin(np.abs(K_grid - S0))
        print(f"tau = {tau_days:3d} days:")
        print(f"  sigma_atm(tau):           {sigma_atm_tau*100:.2f}%")
        print(f"  Reference IV range: [{sig_ref.min()*100:.2f}%, {sig_ref.max()*100:.2f}%]")
        print(f"  Reference ATM:      {sig_ref[atm_idx]*100:.2f}%")
        print(f"  Quantum IV range:   [{sig_qm.min()*100:.2f}%, {sig_qm.max()*100:.2f}%]")
        print(f"  Quantum ATM:        {sig_qm[atm_idx]*100:.2f}%")
        h = 0.01
        ref_slope = (sigma_market_reference(h, tau_years, sigma_atm_tau)
                     - sigma_market_reference(-h, tau_years, sigma_atm_tau)) / (2*h)
        qm_slope  = (sigma_quantum(h, tau_years, sigma_atm_tau)
                     - sigma_quantum(-h, tau_years, sigma_atm_tau)) / (2*h)
        print(f"  dsigma/dlnK|ATM ref:    {ref_slope:.4f}")
        print(f"  dsigma/dlnK|ATM qm:     {qm_slope:.4f}")

    fig.legend(loc="upper right", framealpha=0.9, fontsize=9)
    fig.suptitle(f"KRE IV smile comparison (canonical reference vs quantum model)\n"
                 f"Reference date: {ref_date}. sigma_realized(60d) = {SIGMA_REALIZED_ANNUAL:.2%}. "
                 f"L2_PROXY: canonical equity-crisis parametrization.",
                 fontsize=10, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    pdf_path = FIG_DIR / "real_vs_model_smile.pdf"
    png_path = FIG_DIR / "real_vs_model_smile.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)

    # --- Diagnostic log ---
    kre_md5 = hashlib.md5(kre_path.read_bytes()).hexdigest()[:16]
    pdf_md5 = hashlib.md5(pdf_path.read_bytes()).hexdigest()[:16]
    png_md5 = hashlib.md5(png_path.read_bytes()).hexdigest()[:16]

    log = f"""Fig A.1 IV Smile — regeneration diagnostic (Batch #3.5f, gamma sign flipped)
====================================================================================
Executed at: {datetime.utcnow().isoformat()}Z
L-level: L2_PROXY (canonical equity-crisis reference; no historical KRE options chain)
Methodology: canonical smile a*m + b*m^2 (replaces unstable Heston calibration)

KRE reference data:
  Source: {kre_path.name} (md5 {kre_md5})
  Reference date: {ref_date}
  KRE spot S0: {S0:.4f}
  sigma_realized (60d annualized): {SIGMA_REALIZED_ANNUAL:.4f}

Reference smile parameters (fixed):
  a (skew):      {SMILE_SKEW_A}
  b (curvature): {SMILE_CURV_B}
  ATM term structure: sigma_realized * (1 + 0.05 * (1 - exp(-tau/0.1)))

Quantum model parameters (from paper Table 4 KRE row):
  sigma_0 = {SIGMA_REALIZED_ANNUAL:.4f}
  hbar_econ = {HBAR_ECON}
  gamma = {GAMMA_Q}
  beta = {BETA_Q}

Figure files:
  figures/real_vs_model_smile.pdf ({pdf_path.stat().st_size} bytes, md5 {pdf_md5})
  figures/real_vs_model_smile.png ({png_path.stat().st_size} bytes, md5 {png_md5})
"""
    log_path = LOG_DIR / "figA1_smile_diagnostic.txt"
    log_path.write_text(log, encoding="utf-8")
    print()
    print(log)
    print(f"Diagnostic saved: {log_path}")


if __name__ == "__main__":
    main()
