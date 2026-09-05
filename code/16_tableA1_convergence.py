#!/usr/bin/env python3
"""
Table A.1 — Δ_QM vs Δ_BS Convergence as ℏ → 0
L-level: L3_SIM (pure parameter sweep, no data)

Methodology:
  Numerical parameter sweep of ℏ_econ from 0 to 0.15, showing Δ_QM → Δ_BS as ℏ → 0.

  Fixed parameters:
    S = 38, K = 40, τ = 30 days = 30/252 yr, r = 0.045, σ_0 = 0.35, γ = 0.51

  Compute Δ_QM(ℏ) for ℏ ∈ {0.15, 0.10, 0.05, 0.01, 0.00}:
    x = ln(S/K) = ln(38/40) = -0.0513
    τ_yr = 30/252 = 0.1190
    d1 = (x + σ² τ / 2) / (σ √τ)
    Δ_BS = N(d1)
    Γ_BS = φ(d1) / (S · σ · √τ_yr)
    Δ_QM = Δ_BS + ℏ · Γ_BS · γ · x / τ_yr

Guards:
  G1: Δ_QM(ℏ=0) = Δ_BS exactly (relative error = 0)
  G2: Δ_QM(ℏ=0.15) differs from Δ_BS by < 30%
  G3: Δ_QM monotonic in ℏ (either always increasing or always decreasing)
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import norm

# Fix Windows GBK console for unicode print
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # Python < 3.7

ROOT     = Path(__file__).resolve().parent.parent
LOG_DIR  = ROOT / "logs"
TAB_DIR  = ROOT / "tables"
LOG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)

# =========================================================================
# Fixed parameters
# =========================================================================
S = 38.0
K = 40.0
TAU_DAYS = 30
TAU_YR = TAU_DAYS / 252.0
R = 0.045
SIGMA_0 = 0.35
GAMMA = 0.51

# ℏ_econ sweep values
HBAR_VALUES = [0.15, 0.10, 0.05, 0.01, 0.00]

# Derived
X = np.log(S / K)  # log-moneyness


def compute_delta_bs(S, K, T, sigma, r=0.045):
    """BSM delta."""
    if T <= 0:
        return 1.0 if S >= K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def compute_gamma_bs(S, K, T, sigma, r=0.045):
    """BSM gamma."""
    if T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def compute_delta_qm(S, K, T, sigma_0, hbar, gamma, r=0.045):
    """
    Quantum delta per Prop 4.14:
    Δ_QM = Δ_BS(σ_0) + ℏ · Γ_BS(σ_0) · γ · x/τ
    """
    if T <= 0:
        return 1.0 if S >= K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma_0**2) * T) / (sigma_0 * np.sqrt(T))
    bsm_delta = norm.cdf(d1)
    bsm_gamma = norm.pdf(d1) / (S * sigma_0 * np.sqrt(T))
    x = np.log(S / K)
    correction = hbar * bsm_gamma * gamma * x / T
    return bsm_delta + correction


def main():
    print("=" * 78)
    print("Table A.1 — Δ_QM vs Δ_BS Convergence as hbar → 0")
    print("=" * 78)
    print()
    print(f"Fixed parameters:")
    print(f"  S = {S}, K = {K}, x = ln(S/K) = {X:.6f}")
    print(f"  τ = {TAU_DAYS} days = {TAU_YR:.4f} years")
    print(f"  r = {R}, σ_0 = {SIGMA_0}, γ = {GAMMA}")
    print()

    # Compute reference BSM delta
    delta_bs = compute_delta_bs(S, K, TAU_YR, SIGMA_0, R)
    gamma_bs = compute_gamma_bs(S, K, TAU_YR, SIGMA_0, R)

    print(f"  Δ_BS = {delta_bs:.6f}")
    print(f"  Γ_BS = {gamma_bs:.6f}")
    print()

    # Compute Δ_QM for each ℏ value
    results = []
    for hbar in HBAR_VALUES:
        delta_qm = compute_delta_qm(S, K, TAU_YR, SIGMA_0, hbar, GAMMA, R)
        rel_error = abs(delta_qm - delta_bs) / max(abs(delta_bs), 1e-12) * 100
        results.append((hbar, delta_qm, delta_bs, rel_error))

        print(f"  hbar_econ = {hbar:.2f}: Δ_QM = {delta_qm:.6f}, Δ_BS = {delta_bs:.6f}, "
              f"Rel. error = {rel_error:.4f}%")

    print()

    # =========================================================================
    # GUARDS
    # =========================================================================
    print("=" * 78)
    print("Guard results:")
    print("=" * 78)

    # G1: Δ_QM(ℏ=0) = Δ_BS exactly (relative error = 0)
    hbar_zero_result = [r for r in results if r[0] == 0.0]
    if hbar_zero_result:
        g1_pass = hbar_zero_result[0][3] < 1e-10  # effectively zero
    else:
        g1_pass = False
    print(f"  G1 (Δ_QM(hbar=0) = Δ_BS exactly): {'PASS' if g1_pass else 'FAIL'}")
    if hbar_zero_result:
        print(f"    Δ_QM(0) = {hbar_zero_result[0][1]:.6f}, Δ_BS = {hbar_zero_result[0][2]:.6f}, "
              f"rel_error = {hbar_zero_result[0][3]:.10f}%")

    # G2: Δ_QM(ℏ=0.15) differs from Δ_BS by < 30%
    hbar_015_result = [r for r in results if abs(r[0] - 0.15) < 1e-6]
    if hbar_015_result:
        g2_pass = hbar_015_result[0][3] < 30.0
    else:
        g2_pass = False
    print(f"  G2 (Δ_QM(hbar=0.15) differs from Δ_BS by < 30%): {'PASS' if g2_pass else 'FAIL'}")
    if hbar_015_result:
        print(f"    Δ_QM(0.15) = {hbar_015_result[0][1]:.6f}, Δ_BS = {hbar_015_result[0][2]:.6f}, "
              f"rel_error = {hbar_015_result[0][3]:.4f}%")

    # G3: Δ_QM monotonic in ℏ
    delta_qm_vals = [r[1] for r in results]
    is_monotonic_inc = all(delta_qm_vals[i] <= delta_qm_vals[i+1] for i in range(len(delta_qm_vals)-1))
    is_monotonic_dec = all(delta_qm_vals[i] >= delta_qm_vals[i+1] for i in range(len(delta_qm_vals)-1))
    g3_pass = is_monotonic_inc or is_monotonic_dec
    print(f"  G3 (Δ_QM monotonic in hbar): {'PASS' if g3_pass else 'FAIL'}")
    print(f"    Monotonic increasing: {is_monotonic_inc}, Monotonic decreasing: {is_monotonic_dec}")
    print(f"    Δ_QM values: {[f'{v:.6f}' for v in delta_qm_vals]}")

    if not (g1_pass and g2_pass and g3_pass):
        print("\n*** GUARD VIOLATION: HALTING Table A.1 generation ***")
    else:
        # =========================================================================
        # Write tables/tableA1.tex
        # =========================================================================
        tex = r"""\begin{table}[htbp]
\centering
\caption{Convergence of quantum delta to Black-Scholes-Merton delta as $\hbar_{\mathrm{econ}} \to 0$. Parameters: $S=38$, $K=40$, $\tau=30$ days, $r=4.5\%$, $\sigma_0=0.35$, $\gamma=0.51$.}
\label{tab:delta-convergence}
\begin{tabular}{lccc}
\hline
$\hbar_{\mathrm{econ}}$ & $\Delta_{\mathrm{QM}}$ & $\Delta_{\mathrm{BS}}$ & Relative error \\
\hline
"""
        for hbar, delta_qm, delta_bs, rel_err in results:
            tex += f"${hbar:.2f}$ & ${delta_qm:.4f}$ & ${delta_bs:.4f}$ & ${rel_err:.2f}\\%$ \\\\\n"

        tex += r"""\hline
\end{tabular}

\medskip
\footnotesize
\textit{Note}: The quantum delta converges to the BSM delta as $\hbar_{\mathrm{econ}} \to 0$, confirming that the correction term $\hbar_{\mathrm{econ}} \cdot \Gamma_{\mathrm{BS}} \cdot \gamma \cdot x/\tau$ vanishes in the classical limit. At $\hbar_{\mathrm{econ}}=0$, the two deltas coincide exactly.
\end{table}
"""
        tex_path = TAB_DIR / "tableA1.tex"
        tex_path.write_text(tex, encoding="utf-8")
        print(f"\n  Written: {tex_path}")

    # =========================================================================
    # Diagnostic log
    # =========================================================================
    log = f"""Table A.1 Δ_QM vs Δ_BS Convergence — diagnostic (Batch #3.6c)
============================================================
Executed at: {datetime.utcnow().isoformat()}Z
L-level: L3_SIM (pure parameter sweep, no data)

--- Parameters ---
S = {S}
K = {K}
x = ln(S/K) = {X:.6f}
τ = {TAU_DAYS} days = {TAU_YR:.4f} years
r = {R}
σ_0 = {SIGMA_0}
γ = {GAMMA}

Δ_BS = {delta_bs:.6f}
Γ_BS = {gamma_bs:.6f}

--- Results ---
"""
    for hbar, delta_qm, delta_bs, rel_err in results:
        log += f"hbar={hbar:.2f}: Δ_QM={delta_qm:.6f}, Δ_BS={delta_bs:.6f}, rel_err={rel_err:.4f}%\n"

    log += f"""
--- Guard results ---
G1 (Δ_QM(ℏ=0) = Δ_BS exactly): {'PASS' if g1_pass else 'FAIL'}
G2 (Δ_QM(ℏ=0.15) differs from Δ_BS by < 30%): {'PASS' if g2_pass else 'FAIL'}
G3 (Δ_QM monotonic in ℏ): {'PASS' if g3_pass else 'FAIL'}
"""
    log_path = LOG_DIR / "tableA1_diagnostic.txt"
    log_path.write_text(log, encoding="utf-8")
    print(f"\n  Diagnostic saved: {log_path}")

    # =========================================================================
    # File hashes
    # =========================================================================
    if g1_pass and g2_pass and g3_pass:
        tex_md5 = hashlib.md5(tex_path.read_bytes()).hexdigest()[:16]
        print(f"\n  tableA1.tex md5: {tex_md5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
