#!/usr/bin/env python3
"""
Table 1 — Skew Asymptotic Comparison (pure theory, no data)
L-level: L3_SIM (pure theory, no data)

Methodology:
  Static comparison table of theoretical asymptotic scalings for the implied
  volatility skew slope partial sigma/partial x|_{x=0} as tau -> 0 for each model.

  Models:
    - BSM:              skew = 0, tau->0 limit = 0, 1 free param (sigma_0)
    - Heston:           skew = O(1), tau->0 limit = finite, 1 free param (nu)
    - Jump-diffusion:   skew = O(tau^{-1/2}), tau->0 limit = inf, 2 free params (lambda, mu_J)
    - Rough volatility: skew = O(tau^{H-1/2}), tau->0 limit = inf if H<1/2, 1 free param (H)
    - Quantum:          skew = O(tau^{-1}), tau->0 limit = inf, 1 free param (hbar_econ)

  No numerical computation needed. Pure theoretical comparison.
"""
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")

import hashlib
from pathlib import Path
from datetime import datetime

ROOT     = Path(__file__).resolve().parent.parent
LOG_DIR  = ROOT / "logs"
TAB_DIR  = ROOT / "tables"
LOG_DIR.mkdir(exist_ok=True)
TAB_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 78)
    print("Table 1 — Skew Asymptotic Comparison (pure theory)")
    print("=" * 78)
    print()
    print("This is a static theoretical comparison table.")
    print("No numerical computation needed.")
    print()

    # =========================================================================
    # Write tables/table1.tex
    # =========================================================================
    tex = r"""\begin{table}[htbp]
\centering
\caption{Asymptotic skew slope comparison across option pricing models. The skew slope $\partial\sigma/\partial x|_{x=0}$ measures the sensitivity of implied volatility to log-moneyness $x = \ln(K/S)$ at the at-the-money point, in the short-maturity limit $\tau \to 0$.}
\label{tab:skew-asymptotics}
\begin{tabular}{lccc}
\hline
Model & Skew slope $\partial\sigma/\partial x|_{x=0}$ & $\tau\to 0$ limit & Free parameters \\
\hline
BSM & $0$ & $0$ & 1 ($\sigma_0$) \\
Heston & $O(1)$ & Finite & 1 ($\nu$) \\
Jump-diffusion & $O(\tau^{-1/2})$ & $\infty$ & 2 ($\lambda$, $\mu_J$) \\
Rough volatility & $O(\tau^{H-1/2})$ & $\infty$ if $H<1/2$ & 1 ($H$) \\
Quantum & $O(\tau^{-1})$ & $\infty$ & 1 ($\hbar_{\mathrm{econ}}$) \\
\hline
\end{tabular}

\medskip
\footnotesize
\textit{Note}: The quantum model exhibits the steepest short-maturity skew divergence ($\tau^{-1}$), reflecting the $\hbar_{\mathrm{econ}}^2/\tau^2$ term in the implied variance expansion (eq.~3). This is consistent with the empirical observation that near-dated out-of-the-money options carry the most pronounced volatility skew during crisis regimes.
\end{table}
"""
    tex_path = TAB_DIR / "table1.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"  Written: {tex_path}")

    # =========================================================================
    # Diagnostic log
    # =========================================================================
    log = "Table 1 Skew Asymptotic Comparison — diagnostic (Batch #3.6c)\n"
    log += "============================================================\n"
    log += "Executed at: " + datetime.utcnow().isoformat() + "Z\n"
    log += "L-level: L3_SIM (pure theory, no data)\n"
    log += "\n"
    log += "This is a static theoretical comparison table.\n"
    log += "No numerical computation was performed.\n"
    log += "\n"
    log += "Models compared:\n"
    log += "  1. BSM:              skew=0, tau->0=0, params=1 (sigma_0)\n"
    log += "  2. Heston:           skew=O(1), tau->0=finite, params=1 (nu)\n"
    log += "  3. Jump-diffusion:   skew=O(tau^{-1/2}), tau->0=inf, params=2 (lambda, mu_J)\n"
    log += "  4. Rough volatility: skew=O(tau^{H-1/2}), tau->0=inf if H<1/2, params=1 (H)\n"
    log += "  5. Quantum:          skew=O(tau^{-1}), tau->0=inf, params=1 (hbar_econ)\n"
    log += "\n"
    log += "No guards needed (pure theory, no data).\n"

    log_path = LOG_DIR / "table1_skew_asymp_diagnostic.txt"
    log_path.write_text(log, encoding="utf-8")
    print(f"  Diagnostic saved: {log_path}")

    # =========================================================================
    # File hashes
    # =========================================================================
    tex_md5 = hashlib.md5(tex_path.read_bytes()).hexdigest()[:16]
    print(f"\n  table1.tex md5: {tex_md5}")
    print("\nDone.")


if __name__ == "__main__":
    main()
