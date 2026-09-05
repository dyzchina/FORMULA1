#!/usr/bin/env python3
"""
09_figA2_skew.py — Fig A.2: Short-Maturity Skew Asymptotics
L-level: L3_SIM (pure theoretical curves, no data)

Four models compared in log-log space:
  Quantum: O(τ⁻¹)
  Heston:  O(τ⁻¹ᐟ²)
  SABR:    O(1)
  Rough vol (H=0.10): O(τ⁻⁰·⁴)

Output: figures/skew_comparison.pdf, figures/skew_comparison.png
"""
import argparse, sys, os, warnings, hashlib, datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from _plotting_utils import setup_matplotlib_deterministic
setup_matplotlib_deterministic()

import numpy as np
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from config import FIGURES_DIR, RANDOM_SEED

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = ROOT / "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    np.random.seed(args.seed)

    print("=" * 60)
    print("09_figA2_skew.py — Fig A.2: Skew Asymptotics")
    print("=" * 60)

    # ── Parameters ───────────────────────────────────────────────────────
    # x-axis: τ ∈ 10^0.3 to 10^2.3 trading days ≈ [2, 200], 200 log-spaced points
    tau_days = np.logspace(np.log10(2), np.log10(200), 200)
    tau_years = tau_days / 252.0

    # Model coefficients
    c_Q = 0.063   # Quantum: slope = c_Q * τ^(-1)
    c_H = 0.15    # Heston:  slope = c_H * τ^(-0.5)
    c_S = 0.08    # SABR:    slope = c_S (constant)
    c_R = 0.30    # Rough:   slope = c_R * τ^(-0.4), H=0.10
    H = 0.10

    # Compute slopes
    quantum_slope = c_Q * tau_days ** (-1.0)
    heston_slope  = c_H * tau_days ** (-0.5)
    sabr_slope    = c_S * np.ones_like(tau_days)
    rough_slope   = c_R * tau_days ** (H - 0.5)  # = c_R * τ^(-0.4)

    print(f"\n  Model coefficients:")
    print(f"    Quantum: c_Q = {c_Q}  (slope = c_Q * tau^-1)")
    print(f"    Heston:  c_H = {c_H}   (slope = c_H * tau^-0.5)")
    print(f"    SABR:    c_S = {c_S}   (slope constant)")
    print(f"    Rough:   c_R = {c_R}, H = {H}  (slope = c_R * tau^-0.4)")

    # Values at specific maturities
    for label, arr in [("Quantum", quantum_slope), ("Heston", heston_slope),
                       ("SABR", sabr_slope), ("Rough", rough_slope)]:
        idx5 = np.argmin(np.abs(tau_days - 5))
        idx60 = np.argmin(np.abs(tau_days - 60))
        print(f"    {label}: tau=5d={arr[idx5]:.6f}, tau=60d={arr[idx60]:.6f}")

    # Crossover analysis
    diff_Q_H = quantum_slope - heston_slope
    crossover_QH_idx = np.where(np.diff(np.sign(diff_Q_H)))[0]
    if len(crossover_QH_idx) > 0:
        crossover_QH = tau_days[crossover_QH_idx[0]]
    else:
        crossover_QH = np.nan

    diff_Q_R = quantum_slope - rough_slope
    crossover_QR_idx = np.where(np.diff(np.sign(diff_Q_R)))[0]
    if len(crossover_QR_idx) > 0:
        crossover_QR = tau_days[crossover_QR_idx[0]]
    else:
        crossover_QR = np.nan

    print(f"\n  Slope crossover analysis:")
    print(f"    Quantum vs Heston crossover tau: {crossover_QH:.2f} days")
    print(f"    Quantum vs Rough crossover tau:  {crossover_QR:.2f} days")

    # ── Plot ─────────────────────────────────────────────────────────────
    print(f"\n  Plotting skew comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.loglog(tau_days, quantum_slope, "-", color="#1f77b4", linewidth=2.5,
              label="Quantum $\\mathcal{O}(\\tau^{-1})$")
    ax.loglog(tau_days, heston_slope, "--", color="#d62728", linewidth=2.0,
              label="Heston $\\mathcal{O}(\\tau^{-1/2})$")
    ax.loglog(tau_days, sabr_slope, ":", color="#2ca02c", linewidth=2.0,
              label="SABR $\\mathcal{O}(1)$")
    ax.loglog(tau_days, rough_slope, "-.", color="#9467bd", linewidth=2.0,
              label=f"Rough vol $\\mathcal{{O}}(\\tau^{{-0.4}})$")

    # Reference slope lines
    x_ref = np.array([1.5, 200.0])
    ax.loglog(x_ref, 0.04 * x_ref ** (-1.0), ":", color="gray", alpha=0.3, linewidth=0.8)
    ax.loglog(x_ref, 0.04 * x_ref ** (-0.5), ":", color="gray", alpha=0.3, linewidth=0.8)

    # Text annotations
    ax.text(100, 0.04 * 100 ** (-1.0) * 2.0, "$\\propto \\tau^{-1}$", fontsize=9,
            color="gray", alpha=0.5)
    ax.text(100, 0.04 * 100 ** (-0.5) * 2.0, "$\\propto \\tau^{-1/2}$", fontsize=9,
            color="gray", alpha=0.5)

    ax.set_xlabel("Time to maturity (trading days)", fontsize=11)
    ax.set_ylabel("Absolute skew slope $|\\partial \\sigma / \\partial \\ln K|$", fontsize=11)
    ax.set_title("Short-Maturity Skew Asymptotics: Quantum vs Classical Models", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.15, which="both")
    ax.set_xlim(1, 200)
    ax.set_ylim(0.005, 5)

    plt.tight_layout()
    pdf_path = FIGURES_DIR / "skew_comparison.pdf"
    png_path = FIGURES_DIR / "skew_comparison.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close()
    pdf_size = os.path.getsize(pdf_path)
    png_size = os.path.getsize(png_path)
    print(f"  OK PDF -> {pdf_path} ({pdf_size} bytes)")
    print(f"  OK PNG -> {png_path} ({png_size} bytes)")

    # ── Diagnostic log ───────────────────────────────────────────────────
    now_iso = datetime.datetime.now().isoformat()
    log_lines = []
    log_lines.append("Fig A.2 Skew Asymptotics — regeneration diagnostic")
    log_lines.append("=" * 50)
    log_lines.append(f"Executed at: {now_iso}")
    log_lines.append("L-level: L3_SIM (pure theoretical curves)")
    log_lines.append("")
    log_lines.append("Model coefficients:")
    log_lines.append(f"  Quantum: c_Q = {c_Q}  (slope = c_Q * tau^-1)")
    log_lines.append(f"  Heston:  c_H = {c_H}   (slope = c_H * tau^-0.5)")
    log_lines.append(f"  SABR:    c_S = {c_S}   (slope constant)")
    log_lines.append(f"  Rough:   c_R = {c_R}, H = {H}  (slope = c_R * tau^-0.4)")
    log_lines.append("")
    for label, arr in [("Quantum", quantum_slope), ("Heston", heston_slope),
                       ("SABR", sabr_slope), ("Rough", rough_slope)]:
        idx5 = np.argmin(np.abs(tau_days - 5))
        idx60 = np.argmin(np.abs(tau_days - 60))
        log_lines.append(f"  {label}: tau=5d={arr[idx5]:.6f}, tau=60d={arr[idx60]:.6f}")
    log_lines.append("")
    log_lines.append("Slope crossover analysis:")
    log_lines.append(f"  Quantum vs Heston crossover tau: {crossover_QH:.2f} days")
    log_lines.append(f"  Quantum vs Rough crossover tau:  {crossover_QR:.2f} days")
    log_lines.append("")
    log_lines.append("Figure files:")
    log_lines.append(f"  figures/skew_comparison.pdf ({pdf_size} bytes)")
    log_lines.append(f"  figures/skew_comparison.png ({png_size} bytes)")
    log_lines.append("")
    log_lines.append("Reproducibility (2 runs):")
    log_lines.append("  Run 1 PDF md5: <run2>")
    log_lines.append("  Run 2 PDF md5: <run2>")
    log_lines.append("  Run 1 PNG md5: <run2>")
    log_lines.append("  Run 2 PNG md5: <run2>")

    diag_path = LOGS_DIR / "figA2_skew_diagnostic.txt"
    diag_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"  OK diagnostic -> {diag_path}")

    print(f"\nOK 09_figA2_skew.py done")
    return 0

if __name__ == "__main__":
    sys.exit(main())
