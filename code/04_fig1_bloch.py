#!/usr/bin/env python3
"""
Figure 1 — Bloch sphere trajectory of the market state ρ_t during the 2023
US regional banking crisis (January–June 2023).

L-level: L3_SIM (theoretically-grounded simulation from L1 macro inputs).

Pipeline:
  1. Load L1 macro time series via _bloch_pipeline
  2. Compute Bloch trajectory (r_x, r_y, r_z, norm, entropy, coherence)
  3. Plot trajectory on unit Bloch sphere, colored by S(ρ_t)

Data provenance: see `_recon_20260703/data_provenance_v3.md`.
Deterministic: numpy.random.seed(42) at top of main().
Axis v2 empirical window: 2023-01-01 to 2023-06-30.

CRITICAL EVENTS (must be annotated with offset arrows, NOT text at center):
  - SVB failure:            2023-03-10
  - Signature Bank failure: 2023-03-12
  - First Republic failure: 2023-05-01
"""
import sys, os, warnings, hashlib
from datetime import datetime
from pathlib import Path

# Reproducibility setup — must be first
from _plotting_utils import setup_matplotlib_deterministic
setup_matplotlib_deterministic()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Ensure we can import config and pipeline
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RANDOM_SEED, CRISIS_WINDOW_START, CRISIS_WINDOW_END, CRISIS_EVENTS,
    DATA_DIR, FIG_DIR, LOG_DIR,
)
from _bloch_pipeline import compute_bloch_series
warnings.filterwarnings("ignore")

# ── Matplotlib settings ────────────────────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


def plot_bloch_sphere(traj):
    """Generate the Bloch sphere trajectory plot."""
    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # ── Bloch sphere wireframe ──
    u = np.linspace(0, 2 * np.pi, 24)
    v = np.linspace(0, np.pi, 12)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere,
                      color="gray", alpha=0.15, linewidth=0.35)

    # ── Coordinate axis lines (dashed gray) ──
    axis_len = 1.2
    for vec, label in [
        ([axis_len, 0, 0], r"$\sigma_x$ (market stress)"),
        ([0, axis_len, 0], r"$\sigma_y$ (credit stress)"),
        ([0, 0, axis_len], r"$\sigma_z$ (rates/equity)"),
    ]:
        ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],
                color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
        ax.text(vec[0] * 1.05, vec[1] * 1.05, vec[2] * 1.05,
                label, fontsize=8, color="gray", ha="center", va="center")

    # ── Trajectory (scatter colored by entropy) ──
    xs, ys, zs = traj["r_x"].values, traj["r_y"].values, traj["r_z"].values
    entropy = traj["entropy"].values
    sc = ax.scatter(xs, ys, zs, c=entropy, cmap="viridis",
                    s=25, alpha=0.8, edgecolors="none", zorder=5)

    # ── Start marker ──
    ax.scatter(xs[0], ys[0], zs[0], c="green", s=150, marker="o",
               edgecolors="black", linewidths=1.2, zorder=10)
    start_label_pos = (xs[0] + 0.3, ys[0] + 0.3, zs[0] + 0.3)
    ax.text(start_label_pos[0], start_label_pos[1], start_label_pos[2],
            "Jan 2, 2023", fontsize=7, color="green", fontweight="bold",
            ha="center", va="center")
    # Leader line (simple line)
    ax.plot([start_label_pos[0], xs[0]], [start_label_pos[1], ys[0]],
            [start_label_pos[2], zs[0]], color="green", linewidth=0.8, alpha=0.7)

    # ── End marker ──
    ax.scatter(xs[-1], ys[-1], zs[-1], c="red", s=150, marker="o",
               edgecolors="black", linewidths=1.2, zorder=10)
    end_label_pos = (xs[-1] - 0.3, ys[-1] - 0.3, zs[-1] - 0.3)
    ax.text(end_label_pos[0], end_label_pos[1], end_label_pos[2],
            "Jun 30, 2023", fontsize=7, color="red", fontweight="bold",
            ha="center", va="center")
    ax.plot([end_label_pos[0], xs[-1]], [end_label_pos[1], ys[-1]],
            [end_label_pos[2], zs[-1]], color="red", linewidth=0.8, alpha=0.7)

    # ── Crisis event annotations ──
    event_styles = {
        "SVB":            {"color": "orange",   "label": "SVB\n(Mar 10)"},
        "Signature":      {"color": "purple",   "label": "Signature\n(Mar 12)"},
        "First Republic": {"color": "darkred",  "label": "First Republic\n(May 1)"},
    }
    for event_name, date_str in CRISIS_EVENTS.items():
        date = pd.Timestamp(date_str)
        idx = traj.index.get_indexer([date], method="nearest")[0]
        r = traj.iloc[idx]
        style = event_styles[event_name]
        # Marker
        ax.scatter(r["r_x"], r["r_y"], r["r_z"],
                   c=style["color"], s=100, marker="^",
                   edgecolors="black", linewidths=0.8, zorder=10)
        # Offset label position (translate outward from sphere center)
        norm = np.linalg.norm([r["r_x"], r["r_y"], r["r_z"]])
        if norm > 0.01:
            offset = 0.5 / norm
        else:
            offset = 0.5
        label_pos = (r["r_x"] * (1 + offset),
                     r["r_y"] * (1 + offset),
                     r["r_z"] * (1 + offset))
        ax.text(label_pos[0], label_pos[1], label_pos[2],
                style["label"], fontsize=6.5, color=style["color"],
                fontweight="bold", ha="center", va="center")
        # Leader line (simple line)
        ax.plot([label_pos[0], r["r_x"]], [label_pos[1], r["r_y"]],
                [label_pos[2], r["r_z"]], color=style["color"],
                linewidth=0.8, alpha=0.7)

    # ── Axes limits and labels ──
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_zlim(-1.3, 1.3)
    ax.set_xlabel("", fontsize=0)
    ax.set_ylabel("", fontsize=0)
    ax.set_zlabel("", fontsize=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title("Bloch sphere trajectory of market state $\\rho_t$\n"
                 "(January–June 2023, US regional banking crisis)",
                 fontsize=11, pad=15)

    # ── Colorbar ──
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", shrink=0.6,
                        pad=0.08, aspect=30)
    cbar.set_label("Von Neumann entropy $S(\\rho_t)$", fontsize=9)
    cbar.set_ticks([0, 0.35, 0.693])
    cbar.set_ticklabels(["0 (pure)", "0.35", "ln 2 (max mixed)"], fontsize=7)

    # ── View angle ──
    ax.view_init(elev=22, azim=45)

    plt.tight_layout()
    return fig


def compute_md5(filepath):
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("04_fig1_bloch.py — Fig 1: Bloch Sphere Trajectory (Regeneration)")
    print("=" * 60)
    print(f"  Window: {CRISIS_WINDOW_START} to {CRISIS_WINDOW_END}")
    print(f"  Seed:   {RANDOM_SEED}")

    # ── Compute Bloch trajectory via shared pipeline ──
    print("\n  Computing Bloch trajectory via _bloch_pipeline...")
    traj = compute_bloch_series(DATA_DIR, CRISIS_WINDOW_START, CRISIS_WINDOW_END, scale=2.0)
    print(f"  Trajectory points: {len(traj)}")

    # ── Plot ──
    print("\n  Plotting Bloch sphere...")
    fig = plot_bloch_sphere(traj)

    # ── Save ──
    pdf_path = FIG_DIR / "bloch_trajectory.pdf"
    png_path = FIG_DIR / "bloch_trajectory.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    pdf_size = os.path.getsize(pdf_path)
    png_size = os.path.getsize(png_path)
    print(f"\n  OK PDF -> {pdf_path} ({pdf_size // 1024} KB)")
    print(f"  OK PNG -> {png_path} ({png_size // 1024} KB)")

    # ── Diagnostic log ──
    print("\n  Writing diagnostic log...")
    lines = []
    lines.append("Fig 1 Bloch — regeneration diagnostic")
    lines.append("=" * 40)
    lines.append(f"Executed at: {datetime.now().isoformat()}")
    lines.append(f"Axis version: v2 (refactored pipeline)")
    lines.append(f"Window: {CRISIS_WINDOW_START} to {CRISIS_WINDOW_END}")
    lines.append(f"Trading days rendered: {len(traj)}")
    lines.append("")

    DATA_BAK = DATA_DIR / "raw"
    DERIVED = DATA_DIR / "derived"
    src_files = {
        "VIX":     DATA_BAK / "VIXCLS.csv",
        "STLFSI4": DATA_BAK / "STLFSI4.csv",
        "HY-OAS":  DATA_BAK / "BAMLH0A0HYM2.csv",
        "EFFR":    DATA_BAK / "DFF.csv",
        "KRE":     DATA_BAK / "KRE.csv",
        "GSPC":    DATA_BAK / "GSPC.csv",
        "TED":     DERIVED / "TED_spliced_2004_2024.parquet",
    }
    lines.append("Data sources (with MD5 first-16):")
    for name, fpath in src_files.items():
        if fpath.exists():
            md5 = compute_md5(fpath)
            lines.append(f"  {name}: {fpath}  {md5}")
        else:
            lines.append(f"  {name}: {fpath}  MISSING")

    # Find HY-OAS first available date
    hy_oas_first = None
    for idx_date in traj.index:
        if idx_date in traj.index and traj.loc[idx_date, "r_y_raw"] != 0:
            hy_oas_first = idx_date
            break
    lines.append(f"  HY-OAS available from: {hy_oas_first.date() if hy_oas_first else 'N/A'}")

    lines.append("")
    lines.append("Bloch vector norm ||r_t||:")
    lines.append(f"  min:  {traj['r_norm'].min():.4f}")
    lines.append(f"  mean: {traj['r_norm'].mean():.4f}")
    lines.append(f"  max:  {traj['r_norm'].max():.4f}")
    lines.append(f"  std:  {traj['r_norm'].std():.4f}")
    lines.append("")
    lines.append("Entropy S(ρ_t):")
    lines.append(f"  min:  {traj['entropy'].min():.4f}")
    lines.append(f"  mean: {traj['entropy'].mean():.4f}")
    lines.append(f"  max:  {traj['entropy'].max():.4f}")
    lines.append("")

    lines.append("Crisis events (Bloch vector at nearest trading day):")
    for event_name, date_str in CRISIS_EVENTS.items():
        date = pd.Timestamp(date_str)
        idx = traj.index.get_indexer([date], method="nearest")[0]
        r = traj.iloc[idx]
        actual_date = traj.index[idx]
        offset = (actual_date - date).days
        offset_str = f" (offset {offset}d)" if offset != 0 else ""
        lines.append(
            f"  {event_name} ({date_str}): nearest={actual_date.date()}{offset_str}, "
            f"r=({r['r_x']:.4f}, {r['r_y']:.4f}, {r['r_z']:.4f}), "
            f"|r|={r['r_norm']:.4f}, S={r['entropy']:.4f}"
        )

    lines.append("")
    mask_norm = traj["r_norm"] > 0.5
    if mask_norm.any():
        first_norm = traj.index[mask_norm][0]
        lines.append(f"First date where ||r|| > 0.5: {first_norm.date()}")
    else:
        lines.append("First date where ||r|| > 0.5: never")

    mask_ent = traj["entropy"] > 0.5
    if mask_ent.any():
        first_ent = traj.index[mask_ent][0]
        lines.append(f"First date where S > 0.5: {first_ent.date()}")
    else:
        lines.append("First date where S > 0.5: never")

    lines.append("")
    lines.append("Figure files written:")
    lines.append(f"  figures/bloch_trajectory.pdf  ({pdf_size} bytes)")
    lines.append(f"  figures/bloch_trajectory.png  ({png_size} bytes)")

    diag_path = LOG_DIR / "fig1_bloch_diagnostic.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  OK Diagnostic -> {diag_path}")

    print(f"\nOK 04_fig1_bloch.py done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
