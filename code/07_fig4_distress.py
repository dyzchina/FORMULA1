"""
Figure 4 — Quantum distress amplitude θ_t vs three conventional
crisis indicators (VIX, TED spread, KRE) over 2004–2024.

L-levels:
  - Panel 1 (θ_t): L3_SIM from Bloch pipeline over full 2004–2024 window
  - Panel 2 (VIX): L1_REAL, FRED_VIXCLS
  - Panel 3 (TED): L1_REAL pre-2022 + L2_PROXY post-2022 (SOFR-EFFR spliced)
  - Panel 4 (KRE): L1_REAL from Yahoo, available 2006-06-22 onwards

Purpose: demonstrate that θ_t exhibits contemporaneous or leading
behavior at all three major crises visible in the sample:
  - GFC (2007-12 to 2009-06)
  - COVID pandemic (2020-02 to 2020-04)
  - 2023 US regional banking crisis (2023-03 to 2023-05)

Deterministic: numpy.random.seed(42).
"""
import sys
import os
import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Reproducibility setup — must be first
from _plotting_utils import setup_matplotlib_deterministic
setup_matplotlib_deterministic()

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy.signal import correlate

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "code"))
from config import FIGURES_DIR, LOG_DIR, RANDOM_SEED, DATA_DIR
from _bloch_pipeline import compute_bloch_series

np.random.seed(RANDOM_SEED)
warnings.filterwarnings("ignore")

# ── Font / PDF settings ─────────────────────────────────────────────────────
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "DejaVu Sans"

# ── Constants ───────────────────────────────────────────────────────────────
WINDOW_START = "2004-01-01"
WINDOW_END   = "2024-12-31"
DPI          = 200

# Crisis windows
CRISES = {
    "GFC":           ("2007-12-01", "2009-06-30", "#f8d7da", 0.35),
    "COVID":         ("2020-02-15", "2020-04-30", "#fff3cd", 0.50),
    "2023 Banking":  ("2023-03-01", "2023-06-30", "#ffe4c4", 0.50),
}

# Crisis peak search windows (for lead-lag analysis)
CRISIS_PEAK_WINDOWS = {
    "GFC":          ("2007-12-01", "2009-06-30"),
    "COVID":        ("2020-02-15", "2020-04-30"),
    "2023 Banking": ("2023-03-01", "2023-05-15"),
}

# ── Data loading ────────────────────────────────────────────────────────────
def load_vix(root: Path) -> pd.Series:
    """Load VIX from FRED CSV. Returns Series indexed by datetime."""
    path = root / "data" / "raw" / "all" / "FRED_VIXCLS.csv"
    if path.exists():
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df["VIXCLS"].rename("vix")
    else:
        import sys as _sys
        _sys.path.insert(0, str(root / "code"))
        from config import CACHE_DIR as _CC
        _fraw = pd.read_parquet(_CC / "fred_raw.parquet")[["VIXCLS"]]
        _fraw.index = pd.to_datetime(_fraw.index)
        _fraw.index.name = "date"
        print("  [fallback] loaded VIXCLS from fred_raw.parquet")
        return _fraw["VIXCLS"].rename("vix")


def load_ted(root: Path) -> pd.Series:
    """Load TED spread from spliced parquet. Returns Series indexed by datetime."""
    path = root / "data" / "derived" / "TED_spliced_2004_2024.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["ted_bps"].rename("ted")


def load_kre(root: Path) -> pd.Series:
    """Load KRE from Yahoo CSV. Returns Series indexed by datetime."""
    path = root / "data" / "raw" / "KRE.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    s = df["Close"].rename("kre")
    # Filter to our window
    s = s[s.index <= "2024-12-31"]
    return s


# ── Lead-lag via cross-correlation ──────────────────────────────────────────
def compute_lead_lag(theta: pd.Series, indicator: pd.Series,
                     crisis_start: str, crisis_end: str,
                     max_lag: int = 30) -> int:
    """
    Compute the lag (in trading days) at which cross-correlation between
    θ_t and indicator peaks within the crisis window.
    Positive lag = θ_t leads the indicator.
    """
    # Align both series to the crisis window
    mask = (theta.index >= crisis_start) & (theta.index <= crisis_end)
    t_crisis = theta.loc[mask].dropna()
    i_crisis = indicator.reindex(t_crisis.index).dropna()
    common = t_crisis.index.intersection(i_crisis.index)
    if len(common) < 10:
        return 0
    t_vals = t_crisis.loc[common].values
    i_vals = i_crisis.loc[common].values

    # Cross-correlation
    corr = correlate(t_vals - t_vals.mean(), i_vals - i_vals.mean(), mode="same")
    mid = len(corr) // 2
    # Search within ±max_lag
    half = min(max_lag, mid)
    search = corr[mid - half : mid + half + 1]
    peak_idx = np.argmax(search)
    lag = peak_idx - half
    return lag


# ── Guard checks ────────────────────────────────────────────────────────────
def run_guard_checks(theta: pd.Series, vix: pd.Series, kre: pd.Series,
                     bloch_elapsed: float) -> list:
    """
    Run all guard clauses. Returns list of anomaly messages.
    If non-empty, caller should HALT and write anomaly file.
    """
    anomalies = []

    # Guard 1: θ_t during any crisis is NOT elevated (peak < 0.4)
    for cname, (cs, ce, *_) in CRISES.items():
        mask = (theta.index >= cs) & (theta.index <= ce)
        peak = theta.loc[mask].max() if mask.any() else 0.0
        if peak < 0.4:
            anomalies.append(
                f"GUARD 1 FAIL: θ_t peak during {cname} ({cs}–{ce}) = {peak:.4f} < 0.4"
            )

    # Guard 2: VIX peak during 2020-03 < 40
    vix_2020 = vix.loc["2020-03-01":"2020-03-31"]
    vix_peak = vix_2020.max() if not vix_2020.empty else 0.0
    if vix_peak < 40:
        anomalies.append(
            f"GUARD 2 FAIL: VIX peak during 2020-03 = {vix_peak:.1f} < 40"
        )

    # Guard 3: KRE min during 2023-04 not visibly lower than pre-crisis avg
    kre_2023 = kre.loc["2023-04-01":"2023-04-30"]
    kre_pre = kre.loc["2023-01-01":"2023-02-28"]
    kre_min = kre_2023.min() if not kre_2023.empty else 0.0
    kre_pre_avg = kre_pre.mean() if not kre_pre.empty else 0.0
    if kre_min > kre_pre_avg * 0.85:  # less than 15% drop
        anomalies.append(
            f"GUARD 3 FAIL: KRE min 2023-04 = {kre_min:.2f}, pre-crisis avg = {kre_pre_avg:.2f}, "
            f"drop = {(1 - kre_min/kre_pre_avg)*100:.1f}% < 15%"
        )

    # Guard 4: Bloch pipeline > 5 minutes
    if bloch_elapsed > 300:
        anomalies.append(
            f"GUARD 4 FAIL: Bloch pipeline took {bloch_elapsed:.1f}s > 300s"
        )

    return anomalies


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── 1. Compute Bloch series (θ_t) ──
    print("Computing Bloch series for 2004–2024...")
    t0 = datetime.now()
    bloch = compute_bloch_series(DATA_DIR, WINDOW_START, WINDOW_END, scale=2.0)
    bloch_elapsed = (datetime.now() - t0).total_seconds()
    theta = bloch["r_norm"].copy()
    print(f"  Bloch pipeline completed in {bloch_elapsed:.1f}s")
    print(f"  θ_t shape: {len(theta)}, date range: {theta.index[0]} to {theta.index[-1]}")

    # ── 2. Load conventional indicators ──
    print("Loading conventional indicators...")
    vix = load_vix(BASE_DIR)
    ted = load_ted(BASE_DIR)
    kre = load_kre(BASE_DIR)
    print(f"  VIX: {len(vix)} rows, {vix.index[0]} to {vix.index[-1]}")
    print(f"  TED: {len(ted)} rows, {ted.index[0]} to {ted.index[-1]}")
    print(f"  KRE: {len(kre)} rows, {kre.index[0]} to {kre.index[-1]}")

    # ── 3. Guard checks ──
    print("Running guard checks...")
    anomalies = run_guard_checks(theta, vix, kre, bloch_elapsed)
    if anomalies:
        anomaly_path = BASE_DIR / "_recon_20260703" / "fig4_anomaly.md"
        os.makedirs(anomaly_path.parent, exist_ok=True)
        with open(anomaly_path, "w", encoding="utf-8") as f:
            f.write("# Fig 4 Anomaly Report\n\n")
            f.write(f"Generated at: {datetime.now(timezone.utc).isoformat()}\n\n")
            for a in anomalies:
                f.write(f"- **{a}**\n")
            f.write("\n## Remediation options\n\n")
            f.write("1. Check data source integrity (MD5 hashes)\n")
            f.write("2. Verify Bloch pipeline inputs\n")
            f.write("3. Inspect data loading paths\n")
        print(f"  GUARD FAILED — {len(anomalies)} anomalies written to {anomaly_path}")
        for a in anomalies:
            print(f"    {a}")
        print("  HALTING — waiting for CLAUDE1 decision.")
        return

    print("  All guard checks PASSED.")

    # ── 4. Compute statistics for diagnostic ──
    # θ_t stats
    theta_min = theta.min()
    theta_min_date = theta.idxmin()
    theta_mean = theta.mean()
    theta_max = theta.max()
    theta_max_date = theta.idxmax()
    theta_above_05 = (theta > 0.5).mean() * 100

    # VIX stats
    vix_min = vix.min()
    vix_mean = vix.mean()
    vix_max = vix.max()
    vix_max_date = vix.idxmax()

    # TED stats
    ted_min = ted.min()
    ted_mean = ted.mean()
    ted_max = ted.max()
    ted_max_date = ted.idxmax()
    ted_l1_end = "2022-01-21"
    ted_l2_start = "2022-01-22"

    # KRE stats
    kre_first = kre.index[0]
    kre_min = kre.min()
    kre_min_date = kre.idxmin()
    kre_max = kre.max()
    kre_max_date = kre.idxmax()

    # Crisis-window peak values
    crisis_stats = {}
    for cname, (cs, ce, *_) in CRISES.items():
        m = (theta.index >= cs) & (theta.index <= ce)
        tp = theta.loc[m].max() if m.any() else np.nan
        td = theta.loc[m].idxmax() if m.any() and not np.isnan(tp) else None

        m2 = (vix.index >= cs) & (vix.index <= ce)
        vp = vix.loc[m2].max() if m2.any() else np.nan
        vd = vix.loc[m2].idxmax() if m2.any() and not np.isnan(vp) else None

        m3 = (ted.index >= cs) & (ted.index <= ce)
        tep = ted.loc[m3].max() if m3.any() else np.nan
        ted_d = ted.loc[m3].idxmax() if m3.any() and not np.isnan(tep) else None

        m4 = (kre.index >= cs) & (kre.index <= ce)
        kp = kre.loc[m4].min() if m4.any() else np.nan  # min = worst
        kd = kre.loc[m4].idxmin() if m4.any() and not np.isnan(kp) else None

        crisis_stats[cname] = {
            "theta_peak": tp, "theta_peak_date": td,
            "vix_peak": vp, "vix_peak_date": vd,
            "ted_peak": tep, "ted_peak_date": ted_d,
            "kre_min": kp, "kre_min_date": kd,
        }

    # Lead-lag analysis
    lead_lag = {}
    for cname, (cs, ce) in CRISIS_PEAK_WINDOWS.items():
        ll = {}
        ll["vix"] = compute_lead_lag(theta, vix, cs, ce)
        ll["ted"] = compute_lead_lag(theta, ted, cs, ce)
        ll["kre"] = compute_lead_lag(theta, kre, cs, ce)
        lead_lag[cname] = ll

    # ── 5. Create figure ──
    print("Creating figure...")
    fig, axes = plt.subplots(4, 1, figsize=(10.0, 8.0), sharex=True,
                             gridspec_kw={"hspace": 0.15})

    # Common x-axis limits
    x_start = pd.Timestamp("2004-01-01")
    x_end = pd.Timestamp("2025-01-01")

    # ── Panel 1: θ_t ──
    ax = axes[0]
    ax.plot(theta.index, theta.values, color="#1f77b4", linewidth=1.0, label="θ_t")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.set_ylabel("θ_t = ||r_t||", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xlim(x_start, x_end)

    # ── Panel 2: VIX ──
    ax = axes[1]
    ax.plot(vix.index, vix.values, color="#ff7f0e", linewidth=0.8)
    ax.set_ylabel("VIX (%)", fontsize=9)

    # ── Panel 3: TED ──
    ax = axes[2]
    ax.plot(ted.index, ted.values, color="#2ca02c", linewidth=0.8)
    ax.set_ylabel("TED (bps)", fontsize=9)
    # Splice date vertical line
    splice_date = pd.Timestamp("2022-01-21")
    ax.axvline(x=splice_date, color="gray", linestyle=":", alpha=0.35, linewidth=0.8)

    # ── Panel 4: KRE ──
    ax = axes[3]
    ax.plot(kre.index, kre.values, color="#d62728", linewidth=0.8)
    ax.set_ylabel("KRE ($)", fontsize=9)

    # ── Crisis shading on all panels ──
    legend_patches = []
    for cname, (cs, ce, color, alpha) in CRISES.items():
        for ax in axes:
            ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce), color=color, alpha=alpha, lw=0)
        legend_patches.append(Patch(color=color, alpha=alpha, label=cname))

    # Legend on top panel
    axes[0].legend(handles=legend_patches, loc="upper right", fontsize=7,
                   framealpha=0.8)

    # ── X-axis formatting ──
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))

    # Grid
    for ax in axes:
        ax.grid(axis="x", which="major", color="gray", linestyle="--", alpha=0.15)
        ax.tick_params(axis="both", labelsize=8)

    # Overall title
    fig.suptitle("Quantum distress amplitude θ_t vs conventional crisis indicators (2004–2024)",
                 fontsize=11, y=0.98)

    # ── Save ──
    pdf_path = FIGURES_DIR / "indicator_comparison.pdf"
    png_path = FIGURES_DIR / "indicator_comparison.png"
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved PDF: {pdf_path} ({pdf_path.stat().st_size} bytes)")
    print(f"  Saved PNG: {png_path} ({png_path.stat().st_size} bytes)")

    # ── 6. Write diagnostic log ──
    log_path = LOG_DIR / "fig4_distress_diagnostic.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Fig 4 Distress vs Conventional Indicators — regeneration diagnostic\n")
        f.write("====================================================================\n")
        f.write(f"Executed at: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("Axis version: v2 (refactored pipeline)\n")
        f.write(f"Window: {WINDOW_START} to {WINDOW_END} (full range)\n")
        f.write(f"Trading days rendered: {len(theta)}\n\n")

        f.write("θ_t (distress amplitude) statistics:\n")
        f.write(f"  min:  {theta_min:.4f}  (date {theta_min_date.date()})\n")
        f.write(f"  mean: {theta_mean:.4f}\n")
        f.write(f"  max:  {theta_max:.4f}  (date {theta_max_date.date()})\n")
        f.write(f"  Above-0.5 fraction (proxy for elevated distress):  {theta_above_05:.1f}%\n\n")

        f.write("VIX statistics:\n")
        f.write(f"  min:  {vix_min:.2f}\n")
        f.write(f"  mean: {vix_mean:.2f}\n")
        f.write(f"  max:  {vix_max:.2f}  (date {vix_max_date.date()})\n\n")

        f.write("TED spread statistics:\n")
        f.write(f"  min:  {ted_min:.2f}\n")
        f.write(f"  mean: {ted_mean:.2f}\n")
        f.write(f"  max:  {ted_max:.2f}  (date {ted_max_date.date()})\n")
        f.write(f"  L1 real portion date range: {ted.index[0].date()} to {ted_l1_end}\n")
        f.write(f"  L2 proxy portion date range: {ted_l2_start} to {ted.index[-1].date()}\n\n")

        f.write("KRE statistics:\n")
        f.write(f"  first date: {kre_first.date()}\n")
        f.write(f"  min:  {kre_min:.2f}  (date {kre_min_date.date()})\n")
        f.write(f"  max:  {kre_max:.2f}  (date {kre_max_date.date()})\n\n")

        f.write("Crisis-window peak values (each panel, each crisis):\n")
        for cname, stats in crisis_stats.items():
            f.write(f"  {cname} ({CRISES[cname][0]} to {CRISES[cname][1]}):\n")
            f.write(f"    θ_t peak: {stats['theta_peak']:.4f} on {stats['theta_peak_date'].date() if stats['theta_peak_date'] else 'N/A'}\n")
            f.write(f"    VIX peak: {stats['vix_peak']:.2f} on {stats['vix_peak_date'].date() if stats['vix_peak_date'] else 'N/A'}\n")
            f.write(f"    TED peak: {stats['ted_peak']:.2f} on {stats['ted_peak_date'].date() if stats['ted_peak_date'] else 'N/A'}\n")
            f.write(f"    KRE min:  {stats['kre_min']:.2f} on {stats['kre_min_date'].date() if stats['kre_min_date'] else 'N/A'}\n")
        f.write("\n")

        # MD5 hashes for data sources
        f.write("Data sources with MD5:\n")
        for label, path in [
            ("VIX", BASE_DIR / "data" / "raw" / "all" / "FRED_VIXCLS.csv"),
            ("TED", BASE_DIR / "data" / "derived" / "TED_spliced_2004_2024.parquet"),
            ("KRE", BASE_DIR / "data" / "raw" / "KRE.csv"),
        ]:
            if path.exists():
                h = hashlib.md5()
                h.update(path.read_bytes())
                f.write(f"  {label}:  {path.relative_to(BASE_DIR)}  {h.hexdigest()[:16]}\n")
            else:
                f.write(f"  {label}:  {path.relative_to(BASE_DIR)}  FILE NOT FOUND\n")
        f.write("\n")

        # Lead-lag
        f.write("Cross-crisis lead-lag preview (informal — full analysis in Fig 4 → Table 3 pipeline):\n")
        f.write("  For each crisis, compute lag (in trading days) at which correlation between\n")
        f.write("  θ_t and each conventional indicator (VIX, TED, KRE) peaks.\n\n")
        for cname, ll in lead_lag.items():
            f.write(f"  {cname}:\n")
            f.write(f"    θ_t lead over VIX: {ll['vix']} days\n")
            f.write(f"    θ_t lead over TED: {ll['ted']} days\n")
            f.write(f"    θ_t lead over KRE: {ll['kre']} days\n")
        f.write("\n")

        # Physical sanity checks
        f.write("Physical sanity checks:\n")
        # Check 1: θ_t peak during GFC > θ_t peak during 2015
        gfc_peak = crisis_stats["GFC"]["theta_peak"]
        y2015_peak = theta.loc["2015-01-01":"2015-12-31"].max()
        f.write(f"  1. θ_t peak during GFC ({gfc_peak:.4f}) > θ_t peak during 2015 non-crisis year ({y2015_peak:.4f})? {'YES' if gfc_peak > y2015_peak else 'NO'}\n")
        # Check 2: VIX peak during COVID > VIX peak during 2015
        covid_vix = crisis_stats["COVID"]["vix_peak"]
        y2015_vix = vix.loc["2015-01-01":"2015-12-31"].max()
        f.write(f"  2. VIX peak during COVID ({covid_vix:.2f}) > VIX peak during 2015 ({y2015_vix:.2f})? {'YES' if covid_vix > y2015_vix else 'NO'}\n")
        # Check 3: All panels show elevation during at least 2 of 3 crises
        crises_elevated = 0
        for cname in ["GFC", "COVID", "2023 Banking"]:
            if crisis_stats[cname]["theta_peak"] > 0.4:
                crises_elevated += 1
        f.write(f"  3. θ_t elevated (>0.4) in {crises_elevated}/3 crisis windows? {'YES' if crises_elevated >= 2 else 'NO'}\n\n")

        # File sizes
        f.write("Figure files:\n")
        f.write(f"  {pdf_path}  ({pdf_path.stat().st_size} bytes)\n")
        f.write(f"  {png_path}  ({png_path.stat().st_size} bytes)\n\n")

        f.write("Reproducibility:\n")
        f.write("  (Run 1 and Run 2 hashes will be appended after second run)\n")

    print(f"  Diagnostic log: {log_path}")
    print("=== FIG 4 GENERATION COMPLETE ===")


if __name__ == "__main__":
    main()
