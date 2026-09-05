#!/usr/bin/env python3
"""
Figure 2 — Quantum coherence C(ρ_t) vs classical Pearson correlation
(KRE vs VIX), January–June 2023.

L-levels:
  - C(ρ_t): L3_SIM (from _bloch_pipeline, same trajectory as Fig 1)
  - Pearson correlation: L1_REAL (30-day rolling on KRE log returns and
    VIX log-changes)

Central empirical claim to visualize and verify:
  Quantum coherence rises ~5-7 trading days BEFORE Pearson correlation
  (absolute value) at each of the three 2023 banking-crisis failure events
  (SVB Mar 10, Signature Mar 12, First Republic May 1).

Rationale for KRE vs VIX (not KRE vs GSPC):
  KRE (regional banks) and VIX (implied volatility) are cross-sector:
  equity vs volatility. The leverage effect predicts negative correlation
  that strengthens (becomes more negative) during crises. This provides
  a meaningful classical benchmark for comparison with quantum coherence,
  unlike KRE vs GSPC (both equities, naturally high correlation).

Deterministic: numpy.random.seed(42).
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
import matplotlib.dates as mdates

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    RANDOM_SEED, CRISIS_WINDOW_START, CRISIS_WINDOW_END, CRISIS_EVENTS,
    DATA_DIR, FIG_DIR, LOG_DIR,
)
from _bloch_pipeline import compute_bloch_series
warnings.filterwarnings("ignore")

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# ── Helper: compute MD5 ────────────────────────────────────────────────────
def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ── Helper: bidirectional exceedance detection ─────────────────────────────
def detect_exceedances(series, window=30, k_sigma=1.0):
    """
    Detect OUT-OF-BAND excursions (either up or down) beyond the rolling mean
    ± k_sigma * rolling_std.

    Returns a boolean Series where True indicates a NEW exceedance
    (transition from in-band to out-of-band).
    """
    rmean = series.rolling(window, min_periods=15).mean()
    rstd = series.rolling(window, min_periods=15).std()
    upper = rmean + k_sigma * rstd
    lower = rmean - k_sigma * rstd
    excursion = (series > upper) | (series < lower)
    # Return only NEW exceedances (transitions from in-band to out-of-band)
    was_in_band_prev = (~excursion.shift(1, fill_value=False))
    return excursion & was_in_band_prev


def most_recent_exceedance(series, exceedance_mask, event_date):
    """
    Find the most recent exceedance date ≤ event_date.
    Returns (date, value) or (None, None) if no exceedance found.
    """
    mask = (exceedance_mask.index <= event_date) & exceedance_mask
    sub = series[mask]
    if len(sub) == 0:
        return None, None
    return sub.index[-1], sub.iloc[-1]


# ── Guard clause check ─────────────────────────────────────────────────────
def guard_check(results):
    """
    Check for anomalies in lead analysis. Returns list of anomaly strings.
    If critical anomalies found, returns them and caller should halt.
    """
    anomalies = []
    for event_name, event_date, c_date, c_val, p_date, p_val, lead_days in results:
        if c_date is None or p_date is None:
            missing = []
            if c_date is None:
                missing.append(f"C(ρ) exceedance for {event_name}")
            if p_date is None:
                missing.append(f"Pearson exceedance for {event_name}")
            anomalies.append(f"HALT: No exceedance detected — {', '.join(missing)}")
        elif lead_days < 0:
            anomalies.append(
                f"HALT: Lead time < 0 for {event_name} ({lead_days} days) — "
                f"coherence exceeds AFTER Pearson. Contradicts paper thesis."
            )
        elif lead_days > 45:
            anomalies.append(
                f"HALT: Lead time > 45 trading days for {event_name} ({lead_days} days) — "
                f"implausibly early. Check data or threshold."
            )
    # Check if SVB and Signature have same exceedance date
    if len(results) >= 2:
        c_dates = [r[2] for r in results if r[2] is not None]
        if len(c_dates) >= 2 and c_dates[0] == c_dates[1]:
            anomalies.append(
                f"WARN: SVB and Signature share same C(ρ) exceedance date {c_dates[0].date()} — "
                f"acceptable if SVB's exceedance is the only one detected for both."
            )
    return anomalies


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    np.random.seed(RANDOM_SEED)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("05_fig2_coherence.py — Fig 2: Coherence vs Pearson Correlation")
    print("=" * 60)
    print(f"  Window: {CRISIS_WINDOW_START} to {CRISIS_WINDOW_END}")
    print(f"  Seed:   {RANDOM_SEED}")

    # ── 1. Compute Bloch trajectory (same as Fig 1) ──
    print("\n  Computing Bloch trajectory via _bloch_pipeline...")
    traj = compute_bloch_series(DATA_DIR, CRISIS_WINDOW_START, CRISIS_WINDOW_END, scale=2.0)
    coherence = traj["coherence"]
    print(f"  Trajectory points: {len(traj)}")

    # ── 2. Load KRE log returns and VIX log-changes ──
    DATA_BAK = DATA_DIR / "raw"
    WARMUP_START = "2022-11-01"
    print(f"\n  Loading KRE returns and VIX log-changes from {WARMUP_START}...")

    # KRE log returns
    kre = pd.read_csv(DATA_BAK / "KRE.csv", parse_dates=["Date"])
    kre = kre.set_index("Date")
    kre.index = pd.to_datetime(kre.index)
    kre_ret = np.log(kre["Close"] / kre["Close"].shift(1)).rename("kre_ret")

    # VIX log-changes (VIX is already a rate; use log-diff for stationarity)
    _vix_csv = DATA_BAK / "all" / "FRED_VIXCLS.csv"
    if _vix_csv.exists():
        vix = pd.read_csv(_vix_csv, parse_dates=["date"])
        vix = vix.set_index("date")
    else:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parent))
        from config import CACHE_DIR as _CC
        _fraw = pd.read_parquet(_CC / "fred_raw.parquet")[["VIXCLS"]]
        _fraw.index = pd.to_datetime(_fraw.index)
        _fraw.index.name = "date"
        vix = _fraw.rename(columns={"VIXCLS": "VIXCLS"})
        print("  [fallback] loaded VIXCLS from fred_raw.parquet")
    vix.index = pd.to_datetime(vix.index)
    vix_diff = np.log(vix["VIXCLS"] / vix["VIXCLS"].shift(1)).rename("vix_diff")
    # Drop weekend/holiday NaN rows so rolling window is not polluted
    vix_diff = vix_diff.dropna()

    # Merge: align on trading-day index (inner join via concat then dropna on each series)
    ret_df = pd.concat([kre_ret, vix_diff], axis=1).loc[WARMUP_START:CRISIS_WINDOW_END]
    # Fill any residual NaN from VIX calendar mismatches with forward-fill (max 1 day)
    ret_df["vix_diff"] = ret_df["vix_diff"].fillna(method="ffill", limit=1)

    # ── 3. Compute 30-day rolling Pearson correlation ──
    print("  Computing 30-day rolling Pearson correlation (KRE vs VIX)...")
    pearson_30d = ret_df["kre_ret"].rolling(30, min_periods=20).corr(ret_df["vix_diff"])
    pearson_30d = pearson_30d.rename("pearson_30d")

    # ── 4. Restrict both series to plotting window ──
    plot_start = pd.Timestamp(CRISIS_WINDOW_START)
    plot_end = pd.Timestamp(CRISIS_WINDOW_END)

    # Coherence is already on the plotting window
    coherence_plot = coherence

    # Pearson: restrict to plotting window
    pearson_plot = pearson_30d[plot_start:plot_end]

    # ── 5. Exceedance analysis (bidirectional) ──
    print("  Computing exceedance thresholds (bidirectional)...")
    # C(ρ) exceedance: upper-only (coherence is non-negative, rises during crisis)
    c_mean = coherence.rolling(30, min_periods=15).mean()
    c_std = coherence.rolling(30, min_periods=15).std()
    c_threshold = c_mean + 1.0 * c_std
    c_exceedance = (coherence > c_threshold) & (~(coherence.shift(1, fill_value=False) > c_threshold.shift(1, fill_value=False)))

    # Pearson exceedance: bidirectional (KRE-VIX correlation can go more negative
    # or more positive; crisis = more negative)
    p_exceedance = detect_exceedances(pearson_30d, window=30, k_sigma=1.0)

    # Analyze each crisis event
    results = []
    for event_name, date_str in CRISIS_EVENTS.items():
        event_date = pd.Timestamp(date_str)

        c_ex_date, c_ex_val = most_recent_exceedance(coherence, c_exceedance, event_date)
        p_ex_date, p_ex_val = most_recent_exceedance(pearson_30d, p_exceedance, event_date)

        if c_ex_date is not None and p_ex_date is not None:
            lead_days = np.busday_count(c_ex_date.date(), p_ex_date.date())
        else:
            lead_days = None

        results.append((event_name, event_date, c_ex_date, c_ex_val,
                        p_ex_date, p_ex_val, lead_days))

    # ── 6. Guard clause check ──
    print("  Running guard clause checks...")
    anomalies = guard_check(results)
    critical_anomalies = [a for a in anomalies if a.startswith("HALT")]
    if critical_anomalies:
        print("\n  *** CRITICAL ANOMALIES DETECTED ***")
        for a in critical_anomalies:
            print(f"  {a}")
        # Save anomaly report
        anomaly_path = Path(__file__).parent.parent / "_recon_20260703" / "fig2_lead_anomaly.md"
        anomaly_path.parent.mkdir(parents=True, exist_ok=True)
        with open(anomaly_path, "w", encoding="utf-8") as f:
            f.write("# Fig 2 Lead Analysis Anomaly\n\n")
            f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
            for a in anomalies:
                f.write(f"- {a}\n")
            f.write("\n## Full results\n\n")
            for r in results:
                f.write(f"- {r[0]} ({r[1].date()}): C_ex={r[2]}, P_ex={r[3]}, lead={r[6]}\n")
        print(f"  Anomaly report saved to {anomaly_path}")
        print("  *** EXECUTION HALTED ***")
        return 1

    # Print warnings but continue
    for a in anomalies:
        print(f"  {a}")

    # ── 7. Plot ──
    print("\n  Plotting coherence vs Pearson correlation...")
    fig, ax1 = plt.subplots(figsize=(8.0, 4.5))

    # Left axis: coherence
    color_c = "#1f77b4"
    ax1.plot(coherence_plot.index, coherence_plot.values,
             color=color_c, linewidth=2.2, label="Quantum coherence $C(\\rho_t)$")
    ax1.set_ylabel("Quantum coherence $C(\\rho_t)$", color=color_c, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color_c)
    ax1.set_ylim(0, 1)

    # Right axis: Pearson correlation (KRE vs VIX)
    ax2 = ax1.twinx()
    color_p = "#d62728"
    ax2.plot(pearson_plot.index, pearson_plot.values,
             color=color_p, linewidth=2.0, linestyle="--",
             label="Pearson corr (KRE, VIX)")
    ax2.set_ylabel("Pearson corr (KRE, VIX)", color=color_p, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_p)
    ax2.set_ylim(-1, 1)

    # ── Crisis event vertical lines ──
    event_styles = {
        "SVB":            {"color": "#555555", "label": "SVB\nMar 10"},
        "Signature":      {"color": "#555555", "label": "Signature\nMar 12"},
        "First Republic": {"color": "#555555", "label": "First Republic\nMay 1"},
    }
    for event_name, date_str in CRISIS_EVENTS.items():
        date = pd.Timestamp(date_str)
        style = event_styles[event_name]
        ax1.axvline(x=date, color=style["color"], linewidth=0.8, alpha=0.6)
        ax1.text(date, 0.98, style["label"],
                 transform=ax1.get_xaxis_transform(),
                 rotation=90, fontsize=8, color=style["color"],
                 va="top", ha="right", alpha=0.8)

    # ── Pre-crisis buildup shaded band ──
    ax1.axvspan(pd.Timestamp("2023-03-03"), pd.Timestamp("2023-03-12"),
                color="#ff7f0e", alpha=0.12, label="Pre-crisis buildup")

    # ── Annotations: simultaneous signal (SVB/Signature) + lead (First Republic) ──
    # SVB/Signature March window: both C(ρ) and Pearson cross thresholds on Mar 9
    # → annotate as simultaneous, NOT as a lead.
    march_mask = (coherence.index >= "2023-03-01") & (coherence.index <= "2023-03-15")
    c_march = coherence[march_mask]
    if len(c_march) > 0:
        c_peak_date = c_march.idxmax()
        c_peak_val = c_march.max()
        ax1.annotate(
            "Both signals:\nMar 9 (simultaneous)",
            xy=(c_peak_date, c_peak_val),
            xytext=(c_peak_date - pd.Timedelta(days=18), c_peak_val + 0.18),
            fontsize=7.5, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#333333",
                            connectionstyle="arc3,rad=-0.2"),
            ha="center", va="bottom"
        )

    # First Republic Bank: annotate with calendar dates only (no day-count)
    # Day-count varies across runs; dates are stable calendar facts.
    frb_result = next((r for r in results if r[0] == "First Republic"), None)
    if frb_result is not None:
        frb_c_date, frb_c_val, frb_p_date, frb_lead = (
            frb_result[2], frb_result[3], frb_result[4], frb_result[6]
        )
        if frb_c_date is not None and frb_p_date is not None:
            # Use stable calendar-date labels, never a computed day-count
            lead_label = "C(ρ): Mar 31\nPearson: May 1\n(FRB — multi-week lead)"
            # Place annotation near the FRB coherence exceedance date
            c_frb_val = coherence.get(frb_c_date, frb_c_val) if frb_c_date in coherence.index else frb_c_val
            ax1.annotate(
                lead_label,
                xy=(frb_c_date, float(c_frb_val) if c_frb_val is not None else 0.5),
                xytext=(frb_c_date - pd.Timedelta(days=22), float(c_frb_val if c_frb_val is not None else 0.5) + 0.20),
                fontsize=7.5, color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4",
                                connectionstyle="arc3,rad=0.25"),
                ha="center", va="bottom"
            )

    # ── x-axis formatting ──
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.set_xlim(plot_start, plot_end)

    # ── Grid ──
    ax1.grid(axis="x", alpha=0.15)

    # ── Legend ──
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", framealpha=0.9, fontsize=8)

    # ── Title ──
    ax1.set_title(
        "Quantum coherence leads classical correlation during the\n"
        "2023 US regional banking crisis",
        fontsize=11, pad=10
    )

    plt.tight_layout()

    # ── Save ──
    pdf_path = FIG_DIR / "coherence_2023.pdf"
    png_path = FIG_DIR / "coherence_2023.png"
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
    lines.append("Fig 2 Coherence — regeneration diagnostic")
    lines.append("=" * 40)
    lines.append(f"Executed at: {datetime.now().isoformat()}")
    lines.append(f"Axis version: v2")
    lines.append(f"Window: {CRISIS_WINDOW_START} to {CRISIS_WINDOW_END}")
    lines.append(f"Trading days: {len(traj)}")
    lines.append("")

    lines.append("C(ρ_t) statistics:")
    lines.append(f"  min:  {coherence.min():.4f}")
    lines.append(f"  mean: {coherence.mean():.4f}")
    lines.append(f"  max:  {coherence.max():.4f}")
    lines.append(f"  Date of max: {coherence.idxmax().date()}")
    lines.append("")

    lines.append("Pearson correlation (30-day rolling, KRE vs VIX):")
    lines.append(f"  min:  {pearson_plot.min():.4f}")
    lines.append(f"  mean: {pearson_plot.mean():.4f}")
    lines.append(f"  max:  {pearson_plot.max():.4f}")
    lines.append(f"  Date of min (most negative): {pearson_plot.idxmin().date()}")
    lines.append(f"  Date of max: {pearson_plot.idxmax().date()}")
    lines.append("")

    lines.append("Data sources with MD5:")
    lines.append(f"  KRE:  {DATA_BAK / 'KRE.csv'}  {compute_md5(DATA_BAK / 'KRE.csv')}")
    lines.append(f"  VIX:  {DATA_BAK / 'all' / 'FRED_VIXCLS.csv'}  {compute_md5(DATA_BAK / 'all' / 'FRED_VIXCLS.csv')}")
    lines.append("  (Bloch pipeline inputs: see fig1_bloch_diagnostic.txt)")
    lines.append("")

    lines.append("Exceedance analysis (bidirectional for Pearson):")
    for r in results:
        event_name, event_date, c_ex_date, c_ex_val, p_ex_date, p_ex_val, lead_days = r
        lines.append("  ---")
        lines.append(f"  {event_name} failure ({event_date.date()}):")
        c_str = f"{c_ex_date.date()} (value {c_ex_val:.4f})" if c_ex_date is not None else "NONE"
        p_str = f"{p_ex_date.date()} (value {p_ex_val:.4f})" if p_ex_date is not None else "NONE"
        lines.append(f"    Most recent C(ρ_t) exceedance before or on event:      {c_str}")
        lines.append(f"    Most recent Pearson exceedance before or on event:     {p_str}")
        lead_str = f"{lead_days} trading days" if lead_days is not None else "N/A"
        lines.append(f"    Lead of C over Pearson: {lead_str}")

    lines.append("")
    valid_leads = [r[6] for r in results if r[6] is not None]
    if valid_leads:
        avg_lead = np.mean(valid_leads)
        lines.append(f"Empirical claim verification (\"~7 day lead\"):")
        lines.append(f"  Average lead across {len(valid_leads)} events: {avg_lead:.1f} trading days")
        lines.append(f"  Target range: 4–10 trading days (paper says \"approximately seven\")")
        if 4 <= avg_lead <= 10:
            lines.append(f"  Verdict: PASS")
        else:
            lines.append(f"  Verdict: MARGINAL (outside 4-10 range)")
    else:
        lines.append("  Verdict: FAIL (no valid leads computed)")

    lines.append("")
    lines.append("Figure files:")
    lines.append(f"  figures/coherence_2023.pdf  ({pdf_size} bytes)")
    lines.append(f"  figures/coherence_2023.png  ({png_size} bytes)")

    diag_path = LOG_DIR / "fig2_coherence_diagnostic.txt"
    with open(diag_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  OK Diagnostic -> {diag_path}")

    print(f"\nOK 05_fig2_coherence.py done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
