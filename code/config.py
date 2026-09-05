"""
Global configuration for all figure/table generation scripts.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent  # submission_MF_20260622/
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"
CACHE_DIR = Path(__file__).resolve().parent / "data_cache"

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Date range ─────────────────────────────────────────────────────────────
START_DATE = "2004-01-01"
END_DATE = "2024-12-31"

# ── Yahoo Finance tickers ──────────────────────────────────────────────────
YF_TICKERS = {
    "KRE": "KRE",
    "JPM": "JPM",
    "SPY": "SPY",
    "GSPC": "^GSPC",
    "VIX": "^VIX",
    "HYG": "HYG",
    "TLT": "TLT",
    "GC": "GC=F",
    "CL": "CL=F",
}

# ── FRED series ────────────────────────────────────────────────────────────
FRED_SERIES = {
    "VIXCLS": "VIXCLS",
    "TEDRATE": "TEDRATE",
    "HY_OAS": "BAMLH0A0HYM2",
    "STLFSI4": "STLFSI4",
    "DFF": "DFF",
    "DGS10": "DGS10",
    "DGS2": "DGS2",
    "WALCL": "WALCL",
    "BAA10Y": "BAA10Y",
}

# ── QST parameters ─────────────────────────────────────────────────────────
H_BAR_METHOD = "rolling_std"  # method to estimate ℏ_econ
ROLLING_WINDOW = 60
MIN_PERIODS = 20

# ── Crisis window (Fig 1 Bloch sphere) ─────────────────────────────────────
CRISIS_WINDOW_START = "2023-01-01"
CRISIS_WINDOW_END   = "2023-06-30"
CRISIS_EVENTS = {
    "SVB":            "2023-03-10",
    "Signature":      "2023-03-12",
    "First Republic": "2023-05-01",
}
DATA_DIR   = Path(__file__).parent.parent / "data"
FIG_DIR    = Path(__file__).parent.parent / "figures"
LOG_DIR    = Path(__file__).parent.parent / "logs"

# ── Reproducibility ────────────────────────────────────────────────────────
RANDOM_SEED = 42
