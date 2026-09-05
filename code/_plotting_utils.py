"""
Shared plotting utilities for Fig 1–4 and future figures.
Ensures byte-deterministic PDF outputs and unified matplotlib style.

Usage:
    # Must be called BEFORE any other matplotlib import
    from _plotting_utils import setup_matplotlib_deterministic
    setup_matplotlib_deterministic()

This sets SOURCE_DATE_EPOCH env var (read by matplotlib's PDF backend)
to produce reproducible /CreationDate and /ModDate PDF metadata.
"""
import os

# CRITICAL: set SOURCE_DATE_EPOCH BEFORE matplotlib is imported anywhere else
os.environ.setdefault("SOURCE_DATE_EPOCH", "1704067200")  # 2024-01-01 UTC

import matplotlib as mpl


def setup_matplotlib_deterministic():
    """
    Configure matplotlib for reproducible, publication-quality output.
    Call once at the top of every figure script.
    """
    mpl.rcParams["pdf.compression"] = 6
    mpl.rcParams["pdf.fonttype"] = 42          # embed TrueType (editable text)
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.hashsalt"] = "reproducible_package_v3"
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["axes.unicode_minus"] = False
