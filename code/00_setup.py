#!/usr/bin/env python3
"""
00_setup.py — Environment readiness check for reproducibility_MF_20260622.

Checks:
  1. Python version >= 3.10
  2. All packages in environment/requirements.txt are installed (version warning only)
  3. Critical data files exist in data/raw/

Exit code: 0 if all strong checks pass, 1 otherwise.
"""
import sys
import os
import glob
import importlib.metadata
import warnings

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

REQUIREMENTS_FILE = os.path.join(REPO_ROOT, "environment", "requirements.txt")
DATA_RAW = os.path.join(REPO_ROOT, "data", "raw")
MANIFEST_FILE = os.path.join(REPO_ROOT, "MANIFEST.txt")

errors = []
warnings_list = []

# ── 1. Python version check ──────────────────────────────────────────────
print("[1/3] Checking Python version ...")
py_ver = sys.version_info
if py_ver.major < 3 or (py_ver.major == 3 and py_ver.minor < 10):
    errors.append(
        f"Python >= 3.10 required, got {py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    )
else:
    print(f"       Python {py_ver.major}.{py_ver.minor}.{py_ver.micro} — OK")

# ── 2. Package dependency check ──────────────────────────────────────────
print("[2/3] Checking Python packages ...")
if not os.path.exists(REQUIREMENTS_FILE):
    warnings_list.append(f"requirements.txt not found at {REQUIREMENTS_FILE}")
else:
    with open(REQUIREMENTS_FILE, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    for line in lines:
        # Parse package name (ignore version specifiers for now)
        pkg = line.split(">=")[0].split("==")[0].split("<")[0].strip()
        try:
            installed_ver = importlib.metadata.version(pkg)
            # Check if version constraint is satisfied (>=)
            if ">=" in line:
                req_ver = line.split(">=")[1].strip()
                # Simple tuple comparison
                installed_tuple = tuple(int(x) for x in installed_ver.split(".")[:2])
                req_tuple = tuple(int(x) for x in req_ver.split(".")[:2])
                if installed_tuple < req_tuple:
                    warnings_list.append(
                        f"  {pkg}: installed {installed_ver}, requires >= {req_ver}"
                    )
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"  {pkg}: NOT INSTALLED")

    if not errors:
        print("       All required packages installed — OK")
    else:
        for e in errors:
            print(f"       ERROR: {e}")

# ── 3. Data file check ───────────────────────────────────────────────────
print("[3/3] Checking data files ...")
if not os.path.exists(DATA_RAW):
    errors.append(f"data/raw/ directory not found at {DATA_RAW}")
else:
    csv_files = glob.glob(os.path.join(DATA_RAW, "**", "*.csv"), recursive=True)
    json_files = glob.glob(os.path.join(DATA_RAW, "**", "*.json"), recursive=True)
    jsonl_files = glob.glob(os.path.join(DATA_RAW, "**", "*.jsonl"), recursive=True)
    total_data_files = len(csv_files) + len(json_files) + len(jsonl_files)

    # Try to read expected count from MANIFEST
    expected = None
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("data_raw_file_count:"):
                    expected = int(line.split(":")[1].strip())
                    break

    if expected is not None:
        if total_data_files == expected:
            print(f"       Found {total_data_files} data files (expected {expected}) — OK")
        else:
            warnings_list.append(
                f"  Data file count mismatch: found {total_data_files}, expected {expected}"
            )
    else:
        print(f"       Found {total_data_files} data files (no MANIFEST baseline to compare)")

# ── Summary ──────────────────────────────────────────────────────────────
print()
if errors:
    for e in errors:
        print(f"  [FAIL] {e}")
    print()
    print("[FAIL] Environment NOT ready. Fix errors above and re-run.")
    sys.exit(1)
else:
    if warnings_list:
        print("  Warnings:")
        for w in warnings_list:
            print(f"    {w}")
        print()
    print("[OK] Environment ready.")
    sys.exit(0)
