#!/usr/bin/env python3
"""
run_all.py — Execute all numbered reproduction scripts sequentially.

Workflow:
  1. Change working directory to repo root.
  2. Run all code/0[1-9]_*.py and code/1[0-4]_*.py scripts in sorted order.
  3. Log stdout+stderr to logs/run_<YYYYMMDD_HHMMSS>.log.
  4. Print summary table at the end.

Usage:
    python code/run_all.py
"""
import os
import sys
import subprocess
import time
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)

LOG_DIR = os.path.join(REPO_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(LOG_DIR, f"run_{TIMESTAMP}.log")

# Scripts to run: numbered 01-14, sorted by filename
CODE_DIR = os.path.join(REPO_ROOT, "code")
SKIP_SCRIPTS = {"00_setup.py", "config.py", "verify_fred_csv.py", "run_all.py"}

scripts = sorted(
    f for f in os.listdir(CODE_DIR)
    if f.endswith(".py")
    and f not in SKIP_SCRIPTS
    and (f.startswith("0") or f.startswith("1"))
)

print(f"Reproducibility Run — {TIMESTAMP}")
print(f"Log file: {LOG_FILE}")
print(f"Scripts to execute: {len(scripts)}\n")

results = []
total_start = time.time()

with open(LOG_FILE, "w", encoding="utf-8") as log:
    log.write(f"Reproducibility Run — {TIMESTAMP}\n")
    log.write(f"{'='*70}\n\n")

    for i, script in enumerate(scripts, 1):
        script_path = os.path.join(CODE_DIR, script)
        print(f"[{i}/{len(scripts)}] Running {script} ...")

        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=3600,  # 1 hour per script
            )
            elapsed = time.time() - start
            duration_str = f"{elapsed:.1f}s"

            if proc.returncode == 0:
                status = "OK"
            else:
                status = f"FAIL (rc={proc.returncode})"

            # Write to log
            log.write(f"\n{'─'*70}\n")
            log.write(f"Script: {script}  |  Status: {status}  |  Duration: {duration_str}\n")
            log.write(f"{'─'*70}\n")
            if proc.stdout:
                log.write(f"[stdout]\n{proc.stdout}\n")
            if proc.stderr:
                log.write(f"[stderr]\n{proc.stderr}\n")

            results.append((script, status, duration_str))
            print(f"       → {status} ({duration_str})")

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start
            duration_str = f"{elapsed:.1f}s"
            status = "TIMEOUT"
            log.write(f"\nScript: {script}  |  Status: TIMEOUT after {duration_str}\n")
            results.append((script, status, duration_str))
            print(f"       → TIMEOUT ({duration_str})")

        except Exception as e:
            elapsed = time.time() - start
            duration_str = f"{elapsed:.1f}s"
            status = f"ERROR ({str(e)[:60]})"
            log.write(f"\nScript: {script}  |  Status: {status}\n")
            results.append((script, status, duration_str))
            print(f"       → {status}")

total_elapsed = time.time() - total_start

# ── Summary table ────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"SUMMARY")
print(f"{'='*70}")
print(f"{'Script':30s} {'Status':20s} {'Duration':10s}")
print(f"{'─'*30} {'─'*20} {'─'*10}")
ok_count = 0
fail_count = 0
for script, status, duration in results:
    print(f"{script:30s} {status:20s} {duration:10s}")
    if status == "OK":
        ok_count += 1
    else:
        fail_count += 1
print(f"{'─'*30} {'─'*20} {'─'*10}")
print(f"{'Total':30s} {f'{ok_count} OK, {fail_count} FAIL':20s} {total_elapsed:.1f}s")

# Write summary to log
with open(LOG_FILE, "a", encoding="utf-8") as log:
    log.write(f"\n{'='*70}\n")
    log.write(f"SUMMARY: {ok_count} OK, {fail_count} FAIL, total {total_elapsed:.1f}s\n")
    log.write(f"{'='*70}\n")

print(f"\nLog saved to: {LOG_FILE}")
print()
print("Next step: Compare generated figures/ and tables/ with paper/main.pdf")
print("  e.g.:  figures/bloch_trajectory.pdf  →  Figure 1 in paper")
print("         figures/coherence_2023.tex    →  Figure 2 in paper")
print("         tables/table2.tex             →  Table 2 in paper")
print()

if fail_count > 0:
    sys.exit(1)
else:
    sys.exit(0)
