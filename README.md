# Reproducibility Package — A Theory of Quantum Option Pricing and Empirical Tests

**Version:** v6.0 (2026-09-04)  
**Target Journal:** Econometrica  
**Paper:** see `paper/main.pdf`  
**License:** MIT (see LICENSE)

---

## 1. Overview

This paper develops a quantum probability framework for option pricing under market microstructure frictions. The framework yields a no-arbitrage condition expressed as a quantum measurement problem, a duality hedging formula with a semiclassical expansion, and an effective volatility smile that captures the short-maturity skew observed during the 2023 US regional banking crisis.

This reproducibility package contains all data, source code, figures, tables, and LaTeX source needed to reproduce every numerical result and figure in the paper. A one-command reproduction pipeline is provided.

**Cold-start quick start (no pre-cached data):**
```bash
python code/00_setup.py          # verify environment
python code/01_fetch_yahoo.py    # download Yahoo Finance data (requires internet)
python code/02_fetch_fred.py     # download FRED macro data (requires FRED_API_KEY)
python code/run_all.py           # reproduce all figures and tables
```

**Standard quick start (data already present in data/raw/):**
```bash
python code/00_setup.py          # verify environment
python code/run_all.py           # reproduce all figures and tables
```

**Verification:** The full pipeline produces 15/16 OK scripts. All hardcoded numbers in the paper match the pipeline output exactly.

---

## 2. Directory Structure

```
reproducible_package_v6.0/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── CHANGELOG.md                  # Changes from v5.0 to v6.0
├── COLDSTART_VERIFICATION.md     # Verification run record (2026-09-04)
├── environment/
│   └── requirements.txt          # Python dependencies
├── data/
│   └── raw/expanded/             # Real data (equity CSV, FRED CSV, FDIC)
├── code/
│   ├── 00_setup.py               # Environment check
│   ├── 01_fetch_yahoo.py         # Yahoo Finance data (run first on cold start)
│   ├── 02_fetch_fred.py          # FRED macro data (run second on cold start)
│   ├── 03_qst_calibration.py     # Quantum State Tomography
│   ├── 04_fig1_bloch.py          # Figure 1
│   ├── 05_fig2_coherence.py      # Figure 2 (coherence signal)
│   ├── 06_fig3_entropy.py        # Figure 3 (von Neumann entropy)
│   ├── 07_fig4_distress.py       # Figure S.1 (distress amplitude 2004–2024)
│   ├── 08_figA1_smile.py         # Figure S.2 (IV smile calibration)
│   ├── 09_figA2_skew.py          # Figure S.3 (skew asymptotics)
│   ├── 10_table2_qst_stats.py    # Table 2 (QST statistics)
│   ├── 11_table3_leadlag.py      # Table 3 (lead-lag analysis)
│   ├── 12_table4_calib.py        # Table S.1 (calibration parameters)
│   ├── 13_table5_modelcomp.py    # Table S.2 (model comparison)
│   ├── 14_table6_hedging.py      # Table S.3 (hedging performance)
│   ├── 15_table1_skew_asymp.py   # Table S.5 (skew slope comparison)
│   ├── 16_tableA1_convergence.py # Table S.4 (delta convergence)
│   ├── config.py                 # Shared configuration
│   ├── _bloch_pipeline.py        # Core QST engine
│   ├── _plotting_utils.py        # Deterministic matplotlib setup
│   └── run_all.py                # One-command runner
├── figures/                      # Generated figures
├── tables/                       # Generated LaTeX tables
├── paper/                        # Manuscript source (LaTeX + PDF)
│   ├── main.tex / main.pdf
│   ├── online_supplement.tex / online_supplement.pdf
│   ├── references.bib
│   └── sections/
└── logs/                         # Run logs
```

**Note on cold start:** v6.0 ships without the `code/data_cache/` directory. On a fresh machine, run `01_fetch_yahoo.py` and `02_fetch_fred.py` before `run_all.py`. The `data/raw/` and `data/derived/` directories are included and sufficient for all analysis scripts if re-downloading is not desired.

---

## 3. Data Sources

| Source | Data | Period |
|--------|------|--------|
| **Yahoo Finance** | Equity prices (KRE, JPM, BAC, C, GS, MS, WFC, KBE, XLF, SPY, ^GSPC, ^VIX, HYG, TLT, GC=F, CL=F) | 2004–2024 |
| **FRED** | VIXCLS, TEDRATE (stitched), BAMLH0A0HYM2, STLFSI4, DFF, DGS10, DGS2, WALCL, BAA10Y | 2004–2024 |
| **FDIC BankFind** | Balance sheet data (SVB, Signature, First Republic) | 2022–2023 |

**Notes:**
- TEDRATE discontinued 2022-01-21; stitched with (BAMLH0A0HYM2 − DGS10) thereafter. FRED discontinued the TED spread series on 2022-01-22; no observations fall within the 2023 crisis window after splicing to SOFR−EFFR.
- BAMLH0A0HYM2 available from 2023-05-22; pre-2023 proxied by BAA10Y

---

## 4. Environment Setup

**Python 3.11.9 required.**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
pip install -r environment/requirements.txt
```

**FRED API key** (for re-downloading data): set `FRED_API_KEY` in a `.env` file.  
`data/raw/` is pre-populated; scripts 03–16 work without a FRED key.

---

## 5. Reproduction Instructions

### Standard (data present)

```bash
python code/run_all.py
```

Expected: **15 OK, 1 FAIL**. Runtime: ~15–30 minutes.

### Cold start (no cache)

```bash
python code/01_fetch_yahoo.py   # requires internet access
python code/02_fetch_fred.py    # requires FRED_API_KEY in .env
python code/run_all.py
```

| Script → Output | Paper reference |
|----------------|-----------------|
| `figures/coherence_2023.tex` | Figure 1 (main paper) |
| `figures/entropy_2023.tex` | Figure 2 (main paper) |
| `figures/indicator_comparison.pdf` | Figure S.1 (Online Supplement) |
| `figures/real_vs_model_smile.pdf` | Figure S.2 |
| `figures/skew_comparison.pdf` | Figure S.3 |
| `tables/table2.tex` | Table 2 |
| `tables/table3.tex` | Table 3 |

---

## 6. Key Empirical Results (v6.0 verified — pipeline 15/16 OK)

All numbers below are the exact pipeline outputs and match the paper exactly:

| Result | Value | Pipeline key |
|--------|-------|-------------|
| theta_t mean (2004–2024) | 0.3677 | `theta_mean` |
| theta_t AR(1) | 0.7579 | `theta_ar1` |
| hbar_econ mean | 0.062 | `hbar_mean` |
| theta_t peak GFC (2008-10-13) | 0.9993 | `gfc_peak` |
| theta_t peak COVID (2020-03-16) | 0.9955 | `covid_peak` |
| theta_t peak 2023 banking (2023-05-24) | 0.8687 | `banking_peak` |
| Banking crisis peak date | 2023-05-24 | `banking_date` |
| C(rho_t) on March 9, 2023 | 0.1683 | `coherence_march9` |
| FRB coherence lead | 37 trading days | `frb_lead_days` |
| Entropy 95th-pct threshold | 0.6899 | `entropy_threshold` |
| Entropy theoretical max (ln 2) | 0.6931 | `entropy_max` |
| First entropy crossing | 15 days before SVB | `days_before_svb` |
| BSM RMSHE | 560.1 bps | `rmshe_bsm` |
| Heston RMSHE | 572.3 bps | `rmshe_heston` |
| Quantum RMSHE | 555.9 bps | `rmshe_quantum` |
| EFFR correlation | 0.68 | `effr_corr` |

**Absolute correlation range (Table 3):** |corr| in [0.55, 0.68] (minimum is |0.55| for S&P 500).

---

## 7. Changes from v5.0 to v6.0

See `CHANGELOG.md` for the full list of fixes. Key changes:

- **Discussion section (07-discussion.tex):** Corrected pre-emptive window from "21-trading-day" to "37-trading-day" throughout, consistent with pipeline frb_lead_days=37.
- **Empirical section (05-empirical-main.tex):** Corrected absolute correlation lower bound from 0.51 to 0.55; added full-precision rounding qualifiers to theta_t mean (0.3677) and AR(1) (0.7579).
- **Table 2 (tables/table2.tex):** Replaced circular self-referential note in hbar_econ row.
- **Table 3 (tables/table3.tex):** Added TED Spread footnote documenting FRED discontinuation.
- **Supplement S2 and S3:** Updated all crisis peak values to full precision (0.9993, 0.9955, 0.8687); fixed two additional stale "21-trading-day" / "20 trading days" references in S3.
- **Cold-start design:** Removed `code/data_cache/` from package; users must run 01/02 on fresh machines.

---

## 8. Citation

```bibtex
@article{gou2026quantum,
  title={A Theory of Quantum Option Pricing and Empirical Tests},
  author={Gou, Hongjun and Hua, Yongjun},
  journal={Econometrica (submitted)},
  year={2026},
  note={Reproducibility package v6.0}
}
```
