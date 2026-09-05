# Data Provenance — A Theory of Quantum Option Pricing and Empirical Tests

**Version:** v6.0  
**Paper sample window:** 2004-10-04 to 2024-12-31 (5,096 trading days)  
**Note:** Config sets START_DATE = "2004-01-01" but the effective start is 2004-10-04,
determined by the intersection of VIXCLS and STLFSI4 availability.

---

## 1. Equity Prices and ETFs (Yahoo Finance)

| Series | Ticker | Source | Period in dataset | License |
|--------|--------|--------|-------------------|---------|
| KRE (Regional Banking ETF) | KRE | Yahoo Finance via `yfinance` | 2006-06-22 – 2024-12-31 | Yahoo TOS (non-commercial) |
| JPMorgan Chase | JPM | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| Bank of America | BAC | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| Citigroup | C | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| Goldman Sachs | GS | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| Morgan Stanley | MS | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| Wells Fargo | WFC | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| KBE (S&P Bank ETF) | KBE | Yahoo Finance | 2005-11-08 – 2024-12-31 | Yahoo TOS |
| XLF (Financial Select) | XLF | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| S&P 500 Index | ^GSPC | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| VIX | ^VIX | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |
| SPY | SPY | Yahoo Finance | 2004-10-04 – 2024-12-31 | Yahoo TOS |

**Download method:** `code/01_fetch_yahoo.py` using `yfinance` library.  
**Adjusted prices:** Adjusted Close (split/dividend adjusted).

---

## 2. Macro-Financial Indicators (FRED)

| Series | FRED ID | Description | Period in dataset | Notes |
|--------|---------|-------------|-------------------|-------|
| VIX (close) | VIXCLS | CBOE Volatility Index | 2004-10-01 – 2024-12-31 | Public domain |
| TED Spread | TEDRATE | 3-month LIBOR – 3-month T-bill | 2004-10-04 – **2022-01-21** | FRED discontinued 2022-01-22 |
| TED Splice | — | (BAMLH0A0HYM2 − DGS10) 5-day smoothed | 2022-01-22 – 2024-12-31 | Constructed proxy; see §5.1 |
| HY OAS | BAMLH0A0HYM2 | ICE BofA US HY Master II OAS | **2023-05-22** – 2024-12-31 | Pre-2023: BAA10Y proxy |
| HY proxy | BAA10Y | Moody's BAA–10Y Treasury spread | 2004-10-04 – 2024-12-31 | Used when BAMLH0A0HYM2 unavailable |
| STLFSI4 | STLFSI4 | St. Louis Fed Financial Stress Index (weekly) | 2004-10-04 – 2024-12-31 | Public domain |
| DFF | DFF | Effective Federal Funds Rate (daily) | 2004-10-04 – 2024-12-31 | Public domain |
| DGS10 | DGS10 | 10-Year Treasury Constant Maturity | 2004-10-04 – 2024-12-31 | Public domain |
| DGS2 | DGS2 | 2-Year Treasury Constant Maturity | 2004-10-04 – 2024-12-31 | Public domain |
| WALCL | WALCL | Fed Balance Sheet (weekly) | 2004-10-04 – 2024-12-31 | Public domain |
| BAA10Y | BAA10Y | Moody's BAA–10Y spread | 2004-10-04 – 2024-12-31 | Public domain |

**Download method:** `code/02_fetch_fred.py` using `fredapi` library.  
**API key required:** Set `FRED_API_KEY` in `.env` file.  
**All FRED data:** Public domain (Federal Reserve Bank of St. Louis).

---

## 3. FDIC BankFind Data

| File | Source | Description |
|------|--------|-------------|
| `failures_2023.csv` | FDIC BankFind Suite API | 2023 bank failure records (SVB cert=59017, Signature cert=57053, FRB cert=24735) |
| `institutions_snapshot.csv` | FDIC BankFind Suite | Institution characteristics snapshot |
| `financials_*_*.csv` | FDIC BankFind Suite | Quarterly call report financials for the three failed banks |

**Source URL:** https://banks.data.fdic.gov/api/  
**License:** Public domain (FDIC).

---

## 4. Option Data — IMPORTANT CAVEAT

**The historical KRE options chain for 2023-03-09/10 is NOT publicly available.**

The implied volatility smile calibration in §5 (Robustness) and Online Supplement §S.2.3
uses a **canonical equity-crisis-regime parametric smile**:

```
σ_ref(K, τ) = σ_atm(τ) + a·m + b·m²,   m = ln(K/S₀)
```

with parameters:
- `a = -0.50` (skew): representative of banking-sector equity during crisis
- `b = +1.00` (curvature): representative of equity smirk
- `σ_atm(60d) = 28.41%`: computed from **realized** KRE log-returns, window 2022-12-12 to 2023-03-09

The file `data/raw/expanded/options/KRE_chain_*.csv` contains **2026-08-30 KRE option chains**
(downloaded during package construction for illustrative/robustness purposes).
These files are **not used** in the main paper results.

The `ℏ_econ = 0.087` calibration is derived from the Amihud illiquidity panel
(Tier 1 equity data above), not from the option chain.

---

## 5. Data Snapshot Policy

The `data/raw/` CSV files in this package were downloaded on **2026-09-04** and
represent the state of the data sources at that date. Researchers re-running
`code/01_fetch_yahoo.py` and `code/02_fetch_fred.py` will obtain more recent data
and may observe slightly different values beyond 2024-12-31.

To reproduce the exact paper results, use the cached data in `code/data_cache/*.parquet`
(which were generated from the 2026-09-04 snapshot and truncated to 2024-12-31 per config).

---

## 6. Data Citations

- Yahoo Finance: Data provided by Yahoo Inc. via the `yfinance` open-source library.
  Non-commercial research use only.
- FRED: Federal Reserve Bank of St. Louis. https://fred.stlouisfed.org/. Public domain.
- FDIC BankFind: Federal Deposit Insurance Corporation. https://banks.data.fdic.gov/.
  Public domain.
