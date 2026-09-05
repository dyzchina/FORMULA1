# CHANGELOG — reproducible_package_v6.0

## Version 6.0 (2026-09-04) — Final pre-submission consistency pass

### Summary

Nine targeted fixes applied across paper sections and tables to ensure every
hardcoded number and every policy-implication statement is consistent with the
verified pipeline output (15/16 OK). The single remaining FAIL is a known
data-availability limitation (FRED TED spread discontinued 2022-01-22).

Cold-start design: removed `code/data_cache/` from the distributed package.
Users on fresh machines must run `01_fetch_yahoo.py` and `02_fetch_fred.py`
before invoking `run_all.py`.

---

### File-by-file changes

#### paper/sections/07-discussion.tex — CRITICAL

**Change:** Changed "21-trading-day pre-emptive window" to
"37-trading-day pre-emptive window" in the policy implications paragraph
(line 53).

**Reason:** The pipeline value `frb_lead_days=37` is the authoritative result.
The value 21 was a stale draft figure. The introduction (line 138–139),
section 5 (line 191), and the conclusion's own summary paragraph (line 43)
all already stated 37 trading days; this fix completes consistency across the
manuscript.

---

#### paper/sections/05-empirical-main.tex — MAJOR (two changes)

**Change 1:** Updated the lower bound of the absolute correlation range in
the Table 3 narration paragraph (line 213) from 0.51 to 0.55.
Specifically, `$|	ext{corr}| \in [0.51, 0.68]$` was changed to
`$|	ext{corr}| \in [0.55, 0.68]$`.

**Reason:** The minimum absolute value in Table 3 is |0.55| (S&P 500),
not 0.51. The value 0.51 did not correspond to any entry in the table.

**Change 2:** Added explicit rounding qualifiers to the theta mean and AR(1)
summary in the QST calibration paragraph (lines 121–122).
Changed "mean of 0.37 with strong persistence (AR(1) $= 0.76$)" to
"mean of $pprox 0.37$ (0.3677) with strong persistence
(AR(1) $pprox 0.76$ (0.7579))".

**Reason:** Full-precision values 0.3677 and 0.7579 are reported in Table 2;
the in-text rounded values now carry explicit approximately-equal qualifiers
and parenthetical full-precision values to prevent ambiguity.

**Change 3 (consistency):** Updated the cross-crisis comparison paragraph
peak values from rounded to full-precision pipeline values:
- GFC: 0.999 → 0.9993
- COVID: 0.996 → 0.9955
- Banking 2023: 0.869 → 0.8687

---

#### tables/table2.tex — MINOR

**Change:** Replaced the circular self-referential note in the `hbar_econ` row.
Old note: "scaled so that sample mean equals 0.062 (canonical value from
paper §5.1)". New note: "(scaling constant k chosen so that the full-sample
Amihud-based mean matches the range [0.01, 0.15] documented in
Remark~ef{rem:units})".

**Reason:** The old note referenced the section that contains the table itself,
creating a circular citation. The new note provides the substantive
justification without self-reference.

---

#### tables/table3.tex — MINOR

**Change:** Added footnote marker `$^{*}$` to the TED Spread row and added a
corresponding footnote at the bottom of the table:
"$^{*}$ FRED discontinued the TED spread series on 2022-01-22; no
observations fall within the 2023 crisis window after splicing to
SOFR$-$EFFR."

**Reason:** Parallel treatment to the existing dagger and double-dagger
footnotes for HY-OAS and STLFSI4. Reviewers need to know why the TED Spread
entry has no 2023 banking crisis observation.

---

#### paper/sections/S2-additional-empirical.tex — MAJOR

**Change 1:** Updated Figure S.1 caption peak values from rounded to
full-precision pipeline values:
- GFC: 0.999 → 0.9993
- COVID: 0.996 → 0.9955
- Banking: 0.869 → 0.8687

**Change 2:** Updated in-text reference to banking crisis peak from 0.869
to 0.8687.

**Reason:** Figure captions must match the figure values exactly; full-
precision values are traceable directly to pipeline output.

---

#### paper/sections/06-additional-results.tex — CONSISTENCY

**Change:** Updated crisis indicator comparison paragraph peak values to
full precision matching pipeline:
- 0.999 → 0.9993
- 0.996 → 0.9955
- 0.869 → 0.8687

**Reason:** Every section of the paper should report the same pipeline-
verified figures.

---

#### paper/sections/S3-extended-policy.tex — STALE VALUE CLEANUP

**Change 1:** Changed "21-trading-day lead" to "37-trading-day lead"
(line 42).

**Change 2:** Changed "lead extends to 20 trading days" to
"lead extends to 37 trading days" (line 77).

**Reason:** Both were stale draft values left over from an earlier calibration
run. The authoritative pipeline value is `frb_lead_days=37`. These were the
last two remaining occurrences of the incorrect lead value in the manuscript.

---

### Cold-start package change

**Removed:** `code/data_cache/` directory.

**Reason:** Cold-start reproducibility requires users to run `01_fetch_yahoo.py`
and `02_fetch_fred.py` to regenerate this cache from scratch. Shipping a
pre-built cache hides potential data-version dependencies and inflates the
package size. The `data/raw/` and `data/derived/` directories are retained and
are sufficient to run all 16 analysis scripts without re-downloading.

---

## Version 5.0 (2026-09-04) — Previous release

See the v5.0 package for the corresponding CHANGELOG and README.
