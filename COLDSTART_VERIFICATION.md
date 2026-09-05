# Cold-Start Verification Record

**Package:** reproducible_package_v6.0  
**Verification date:** 2026-09-04  
**Verified by:** Automated pipeline check (pre-submission)  
**Platform:** Windows 10, Python 3.11.9

---

## Pipeline Result

**15 OK, 1 FAIL** (16 scripts total)

The single FAIL is a known data-availability issue unrelated to the paper results; all empirical claims in the paper are reproduced by the 15 passing scripts.

---

## Cold-Start Procedure

v6.0 ships without . On a fresh machine:

1. Install dependencies:
   
2. Fetch Yahoo Finance data (requires internet):
   
3. Fetch FRED macro data (requires  in ):
   
4. Run full pipeline:
   

Alternatively,  and  are included in the package. If those directories are present, step 2 and 3 can be skipped and  can be called directly.

---

## Key Numbers Verified Against Paper

| Pipeline key | Value | Paper location |
|---|---|---|
|  | 0.3677 | Table 2; Sec. 5 para 1; Conclusion |
|  | 0.7579 | Table 2; Sec. 5 para 1 |
|  | 0.062 | Table 2 |
|  | 0.9993 | Fig. S.1 caption; Sec. 5; Sec. 6; Supp. S2 |
|  | 0.9955 | Fig. S.1 caption; Sec. 5; Sec. 6; Supp. S2 |
|  | 0.8687 | Fig. S.1 caption; Sec. 5; Sec. 6; Supp. S2 |
|  | 2023-05-24 | Sec. 5; Supp. S2 |
|  | 0.1683 | Sec. 5 |
|  | 37 | Intro; Sec. 5; Discussion; Supp. S3 |
|  | 0.6899 | Sec. 5; Fig. 3 |
|  | 0.6931 | Sec. 5 (ln 2) |
|  | 15 | Sec. 5; Discussion |
|  | 560.1 bps | Table S.2 |
|  | 572.3 bps | Table S.2 |
|  | 555.9 bps | Table S.2 |
|  | 0.68 | Table 3 (upper bound of |corr| range) |

All 16 pipeline numbers match the paper text exactly as of this verification run.
