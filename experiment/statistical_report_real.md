# Statistical Report — Real Embedder Retrieval Experiment

**Date:** 2026-07-09

## Experiment Overview

- Total queries: 1000
- Embedding model: BAAI/bge-small-zh-v1.5
- Bootstrap resamples: 10000
- Confidence level: 95% (α = 0.05)

## Overall Results (All 1000 Queries)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| MRR | 0.560658 | 0.704542 |
| 95% CI | [0.533510, 0.587670] | [0.680477, 0.728261] |
| Not Recalled (rank=-1) | 248 / 1000 | 138 / 1000 |

**Delta (Hybrid − Baseline) = 0.143885**
- 95% CI: [0.127451, 0.161199]
- Significant at α=0.05: **YES** (CI does not cross zero)

## Stratified Analysis

### Exact（精确查询） (500 Queries)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| MRR | 0.891674 | 0.936852 |
| 95% CI | [0.870099, 0.912194] | [0.920971, 0.952000] |
| Not Recalled | 2 / 500 | 0 / 500 |

**Delta (Hybrid − Baseline) = 0.045179**
- 95% CI: [0.028761, 0.061892]
- Significant at α=0.05: **YES** (CI does not cross zero)

### Fuzzy（模糊查询） (500 Queries)

| Metric | Baseline | Hybrid |
|--------|----------|--------|
| MRR | 0.229642 | 0.472233 |
| 95% CI | [0.202218, 0.259227] | [0.437140, 0.508027] |
| Not Recalled | 246 / 500 | 138 / 500 |

**Delta (Hybrid − Baseline) = 0.242590**
- 95% CI: [0.215628, 0.270106]
- Significant at α=0.05: **YES** (CI does not cross zero)

## Summary

| Subset | N | Baseline MRR | Hybrid MRR | Delta | 95% CI (Delta) | Significant? |
|--------|---|---|---|------|-----------------|-------------|
| Overall (All 1000) | 1000 | 0.560658 | 0.704542 | 0.143885 | [0.127451, 0.161199] | ✓ |
| Exact（精确查询） | 500 | 0.891674 | 0.936852 | 0.045179 | [0.028761, 0.061892] | ✓ |
| Fuzzy（模糊查询） | 500 | 0.229642 | 0.472233 | 0.242590 | [0.215628, 0.270106] | ✓ |

---

_Generated automatically by analyze_real.py_