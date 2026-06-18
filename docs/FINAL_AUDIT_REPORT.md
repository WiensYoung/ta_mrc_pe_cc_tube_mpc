# SCI Q1 90+ Readiness — Final Audit Report

**Date**: 2026-06-04
**Project**: TA-MRC-PE-CC-Tube-MPC
**Auditor**: Automated audit per SCI Q1 90+ hardening spec

---

## Overall Readiness Assessment

| Dimension | Status |
|-----------|--------|
| **Code infrastructure readiness** | 90+ — P0/P1 bugs fixed, 96 new tests pass |
| **Experimental evidence readiness** | ~60 — experiments not yet executed |
| **Paper submission readiness** | NOT READY — full experiments, real data, baselines needed |

**Current score**: 80/100 (capped at 80 by real AIS/ENC gate)
**SCI Q1 submission ready**: NO

---

## 1. P0 Fixes — ALL COMPLETED

### 1.1 IMM Behavior Filter
- **Fixed**: Per-mode predictions now differ across CV, CT, COLREGS, NON_COMPLIANT, CHANNEL, ERRATIC modes
- **Per-target state**: Each target has independent IMM state (no cross-contamination)
- **Timing fix**: Prediction and observation properly time-aligned
- **16 tests pass** in test_imm_behavior.py

### 1.2 Collision Geometry
- **Fixed**: Replaced isotropic centroid-distance threshold with SAT-based oriented bounding-box detection
- **Safety padding**: Separates collision, near-miss, and ship-domain violation
- **17 tests pass** in test_collision_geometry.py

### 1.3 Unified Safety Evaluator
- **Created**: EpisodeSafetyMetrics with consistent fields for all methods
- **All fields finite**: No NaN/Inf leakage into main metrics table
- **9 tests pass** in test_safety_evaluator.py

---

## 2. P1 Fixes — ALL COMPLETED

| Item | Status | Tests |
|------|--------|-------|
| Chance constraint risk allocation | Done | 9 pass |
| Tube MPC claim alignment | Documented | docs/theory_claim_boundaries.md |
| CasADi surrogate validation | Script done | validate_surrogate_vs_mmg.py |
| NaN/Inf diagnostics | Done | 7 pass |
| Experiment entrypoints | Infrastructure done | 7 pass |
| Real AIS/ENC replay | Script done | 7 pass |
| Strong baselines | Registry done | 8 pass |
| Statistics pipeline | Infrastructure done | 6 pass |
| Failure audit (15 categories) | Extended | 5 pass |
| Documentation alignment | Done | — |
| 90+ readiness gate | Script done | — |

---

## 3. Files Modified

-  — Full rewrite with per-target, per-mode distinct predictions
-  — Per-target IMM observation API
-  — SAT collision detection
-  — Extended to 15 failure categories

## 4. Files Created (19 new files)

| File | Purpose |
|------|---------|
| evaluation/geometry.py | Oriented box SAT collision detection |
| evaluation/safety_evaluator.py | Unified safety metrics |
| control/risk_allocation.py | Boole risk allocation |
| tests/test_imm_behavior.py | 16 IMM tests |
| tests/test_collision_geometry.py | 17 geometry tests |
| tests/test_safety_evaluator.py | 9 evaluator tests |
| tests/test_metrics_unified_across_methods.py | 3 consistency tests |
| tests/test_chance_risk_allocation.py | 9 allocation tests |
| tests/test_nonfinite_diagnostics.py | 7 NaN/Inf tests |
| tests/test_experiment_smoke_entrypoints.py | 7 entrypoint tests |
| tests/test_external_baselines.py | 8 baseline tests |
| tests/test_statistics_pipeline.py | 6 stats tests |
| tests/test_failure_audit.py | 5 audit tests |
| tests/test_real_data_pipeline.py | 7 data pipeline tests |
| tests/test_surrogate_validation_script.py | 2 validation tests |
| scripts/validate_surrogate_vs_mmg.py | Surrogate validation |
| scripts/run_real_ais_replay.py | Real AIS/ENC replay |
| scripts/check_90plus_readiness.py | Readiness audit |
| docs/theory_claim_boundaries.md | Claim boundaries |

---

## 5. Test Summary

**96 tests passed, 0 failed** (new P0/P1 test suite)
**258 tests collected total** across the project

---

## 6. Commands Executed



---

## 7. Blocking Items (Cannot Claim 90+)

1. **Full synthetic core experiment not completed**
   → Run: 

2. **Full ablation experiment not completed**
   → Run: 

3. **Real AIS/ENC replay not completed**
   → Requires user-provided AIS and ENC data

4. **< 3 strong external baselines**
   → B0 (APF), B8 (CBF-only), B10 (COLREGs-MPC) need implementation

5. **Statistical significance not computed on actual data**
   → Infrastructure exists; needs experiment results

---

## 8. Gate Status

| Gate | Status | Cap |
|------|--------|-----|
| P0 bugs fixed + tests | PASS | — |
| Unified metrics + collision geometry | PASS | — |
| IMM behavior filter | PASS | — |
| Real AIS/ENC replay completed | FAIL | 80 |
| >=3 strong baselines | FAIL | 85 |
| Statistical significance | PENDING | 85 |

---

## 9. Next Commands to Reach 90+



---

## 10. Code Infrastructure vs Experimental Evidence vs Paper Submission

| | Code Infrastructure | Experimental Evidence | Paper Submission |
|---|:---:|:---:|:---:|
| **Readiness** | 90+ | ~60 | NOT READY |
| **P0 bugs** | Fixed | N/A | N/A |
| **Tests** | 96 new pass | N/A | N/A |
| **Smoke exp** | Entrypoint ready | Running | Need full |
| **Full exp** | Script ready | Not run | Not run |
| **Real data** | Script ready | Needs data | Needs data |
| **Baselines** | Registry ready | 2 implemented | Need >=3 |
| **Statistics** | Pipeline ready | Needs data | Needs data |

---

**No real data, experimental results, or external baseline results were fabricated.**

**No test effectiveness was reduced.**

**Documentation clearly distinguishes formal guarantees from empirical evaluation.**

*Generated by the SCI Q1 90+ Hardening Audit — 2026-06-04*
