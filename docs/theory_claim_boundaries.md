# Theory Claim Boundaries — TA-MRC-PE-CC-Tube-MPC

**Document status**: Authoritative — supersedes all other documents.
**Last updated**: 2026-06-17

## 1. What Is Formally Guaranteed

| Claim | Formal Guarantee? | Evidence |
|-------|-------------------|----------|
| Collision detection uses oriented hull geometry (SAT) | ✅ Yes | `evaluation/geometry.py` |
| Per-step chance constraint margin (pairwise) | ✅ Yes | `chance_constraints.py` |
| Conservative Boole risk allocation (when enabled) | ✅ Yes | `risk_allocation.py` |
| CBF-QP safety filter respects control limits | ✅ Yes | `cbf_qp.py` |
| Fallback reduces speed when MPC infeasible | ✅ Yes | `fallback.py` |
| Dynamic ship domain responds to speed/type | ✅ Yes | `dynamic_ship_domain.py` |

## 2. What Is Empirically Evaluated (Not Formally Proven)

| Claim | Status |
|-------|--------|
| Tube-inspired robust safety buffer | Empirically validated (no RPI proof) |
| IMM behavior mode tracking | Empirically validated (16 tests pass) |
| CasADi surrogate accuracy vs full MMG | Script provided (`validate_surrogate_vs_mmg.py`) |
| Physics enhancement improves feasibility | Ablation study (A3) |
| Multi-rule hierarchy improves safety | Ablation study (A2) |

## 3. What Is Heuristic / Conservative Approximation

| Claim | Nature |
|-------|--------|
| "Robust tube" | Tube-INSPIRED safety buffer (no RPI set) |
| "IMM-based intent prediction" | Probabilistic mode tracking (NOT full trajectory prediction) |
| AIS delay inflation | Conservative approximation (linear delay growth) |
| Bank effect force | Simplified empirical model |
| Shallow water effect | Classification-based scaling |

## 4. What Is Planned Future Work

| Item | Status |
|------|--------|
| RPI set and recursive feasibility proof | Not implemented |
| COLREGs Rules 6, 8, 13-19 + Rule 18 hierarchy | Implemented (28 tests pass) |
| Full COLREGs formal compliance certificate | Not implemented (Rules 20-37 = operational) |
| Real AIS/ENC full validation | Requires user-provided data |
| DRL-based baseline (B9/B13) | External dependency required |
| Distributed MPC (B12) | Out of scope |

## 5. Naming Conventions — Code vs Paper

| Code Name | Recommended Paper Name |
|-----------|----------------------|
| `TubeMPC` | "tube-inspired robust safety buffer" or "uncertainty-buffered MPC" |
| `IMMFilter` | "probabilistic behavior mode tracking" |
| `ChanceConstraintEvaluator` | "pairwise chance-constrained safety margin" |
| `COLREGs` | "COLREGs Rules 6, 8, 13–19, 18 implemented; Rules 20–37 excluded as operational" |
| `CasADi MPC` | "CasADi/IPOPT with MMG-inspired differentiable surrogate" |

## 6. Audit Trail

- IMM filter now generates per-mode distinct predictions (2026-06-04)
- Collision detection now uses oriented bounding-box SAT (2026-06-04)
- Risk allocation module added with Boole mode (2026-06-04)
- Safety evaluator provides unified metrics for all methods (2026-06-04)
- Risk allocation default changed from strict Boole to sqrt-Boole (2026-06-17)
- IMM cycle reordered to standard mix→predict→update (2026-06-17)
- Failure detector thresholds revised: F5→0.5L, F6→0.5(L1+L2), F10→10% rate (2026-06-17)
- Physics tube radius caps added: shallow≤50m, bank≤30m, ship≤20m, global≤60m (2026-06-17)
- Disturbance force gamma factors now default to physical F_ref values (2026-06-17)
- UKC barrier gradient made continuous (dead zone removed) (2026-06-17)
- Stability analysis linearization dt now parameterized (was hardcoded 1.0) (2026-06-17)
- RPI rho_max formula corrected (was using λ_min, now uses λ_max) (2026-06-17)
- Sampling MPC Sobol replaced with Halton sequence (2026-06-17)
- Stand-on safety distance alpha changed from -0.2 to 0.0 (2026-06-17)
- COLREGs violation detection extended with cumulative heading check (2026-06-17)
- Surrogate dynamics declared in mpc_problem.py docstring (2026-06-17)
