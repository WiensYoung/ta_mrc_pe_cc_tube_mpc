# Limitations and Scope Boundaries

This document provides an honest assessment of the limitations of the
TA-MRC-PE-CC-Tube-MPC framework.  These limitations should be cited in the
paper's Discussion section to pre-empt reviewer concerns.

---

## 1. No Formal Robust Tube Certification

The "tube" component is a **tube-inspired adaptive safety buffer**, not a
formally certified tube-MPC with Robust Positively Invariant (RPI) set
guarantee.  There is no proof that the tube radius ρ_tube bounds the
worst-case disturbance realization over the prediction horizon.

**Mitigation**: The tube radius is computed from physics-based disturbance
models (shallow water, bank effect, ship interaction, wind/current) with
empirically calibrated scaling factors.  Conservatism analysis is provided
in `analysis/conservatism.py`.

**Comparison**: Köhler et al. (2024) provide a formally certified tube-MPC
for marine vehicles with RPI proof.  Our approach trades formal guarantees
for computational tractability and multi-physics integration.

---

## 2. Conservative Chance Constraint Allocation

The risk allocation uses a square-root Boole inequality:
`ε_step = ε_total / √(N × T)`.  For typical problems (N=2, T=12, ε=0.10),
this gives κ ≈ 2.75, which is more conservative than the unallocated
κ ≈ 2.15.  The strict Boole allocation (κ ≈ 3.3) was even more conservative
and has been replaced.

**Impact**: In tight waterways (channel width < 200m), the conservative
margins may cause MPC infeasibility, triggering fallback.  This is a
known trade-off between probabilistic safety guarantee and operational
feasibility.

---

## 3. Surrogate Dynamics in CasADi Backend

The CasADi+IPOPT backend (default) uses a **simplified surrogate dynamics
model** with linear damping and no cross-coupling terms, rather than the
full nonlinear MMG model.  The surrogate is parameterized from config but
uses default coefficients that may not match the actual vessel.

The scipy SLSQP backend uses the full MMG model but is significantly slower.

**Impact**: The optimizer plans with a different dynamics model than the
one used for simulation.  For comparative evaluation (all methods share
the same simulation dynamics), this is acceptable.  For claims about
absolute maneuvering performance, the surrogate approximation error
should be quantified.

---

## 4. Single-Ship Optimization Assumption

The framework optimizes the ownship's trajectory while treating target
ships as predicted (not optimized).  There is no multi-ship coordination
or game-theoretic interaction.  In multi-vessel scenarios, this may lead
to:
- Overly conservative behavior when multiple targets are present
- Sub-optimal solutions when coordination would benefit all parties
- Potential deadlocks in symmetric encounter scenarios

---

## 5. IMM Is a Simplified Multi-Model Filter

The target behavior predictor uses a simplified IMM variant where:
- Mode probabilities are mixed via Markov transition (correct)
- Per-mode state estimates evolve independently (no cross-mode blending)
- The blending formula uses `sqrt(μ)` rather than standard Kalman mixing

This is sufficient for weighting trajectory hypotheses but does not
provide the full state estimation benefits of a standard IMM.

---

## 6. Disturbance Model Simplifications

- **Bank effect**: Simplified empirical model with fixed yaw coupling
  factor (0.1).  Real bank effects depend on bank slope, under-keel
  clearance profile, and vessel speed.
- **Shallow water**: Piecewise linear resistance approximation without
  physical basis beyond PIANC/ITTC qualitative guidance.
- **Ship interaction**: Risk score provides limited discrimination
  between moderate and severe interactions at default parameters.
- **Wind/current**: Uniform wind field assumed (no spatial gradient).

---

## 7. Scenario and Data Limitations

- All core experiments use **synthetic scenarios**, not real AIS replay.
- AIS data from 3 waterways (Puget Sound, NY Harbor, SF Bay) is available
  but has not been fully integrated into closed-loop validation.
- The S-57 ENC parser covers depth, land, buoys, TSS, fairways, and
  bridges, but not all S-57 feature types.
- Target ship MMG parameters are generic defaults by vessel type, not
  vessel-specific.

---

## 8. Computational Cost

- Mean runtime: ~0.27s per control step (CasADi backend, 192-core machine).
- The 0.5s control period is tight for real-time operation on commodity
  hardware.  JAX GPU acceleration is available but not benchmarked in
  the current results.
- IPOPT convergence depends on warm-starting quality; cold starts may
  take significantly longer.

---

## 9. CBF Limitations

- UKC barrier has limited control authority (can only reduce speed, not
  directly change water depth).
- Lane boundary barrier assumes channel centerline at y=0 in the local
  coordinate frame.
- CBF uses current (not predicted) target positions — it is myopic.
- The CBF safety distance (layer 1 only) differs from the MPC safety
  distance (layers 1+2+3) by design, but this means CBF may allow
  situations the MPC was trying to avoid.

---

## 10. Statistical Validation Gaps

- Surrogate vs MMG validation: n=2 samples (insufficient for statistical
  claims).  Needs n≥100 with horizon≥10.
- No real-world sea trial validation.
- No comparison with learning-based methods (DRL, GNN+RL).
- No comparison with distributed/game-theoretic MPC approaches.

---

## 11. JAX vs CasADi Backend Divergence

The JAX backend (`physics/jax/tube_boundary.py`) uses a different tube
radius formula — proportional to `rho_0` with fixed fractional coefficients
(e.g., `rho_0 * 0.25` for shallow water) — rather than the physics-based
disturbance models used by the CasADi backend.  Switching backends changes
the tube radius computation substantially.  All published results use the
CasADi backend.

---

## 12. Fallback Escalation Hardcoded Multipliers

`fallback.py` multiplies `extend_safety_domain_factor` by 1.2 (STOP level)
and 1.5 (EMERGENCY level) with hard caps at 1.5 and 2.0.  These multipliers
are hardcoded, not configurable, and were not documented until this revision.

---

## 13. Stability Analysis Scope

The stability analysis in `analysis/stability.py`:
- Linearizes around a single cruise operating point (7 m/s, zero drift)
- Uses a scalar worst-case disturbance bound W_max
- Verifies recursive feasibility against a single realized trajectory
- Does NOT prove universal recursive feasibility for all disturbance
  realizations

The analysis provides supporting evidence but is NOT a formal stability
proof.  The RPI set computation and tube contraction check are
implementation scaffolds for future formal analysis.

---

## How to Cite These Limitations

Suggested phrasing for the paper:

> "We acknowledge the following limitations: (1) the tube component is an
> empirical safety buffer without formal RPI certification; (2) the CasADi
> backend uses a simplified surrogate dynamics model; (3) the multi-ship
> formulation is single-agent with predicted (not optimized) target
> behavior; (4) validation uses synthetic scenarios rather than real-world
> sea trials.  These limitations represent directions for future work."
