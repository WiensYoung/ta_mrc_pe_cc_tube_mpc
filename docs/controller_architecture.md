# Controller Architecture

## Overview

The TA-MRC-PE-CC-Tube-MPC controller is a layered collision-avoidance system:

```
┌─────────────────────────────────────────────────┐
│  Layer 5: Fallback Supervisor                    │
│  (emergency stop, safety-domain expansion)       │
├─────────────────────────────────────────────────┤
│  Layer 4: CBF-QP Safety Filter                   │
│  (runtime barrier-function correction)           │
├─────────────────────────────────────────────────┤
│  Layer 3: Chance-Constrained Tube-MPC            │
│  (optimization-based trajectory planning)        │
├─────────────────────────────────────────────────┤
│  Layer 2: Physics Enhancement                    │
│  (shallow water, bank, ship interaction, wind)   │
├─────────────────────────────────────────────────┤
│  Layer 1: Multi-Rule + Dynamic Ship Domain       │
│  (P0-P5 priority, COLREGs, VTS, ENC, TSS)       │
└─────────────────────────────────────────────────┘
```

## MPC Backends

| Backend | Type | Gradient | Default for |
|---------|------|----------|-------------|
| `casadi` | Nonlinear MPC (IPOPT) | Auto-diff | B4–B7, Proposed |
| `sampling` | Random-exploration MPC | None (zero-order) | B3 (deterministic baseline) |
| `scipy` | SLSQP | Finite-difference | Automatic fallback |

**Default**: `casadi` (set via `mpc.backend` in `configs/default.yaml`).

**Fallback behaviour**: If CasADi+IPOPT fails (numerical issue, timeout), the solver
automatically degrades to scipy SLSQP and records `degraded_backend` in the result
metadata.  **This is never silent** — check `info["degraded_backend"]` in logs.

The `sampling` backend is used **only** for B3 (deterministic MPC baseline) and is
explicitly labelled as a baseline, not the claimed nonlinear MPC.

## Safe-Distance Stack

The total safe distance enforced in MPC constraints is the sum of three layers:

```
d_safe_total = d_safe_eff          ← DynamicShipDomain (scalar, additive)
             + rho_tube             ← compute_tube_radius (8-term conservative sum)
             + κ·√(λ_max(Σ_rel))   ← chance-constraint margin (chi-squared, 2-DOF)
```

See `src/ta_mrc_pe_cc_tube_mpc/control/tube_mpc.py` for the full documentation.

## Physics Effect Pathways

Each physics module output explicitly tags which pipeline stage it feeds:

| Result field | Pipeline stage |
|-------------|---------------|
| `disturbance_bound` | → dynamics (force added to MMG) |
| `rho_*` | → tube_radius (safety margin inflation) |
| `*_violation` | → P0 hard constraint |
| `I_*` / indicator | → MPC cost penalty |
| classification strings | → metadata only |

See `src/ta_mrc_pe_cc_tube_mpc/physics/` module docstrings.

## Key Modules

| Module | Role |
|--------|------|
| `control/controller.py` | Top-level orchestrator |
| `control/tube_mpc.py` | Tube-MPC wrapper, backend dispatch |
| `control/mpc_problem.py` | CasADi / SLSQP / sampling MPC solvers |
| `control/cbf_qp.py` | CBF-QP runtime safety filter (cvxpy OSQP / scipy) |
| `control/fallback.py` | 6-level fallback supervisor |
| `control/chance_constraints.py` | Relative-covariance chance-constraint evaluator |
| `physics/tube_boundary.py` | 8-term additive tube radius |
| `risk/dynamic_ship_domain.py` | Additive scalar safe distance |
| `rules/rule_engine.py` | Multi-source rule assessment (P0-P5) |
