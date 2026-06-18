# Baselines and Ablations

**Single source of truth**: `src/ta_mrc_pe_cc_tube_mpc/experiments/baseline_registry.py`
(version 2.0). All experiment scripts import from this module.

## Baselines (8 methods)

| ID | Name | Controller | Key features | Backend |
|----|------|-----------|-------------|---------|
| B1 | COLREGs + VO/OZT | `VelocityObstacleController` | Geometric velocity obstacle, COLREGs rules | N/A |
| B2 | Rule-based DWA | `DynamicWindowController` | Dynamic window search, cost-based selection | N/A |
| B3 | Deterministic MMG-MPC | `TAMRCPECCTubeMPCController` | Multi-rule P0-P5, NO CC/PE/Tube/CBF/fallback | sampling |
| B4 | CC-MPC | same | B3 + chance constraint | casadi |
| B5 | PE-CC-MPC | same | B4 + physics enhancement | casadi |
| B6 | PE-CC-Tube-MPC | same | B5 + tube MPC | casadi |
| B7 | PE-CC-Tube-MPC + CBF | same | B6 + CBF-QP | casadi |
| Proposed | TA-MRC-PE-CC-Tube-MPC | same | B7 + fallback (full system) | casadi |

**Critical constraint**: `enable_fallback` is **only** `True` in Proposed.
B3–B7 all have `enable_fallback=False`.  This ensures the ablation "w/o fallback"
(A7) has a valid control group.

## Progressive Chain

```
B3 ⊂ B4 ⊂ B5 ⊂ B6 ⊂ B7 ⊂ Proposed
```

Each step adds exactly **one** feature, making ablation effects uniquely attributable.

## Ablations (12)

Each ablation removes **one** feature from Proposed. All other features remain enabled.

| ID | Removes | Verifies |
|----|---------|----------|
| A1 | Target vessel awareness (`enable_target_size`) | H1 |
| A2 | Multi-source rules (`enable_multi_rule`) | H2 |
| A3 | Physics enhancement (shallow, bank, ship, wind) | H3 |
| A4 | Chance constraints (`enable_chance_constraint`) | H4 |
| A5 | Tube MPC (`enable_tube_mpc`) | H4 |
| A6 | CBF-QP (`enable_cbf_filter`) | H5 |
| A7 | Fallback (`enable_fallback`) | H5 |
| A8 | Bank effect (`enable_bank_effect`) | H3 |
| A9 | Shallow water (`enable_shallow_water`) | H3 |
| A10 | Ship interaction (`enable_ship_interaction`) | H3 |
| A11 | Dynamic ship domain → scalar only | H1 |
| A12 | Multi-modal target prediction | H1/H4 |

## Usage

```python
from ta_mrc_pe_cc_tube_mpc.experiments.baseline_registry import (
    get_baseline,
    get_ablation,
    build_ablation_features,
    list_baseline_methods,
)
```
