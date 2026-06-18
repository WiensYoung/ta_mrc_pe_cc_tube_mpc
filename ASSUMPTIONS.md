# ASSUMPTIONS.md — Modeling Assumptions and Parameter Sources

This document records all significant modeling assumptions, parameter sources, and limitations. Each assumption is tagged with the module it affects and the expected impact on results.

---

## 1. 3-DOF / MMG Ship Model (`models/mmg_3dof.py`)

### Simplified Nonlinear Maneuvering Model
- **Assumption**: The implementation uses a simplified nonlinear maneuvering model rather than a full MMG (Maneuvering Modeling Group) model with empirically fitted coefficients.
- **Rationale**: Full MMG requires vessel-specific hydrodynamic derivatives obtained from captive model tests or CFD, which are not publicly available for most vessel types.
- **Parameterization**: The model uses configurable nondimensional derivatives (Y_v', N_r', etc.) with example values drawn from published benchmark vessels (e.g., KVLCC2, KCS container ship, Series 60).
- **Impact**: Maneuvering trajectories are qualitatively correct (turning circle, zigzag) but may differ quantitatively from a specific hull's behavior. Sensitivity experiments perturb key derivatives by +-20%.
- **Config path**: `configs/vessel.yaml`

### Propeller and Rudder Models
- **Assumption**: Simplified first-order propeller thrust model (T = K_T * rho * n^2 * D^4) and linear rudder lift model with stall clipping.
- **Rationale**: Sufficient for control design validation; detailed propulsion models would require engine-specific data.

---

## 2. Bank Effect (`physics/bank_effect.py`)

### Conservative Bounded Disturbance
- **Assumption**: Bank effect is treated as a conservative bounded disturbance rather than a precise hydrodynamic force.
- **Formula**: `||w_bank|| <= gamma_b * (U_i/U_ref)^2 * (d_ref/max(d_bank, d_min))^2`
- **Parameter sources**:
  - `gamma_b0`: conservative upper bound from published ship-bank interaction studies (e.g., Vantorre et al., Lataire et al.)
  - `k_b`: calibrated from near-bank AIS track deviations in available datasets
- **Sensitivity range**: `gamma_b in {0.5, 1.0, 1.5, 2.0} * gamma_b0`
- **Impact**: The disturbance bound is a safe upper envelope; individual trajectories may experience less bank effect than the bound. This is conservative for safety.
- **Limitation**: Does not model the sign reversal of bank suction at very close distances (asymmetric bank effect). The CBF-QP safety filter compensates by maintaining minimum clearance regardless of disturbance direction.

---

## 3. Ship Interaction (`physics/ship_interaction.py`)

### Activation Criteria
- **Assumption**: Ship-ship hydrodynamic interaction activates when:
  - `d_ij < k_L * (L_i + L_j)` OR
  - `d_lat_ij < k_B * (B_i + B_j)` OR
  - `TCPA_ij < T_crit`
- **Parameter sources**:
  - `gamma_s, k_s`: derived from published overtaking/passing simulations and empirical models
- **Sensitivity range**: `k_s in {0.5, 1.0, 1.5, 2.0} * k_s0`
- **Impact**: The disturbance bound captures worst-case interaction; actual interaction depends on detailed relative geometry.
- **Limitation**: Does not implement full potential-flow or RANS-based interaction modeling.

---

## 4. AIS Uncertainty (`risk/uncertainty.py`)

### Covariance Model
- **Assumption**: Target ship position uncertainty `Sigma_j` is modeled as a diagonal covariance in local ENU coordinates, inflated by AIS delay and dropout.
- **Sources**:
  - Base AIS accuracy: ~10 m GPS precision (95%)
  - Delay-induced inflation: `Sigma_delay = v_target * v_target^T * delay^2`
  - Dropout: linear interpolation with inflated covariance during gaps
- **Limitation**: Does not model systematic AIS errors (e.g., wrong MMSI, heading offset) or spoofing.

---

## 5. Shallow Water (`physics/shallow_water.py`)

### Depth/Draft Ratio Classification
- **Assumption**: Shallow water effects are classified by `lambda_h = h / T_i`:
  - `lambda_h > 4.0`: deep water (no correction)
  - `3.0 < lambda_h <= 4.0`: mild shallow
  - `2.0 < lambda_h <= 3.0`: moderate shallow
  - `1.5 < lambda_h <= 2.0`: strong shallow
  - `lambda_h <= 1.5`: high grounding risk / forbidden
- **rho_shallow**: Conservative estimate of added resistance and reduced controllability in shallow water.
- **Sources**: PIANC guidelines, ITTC recommended procedures for shallow water maneuvering.
- **Limitation**: Does not implement full MMG shallow-water corrections for hydrodynamic derivatives.

---

## 6. Rule Weights (`rules/rule_priority.py`)

### Priority Hierarchy
- **Assumption**: Rule priorities P0-P5 are strictly hierarchical for hard constraints.
- P0 and P1 are always hard constraints and cannot be relaxed.
- P2-P4 rules can be configured as hard or soft constraints depending on waterway context.
- P5 terms contribute to the MPC cost function but never override safety constraints.
- **Weight calibration**: Default weights are set to produce qualitatively correct COLREGs-compliant behavior. If calibrated against specific waterways, weights should be adjusted with domain expert input.
- **Impact**: Changing P2-P4 weights may alter trajectory details but should not change collision avoidance outcomes, which are enforced by hard constraints.

---

## 7. Tube-MPC Boundary (`physics/tube_boundary.py`)

### Additive Tube Radius Model
- **Assumption**: The tube radius `rho_tube` is the sum of independent contributions. This assumes disturbances are additive and the worst-case combination is the sum of individual bounds.
- **Formula**: `rho_tube = rho_0 + rho_AIS + rho_current + rho_wind + rho_shallow + rho_bank + rho_ship + rho_maneuver`
- **Impact**: This is conservative (assumes all disturbances align constructively). In practice, disturbances may partially cancel, but the sum-of-bounds approach guarantees safety at the cost of potentially larger safety margins.
- **Limitation**: Does not model cross-correlations between disturbance sources.

---

## 8. Chance Constraints (`control/chance_constraints.py`)

### Gaussian Approximation
- **Assumption**: Relative position uncertainty is approximated as Gaussian with covariance `Sigma_rel = Sigma_j + Sigma_i`.
- **kappa_epsilon**: Uses chi-squared quantile for 2-DOF: `kappa_epsilon = sqrt(chi2.ppf(1-epsilon, df=2))`
- **epsilon values**: Based on waterway risk tolerance (0.10 for normal, 0.01 for tanker lanes).
- **Impact**: The Gaussian approximation is standard in maritime collision risk (see COLREGS-compliant collision avoidance literature). Non-Gaussian effects (e.g., heavy-tailed AIS errors) are partially captured by the tube radius inflation.

---

## 9. Environment Conditions

### Wind and Current
- **Assumption**: Wind and current are modeled as uniform fields over the simulation domain.
- **Current**: Enters the kinematics additively (relative velocity in the MMG model).
- **Wind**: Treated as a bounded disturbance force.
- **Current levels**: 0, 1, 2 kn for normal experiments; 3 kn only for stress testing.
- **Limitation**: Does not model spatial/temporal variability of current or gust effects.

### Bathymetry and ENC Data
- **Assumption**: When real ENC data is unavailable, synthetic depth grids and channel boundaries are generated via `synthetic_generator.py`.
- **Data interface**: Standardized `EncLayer` dataclass supports both real and synthetic data through the same API.
- **Limitation**: Synthetic bathymetry is simplified (uniform slope, rectangular channels) and may not capture complex bathymetric features of real waterways.

---

## 10. Baseline Methods

### Simplified Implementations
- **B1 (COLREGs + VO/OZT)**: Uses a geometric velocity obstacle approach with COLREGs rule-based action selection.
- **B2 (Rule-based DWA)**: Dynamic Window Approach with rule-based cost terms.
- **B3-B7**: Progressive implementations sharing the same MMG model and environment configuration.
- **Assumption**: Baseline implementations are research-grade, not production-grade. They capture the essential algorithmic differences while sharing common infrastructure.
- Each baseline's `unsupported_features` metadata records which components it cannot handle.

---

## 11. Statistical Testing

### Independence Assumptions
- **Assumption**: Synthetic perturbation episodes from the same real-AIS-derived episode are treated as clustered (non-independent) for significance testing.
- Cluster bootstrap resamples at the real-episode level, not the perturbation level.
- Mixed-effects model uses `(1|Seed)` random intercept to account for seed variability.
- **Limitation**: Statsmodels MixedLM is used as an approximation; the formula `Metric ~ Method + Scenario + Waterway + ShipType + Disturbance + (1|Seed)` may be simplified if convergence fails.

### Multi-Ship Scenario Coverage (GAP #10 — 2026-06-10)
- **Assumption**: Multi-ship encounter scenarios (3–4 target vessels) are weighted (2×–4× oversampling) to ensure sufficient statistical power relative to pairwise (1-ship) scenarios.
- **Weighting config**: `configs/statistics.yaml::scenario_weights`
- **Coverage**: 35 scenarios total, 0–4 target ships. Multi-ship (3+) scenarios constitute 5/35 by count but ~30% of total episodes after weighting.
- **Limitation**: Weighting addresses statistical imbalance but does not create new real-world multi-ship encounter types beyond what the AIS data or Imazu benchmark provide.

---

## 12. Known Limitations

1. **No sensor modeling**: Assumes perfect ownship state knowledge. Positioning noise enters only through Sigma_i (configurable).
2. **No communication delays**: VHF/ASM communication delays between ships are not modeled.
3. **Single-agent control**: Only ownship is controlled; target ships follow predefined trajectories. Multi-vessel cooperative control is outside scope. Our framework uses the most-conservative target behavior hypothesis for safety, consistent with the IMO MASS Code transparency principle — an autonomous ship must navigate safely without assuming cooperation from other vessels. See also `risk/intent_predictor.py` for multi-modal target behavior prediction (constant velocity, COLREGs-compliant, non-compliant, channel-following, emergency evasion, erratic).
4. **No human factors**: No modeling of watchkeeper fatigue, OODA loop delays, or bridge resource management.
5. **Simplified geometry**: Ships are approximated as ellipses or rectangles for domain calculations, not full 3D hull forms.
6. **No ice or restricted visibility**: Ice navigation and fog are mentioned in the framework but not implemented in this version.
7. **Additive tube conservatism**: The tube radius is the sum of independent contributions. This is a conservative overestimate (assumes all disturbances align). Section 13 below documents the conservatism analysis and calibration methodology to quantify this conservatism.

---

## 13. Conservatism Analysis and Recursive Feasibility

### Conservatism of Additive Tube Radius

The additive tube radius `rho_tube = Σ rho_i` assumes all eight disturbance components simultaneously achieve their worst-case values and align constructively. This is a **safe upper envelope** that guarantees constraint satisfaction at the cost of larger safety margins.

The `analysis/conservatism.py` module provides:

- **Tube utilization analysis** (`tube_utilization_analysis`): For each simulation timestep, `utilization_k = actual_error_k / tube_bound_k`. A low utilization ratio (< 0.3) confirms the additive bound is conservative. Results are reported as mean, P95, and P99 utilization.
- **Component correlation analysis** (`component_correlation_analysis`): Pearson correlation matrix between tube components. If components are uncorrelated (|r| < 0.3), an RSS-based fusion `sqrt(Σ rho_i²)` is statistically justified and provides approximately `(1 - sqrt(8)/8) ≈ 65%` reduction. See discussion in `conservatism.py` docstring.
- **Monte Carlo joint bound** (`monte_carlo_joint_bound`): Bootstrap estimate of the joint distribution of total disturbance magnitude, providing 95th and 99th percentile bounds for comparison with the additive bound.

### Recursive Feasibility

The `analysis/stability.py` module implements a tube-MPC recursive feasibility analysis following Mayne et al. (2005):

- **RPI set computation** (`compute_robust_positive_invariant_set`): λ-contractive robust positively invariant set via discrete Lyapunov equation.
- **Ancillary feedback gain** (`compute_ancillary_feedback_gain`): LQR-based gain K for the error dynamics `u = -K * (x - x_nominal)`.
- **Terminal cost matrix** (`solve_terminal_cost_matrix`): P_terminal for `||x_N||²_P` ensuring closed-loop stability.
- **Unified framework feasibility verification** (`verify_unified_framework_recursive_feasibility`): End-to-end check covering tube coverage, RPI certification, terminal constraint satisfaction, and constraint margins.

The theoretical framework extends standard tube-MPC to chance-constrained tube-MPC by noting that the chance-constrained safety margin `kappa_epsilon * sqrt(lambda_max(Sigma_rel))` enters as an additive expansion to the tube radius, preserving the contraction property when `rho_tube(t+1) <= λ * rho_tube(t) + w_max`.

### Experimental Validation Interface

The `data/experimental_validator.py` module provides a standardized interface for comparing simulation outputs against published experimental data (SIMMAN 2020 benchmarks: KVLCC2, KCS, KVLCC1). The `tests/validation/test_mmg_validation.py` test suite implements:

- Qualitative validation: turning direction, overshoot existence, proportional rudder response
- Quantitative validation: standard MMG model against SIMMAN benchmark turning circle and zigzag data (±2σ tolerance per ITTC 2021 recommendations)
- Sensitivity analysis: ±20% derivative perturbation, verifying bounded trajectory degradation

### Key References (added with this update)

- Delefortrie, G., Verwilligen, J., Eloot, K., Lataire, E. (2024). "Bank interaction effects on ships in 6 DOF." *Ocean Engineering*, 310, 118614.
- Zhang, C., Dhyani, A., Ringsberg, J.W., et al. (2025). "Nonlinear Model Predictive Control for Path Following of Autonomous Inland Vessels in Confined Waterways." *Ocean Engineering*, 334, 121592.
- Cao, Y., Li, T. (2024). "Nonlinear antiswing control for shipboard boom cranes with full state constraints." *Applied Ocean Research*, 146, 103971.
- Lee, H., Tran, H., Kim, J. (2024). "Safety-Guaranteed Ship Berthing Using Cascade Tube-Based Model Predictive Control." *IEEE Trans. Control Systems Technology*, 32(4), 1504-1511.
- Villagómez, Van Zwijnsvoorde, Lataire, Delefortrie (2025). "COLREGs-compliant model predictive collision avoidance for autonomous ships in restricted environments." *Ocean Engineering*, 338, 121966.
- Yasukawa, H., Yoshimura, Y. (2015). "Introduction of MMG standard method for ship maneuvering predictions." *J. Marine Science and Technology*, 20, 37-52.
- PIANC (2014). Report 121 — Harbour Approach Channels: Design Guidelines.
- ITTC (2021). Recommended Procedures and Guidelines: Manoeuvrability in Shallow Water.

---

*Last updated: 2026-06-02*

