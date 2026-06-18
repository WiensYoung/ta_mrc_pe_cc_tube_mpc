# Known Limitations

This document records all significant limitations of the current implementation.
These are **honest** limitations — claims in papers or presentations must not
exceed what the code actually does.

## Data

- **AIS data**: Real AIS trajectory data has been obtained from
  [MarineCadastre.gov](https://marinecadastre.gov) (U.S. Bureau of Ocean Energy
  Management / NOAA), covering the full calendar year 2025.  The
  `data/ais_preprocess.py` pipeline ingests standard AIS CSV/Parquet files and
  produces episode configurations.  A synthetic data generator
  (`data/synthetic_generator.py`) is also available as a fallback for rapid
  prototyping.
- **ENC data**: Real S-57/S-100 ENC data has been obtained from the
  [NOAA Office of Coast Survey](https://nauticalcharts.noaa.gov).  The
  `data/extract_enc.py` script parses these into `EncLayer` objects.  When real
  ENC data is not yet processed, the framework falls back to synthetic ENC layers
  (rectangular channels) via `make_synthetic_enc()`.
- **Geographic coverage**: The specific waterway regions covered are annotated
  within the data files themselves (chart / region metadata).
- **Data is not bundled**: Real AIS and ENC datasets are **not included** in this
  repository due to size and licensing constraints.  They must be placed in a
  configured data directory before running experiments.

## Ship Dynamics

- **MMG model**: The default model uses a simplified nonlinear manoeuvring model with
  empirically scaled coefficients.  It produces **qualitatively correct** trajectories
  (turning direction, speed response) but is not quantitatively validated against
  specific hull forms.  The standard MMG model (KVLCC2, KCS coefficients) is
  partially implemented — full U-dependent nondimensionalization (Yasukawa & Yoshimura
  2015 convention) is pending.
- **Single-agent control**: Only the ownship is controlled.  Target ships follow
  predefined trajectories.  Multi-vessel cooperative control is out of scope.
- **No sensor model**: Perfect ownship state knowledge is assumed.  Positioning
  noise enters only through the configurable `own_position_std` parameter.
- **No communication delays**: VHF/ASM communication delays are not modelled.

## Ship Domain

- **Scalar additive model**: The dynamic ship domain is a scalar additive distance
  per target ship.  It is **not** a full elliptical, quaternion, or heading-dependent
  domain (Fujii, Goodwin, etc.).  Claims should use "additive scalar safe distance"
  or "scalar dynamic safety domain".
- **Simplified geometry**: Ships are approximated by their length/beam/draught
  dimensions for domain calculations.  Full 3D hull forms are not used.

## Physics Enhancement

- **Shallow water**: The disturbance bound is a conservative estimate.  Full MMG
  shallow-water derivative corrections are not implemented.
- **Bank effect**: Conservative bounded disturbance based on published coefficients
  (Delefortrie et al. 2024).  Not a precision CFD model.
- **Ship interaction**: Conservative bounded disturbance.  Full potential-flow or
  RANS-based interaction modelling is not implemented.
- **Wind**: Uniform field, treated as bounded disturbance.  No gust or spatial
  variability modelling.
- **Current**: Uniform field, additive kinematics.  No spatial/temporal variability.
- **No ice or restricted visibility**: Not implemented in this version.

## Chance Constraints

- **Gaussian approximation**: Relative position uncertainty is approximated as
  Gaussian with diagonal covariance.  Non-Gaussian effects (heavy-tailed AIS errors)
  are partially captured by the tube radius inflation.
- **No systematic AIS errors**: Wrong MMSI, heading offset, or spoofing are not modelled.

## Tube Radius

- **Additive conservatism**: The tube radius is the sum of eight independent
  contributions, assuming all disturbances simultaneously achieve their worst-case
  values and align constructively.  This is a **safe upper envelope** that guarantees
  constraint satisfaction at the cost of larger safety margins.  The conservatism
  analysis module (`analysis/conservatism.py`) quantifies this overhead.

## CBF / Fallback

- **CBF-QP conservatism**: The CBF safety filter may intervene earlier than necessary,
  adding a layer of conservatism on top of the MPC solution.
- **Fallback coverage**: The 6-level fallback strategy handles MPC infeasibility,
  CBF over-intervention, emergency proximity, AIS loss, near-bank, and runtime
  deadline misses.  It does **not** cover all possible failure modes (e.g., propulsion
  failure, sensor spoofing).

## Validation

- **No real-ship trials**: The framework has not been validated with real ship
  manoeuvring data beyond the SIMMAN 2020 benchmark comparisons (partial, standard MMG
  only).
- **No towing-tank validation**: Physics enhancement models use published coefficients
  but have not been independently validated against basin experiments.
- **Synthetic scenarios only**: All packaged scenarios use synthetic data.
  Real-world waterway scenarios require external AIS/ENC data.

## Current Experimental Status (2026-06-03)

- **CasADi/IPOPT backend**: Requires an environment with CasADi >= 3.6.0 installed.
  When CasADi is unavailable or fails, the system falls back to scipy SLSQP.
  Results using SLSQP fallback MUST NOT be reported as "CasADi/IPOPT" results.
  The metadata now distinguishes `requested_backend` from `actual_backend`.
- **Core experiment suite**: NOT YET EXECUTED.  The 32,000-run experiment matrix
  described in `docs/SCI_Q1_EXPERIMENT_PLAN.md` is a PLANNED experiment design.
  No results are available yet.
- **Real AIS/ENC experiments**: NOT YET EXECUTED.  Real data has been acquired from
  MarineCadastre.gov and NOAA OCS but has not yet been processed into episode
  configurations.  The infrastructure (AIS preprocessor, ENC parser) is in place.
- **Ablation experiments**: NOT YET EXECUTED.
- **Sensitivity analysis**: NOT YET EXECUTED.
- **Statistical analysis**: Scripts are ready but require experiment outputs.

## References

Verified references used by this project:

- Delefortrie, G., Verwilligen, J., Eloot, K., Lataire, E. (2024). "Bank interaction
  effects on ships in 6 DOF." Ocean Engineering, 310, 118614.
- Yasukawa, H., Yoshimura, Y. (2015). "Introduction of MMG standard method for ship
  maneuvering predictions." J. Marine Science and Technology, 20, 37-52.
- Mayne, D. Q., Seron, M. M., Raković, S. V. (2005). "Robust model predictive control
  of constrained linear systems with bounded disturbances." Automatica, 41(2), 219-224.
- PIANC (2014). Report 121 — Harbour Approach Channels: Design Guidelines.
- ITTC (2021). Recommended Procedures and Guidelines: Manoeuvrability in Shallow Water.
- Zhang, C., Dhyani, A., Ringsberg, J.W., et al. (2025). "Nonlinear Model Predictive
  Control for Path Following of Autonomous Inland Vessels in Confined Waterways."
  Ocean Engineering, 334, 121592.
- Lee, H., Tran, H., Kim, J. (2024). "Safety-Guaranteed Ship Berthing Using Cascade
  Tube-Based Model Predictive Control." IEEE Trans. Control Systems Technology,
  32(4), 1504-1511.
