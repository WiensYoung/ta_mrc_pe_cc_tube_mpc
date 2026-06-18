# TA-MRC-PE-CC-Tube-MPC

**Multi-source rule-aware, physics-enhanced, chance-constrained tube MPC**
with CBF safety filter and fallback supervisor for ship collision avoidance
in restricted waterways.

## Overview

This project provides a reproducible experimental framework for evaluating
collision-avoidance controllers under realistic multi-rule, multi-physics,
and stochastic conditions.  The methodology combines:

1. **3-DOF MMG ship model** — simplified nonlinear manoeuvring dynamics
2. **Additive scalar dynamic ship domain** — target-vessel-aware safety envelope
3. **Multi-source rule engine** — P0-P5 priority hierarchy (COLREGs, Inland Rules, TSS, VTS, ENC)
4. **Physics-enhanced disturbance bounds** — shallow water, bank effect, ship interaction, wind/current
5. **Chance-constrained tube-MPC** — relative-covariance probabilistic safety margins
6. **CBF-QP safety filter** — runtime barrier-function correction
7. **Fallback supervisor** — 6-level recovery from infeasibility / emergencies

> **Scope**: This is a **simulation-based research framework**, not a production
> autopilot.  See [Known Limitations](docs/known_limitations.md).

## Documentation

| Document | Content |
|----------|---------|
| [Installation & Reproducibility](docs/reproducibility.md) | Install, quick test, run experiments |
| [Controller Architecture](docs/controller_architecture.md) | Layer diagram, backend dispatch, safe-distance stack |
| [Baselines & Ablations](docs/baselines_and_ablations.md) | 8 methods, 12 ablations, progressive chain |
| [Experiment Protocol](docs/experiment_protocol.md) | Scenarios, metrics, statistics (English summary) |
| [Known Limitations](docs/known_limitations.md) | Honest scope boundaries |
| [SCI Q1 Experiment Plan](docs/SCI_Q1_EXPERIMENT_PLAN.md) | **Q1 journal submission strategy** — acceptance criteria, red flags, figure/table plan |
| [实验方案 V5](实验方案_V5_代码对齐版.md) | Full experimental design — code-aligned (Chinese) |
| [Assumptions](ASSUMPTIONS.md) | Modelling assumptions and parameter sources |

## Quick Start

```bash
pip install -e ".[solver]"         # core + CasADi+IPOPT
pytest tests/ -q                    # run the test suite

# Smoke test (single scenario, ~10 s)
python scripts/run_single_scenario.py --scenario S2 --method Proposed --seed 1
```

## Experiment Scenarios

| Category | Scenarios | Target Ships | Description |
|----------|-----------|-------------|-------------|
| Core (restricted waterway) | S1–S9 | 0–3 | TSS, crossing, precautionary area, channel, bank, shallow, multi-vessel |
| Extended (stress tests) | E1–E4 | 1–4 | Tanker lane, NY Harbor, bridge bend, **4-vessel harbor convergence** |
| Imazu benchmark | Imazu_01–22 | 1–4 | **Complete 22-scenario** COLREGs benchmark (Imazu & Koyama, 1987) |
| **Total** | **35 scenarios** | **0–4** | Full coverage from single-ship channel keeping to 4-vessel multi-encounter |

Multi-ship scenarios (3–4 targets) are weighted to ensure sufficient statistical power
(see `configs/statistics.yaml::scenario_weights`).

## Running Experiments

```bash
# Quick validation
python scripts/run_all_core.py --quick --output results/quick

# Full core experiment (9 scenarios, weighted multi-ship coverage)
python scripts/run_all_core.py --n-seeds 5 --output results/raw/core_results.csv

# Imazu benchmark
python scripts/run_all_core.py --scenarios-file configs/scenarios_imazu.yaml --output results/imazu

# Ablations
python scripts/run_all_ablations.py --output results/raw/ablation_results.csv

# Analysis
python scripts/analyze_results.py --input results/raw/core_results.csv --output results/analysis
python scripts/run_statistics.py --input results/raw/core_results.csv --output_dir results/analysis
```

## Backend Selection

| Backend | Type | Default for |
|---------|------|-------------|
| `casadi` | Nonlinear MPC (IPOPT, auto-diff) | B4–B7, Proposed |
| `sampling` | Random-exploration baseline | B3 only |
| `scipy` | SLSQP (automatic fallback) | degraded |

Set via `mpc.backend` in `configs/default.yaml`.  Sampling is **never** the default
for Proposed.

## Project Structure

```
ta_mrc_pe_cc_tube_mpc/
├── configs/              # YAML (default, vessel, rules, scenarios, statistics)
├── docs/                 # Architecture, baselines, reproducibility, limitations
├── src/ta_mrc_pe_cc_tube_mpc/
│   ├── analysis/         # Conservatism, stability, Sobol sensitivity
│   ├── control/          # MPC, tube-MPC, CBF-QP, fallback, DWA, VO
│   ├── data/             # AIS, ENC, VTS, synthetic generator, perturbation
│   ├── evaluation/       # Metrics, failure taxonomy, statistics, plots
│   ├── experiments/      # baseline_registry, core/ablation/sensitivity runners
│   ├── models/           # MMG 3-DOF (simplified + standard)
│   ├── physics/          # Shallow, bank, ship interaction, wind/current, tube
│   ├── risk/             # CPA, encounter, IMM, intent prediction, domain, uncertainty
│   ├── rules/            # COLREGs, Inland, waterway, rule engine, priority
│   └── simulation/       # Simulator, runner, trajectory_io, failure detector
├── scripts/              # CLI entry points
├── tests/                # 231 tests + 2 xfail
└── results/              # Experiment output
```

## Key Mathematical Formulations

| Formulation | Reference | Caveat |
|-------------|-----------|--------|
| TCPA/DCPA with boundary handling | Section 5 of experiment protocol | — |
| Relative covariance chance constraint | Σ_rel = Σ_j + Σ_i, chi-squared 2-DOF | Gaussian approximation |
| Additive scalar dynamic ship domain | 9-term sum, `control/tube_mpc.py` docstring | Scalar, not directional |
| P0-P5 rule priority hierarchy | `configs/rules.yaml` | — |
| 8-term additive tube radius | `physics/tube_boundary.py` | Tube-inspired, not formally RPI |
| CasADi+IPOPT nonlinear MPC | `control/mpc_problem.py` | Uses surrogate dynamics |

## Data

| Dataset | Source | Coverage |
|---------|--------|----------|
| AIS trajectories | [MarineCadastre.gov](https://marinecadastre.gov) | Full year 2025 |
| ENC (S-57/S-100) | [NOAA Office of Coast Survey](https://nauticalcharts.noaa.gov) | Full year 2025 |

Geographic coverage is annotated in the data files.  These datasets are **not
bundled** with this repository (size / licensing).  Place them in a configured
data directory before running experiments.  A synthetic data generator is
available as a fallback.

See `docs/reproducibility.md` for the full experiment workflow.

### Large File Downloads

Due to GitHub's file size limits, the following large datasets and experiment results
are hosted on cloud storage. Download and place them in the project root directory.

| Directory | Size | Download Link |
|-----------|------|---------------|
| `data/processed/` | ~47 GB | [百度网盘](https://pan.baidu.com/s/1F3q5yHJnnVyXxl_ZJa_tlA) (提取码: r25x) |
| `results/checkpoints/` | ~66 GB | [百度网盘](https://pan.baidu.com/s/1F3q5yHJnnVyXxl_ZJa_tlA) (提取码: r25x) |
| `results/raw/` | ~12 GB | [百度网盘](https://pan.baidu.com/s/1F3q5yHJnnVyXxl_ZJa_tlA) (提取码: r25x) |

> **Note**: After downloading, ensure the directory structure matches the expected
> paths (e.g., `data/processed/ais_juan_de_fuca_puget_sound.csv`).
> A synthetic data generator (`scripts/build_episodes.py`) is available as a fallback.

## License

MIT


