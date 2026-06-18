# Reproducibility

## Installation

```bash
git clone <repo-url>
cd ta_mrc_pe_cc_tube_mpc
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Core (CBF-QP works, MPC falls back to SLSQP)
pip install -e .

# With CasADi+IPOPT (recommended — nonlinear MPC as in paper)
pip install -e ".[solver]"

# Everything (CasADi + Ray + dev tools)
pip install -e ".[all]"
```

## Data Preparation

Real AIS and ENC data have been obtained from official sources:

| Dataset | Source | Coverage |
|---------|--------|----------|
| AIS trajectories | [MarineCadastre.gov](https://marinecadastre.gov) | Full year 2025 |
| ENC (S-57/S-100) | [NOAA Office of Coast Survey](https://nauticalcharts.noaa.gov) | Full year 2025 |

Geographic coverage is annotated in the data files themselves.

These datasets must be placed in a configured data directory before running
full experiments.  See `configs/default.yaml` for data path configuration.

```bash
# Preprocess AIS data
python scripts/preprocess_ais.py --input /path/to/ais_2025/ --output data/processed/

# Extract ENC layers
python scripts/extract_enc.py --input /path/to/enc_2025/ --output data/enc/

# Build episodes from processed data
python scripts/build_episodes.py --ais data/processed/ --enc data/enc/ --output data/episodes/
```

When real data is not yet processed, the framework falls back to synthetic data
generation (automatically enabled when no data directory is configured).

## Verify Installation

```bash
pytest tests/ -q
# Test count depends on environment (CasADi availability, Python version).
# CI should be consulted for authoritative pass/fail status.
```

## Quick Smoke Test

```bash
python scripts/run_single_scenario.py \
  --config configs/default.yaml \
  --scenario S2 \
  --method Proposed \
  --seed 1
```

## Core Experiments

```bash
# Quick validation (1 episode × 1 seed per scenario, B3 + Proposed only)
python scripts/run_all_core.py --quick --output results/quick

# Full experiment: 100 episodes × 5 seeds, all 8 methods
python scripts/run_all_core.py \
  --n-seeds 5 \
  --output results/raw/core_results.csv
```

## Ablation Experiments

```bash
python scripts/run_all_ablations.py \
  --output results/raw/ablation_results.csv
```

## Analysis

```bash
# From core experiment output
python scripts/analyze_results.py \
  --input results/raw/core_results.csv \
  --output results/analysis

# Or run the full statistical pipeline (includes cluster bootstrap, mixed models)
python scripts/run_statistics.py \
  --input results/raw/core_results.csv \
  --output_dir results/analysis
```

## Output Files

```
results/
  raw/
    core_results.csv         # Raw episode-level metrics
    ablation_results.csv     # Ablation episode results
  trajectories/
    <episode_id>.npz         # Per-episode state/command/target histories
  analysis/
    summary.csv              # Per-method descriptive statistics
    pairwise_comparisons.csv # Paired tests + effect sizes
    significance_tests.csv   # Holm-Bonferroni / BH corrected
    effect_sizes.csv         # Cohen's d and Cliff's delta
    failure_taxonomy.json    # Failure counts by method and type
    metadata.json            # Analysis parameters
```

## Deterministic Seeds

All seeds use MD5 hashing of `(scenario_id, method, seed_index)` — reproducible
across Python versions and operating systems.  Python's built-in `hash()` is
**never** used for seed generation.

## Key Configuration

Single config file: `configs/default.yaml`

```yaml
mpc:
  backend: "casadi"     # "casadi" | "scipy" | "sampling"
  horizon: 20
  dt: 0.5

controller:             # Feature flags — also in baseline_registry.py
  enable_multi_rule: true
  enable_chance_constraint: true
  ...
```

## Re-running Paper Figures

1. Run core experiments → `results/raw/core_results.csv`
2. Run `scripts/analyze_results.py` → `results/analysis/`
3. Use `evaluation/pub_plots.py` for publication-quality trajectory/sensitivity plots
