# Experiment Protocol

**Primary reference**: `实验方案_V4_最终版.md` (Chinese, 986 lines) — the
authoritative experimental design document.

This file is a brief English summary for quick reference.

## Scenarios

8 core scenarios (`configs/scenarios_core.yaml`), 3 extended stress tests:

| ID | Name | Key factors |
|----|------|-------------|
| S1 | TSS co-directional following | Lane keeping, speed control |
| S2 | Large cargo + ferry crossing | High-speed crossing, vessel-type weighting |
| S3 | Precautionary area multi-vessel | Multi-ship conflict, rule priority |
| S4 | Cross-current channel keeping | Flow disturbance, tube radius |
| S5 | Near-bank passage / bend | Bank effect, CBF intervention |
| S6 | Close parallel / overtaking | Ship interaction, rudder peak |
| S7 | AIS delay / dropout | Chance constraint, fallback recovery |
| S8 | Shallow water / grounding risk | UKC, shallow-water safety domain |
| E1 | Prince William Sound tanker lane | Extreme UKC, grounding |
| E2 | NY Harbor extreme restricted water | Bank/ship coupling |
| E3 | Bridge area + bend + tug-barge | Pier/bank/ship multi-hazard |

## Methods

8 baselines + 12 ablations.  See `docs/baselines_and_ablations.md`.

## Metrics

Six dimensions: safety, rule compliance, target awareness, physical feasibility,
restricted-waterway risk, robustness & runtime.

## Statistics

Paired t-test, Wilcoxon signed-rank, Cohen's d, Cliff's delta,
Holm-Bonferroni, Benjamini-Hochberg, cluster bootstrap (by real AIS episode),
mixed-effects model.

See `docs/reproducibility.md` for run commands.
