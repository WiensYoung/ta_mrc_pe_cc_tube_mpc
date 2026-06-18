# SCI Q1 Experiment Plan

## TA-MRC-PE-CC-Tube-MPC — 面向一区期刊投稿的实验矩阵

**目标期刊等级**: Ocean Engineering, IEEE Trans. Intelligent Transportation Systems,
Reliability Engineering & System Safety, 或同等级 SCI Q1 期刊

**核心贡献声明**: 在受限水域多船会遇场景下，一个**可复现的**、多源规则感知的、
物理增强的、机会约束管 MPC 框架，通过系统消融实验证明各模块（CC, Tube, CBF, Fallback）
对安全-效率 tradeoff 的独立贡献。

> **⚠️ 当前状态 (2026-06-03)**: 本计划是**投稿路线图 (roadmap)**，不是实验报告。
> 以下描述的 32,000-run 实验矩阵尚未执行。代码基础设施已完成（见 build system
> 和 baseline registry），但实验运行、结果分析和论文撰写均未开始。
> 当前项目距离 SCI 一区投稿还需要：
> 1. 修复 CasADi/IPOPT backend（P0，已完成）
> 2. 完成核心实验运行（8 scenarios × 100 eps × 8 methods × 5 seeds = 32,000 runs）
> 3. 完成真实 AIS/ENC 数据预处理和实验
> 4. 增加前沿 baseline
> 5. 给出完整统计检验结果
> 6. 报告运行时间和 fallback rate

---

## 目录

1. [Minimum Publishable Experiment Set](#1-minimum-publishable-experiment-set)
2. [Strong Q1 Experiment Set](#2-strong-q1-experiment-set)
3. [Baseline Hierarchy](#3-baseline-hierarchy)
4. [Statistical Power and Confidence](#4-statistical-power-and-confidence)
5. [Figure and Table Plan](#5-figure-and-table-plan)
6. [Acceptance Criteria](#6-acceptance-criteria)
7. [Red Flags](#7-red-flags)

---

## 1. Minimum Publishable Experiment Set

以下为**最低**可支撑一篇 Q2/Q3 论文的实验规模。Q1 需要 Section 2 的加强版。

### 1.1 Scenarios

| # | Scenario | 目的 |
|---|----------|------|
| S1 | TSS co-directional following | lane keeping, speed control |
| S2 | Cargo + ferry crossing | high-speed, vessel-type weight |
| S3 | Precautionary area multi-vessel | multi-conflict, rule priority |
| S4 | Cross-current channel keeping | flow disturbance, tube radius |
| S5 | Near-bank passage / bend | bank effect, CBF intervention |
| S6 | Close parallel / overtaking | ship interaction |
| S7 | AIS delay / dropout | chance constraint, fallback |
| S8 | Shallow water / grounding risk | UKC, non-navigable zone |

### 1.2 Methods

| Method | Purpose |
|--------|---------|
| B1: VO/OZT | reactive heuristic baseline |
| B2: DWA | local search baseline |
| B3: Deterministic MPC | optimization baseline |
| B4: CC-MPC | chance constraint contribution |
| B5: PE-CC-MPC | physics enhancement contribution |
| B6: PE-CC-Tube-MPC | tube robustness contribution |
| B7: PE-CC-Tube-MPC + CBF | CBF safety filter contribution |
| Proposed | full system with fallback |

### 1.3 Scale

| Parameter | Value |
|-----------|-------|
| Episodes per scenario | ≥100 (synthetic perturbation) |
| Seeds per (episode, method) | ≥5 |
| Total runs | 8 scenarios × 100 episodes × 8 methods × 5 seeds = **32,000** |
| Statistical tests | Paired t-test, Wilcoxon, Cohen's d, Holm-Bonferroni |
| Reproducibility | MD5 deterministic hash seeds, resolved_config.yaml saved |

### 1.4 Minimum Acceptable Results

- Proposed collision rate < B3 collision rate (p < 0.05, paired t-test)
- At least 4 of 7 ablations show statistically significant effects in expected direction
- p95 runtime < 0.5× control period (i.e., < 0.25s for 0.5s dt)
- Fallback trigger rate < 10% (Proposed not relying on fallback)
- No silent backend degradation in ≥95% of control steps

---

## 2. Strong Q1 Experiment Set

Q1 期刊需要**超越**最低实验集。以下为加强要求：

### 2.1 真实 AIS Replay

- 从 MarineCadastre.gov (2025) 和 NOAA ENC 数据提取 ≥50 个真实会遇片段
- 每片段 ≥5 扰动变体（初始位置 ±σ, 速度 ±Δv, 航向 ±Δψ）
- 报告合成 vs 真实场景的性能差异
- 对真实 AIS 片段使用 **cluster bootstrap** (按 `real_episode_id` 聚类)

### 2.2 ENC/TSS/本地水道几何

- 使用真实 S-57/S-100 ENC 数据（非合成矩形水道）
- 报告中至少包含 2 个真实水道（如 Puget Sound TSS, New York Harbor）
- 对比合成 vs 真实 ENC 的结果差异

### 2.3 Non-Compliant Target Vessels

- 每场景至少 1 个 non-compliant target variant
- Non-compliant 行为: 错误方向转向、不让路、加速增加碰撞风险
- 预期: Proposed 在 non-compliant 场景下优势更大

### 2.4 Stress Tests

| 测试 | 条件 | 预期 |
|------|------|------|
| 极端横流 | 3 kn cross-current | tube radius 自适应增大 |
| 极浅水 | UKC < 2m | P0 约束触发, fallback 激活 |
| 多船同时逼近 | ≥3 targets within 5× d_safe | MPC feasibility 压力测试 |
| AIS 完全丢失 | dropout_prob = 1.0, delay > 30s | fallback 接管, 恢复成功率 |
| 连续 MPC infeasibility | 注入不可行初始条件 | fallback 升级链完整 |

### 2.5 Runtime Benchmark

- 报告每 backend 的 mean/P95/P99/P99.9 runtime
- 报告 deadline miss rate (threshold = 0.1s, 0.2s, 0.5s)
- 对比 CasADi vs sampling vs SLSQP 的 runtime 分布
- 报告 IPOPT iteration count 分布

### 2.6 Failure Taxonomy

- F1-F10 完整失败分类，按方法×场景交叉表
- 失败案例分析: 至少展示 3 个典型失败的轨迹图
- Fallback 升级链统计: CAUTION→REDUCE_SPEED→STOP→EMERGENCY 各级触发率

### 2.7 Rule Violation Audit

- 报告每方法的 COLREGs 违规类型分布 (head-on, crossing, overtaking)
- TSS/channel/VTS/ATBA 违规按场景统计
- 展示至少 1 个规则冲突消解的定性示例

---

## 3. Baseline Hierarchy

### 3.1 必需基线 (Minimum: 8 methods)

已在代码中实现并通过运行时验证 (`test_baseline_registry.py`):

| ID | Type | Controller | Key differentiator |
|----|------|-----------|-------------------|
| B1 | Reactive | VO/OZT | COLREGs + geometric velocity obstacle |
| B2 | Local search | DWA | Dynamic window + cost-based selection |
| B3 | Optimization | Deterministic MPC | Multi-rule P0-P5, no prob/PE/robust |
| B4 | +Probabilistic | CC-MPC | Relative-covariance chance constraint (χ²) |
| B5 | +Physics | PE-CC-MPC | Shallow/bank/ship/wind-current |
| B6 | +Robustness | PE-CC-Tube-MPC | 8-term additive tube + adaptive scaling |
| B7 | +Safety filter | +CBF | CBF-QP runtime barrier correction |
| Proposed | Full system | TA-MRC-PE-CC-Tube-MPC | All above + fallback supervisor |

### 3.2 推荐增加基线 (Strong Q1: +2-4 methods)

| ID | Type | Justification | Implementation effort |
|----|------|--------------|----------------------|
| B0 | APF | Artificial Potential Field — classic reactive method | Low (new standalone controller) |
| B8 | CBF-only | CBF-QP without MPC (verify CBF vs MPC contribution) | Medium (new controller config) |
| B9 | DRL | Deep RL baseline (e.g., PPO/SAC trained on same scenarios) | High (external dependency) |
| B10 | COLREGs-MPC (prior art) | Villagómez et al. (2025) style — COLREGs-only MPC | Medium (B3 with multi_rule=False) |

> B10 is already achievable by running B3 with `enable_multi_rule=False` —
> effectively a "COLREGs-geometric-only MPC" variant.

### 3.3 基线有效性检查清单

- [ ] 每个基线的 feature flags 在运行时实例化并验证 (✅ `TestRuntimeFeatureFlags`)
- [ ] B3-B7 无 fallback 泄漏 (✅ `validate_registry`)
- [ ] backend_override 强制生效 (✅ `test_backend_override_sampling_is_applied`)
- [ ] Ablation override 优先级最高 (✅ `test_ablation_override_takes_final_precedence`)

---

## 4. Statistical Power and Confidence

### 4.1 样本量论证 (A Priori Power Analysis)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n per (scenario, method) | ≥500 (100 eps × 5 seeds) | 检测 Cohen's d ≥ 0.125, power ≥ 0.80, α = 0.05 |
| Ablation n | ≥150 (30 eps × 5 seeds) | 检测 Cohen's d ≥ 0.3 |
| Sensitivity n | ≥50 per level | OFAT 参数扫描 |
| Sobol n | 1024 base samples | Saltelli sampling, D ≤ 8 |

### 4.2 Rare Event Handling

**Collision** 是小概率事件（预期 < 5%）。标准 t-test 不合适。

| Method | Use case |
|--------|----------|
| Wilson score interval | collision_rate 的 95% CI (二项比例) |
| Clopper-Pearson (exact) | 保守上界, n 较小或 rate=0 时 |
| Fisher's exact test | 2×2 方法间 collision 计数比较 |
| Zero-collision upper bound | 若某方法碰撞=0, 报告 3/n rule (95% 上界) |

### 4.3 Continuous Metrics

| Method | Metric |
|--------|--------|
| Bootstrap CI (BCa) | min_cpa, route_efficiency, runtime P95 |
| Cluster bootstrap | 真实 AIS episode 场景 |
| Paired t-test / Wilcoxon | 配对 (same scenario, same seed) |
| Cohen's d / Cliff's delta | 效应量 |
| Holm-Bonferroni | 28 pairwise × 8 metrics = 多重比较校正 |
| Mixed-effects model | Metric ~ Method + Scenario + (1\|Seed), 用于重复场景 |

### 4.4 Visual Reporting Standard

每个 pairwise comparison 必须报告:
- 检验统计量 (t 或 W)
- 原始 p 值 + Holm 校正后 p 值
- 效应量 (Cohen's d 或 Cliff's delta) 及 95% CI
- 均值差异 及 95% CI
- 有效样本量 n + 排除量 n_excluded

---

## 5. Figure and Table Plan

### 5.1 Required Figures (8-10)

| # | Type | Content | Priority |
|---|------|---------|----------|
| Fig 1 | Architecture diagram | 5-layer controller + safe-distance stack | P0 |
| Fig 2 | Scenario map | Geographic layout of 8 core scenarios | P0 |
| Fig 3 | Safety-efficiency Pareto | collision_rate vs route_efficiency, all methods | P0 |
| Fig 4 | min CPA distribution | Violin/box plot per method across all scenarios | P0 |
| Fig 5 | Ablation bar chart | Cohen's d forest plot for 12 ablations | P0 |
| Fig 6 | Runtime P95/P99 | Bar chart per method × backend | P1 |
| Fig 7 | Failure case visualization | 3 trajectory plots with annotations | P1 |
| Fig 8 | Qualitative trajectory examples | S2 ferry crossing, S3 multi-vessel, S5 near-bank | P1 |
| Fig 9 | Tube utilization | Box plot + correlation heatmap (8 components) | P2 |
| Fig 10 | Fallback trigger chain | Sankey or stacked bar of CAUTION→...→EMERGENCY | P2 |

### 5.2 Required Tables (5-7)

| # | Content | Priority |
|---|---------|----------|
| Table 1 | Baseline feature-flag matrix (8 methods × 13 flags) | P0 |
| Table 2 | Core safety metrics: collision_rate, near_miss, min_cpa, domain_violation | P0 |
| Table 3 | Rule compliance: colregs, tss, channel, forbidden_zone, vts | P0 |
| Table 4 | Ablation results: 12 rows × (Δcollision, Δmin_cpa, Δruntime, p-value, Cohen's d) | P0 |
| Table 5 | Runtime benchmark: mean/P95/P99/P99.9 per backend | P1 |
| Table 6 | Failure taxonomy: F1-F10 count × method | P1 |
| Table 7 | Comparison with prior art (literature table) | P2 |

### 5.3 Supplementary Material

- All raw metrics CSV (for reviewer verification)
- Per-episode trajectory plots (multi-page PDF)
- Statistical test outputs (full pairwise matrix)
- resolved_config.yaml for every experiment run
- `method_feature_table.csv`
- `README_run.md`

---

## 6. Acceptance Criteria

以下为判断实验结果是否值得投向 Q1 期刊的标准：

### 6.1 安全指标 (P0 — 必须满足)

- [ ] Proposed collision_rate **显著低于** B3 (p < 0.01, 经多重比较校正后)
- [ ] Proposed near_miss_rate 最低或与最优基线无显著差异
- [ ] Proposed worst_5%_cpa **显著高于** B3/B4
- [ ] 在 ≥80% 的场景中 Proposed min_cpa > 安全阈值

### 6.2 消融一致性 (P0 — 必须满足)

- [ ] ≥8/12 ablations 在预期方向上产生统计显著差异
- [ ] A4 (无 CC) 在 S7 (AIS 不确定性) 中 min_cpa 显著降低
- [ ] A5 (无 Tube) 在 S4/S5 中 infeasibility_rate 显著升高
- [ ] A6 (无 CBF) 在 S5/S6 中 collision_rate 或 near_miss_rate 显著升高
- [ ] A7 (无 Fallback) 在 S7/S8 中 recovery_success_rate 显著降低

### 6.3 效率指标 (P1 — 应该满足)

- [ ] Proposed route_efficiency 与 B3 无显著差异（或损失 < 5%）
- [ ] Proposed control_effort (mean |rudder|²) 不高于 B3
- [ ] Fallback trigger_rate < 10% (Proposed 安全性不由 fallback 主导)
- [ ] CBF intervention_rate < 20% (CBF 是监督者而非替代者)

### 6.4 实时性 (P1 — 应该满足)

- [ ] CasADi backend mean_runtime < 0.5 × dt (即 < 0.25s for dt=0.5s)
- [ ] CasADi P95_runtime < 1.0 × dt (即 < 0.5s)
- [ ] deadline_miss_rate < 5% at 0.5s threshold
- [ ] IPOPT iteration count median < 50

### 6.5 鲁棒性 (P2 — 加分项)

- [ ] 在 non-compliant target 变体中 Proposed 优势 ≥ compliant 场景
- [ ] 真实 AIS replay 与合成场景的性能排名一致
- [ ] 管利用率均值 < 0.5（保守性在合理范围）
- [ ] 分量相关性 |r| < 0.3 → RSS 融合可行

---

## 7. Red Flags

以下为**不应投稿 Q1** 的信号。如果出现，应先修复再考虑投稿。

### 7.1 实验可信度

- [ ] ❌ Baseline feature flags 再次失效（如 B3 意外启用 fallback）
  → **必须修复**: 运行 `test_baseline_registry.py` 确认 22/22 通过
- [ ] ❌ Ablation 结果与假设方向相反或全无差异
  → **必须排查**: (a) 消融 flag 是否真实生效; (b) 场景是否有碰撞风险
- [ ] ❌ 合成场景效果好但 AIS replay 失败
  → **必须排查**: 合成数据生成器是否过度简化; AIS 预处理是否有 bug
- [ ] ❌ 统计显著性依赖个别场景
  → **必须排查**: 使用 leave-one-scenario-out 敏感性分析

### 7.2 方法问题

- [ ] ❌ Proposed 主要依赖 fallback 才安全 (fallback trigger > 30%)
  → **论文不能声称 MPC 方案有效** — fallback 是安全网,不是核心控制器
- [ ] ❌ CBF 过度介入 (intervention_rate > 50%)
  → **MPC 优化器产生的轨迹本身不安全** — 需要重新调优
- [ ] ❌ CasADi 频繁降级到 SLSQP (degrade_rate > 20%)
  → **IPOPT 参数或 surrogate 模型需要调整**
- [ ] ❌ Runtime 超过控制周期 (P95 > dt)
  → **实时性不可行** — 考虑减小 horizon 或使用 sampling backend

### 7.3 统计问题

- [ ] ❌ 零碰撞方法未报告置信上界
  → **必须报告**: Wilson 或 Clopper-Pearson 上界
- [ ] ❌ NaN 被静默填充为 0（检查 `_safe_float` 是否仍返回 0.0）
  → **运行** `test_safe_float_nan_returns_none` 确认
- [ ] ❌ 配对检验未报告 n_excluded
  → **检查** `paired_ttest` 输出是否包含 `n_excluded` 字段

### 7.4 声称问题

- [ ] ❌ 论文声称 "nonlinear MMG-MPC" 但未说明 CasADi surrogate
  → **修改论文**: 使用 V5 实验方案中的精确表述
- [ ] ❌ 论文声称 "dynamic ship domain" 但实际是标量
  → **修改论文**: "additive scalar safe distance"
- [ ] ❌ 论文声称 "COLREGs-compliant" 但未在 limitations 中说明边界
  → **修改论文**: 说明覆盖 Rules 13-18, 不覆盖所有地方规则

---

## 附录: 与代码的对应关系

| 本计划条目 | 代码/配置位置 |
|-----------|-------------|
| 8 baseline methods | `experiments/baseline_registry.py::BASELINE_REGISTRY` |
| 12 ablations | `experiments/baseline_registry.py::ABLATION_REGISTRY` |
| 13 feature flags | `experiments/baseline_registry.py::_ALL_FEATURE_KEYS` |
| Feature flag runtime verification | `tests/test_baseline_registry.py::TestRuntimeFeatureFlags` |
| Backend selection | `configs/default.yaml` → `mpc.backend: "casadi"` |
| CasADi surrogate dynamics | `control/mpc_problem.py::_solve_casadi` |
| SLSQP full MMG | `control/mpc_problem.py::_solve_slsqp` |
| Safe-distance 3-layer stack | `control/tube_mpc.py::compute_control` docstring |
| Tube radius 8 components | `physics/tube_boundary.py::compute_tube_radius` |
| COLREGs target_role fix | `risk/intent_predictor.py::_predict_colregs_maneuver` |
| NaN preservation | `simulation/closed_loop_runner.py::_safe_float` |
| Experiment dry-run | `scripts/run_all_core.py --dry-run` |
| resolved_config save | `scripts/run_all_core.py` → `resolved_config.yaml` |
| Statistical tests | `evaluation/statistics.py` |
| Deterministic seeding | MD5 hash in `run_core_experiments.py::_deterministic_hash` |

---

*本计划应与 `实验方案_V5_代码对齐版.md` 配合使用。V5 提供完整的实验设计蓝图；
本计划提供 Q1 投稿的策略性指导。*
