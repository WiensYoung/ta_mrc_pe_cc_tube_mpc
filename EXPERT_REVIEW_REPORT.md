# SCI Q1 审稿专家综合评审报告

## TA-MRC-PE-CC-Tube-MPC：目标感知多规则约束物理增强机会约束管式模型预测控制

**审稿日期**: 2026-06-17
**审稿人身份**: 中科院SCI一区资深审稿专家（海洋自主系统 / 模型预测控制 / 海事安全领域）

---

## 一、总体评价 (Overall Assessment)

### 评分: Major Revision (大修)

本工作提出了一个面向受限水域多船避碰的分层控制框架，融合了MMG船舶运动模型、COLREGs规则引擎、IMM多模型滤波、机会约束管式MPC、CBF安全滤波及多级回退策略。框架设计的系统性和工程完整性值得肯定——68个源文件、~38,000行代码、40个测试文件，体现了较高的软件工程水准。

然而，从SCI一区论文的标准审视，本工作存在**致命性实验缺陷**、**若干理论硬伤**和**多处与当前前沿的差距**，距离可发表状态尚有显著距离。

---

## 二、致命性问题 (Critical / Fatal Issues)

### C1. 实验结果全部方法成功率均为0%——论文核心主张无法成立

Phase 3实验结果显示：**所有9种方法（B1-B8 + Proposed）在270个episode中的成功率均为0%**。这意味着：

- **论文核心假设H1-H4无法验证**："每个模块独立贡献安全-效率权衡"——当所有方法均100%失败时，无法进行任何有意义的统计比较。
- **消融实验失去意义**：12个消融变体中，9个的碰撞率与Proposed完全相同（25/270 = 9.26%）。移除任何单一组件几乎不改变结果——这要么说明场景过于确定性，要么说明消融机制存在根本缺陷。
- **验收标准全部未达标**：
  - ❌ "Proposed碰撞率显著低于B3 (p < 0.01)"——无法检验
  - ❌ "至少8/12消融显示统计显著效果"——无信号可检测
  - ❌ "p95 runtime < 0.25s"——未计算
  - ❌ "Fallback触发率 < 10%"——未报告

**审稿意见**: 在解决"为什么所有方法均100%失败"这一根本问题之前，本文不具备发表条件。这可能源于：(a) 场景配置错误（如终止条件过于严苛）；(b) 失败分类逻辑存在bug（Phase 3报告中B1/B2在失败分布表中有SUCCESS条目，但汇总表报告0%成功率——存在内部矛盾）；(c) 控制器实现的根本性缺陷。

### C2. 碰撞率数据的"假阳性"问题

从`verify_ablation_results.csv`分析：
- 碰撞**仅发生在F1（head-on failure）类型的episode中**，其他所有failure类型（F2, F4, F5, F6, F10）碰撞率均为0。
- F4和F5类型的path_efficiency=0.0、safety_margin=0.0、ship_domain_violation=0.0——这些episode可能是无遭遇场景，metrics计算默认为零。
- COLREGs违反率在所有方法、所有episode中**恒为0.0**——但存在9.3%的碰撞率。碰撞场景中COLREGs违反率应显著大于零，这强烈暗示metrics计算逻辑存在bug。

### C3. 消融实验不具区分度

| 指标 | Proposed | 消融范围 | Δ |
|------|----------|----------|---|
| 碰撞率 | 0.0926 | 0.0741–0.1000 | 0.026 |
| COLREGs违反 | 0.0 | 0.0–0.0 | 0.0 |
| Path效率 | 0.9998 | 0.9994–1.0001 | 0.0007 |
| 平均运行时(s) | 0.2681 | 0.2660–0.02922 | 0.026 |

- **9/12消融变体的碰撞数与Proposed完全相同（25/270）**
- **A7（移除fallback）反而减少了碰撞**（20/270 vs 25/270）——与论文假设矛盾
- **A6（移除CBF-QP）碰撞率反而略降**（24/270 vs 25/270）

唯一显示有意义差异的指标是CBF干预率（A6=0，设计如此）和ship domain violation rate（A1显著不同）。这远未达到"8/12消融统计显著"的要求。

### C4. Baseline结果文件损坏

`verify_core_all_results.csv`的第6-8列（state_history, command_history, target_histories）包含原始Python dict/list字符串，内嵌逗号，导致CSV解析失败。B1-B8的全部metrics无法从此文件中提取。**核心对比实验的数据不可用**。

### C5. 统计分析完全缺失

Phase 5因CSV解析错误（"field larger than field limit 131072"）而完全失败。实验计划中描述的所有统计检验（paired t-test, Wilcoxon, Cohen's d, Holm-Bonferroni校正）均未执行。**没有任何统计显著性证据**。

---

## 三、理论与算法硬伤 (Theoretical / Algorithmic Issues)

### T1. "管式MPC"名不副实——缺乏RPI证明

`analysis/stability.py`中实现了`compute_robust_positive_invariant_set`，但`docs/theory_claim_boundaries.md`明确承认"**没有RPI证明**"。当前实现的"tube"本质上是一个**启发式安全裕度**（additive buffer），而非严格的管式MPC（Mayne et al., 2000; Köhler et al., 2024）。

**审稿意见**: 论文必须明确区分"tube-inspired robust safety buffer"与"formally robust tube-MPC with RPI certification"。建议在文中使用"uncertainty-buffered MPC"或"tube-inspired MPC"的表述，并诚实声明这是经验性方法而非形式化保证。当前的命名在审稿中将被视为**过度声明（overclaiming）**。

**最新前沿对标**: Köhler et al. (2024)在Ocean Engineering上发表了真正的管式MPC用于船舶轨迹跟踪，其中包含完整的RPI集构造和收缩证明。本文若声称"tube-MPC"，审稿人必然会要求与之对比。

### T2. CasADi后端使用代理动力学——非真正的非线性MMG-MPC

`mpc_problem.py:417-438`的CasADi后端使用**简化线性阻尼代理模型**，而非完整的MMG非线性模型。具体而言：
- 代理模型省略了交叉耦合项（v-r耦合）
- 使用硬编码的无量纲系数（`_surge_Xuu = -2e-4`）
- 缺少标准MMG水动力导数

而SLSQP后端使用完整MMG模型。这意味着**论文声称的"nonlinear MMG-MPC"实际上在默认CasADi后端中是线性化代理MPC**。

**审稿意见**: 这是一个严重的声明-实现不一致问题。建议：(a) 将surrogate dynamics在论文中明确声明并分析其近似误差；或 (b) 实现真正的CasADi符号化MMG动力学（计算量会显著增加但技术上可行）。

### T3. 机会约束Boole分配过于保守

`risk_allocation.py`使用Boole不等式分配：`ε_step = ε_total / (N × T)`。

对于 `ε_total=0.10, N=2, T=12`：`ε_step = 0.10/24 ≈ 0.0042`，对应 `κ = √(χ².ppf(0.9958, 2)) ≈ 3.3`。相比不分配时的 `κ ≈ 2.15`，裕度增加了53%。

在受限水域（channel width < 200m）中，这种保守性可能导致MPC频繁不可行，进而触发fallback。**这可能是所有方法成功率低的一个重要原因**。

**最新前沿对标**: 近期文献（Mesbah, 2016的后续工作）提出了基于分布式风险分配的改进方案（如Scalable Risk Allocation, Zhang et al., 2024），以及基于场景采样的近似机会约束MPC（Ahmed et al., 2025），可以在保持概率保证的同时显著降低保守性。

### T4. IMM滤波实现非标准——缺少交互/混合步骤

`imm_filter.py`的循环顺序为：**更新 → Markov转移 → 预测**，而标准IMM（Blom & Bar-Shalom, 1988）的顺序为：**交互/混合 → 预测 → 更新**。

关键差异：
- 缺少**状态混合步骤**（state mixing/interaction）——这是IMM算法的核心创新
- Markov转移作用于后验概率而非先验——导致模式判别延迟
- 等效于"独立模型滤波器+概率跟踪"，而非真正的IMM

**审稿意见**: 如果论文声称"IMM-based intent prediction"，审稿人（尤其是信号处理/估计领域的审稿人）会要求对照标准IMM实现验证。建议：(a) 实现标准IMM cycle；或 (b) 明确声明使用的是"simplified multi-model filter"而非严格IMM。

### T5. 管式边界半径可能不合理

`physics/tube_boundary.py`的浅水管半径：
```python
rho_shallow = rho_factor * I_shallow * vessel_length * (vessel_speed / U_ref)
# = 2.0 * 1.0 * 180 * 1.0 = 360m
```

360米的管半径**超过了许多航道的宽度**（如East River ~200m），直接导致MPC不可行。这与bank effect的类似问题（dimensionless default treated as force）叠加，可能系统性地使问题不可解。

### T6. 扰动力量纲不一致

`physics/shallow_water.py`、`physics/bank_effect.py`、`physics/ship_interaction.py`的默认扰动边界是**无量纲量**，但下游代码（MMG模型）将其作为牛顿（力的单位）处理。除非显式配置`gamma_shallow_factor`、`gamma_b_factor`、`gamma_s_factor`为物理值，否则注入MMG模型的扰动力量级错误。

---

## 四、与当前前沿的差距 (Gaps vs. State-of-the-Art)

### G1. 缺乏与DRL方法的对比

2024-2026年，基于深度强化学习的船舶避碰方法在IEEE Trans. ITS、Ocean Engineering等顶刊大量涌现：
- **Graph Neural Network + RL**: 处理可变数量目标船、编码多船交互关系（Xie et al., 2025, IEEE TITS）
- **Multi-agent RL**: 分布式决策、无需集中式协调（Chen et al., 2025, Ocean Engineering）
- **Hierarchical RL**: 高层策略选择COLREGs动作，低层MPC执行（Li et al., 2025）

本文的B1-B8全部是传统方法（VO、DWA、MPC变体），**没有任何学习方法作为baseline**。在一区审稿中，这将被视为重大遗漏。

### G2. 缺乏与High-Order CBF (HOCBF)的对比

当前CBF实现使用一阶CBF（`h_ship = d_ij - d_safe`，相对度=1）。但船舶动力学的控制输入（舵角→航向→位置）相对度>1。近期文献（Xiao & Belta, 2022; Liu et al., 2025, IEEE TAC）提出了HOCBF用于高阶系统，可以更精确地处理船舶安全约束。

当前实现中UKC barrier返回`hdot=0, dhdot=[0,0]`（`cbf_qp.py:396-398`），意味着**UKC约束的CBF完全不功能**——要么trivially满足，要么unfixable。

### G3. 缺乏与分布式/多智能体MPC的对比

受限水域多船避碰的核心挑战是**多智能体协调**。近期工作：
- **Distributed MPC**: 每艘船独立求解MPC但通过通信协调约束（Johansen et al., 2024, Control Engineering Practice）
- **Game-theoretic MPC**: 将多船交互建模为Nash/Stackelberg博弈（Zhou et al., 2025, IEEE TITS）

本文采用单船优化+目标船预测的架构，忽略了多船之间的策略交互。在多船场景（S4, S8）中，这可能导致过度保守或不可行。

### G4. 缺乏真实水域验证

实验计划的"Strong Q1 Additions"要求50+真实AIS replay episodes和至少2个真实水域的ENC数据。当前**所有实验均使用合成场景**，AIS数据仅用于episode提取而非闭环回放。真实水域验证是Ocean Engineering一区论文的标配要求。

### G5. Surrogate vs MMG验证样本量不足

`surrogate_vs_mmg_summary.json`显示仅使用**n=2个样本**进行验证，horizon=3, dt=0.5s。这在统计上毫无意义。SIMMAN 2020 benchmark（`tests/validation/test_mmg_validation.py`中引用）要求完整的PMM测试对比。

### G6. 缺乏计算复杂度分析

当前仅报告平均运行时（~0.27s），但缺乏：
- IPOPT迭代次数分布
- P95/P99 runtime
- Deadline miss的episode分析
- JAX vs CasADi vs SLSQP后端的系统对比
- 随目标船数量的可扩展性分析

---

## 五、代码质量问题 (Code Quality Issues)

### Q1. 大量硬编码魔法数字

| 位置 | 硬编码值 | 影响 |
|------|----------|------|
| `controller.py:548` | IMM entropy阈值=0.5, 缩放=0.3 | 管半径调整不连续 |
| `mpc_problem.py:536` | `n_rps = pk * 3.0` | 螺旋桨转速假设 |
| `cbf_qp.py:64-86` | `alpha_cbf=1.0`, `yaw_gain=0.015` | CBF行为 |
| `tube_boundary.py:204` | `speed / 7.0` | 速度归一化未使用U_ref |
| `dynamic_ship_domain.py:171` | `alpha_rule_standon = -0.2` | stand-on安全距离为负 |
| `uncertainty.py:90` | `100.0 * I` | 默认协方差 |

### Q2. CBF与MPC使用不同的安全距离

MPC使用三层安全裕度（domain + tube + chance），而CBF仅使用第一层（domain）。这是有意设计（`cbf_qp.py:111-119`），但会导致CBF允许MPC试图避免的情况发生。

### Q3. 数据缓存粒度问题

- 域缓存hash精度10m——在近距离遭遇中可能返回过时结果
- 扰动缓存position精度0.1m——连续步骤可能产生相同hash

### Q4. 命名不一致

- `theory_claim_boundaries.md`建议使用"tube-inspired"，但代码中所有类名和变量名使用"tube_mpc"
- Surrogate dynamics在代码中存在但论文中未声明

---

## 六、实验设计建议 (Experimental Design Recommendations)

### 必须完成（Mandatory）

1. **诊断并修复0%成功率问题**：
   - 检查failure detector的终止条件是否过于严苛
   - 检查Phase 3报告中B1/B2的SUCCESS/Failure计数不一致
   - 在至少1个简单场景（如S2无遭遇）上验证100%成功率

2. **修复`verify_core_all_results.csv`**：
   - 使用Python csv模块的`quoting`参数正确处理嵌入式dict
   - 重新生成baseline对比表

3. **运行统计分析**：
   - 修复Phase 5的CSV解析错误
   - 执行完整的假设检验pipeline

4. **扩大实验规模**：
   - 至少100 episodes × 5 seeds × 8 methods × 8 scenarios = 32,000 runs
   - 当前仅完成~2,430 runs（7.5%）

5. **Surrogate验证**：
   - 至少n=100个样本
   - 扩展horizon到至少10步
   - 报告位置/航向/速度误差的分布（不仅是均值）

### 强烈建议（Strongly Recommended）

6. **添加DRL baseline**：至少实现一个PPO/SAC-based collision avoidance agent
7. **实现标准IMM**：添加交互/混合步骤，对比模式判别性能
8. **真实水域AIS replay**：使用已有的3个水域AIS数据进行闭环验证
9. **修复IMM cycle顺序**：从"update→transition→predict"改为标准"mix→predict→update"
10. **消除量纲不一致**：为所有扰动模型配置物理单位的gamma factors

### 可选改进（Nice to Have）

11. **HOCBF替代一阶CBF**：处理船舶动力学的高阶特性
12. **分布式MPC对比**：至少在多船场景中与distributed MPC对比
13. **计算复杂度完整分析**：IPOPT迭代分布、可扩展性、后端对比
14. **JAX GPU加速实际对比**：当前GPU benchmark结果未见报告

---

## 七、论文写作建议 (Paper Writing Recommendations)

### 必须修改

1. **命名诚实化**：
   - "tube-MPC" → "tube-inspired robust safety buffer" 或 "uncertainty-buffered MPC"
   - "IMM-based intent prediction" → "multi-model behavior hypothesis generator"
   - "nonlinear MMG-MPC" → 如果使用surrogate则需明确声明

2. **声明surrogate dynamics**：论文必须说明CasADi后端使用简化代理模型，并分析其近似误差

3. **添加limitations节**：
   - 无RPI证明
   - Boole分配的保守性
   - 单船优化假设
   - 未验证真实水域

### 推荐对标文献

| 主题 | 推荐对标文献 | 与本文的关系 |
|------|-------------|-------------|
| 管式MPC | Köhler et al., 2024, Ocean Engineering | 有RPI证明的真正管式MPC |
| 机会约束MPC | Mesbah, 2016, IEEE CST | 本文的基础理论参考 |
| CBF船舶避碰 | Liu et al., 2025, IEEE TAC | HOCBF处理高阶船舶动力学 |
| DRL避碰 | Xie et al., 2025, IEEE TITS | GNN+RL处理多船交互 |
| IMM滤波 | Blom & Bar-Shalom, 1988 | 标准IMM cycle |
| 船舶MMG模型 | Yasukawa & Yoshimura, 2015, J Marine Sci Tech | 标准MMG公式 |

---

## 八、总结

### 优势
- ✅ 系统架构设计完整，模块化程度高
- ✅ 软件工程实践良好（测试覆盖、检查点恢复、配置管理）
- ✅ 理论声明边界清晰（`theory_claim_boundaries.md`）
- ✅ 消融实验设计合理（12个消融，feature-flag机制）
- ✅ 物理模型考虑周全（bank effect, shallow water, ship interaction）

### 必须解决的问题
- ❌ 所有方法成功率0%——核心实验不可用
- ❌ 消融实验无区分度——无法证明各模块贡献
- ❌ Baseline数据文件损坏——对比实验不可用
- ❌ 统计分析完全缺失——无显著性证据
- ❌ "管式MPC"声明缺乏RPI证明
- ❌ CasADi后端使用surrogate而非MMG——与论文声称不一致
- ❌ 实验规模仅为计划的7.5%

### 审稿结论

**本文的核心创新点（多规则约束+物理增强+机会约束管式MPC+CBF+回退）在概念层面是有价值的，但当前的实验结果完全无法支撑任何科学主张。** 建议作者：(1) 首先诊断并修复0%成功率的根本原因；(2) 完成完整的32,000 run实验；(3) 运行统计分析pipeline；(4) 修正论文中的过度声明；(5) 添加DRL baseline和真实水域验证。

在上述问题全部解决之前，本文不适合投稿至SCI一区期刊。

---

*审稿完成日期: 2026-06-17*
*审稿人: SCI Q1 Reviewer (Maritime Autonomous Systems & MPC)*
