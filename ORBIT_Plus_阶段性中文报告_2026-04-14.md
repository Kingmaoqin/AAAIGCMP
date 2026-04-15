# ORBIT+ 阶段性中文报告

日期：2026-04-14

文档位置：
- 论文草案：`/home/xqin5/llmtreatment/ORBIT_Plus_v2_Revised.md`
- 执行计划：`/home/xqin5/llmtreatment/ORBIT_Plus_Execution_Plan.md`
- 代码目录：`/home/xqin5/llmtreatment/orbit_plus`

## 1. 项目目标与当前结论

本阶段工作的目标，是将论文《ORBIT+ v2 Revised》中提出的高维电子病历因果推断框架，按照“先计划、后实现、边修复边验证”的方式，落地为一套可运行的 PyTorch 原型系统。系统不仅要覆盖合成数据实验，还要具备对真实 ICU 数据资产进行半合成验证和临床锚点验证的能力，并把 ORBIT+ 中最核心的 Agentic Workflow 实现出来，即：

- 由 `Role_Prior_Agent` 负责初始化特征角色先验。
- 由 `LLM_Auditor_Agent` 在训练过程中周期性审计 routing 状态。
- 由 `Training_Agent` 执行 warm-up、全训练、audit loop 和 cross-fitting。

当前阶段的总体结论如下：

1. ORBIT+ 主体代码已经搭建完成，包含数据、模型、目标函数、训练、评估和实验入口。
2. 合成数据主链条已经可以跑通，从 `M0` 到 `M5` 以及 `M5-noProxy`、`M5-noBottleneck` 均已有实验结果。
3. 真实数据部分已经接入可用资产，其中 MIMIC 侧使用现有 `mimic3_trajectories.csv` 跑通了 semi-synthetic 与 anchor 两类验证；eICU 接口已完成，但由于缺少 harmonized CSV，目前是规范化 skip。
4. `M5` 的 LLM 审计接口经历了多轮修复，当前已经能在正式 synthetic 配置上稳定进入 LLM 路径，`llm_audit_fraction` 已从早期的近似 0 提升到正式配置下 `0.67`，进一步修复后在 `M5-only` 复跑中达到 `0.889`。
5. 但从实验结果看，正式 synthetic 配置上 `M5` 目前尚未稳定优于 `M4`，这意味着“LLM 带来的语义增益”已经在工程上接通，但论文层面的优势还没有完全打实。

## 2. 实现过程梳理

### 2.1 Phase 1：执行计划与实现约束固化

首先根据论文草案全文，尤其是方法、伪代码和 M0-M5 实验设计，写出了独立的执行计划文档 `ORBIT_Plus_Execution_Plan.md`。该文档明确了以下硬约束：

- 所有与干预相关的变量、注释、模块，一律使用 `intervention` 或 `assignment logic`，避免使用 `treatment`。
- `E_audit` 必须严格由 `E_route + E_temporal + E_support` 三部分组成。
- temporal eligibility 约束必须采用硬约束式实现，而不能退化为普通 soft penalty。
- `lambda_audit` 必须使用论文中的高维衰减调度式。

这一阶段的意义，是把“论文设定”转化成“代码不可越界的实现合同”，避免后续实现偏离论文。

### 2.2 Phase 2：项目结构落地

按照执行计划，已经在 `/home/xqin5/llmtreatment/orbit_plus` 下完成以下模块结构：

- `data/`：数据 schema、feature metadata、synthetic generator、MIMIC/eICU loader。
- `agents/`：LLM client、prompt、role prior agent、heuristic auditor、oracle auditor、LLM auditor。
- `models/`：feature encoder、temporal gate、role router、subspace aggregator、proxy bridge、nuisance heads。
- `objectives/`：balance、orthogonality、audit energy、bridge bottleneck。
- `training/`：warm-up、主训练器、audit 调度、cross-fitting。
- `evaluation/`：CAIE 指标、routing 指标、overlap 指标、attribution。
- `experiments/`：progressive ablation、degradation、sensitivity、MIMIC confirmation、eICU confirmation。

这一步完成后，ORBIT+ 已经具备完整的“研究代码骨架”，而不是零散脚本。

## 3. 数据处理与数据实现

### 3.1 合成数据设计

论文对合成数据的要求并不是普通 tabular synthetic，而是必须满足：

- 五类角色划分：`confounding`、`intervention`、`outcome`、`proxy`、`post_intervention`
- 多级 intervention
- 完整潜在结局真值，用于 CAIE 评估
- 每个特征的 metadata，用于 temporal gate 和 role prior
- oracle role labels，用于计算 misrouting 指标 `epsilon(g)`

因此没有直接复用 `/home/xqin5/ICMLcompare/proposalversion/advance/confound_diffusion_opm/data/synthetic.py`，而是单独实现了：

- `data/synthetic_generator.py`
- `data/metadata_builder.py`
- `data/schemas.py`

其中 `metadata_builder.py` 为 synthetic feature 生成了具有临床语义的名称和描述，而不是简单占位名。例如：

- confounding 类：`age_at_icu_admission`、`baseline_sofa_score`、`baseline_creatinine`
- intervention 类：`vasopressor_readiness_flag`、`clinician_escalation_signal`
- outcome 类：`baseline_mortality_risk_marker`、`pre_index_metabolic_acidosis_pattern`
- proxy 类：`latent_acuity_surrogate`、`pre_index_monitoring_density`
- post_intervention 类：`post_index_lactate_clearance`、`post_index_pressor_dose_adjustment`

这一步的关键意义在于，后续 `Role_Prior_Agent` 和 `LLM_Auditor_Agent` 不再面对无语义的 feature id，而是面对可区分“基线严重度”“赋值逻辑信号”“结果风险线索”“隐藏严重度代理”“干预后响应”的语义输入。

### 3.2 可复用数据逻辑

从 `/home/xqin5/ICMLcompare/proposalversion/advance/confound_diffusion_opm/data` 中，实际复用了或借鉴了：

- tabular 预处理模式
- split 逻辑
- dataset wrapper 模式
- 离散 intervention 编码思路

但 ORBIT+ 的 synthetic 生成器是全新实现，因为原始合成逻辑不包含论文要求的五角色、metadata、oracle 潜在结局和 post-intervention block。

### 3.3 MIMIC 数据处理

当前机器上可稳定使用的 ICU 真实资产不是论文目标中的 MIMIC-IV vasopressor cohort，而是：

- `/home/xqin5/RLDTALL/RL0903/data_mimic/mimic3_trajectories.csv`

已经基于该文件实现了：

- `data/mimic_loader.py`
- `experiments/run_mimic_confirmation.py`

当前实际使用的列包括：

- 生理变量：`HR`, `MAP`, `SBP`, `DBP`, `RR`, `TEMP`, `SPO2`
- 实验室变量：`CREATININE`, `BILIRUBIN`, `PLATELET`, `LACTATE`
- 支持变量：`FLUID_ML`, `sofa_approx`
- 干预标签：`action_id`
- 真实结局锚点：`terminal`

MIMIC 实验分成两部分：

1. `semi_synthetic`
   - 使用真实 covariates 和 intervention。
   - 通过程序附加潜在结局真值，构造可计算 RMSE 的半合成任务。

2. `anchor`
   - 直接使用现成 `terminal` 作为真实数据锚点。
   - 不再有严格真值，而是给出 pairwise effect 和标准误。

### 3.4 eICU 数据处理

已实现：

- `data/eicu_loader.py`
- `experiments/run_eicu_confirmation.py`
- `configs/eicu_confirmation.yaml`

但当前机器上没有 harmonized eICU CSV，因此实验输出为：

- `status = skipped`
- `reason = No harmonized eICU CSV was found`

这意味着 eICU 入口已经完成，但尚未执行真实 transport/confirmation。

### 3.5 HiRID 预留接口

已实现：

- `data/hirid_gate_calibration.py`

该文件主要用于后续 gate calibration 的真实数据接入，目前还属于接口预留阶段，尚未完成基于真实标注或真实时间戳的 gate 标定实验。

## 4. 算法实现梳理

### 4.1 特征编码与 metadata 融合

对应论文中的 `e_ij = f_theta(x_ij, m_j)`，实现为：

- `models/feature_encoder.py`

该模块将原始 covariate 值与 metadata 矩阵结合，得到每个 feature 的 embedding。metadata 中编码了：

- `relative_time`
- `post_intervention_keyword`
- `missingness_pre_index`
- `always_missing_pre_index`
- `measurement_window`

这一步使 temporal gate 和 role routing 都不再只依赖数值，而是依赖“数值 + feature 类型语义”。

### 4.2 Temporal Eligibility Gate

对应论文中的 temporal eligibility gate：

- `models/temporal_gate.py`

该模块通过 metadata 估计每个特征是否应该进入 pre-intervention confounding 通道。当前 gate 的硬约束作用体现在：

- 如果 gating 关闭，则所有特征默认 eligible。
- 如果 gating 开启，则 confounding role 的质量会受 eligibility 限制。

需要说明的是，现阶段 `gate_precision/gate_recall` 在现有输出中仍然接近 `0.0`。这不是代码没实现，而是说明当前 synthetic 上的 gate calibration 和评价方式还没有打磨到位。

### 4.3 Hard Eligibility Role Router

对应论文中的 hard eligibility-constrained routing：

- `models/role_router.py`

该模块将特征分配到五个角色子空间：

- confounding
- intervention
- outcome
- proxy
- remainder

其中 `remainder` 对应论文中的 post-intervention 或应排除特征。实现上并没有显式命名为 `post_intervention` 子空间，而是通过 `remainder` 承接被排除的 feature，以满足训练图结构的一致性。

### 4.4 子空间聚合与 nuisance heads

实现文件：

- `models/subspace_aggregator.py`
- `models/nuisance_heads.py`

作用包括：

- 将各 role 下的 feature embedding 聚合为 role-specific sample representation
- 用 `confounding + intervention + latent_proxy` 预测 intervention assignment logic
- 用 `confounding + outcome + latent_proxy` 预测各 intervention level 的 potential outcomes

### 4.5 Proxy Bridge

实现文件：

- `models/proxy_bridge.py`
- `objectives/bridge_loss.py`

该模块的作用不是提供严格 proximal identification，而是实现论文中所说的 controlled proxy channel：

- 从 proxy 子空间中提取 latent proxy
- 配合 bridge bottleneck loss 限制不希望泄漏的路径

当前实验中 `M5-noProxy` 和 `M5-noBottleneck` 也都已跑通，因此 proxy bridge 的边际贡献可以单独检查。

### 4.6 目标函数

实现文件：

- `objectives/balance_loss.py`
- `objectives/orthogonality_loss.py`
- `objectives/audit_energy.py`

其中 `audit_energy.py` 是最关键的约束实现，严格按论文拆成三部分：

1. `E_route`
   - 约束 role prior 和当前 routing 之间的一致性。

2. `E_temporal`
   - 对 ordering pairs 逐对施加时序一致性约束。

3. `E_support`
   - 对低 overlap 样本的 unit risk 进行支持约束。

同时 `training/audit_loop.py` 中实现了 `lambda_audit` 的衰减调度。

### 4.7 Cross-fitting 与 DR Estimation

实现文件：

- `training/crossfit.py`

当前的 cross-fitting 会：

- 做 K-fold 切分
- 每折在训练集上训练 ORBIT+
- 在 holdout 上预测 propensity 与 potential outcomes
- 构造 DR pseudo-outcome
- 输出 pairwise effect 和标准误

这一步保证实验结果不是简单的同分布内过拟合评估，而是更接近论文中强调的 cross-fitted DR 估计。

## 5. Agentic Workflow 的实现

### 5.1 Role_Prior_Agent

实现文件：

- `agents/role_prior_agent.py`

它负责论文中的 `LLMInit`。实现上采用：

- 若本地 LLM 可用，则从 feature metadata 生成初始角色先验和 ordering pairs
- 若 LLM 不可用或解析失败，则回退到 heuristic prior

后续修复中，对 ordering pairs 做了上限控制：

- `max_ordering_pairs = 500`

这是为了避免 `p=500` 时 ordering pair 数量爆炸导致 audit energy 中逐对计算过慢。

### 5.2 Heuristic Auditor

实现文件：

- `agents/heuristic_auditor.py`

它是 `M4` 的实现。主要基于：

- weighted SMD
- attribution_intervention
- attribution_outcome
- relative_time

进行规则式角色修正和 exclusion strength 估计。它的优点是稳定，缺点是几乎不使用 feature 语义。

### 5.3 Oracle Auditor

实现文件：

- `agents/oracle_auditor.py`

这是 upper bound 对照，用 metadata 中的 oracle_role 直接构造理想更新，主要用于验证系统上限和理论可达性。

### 5.4 LLM Auditor

实现文件：

- `agents/audit_agent.py`
- `agents/prompts.py`
- `agents/llm_client.py`

这是当前修复最集中的部分。整个修复过程大致经历了以下阶段：

#### 阶段 A：最初版本

最早的 `M5` prompt 非常薄，只喂入少量统计摘要，且要求 LLM 返回整张 `role_prior` 大表。这会导致两个问题：

- LLM 实际获得的语义信息不足，与 heuristic 差异很小
- 本地 7B 模型容易在大 JSON 输出时失败

结果就是 `M5` 经常与 `M4` 几乎重合。

#### 阶段 B：稀疏更新协议

随后将接口改成：

- 不要求返回全量 `role_prior`
- 只要求返回少量 `feature_updates`

同时把输入改成带语义的 feature cards，包括：

- feature name
- description
- relative time
- metadata gate
- weighted SMD
- attribution_intervention
- attribution_outcome
- current top role
- semantic hints

这一步让 LLM 开始真正面对“哪些特征像 assignment logic，哪些像 downstream risk，哪些像 hidden acuity proxy，哪些像 post-index response”。

#### 阶段 C：completion-style 接口

由于 `/home/xqin5/llm/local_llm_server.py` 的 `/v1/chat/completions` 实际只是把消息拼成普通文本再采样，不适合 Llama-2 chat 结构，因此将 `llm_client.py` 改成：

- 使用 `/v1/completions`
- 用 `[INST] <<SYS>> ... [/INST]` 构造 prompt
- 强制要求只返回 JSON

这一步显著提升了输出可控性。

#### 阶段 D：输出压缩与 JSON 容错

为了进一步降低失败率，又进行了三项修复：

1. 把 role 输出压缩成单字符：
   - `c`, `i`, `o`, `p`, `r`

2. 把返回协议进一步压缩成：
   - `[idx, role, strength, exclusion]`

3. 在 `llm_client.py` 中实现：
   - balanced-brace JSON 提取
   - partial JSON salvage

即便 LLM 只返回了接近正确但不完全合法的 JSON，也尽量恢复其中的 `feature_updates`。

#### 阶段 E：重试与复用

在 `audit_agent.py` 中加入：

- compact retry：如果主 prompt 失败，用更小的 prompt 再试一次
- `llm_reuse_last_success`：如果某轮失败，优先复用上一轮成功的 LLM 更新，而不是立即回退 heuristic

这一步的实际效果，是让 `M5` 在正式配置上的 `llm_audit_fraction` 从早期接近 0，提升到：

- formal full ablation 中：`0.667`
- salvage 修复后 `M5-only` 重跑中：`0.889`

## 6. 训练与实验设计

### 6.1 Progressive Isolation Ablation

核心入口：

- `experiments/run_progressive_ablation.py`

已实现的版本包括：

- `M0`: 无 gate、无 routing、无 LLM、无 proxy bridge
- `M1`: 只有 gate
- `M2`: gate + routing
- `M3`: gate + routing + LLMInit + proxy bridge
- `M4`: gate + routing + LLMInit + heuristic audit + proxy bridge
- `M5`: gate + routing + LLMInit + LLM audit + proxy bridge
- `M5-noProxy`
- `M5-noBottleneck`

### 6.2 Smoke 实验

smoke 配置：

- `configs/synthetic_smoke.yaml`

主要用途：

- 快速检查代码链路是否可跑
- 快速定位 `M5` 的回退与 prompt 问题

典型结果见：

- `outputs/progressive_ablation_smoke_m0.json`
- `outputs/progressive_ablation_smoke_m1_m2.json`
- `outputs/progressive_ablation_smoke_m3_m5_gpu.json`

在这类小规模 smoke 中，曾经出现过：

- `M4` 与 `M5` 完全相同

后来通过多轮接口修复，才把 `M5` 真正从 heuristic 中分离出来。

### 6.3 正式 synthetic ablation

正式配置：

- `configs/synthetic.yaml`

当前最重要的结果文件是：

- `outputs/progressive_ablation_llm_v2.json`

其中关键结果如下：

- `M4`
  - `caie_rmse = 0.5615`
  - `epsilon_total = 11.5868`
  - `audit_source_counts = {'heuristic': 9}`

- `M5`
  - `caie_rmse = 0.5621`
  - `epsilon_total = 11.6107`
  - `audit_source_counts = {'llm': 4, 'llm_reuse_last_success': 2, 'heuristic_fallback_error': 3}`
  - `llm_audit_fraction = 0.667`

- `M5-noProxy`
  - `caie_rmse = 0.5606`
  - `llm_audit_fraction = 0.778`

- `M5-noBottleneck`
  - `caie_rmse = 0.5623`
  - `llm_audit_fraction = 0.556`

从这些结果可以得到三点判断：

1. `M5` 已经是真正走 LLM 路径，而不是假性等于 `M4`。
2. 但在正式 synthetic 上，`M5` 目前还没有稳定优于 `M4`。
3. proxy bridge 与 bottleneck 的贡献仍需进一步分析，因为当前 `M5-noProxy` 并未明显更差。

### 6.4 `M5` 单独 salvage 复跑

为了验证 partial JSON salvage 修复的效果，又单独重跑了 `M5`：

- 输出：`outputs/progressive_ablation_m5_salvage.json`

结果：

- `caie_rmse = 0.5628`
- `llm_audit_fraction = 0.889`
- `audit_source_counts = {'llm': 7, 'llm_reuse_last_success': 1, 'heuristic_fallback_error': 1}`

该结果说明：

- 新修复显著提升了 LLM 实际参与比例
- 但更高的 LLM 参与率并没有自动转化为更低的 RMSE

也就是说，当前的主要瓶颈已从“接口打不通”转向“LLM 提供的语义更新是否真的优于 heuristic”。

### 6.5 Degradation 与 Sensitivity

当前已经有 smoke 结果：

- `outputs/degradation_smoke.json`
- `outputs/sensitivity_smoke.json`

degradation smoke 中：

- random corruption 从 `0.0` 到 `0.3` 时，RMSE 从 `0.5781` 升到 `0.5797`
- adversarial corruption 下，`epsilon_total` 升到 `5.0380`
- `post_intervention_recall` 降到 `0.75`

说明系统在对抗性误导下会明显恶化，但不会完全崩溃。

sensitivity smoke 中：

- `audit_interval=2` 与 `4` 差异较小
- `latent_proxy_dim=0` 时 RMSE 反而优于 `4`

需要注意，这部分 smoke 结果主要用于工程调试，不能直接当成论文主结论。

### 6.6 MIMIC confirmation

结果文件：

- `outputs/mimic_confirmation.json`

semi-synthetic 结果：

- `M0`: RMSE `0.0415`
- `M2`: RMSE `0.0243`
- `M4`: RMSE `0.0275`
- `M5`: RMSE `0.0275`

anchor 结果：

- `M2`, `M4`, `M5` 都已输出 pairwise effect 与标准误

解读：

- ORBIT+ 的结构化版本在 MIMIC 真实 covariates 上优于最基础的 `M0`
- 但 `M4` 和 `M5` 仍然重合，说明真实资产上 LLM 的边际收益尚未被显著拉开

### 6.7 eICU confirmation

结果文件：

- `outputs/eicu_confirmation.json`

当前状态：

- `status = skipped`

原因：

- 没有找到 harmonized eICU CSV

因此这部分只能算接口和 schema 已完成，还没有实际实验结果。

## 7. 当前已经解决的问题

截至目前，已经明确解决了以下工程问题：

1. ORBIT+ 从论文草案变成了完整的代码骨架，不再停留在设计层。
2. synthetic generator 不再是无语义占位，而是具备临床语义 feature metadata。
3. GPU 训练路径已经打通。
4. 本地 7B LLM server 已经能作为 ORBIT+ 的 auditor 后端工作。
5. `M5` 最关键的工程问题，即“频繁回退到 heuristic 导致名义上是 M5，实际上还是 M4”，已经被显著缓解。
6. 审计来源、更新数量、LLM 参与比例都已经被纳入日志，可观测性明显增强。

## 8. 当前仍然存在的 gap

### 8.1 理论与实验的核心 gap

从论文视角看，目前最大的 gap 不是“系统跑不起来”，而是：

- `M5` 已经能调用 LLM
- 但还没有稳定、清晰地优于 `M4`

这意味着论文中“LLM 提供超越 heuristic 的语义判断”这一主张，在正式 synthetic 上还没有被充分证明。

### 8.2 gate 指标不足

当前输出中 `gate_precision` 和 `gate_recall` 普遍接近 `0.0`。这说明：

- gate 模块已经存在
- 但评估方式、标定方式或 synthetic 对应关系还不够好

这一点如果不修，会削弱论文关于 temporal eligibility 的说服力。

### 8.3 真实数据仍不够完整

目前真实数据部分的缺口包括：

- 尚未使用论文目标中的 MIMIC-IV vasopressor cohort
- eICU 尚未真正跑通
- HiRID gate calibration 尚未完成

因此真实数据部分目前更适合作为“初步确认”，而不是完整论文级证据。

## 9. 下一步建议

下一阶段建议按照优先级推进：

1. 继续精简 `M5` 输出协议
   - 尝试只返回 `[idx, role, exclusion]`
   - 进一步压低 `heuristic_fallback_error`

2. 跑 formal degradation
   - 检查高 `llm_audit_fraction` 下，系统对错误审计的鲁棒性是否更合理

3. 跑 formal sensitivity
   - 现在 sensitivity 已切到 `llm`
   - 可以开始正式比较 audit frequency 与 proxy dim 对 `M5` 的影响

4. 用更贴近论文设定的真实 cohort 替换当前 MIMIC-III 资产
   - 尤其是 MIMIC-IV vasopressor cohort

5. 补 eICU harmonized 数据
   - 真正完成 transport/confirmation

## 10. 阶段性评价

如果把当前成果分成“工程完成度”和“论文说服力”两个维度来评价：

- 工程完成度：已经达到中高水平
  - 主系统、实验脚本、真实数据入口、LLM agent、GPU 跑批都具备了

- 论文说服力：目前是中等
  - `M0-M5` 证据链已经存在
  - 但 `M5 > M4` 还不够稳定
  - 真实数据部分还需要更强 cohort 和更多验证

因此当前状态可以概括为：

“ORBIT+ 已经从概念稿进入可反复实验的系统原型阶段；最重要的工程阻塞已基本清除，但要达到强论文结论，还需要继续把 LLM 的语义增益、gate 标定和真实数据验证做扎实。”

## 11. 2026-04-14 夜间补充分析

在初版阶段性报告完成后，又补做了三项更关键的分析，用来判断论文主叙事应该继续强调 “LLM strong gain”，还是暂时转向 “bounded LLM harm + structured fallback safety”。

### 11.1 `M4` vs `M5` 的 feature subset 级差异分析

新增输出：

- `outputs/m4_m5_feature_subset_analysis.json`

该分析直接比较 formal synthetic 配置下 `M4` 与 `M5` 的 feature-level routing 差异，并按 oracle subset 汇总：

- `J_I`：intervention features
- `J_P`：proxy features
- `J_post`：post-intervention features

核心结果：

1. `M4` 与 `M5` 只有 **1 个 feature** 出现 top-role 差异。
2. 这 1 个 feature 位于 `outcome` subset，而不是 `intervention`、`proxy` 或 `post_intervention`。
3. 具体差异 feature 为：
   - `baseline_mortality_risk_marker_018`
   - `M4` 将其 top-role 判为 `intervention`
   - `M5` 将其 top-role 判为 `outcome`
4. 对最关心的三个 subset：
   - `J_I`：`M4` 与 `M5` 几乎完全相同
   - `J_P`：`M4` 与 `M5` 几乎完全相同
   - `J_post`：`M4` 与 `M5` 完全相同，且两者 top1 accuracy 都为 `1.0`

具体数值：

- `intervention`
  - `M4 oracle_role_mass_mean = 0.2960`
  - `M5 oracle_role_mass_mean = 0.2961`
  - `M4 top1_accuracy = 0.0`
  - `M5 top1_accuracy = 0.0`

- `proxy`
  - `M4 oracle_role_mass_mean = 0.3026`
  - `M5 oracle_role_mass_mean = 0.3047`
  - `M4 top1_accuracy = 1.0`
  - `M5 top1_accuracy = 1.0`

- `post_intervention`
  - `M4 oracle_role_mass_mean = 0.7059`
  - `M5 oracle_role_mass_mean = 0.7059`
  - `M4 top1_accuracy = 1.0`
  - `M5 top1_accuracy = 1.0`

这一结果的论文含义非常明确：

- 当前版本还不能支持“LLM 的优势主要发生在 heuristic 难以区分 intervention predictor、proxy 与 collider 的边界区域”这一主张。
- 至少在当前 formal synthetic 配置下，`M5` 的局部增益并没有出现在 `J_I / J_P / J_post`，而只出现在 1 个 `outcome` feature 上。

这意味着后续论文表述需要谨慎。除非后面在更强的语义 synthetic 或真实数据上找到更明确的局部收益，否则 “LLM strong gain” 不应作为当前版本的主结论。

### 11.2 gate evaluation 失效原因诊断

新增输出：

- `outputs/gate_diagnostics.json`

这项分析不是只看旧的 `gate_precision/gate_recall`，而是直接检查：

- gate 的 ground truth 是什么
- 各 oracle role 的 gate score 分布是什么
- threshold sweep 下 precision/recall 会如何变化

首先，ground truth 在当前 synthetic 中应理解为：

- 如果从 safety exclusion 角度评价：
  - `post_intervention` 为正类
  - 目标是让这些特征 gate 低

- 如果从 pre-index eligibility 角度评价：
  - `non-post_intervention` 为正类
  - 目标是让这些特征 gate 高

当前旧指标为 `post_threshold = 0.2`，结果是：

- `precision = 0.0`
- `recall = 0.0`

但进一步诊断发现，问题不只是 threshold 选错，而是 gate 本身也没有学到预期方向。formal synthetic 下各角色平均 gate 分数为：

- `confounding = 0.0788`
- `proxy = 0.1571`
- `intervention = 0.2199`
- `outcome = 0.3135`
- `post_intervention = 0.5019`

也就是说，当前 gate 的排序方向与“post-intervention 应该更低”这一安全直觉相反，`post_intervention` 反而拥有最高的平均 gate score。

因此，gate 指标接近 0 的原因是两个问题叠加：

1. 旧的评价方式和阈值本身不合理。
2. 当前训练下 gate 自身确实没有学到正确的安全方向。

进一步的 threshold sweep 结果：

- 最佳 `eligible` threshold 在当前配置下接近 `0.1`
  - precision `0.7105`
  - recall `0.6923`

- 最佳 `post` threshold 在当前配置下接近 `0.6`
  - precision `0.22`
  - recall `1.0`

这说明：

- 旧的 `0.2` post threshold 不能再作为正式论文指标使用。
- 但即使换阈值，gate 当前也只是“有一定区分度”，还远不能算真正可靠的 primary safety mechanism。

因此，后续最值得修的不是 prompt，而是 gate 的学习目标、初始化或标定方式。

### 11.3 formal degradation：错误 auditor 下是否平滑退化

新增输出：

- `outputs/degradation_formal.json`

本次 formal degradation 覆盖：

1. random corruption
   - `corruption_fraction = 0.0, 0.1, 0.3, 0.5`

2. adversarial corruption
   - `D2-adversarial`
   - `D3-adversarial-fixed-confidence`

关键结果如下。

#### random corruption

- `0.0`
  - `caie_rmse = 0.5619`
  - `epsilon_total = 11.6038`
  - `post_intervention_recall = 1.0`

- `0.1`
  - `caie_rmse = 0.5624`
  - `epsilon_total = 11.8477`
  - `post_intervention_recall = 0.9091`

- `0.3`
  - `caie_rmse = 0.5624`
  - `epsilon_total = 12.3710`
  - `post_intervention_recall = 0.9091`

- `0.5`
  - `caie_rmse = 0.5628`
  - `epsilon_total = 12.4570`
  - `post_intervention_recall = 0.8182`

#### adversarial corruption

- `D2-adversarial`
  - `caie_rmse = 0.5615`
  - `epsilon_total = 12.0992`
  - `post_intervention_recall = 0.8182`

- `D3-adversarial-fixed-confidence`
  - `caie_rmse = 0.5625`
  - `epsilon_total = 12.1055`
  - `post_intervention_recall = 0.8182`

这组结果的意义是：

1. 当 auditor 逐步被污染时，`epsilon_total` 确实单调上升。
2. `post_intervention_recall` 从 `1.0` 下降到 `0.8182`，说明错误审计会侵蚀安全边界。
3. 但 `caie_rmse` 只是在 `0.5619 -> 0.5628` 的范围内轻度波动，并没有出现灾难性失控。

因此，在当前版本下，比 “LLM strong gain” 更容易先成立的主张，确实是：

“Even when the auditor is wrong, the system degrades gracefully rather than catastrophically collapsing.”

当然，这条结论仍需要和 gate 的真实作用拆开看，因为当前 gate 本身还不够理想。更准确地说，当前 formal degradation 更支持：

- structured audit corruption 不会立刻把系统打崩
- 但这还不能直接等价为 “eligibility gate 已经提供了强安全下界”

因为 gate 的学习方向本身还存在问题。

### 11.4 gate sanity：如果直接给 oracle metadata gate，会发生什么

新增输出：

- `outputs/gate_sanity.json`

该实验保持 `M2` 的主体结构不变，只比较两种 gate 来源：

1. `M2-learnedGate`
2. `M2-oracleMetadataGate`

其中 `oracleMetadataGate` 不是新的论文方法，而是一个纯诊断上界：

- 对于 `post_intervention` 特征，gate 固定为 0
- 对于非 `post_intervention` 特征，gate 固定为 1

结果如下：

- `M2-learnedGate`
  - `caie_rmse = 0.5639`
  - `epsilon_total = 11.6434`
  - `gate_precision = 0.0`
  - `gate_recall = 0.0`

- `M2-oracleMetadataGate`
  - `caie_rmse = 0.5651`
  - `epsilon_total = 6.1133`
  - `gate_precision = 1.0`
  - `gate_recall = 1.0`

这组结果非常重要，因为它说明：

1. 如果只看 routing safety，当前 learned gate 的确是明显瓶颈。
   - `epsilon_total` 几乎减半：`11.64 -> 6.11`

2. 但如果只看 CAIE RMSE，oracle gate 并没有自动带来更低误差。
   - `0.5639 -> 0.5651`

因此可以得到更细的判断：

- gate 现在首先是一个 **routing safety bottleneck**
- 但不是当前 **overall CAIE error** 的唯一决定因素

这意味着下一阶段修 gate 是必要的，但即便 gate 修好，也不保证 `M5` 会自动超过 `M4`。后续仍然需要同时检查：

- nuisance heads 是否足够强
- outcome / intervention / proxy 三个子空间是否真的被学开
- audit update 是否在真正有信息量的 subset 上发生
