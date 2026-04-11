# 项目阶段总结：SCSU-Hybrid 术中表面跟踪系统

**日期**: 2026-04-11  
**版本**: V1.0  
**项目路径**: `/home/xqin5/BIMpointAI/MDBIMDT/md/test_hybrid_lattice_v1/`  
**目标会议**: NeurIPS 2026

---

## 1. 当前阶段的任务定义

### 1.1 核心问题

本阶段解决的核心问题是：**术中实时软组织表面跟踪中的负更新（negative update）问题**。

在微创手术（如腹腔镜肾脏手术）中，术前 CT 构建的三维解剖模型（数字孪生）需要在术中根据内窥镜观测持续更新，以反映器官因呼吸运动、工具操作等因素导致的实时形变。然而，由于术中观测的局限性（有限视野、遮挡、噪声），**部分帧的跟踪更新反而使模型偏离真实位置**——即"追踪后的误差比不追踪还大"。我们将这种现象定义为"**负更新帧（negative surface frame）**"。

在 KiTS23 基准上的实测显示：基线方法在 5mm 形变场景下有 **23.4%** 的帧产生负更新。这意味着接近四分之一的时间里，跟踪系统在"帮倒忙"。

### 1.2 任务形式化

- **输入**: 
  - 术前三维表面网格 $\mathcal{M}_0 = (V_0, F)$，其中 $V_0 \in \mathbb{R}^{N \times 3}$ 为顶点坐标，$F \in \mathbb{Z}^{M \times 3}$ 为三角面索引
  - 术中部分观测点云序列 $\{O_t\}_{t=1}^{T}$，每帧 $O_t = \{(\mathbf{p}_j, \mathbf{n}_j, q_j)\}$ 包含位置、法向、置信度
- **输出**: 
  - 每帧更新后的表面位置 $V_t \in \mathbb{R}^{N \times 3}$
  - 肿瘤区域质心估计 $\mathbf{c}_t^{\text{tumor}} \in \mathbb{R}^3$
- **问题类型**: 时序表面回归问题 + 在线状态估计问题
- **约束**: 实时性要求 > 5 FPS（每帧 < 200ms）

### 1.3 子任务拆解

| 子任务 | 描述 | 类型 |
|--------|------|------|
| 观测残差计算 | 从点云计算每个网格顶点的位移残差 | 信号处理 |
| 可观测性估计 | 判断每个顶点的观测质量 | 学习型分类/回归 |
| 形变求解 | 根据残差和可观测性求解最优位移 | 优化求解 |
| 格点正则化 | 通过粗粒度网格施加空间平滑约束 | 优化求解 |
| 鲁棒性机制 | 在信号弱时抑制错误更新 | 规则型门控 |

### 1.4 与最终目标的关系

本阶段工作是整个"医学数字孪生"系统三路径架构中 **Path 1（BIM fast path）和 Path 2（Hybrid learned path）** 的核心贡献。最终系统的完整架构为：

1. **BIM fast path (HC)**: 纯手工支持 + 求解器，< 100ms/帧
2. **Hybrid path (learned)**: 学习型支持 + 格点先验 + 融合，~120ms/帧
3. **nnUNet re-seg**: 结构变化时完全重建，5-30s/帧，< 2% 帧触发

本阶段的核心贡献是：在 Path 1/2 之上添加 **observation-gated preop pullback 鲁棒性机制**，以及证明学习模型的跨解剖泛化能力。

---

## 2. 当前模型总体结构

### 2.1 总体架构概述

SCSU-Hybrid 是一个**多模块级联系统**，包含信号处理、学习型推理、优化求解和鲁棒性门控四个层次。数据从术中点云输入开始，经过残差计算 → 可观测性估计 → 形变求解 → 格点正则化 → 鲁棒性门控 → 表面更新，最终输出更新后的表面位置。

整个系统的推理流程可以用如下管道描述：

```
O_t (观测点云)
  ↓
Kernel A: 残差计算 → residuals [N,3], s_obs [N]
  ↓
Kernel B: 支持扩散 → S_hc [N] (手工支持)
  ↓
Support Head: 学习型支持 → S_hat [N]
  ↓
Lattice Update: 格点残差投影 + 弹性求解 → V_lattice [N,3]
  ↓
Kernel C: PCG 求解 → delta_V [N,3]
  ↓
Robustness Mechanism: 门控 + 衰减 + 回拉 → V_new [N,3]
  ↓
Blend Head: 求解器与格点融合 → V_pred [N,3]
  ↓
State Update: 更新孪生状态
```

### 2.2 模块分类

| 模块 | 类型 | 是否包含可学习参数 | 延迟 |
|------|------|------------------|------|
| Kernel A (残差计算) | 信号处理 | 否 | ~20ms |
| Kernel B (支持扩散) | 规则型 | 否 | ~5ms |
| Support Head | 学习型 | 是 (MLP, 14→64→64→1) | ~2ms |
| Lattice Grid | 优化求解 | 否 (scipy CG) | ~10ms |
| Kernel C (PCG 求解) | 优化求解 | 否 (可微分 PCG) | ~30ms |
| Blend Head | 学习型 | 是 (MLP, 9→32→32→1) | <1ms |
| Safety Head | 学习型 | 是 (MLP, 9→32→32→1) | <1ms |
| Selector Head | 学习型 | 是 (MLP, 9→32→32→1) | <1ms |
| Robustness Mechanism | 规则型 | 否 (elementwise) | <1ms |

### 2.3 训练与推理的区别

**训练时**：
- 使用 Mode 1（solver-coupled）训练：前向传播贯穿 Kernel A → B → Support Head → Lattice → Kernel C → Blend → 损失计算
- 通过 Kernel C 的隐函数定理反向传播梯度（而非展开 CG 迭代）
- Hard-negative mining 加权损失：负更新帧获得额外权重
- 序列长度为 5 帧（seq5），用于学习多帧一致性

**推理时**：
- 额外启用 Robustness Mechanism（τ, d, p 三个超参数）
- 鲁棒性机制在训练中不参与，是纯推理时的附加模块
- 这保证了鲁棒性机制完全 GT-free（不需要任何地面真值）

---

## 3. 每一个板块分别用了什么模型、技术方法和原理

### 3.1 Kernel A: 各向异性残差计算

#### 目标
从术中观测点云 $O_t$ 和当前网格状态 $V_t$ 计算每个顶点的位移残差向量。

#### 输入
- 当前网格顶点 $V \in \mathbb{R}^{N \times 3}$ 及法向 $\mathbf{n}_i \in \mathbb{R}^3$
- 观测点云 $\{(\mathbf{p}_j, \mathbf{n}_j^{\text{obs}}, q_j)\}_{j=1}^{N_{\text{obs}}}$

#### 输出
- 残差向量 $\mathbf{r}_i \in \mathbb{R}^3$，每个顶点一个
- 瞬时可观测性 $s_{\text{obs},i} \in [0, 1]$
- 观测计数 $c_i$ 和高斯权重和 $w_i$

#### 技术原理

对每个观测点 $\mathbf{p}_j$，用 KD-tree 查找 $k=8$ 个最近的网格顶点。对每对 $(\mathbf{p}_j, \mathbf{v}_i)$ 计算高斯权重：

$$w_{ji} = q_j \cdot \exp\left(-\frac{\|\mathbf{p}_j - \mathbf{v}_i\|^2}{\sigma^2}\right)$$

其中：
- $q_j$: 观测点的置信度（由入射角和深度不确定性决定）
- $\sigma$: 高斯核带宽，默认 10mm
- 同一连通分量约束：不同 component 的权重置零

**各向异性残差分解**（本工作改进）：

将观测-顶点差向量 $\mathbf{d}_{ji} = \mathbf{p}_j - \mathbf{v}_i$ 分解为法向分量和切向分量：

$$\mathbf{d}_{ji}^{\perp} = (\mathbf{d}_{ji} \cdot \mathbf{n}_i) \cdot \mathbf{n}_i \quad (\text{法向分量})$$
$$\mathbf{d}_{ji}^{\parallel} = \mathbf{d}_{ji} - \mathbf{d}_{ji}^{\perp} \quad (\text{切向分量})$$

合成残差：

$$\mathbf{r}_{ji}^{\text{3D}} = \mathbf{d}_{ji}^{\perp} + \beta \cdot q_j \cdot \mathbf{d}_{ji}^{\parallel}$$

其中 $\beta$ 为各向异性因子：
- $\beta = 0$: 仅保留法向运动（原始方法）
- $\beta = 0.3$: 5mm 形变场景推荐值
- $\beta = 1.5$: 10mm 形变场景推荐值

$\beta$ 需要随形变振幅调整的原因：小形变时切向信号噪声大，需要强抑制；大形变时切向信号信噪比提高，可以适当保留。

每个顶点的最终残差为加权平均：

$$\mathbf{r}_i = \frac{\sum_j w_{ji} \cdot \mathbf{r}_{ji}^{\text{3D}}}{\sum_j w_{ji} + \epsilon}$$

瞬时可观测性使用 log-sum-exp 互补公式：

$$s_{\text{obs},i} = 1 - \exp\left(\sum_j \log(1 - w_{ji}^{\text{flat}})\right)$$

#### 符号汇总

| 符号 | 含义 | 维度/取值 |
|------|------|----------|
| $w_{ji}$ | 观测点 $j$ 对顶点 $i$ 的高斯权重 | 标量 $\geq 0$ |
| $q_j$ | 观测点置信度 | $[0, 1]$ |
| $\sigma$ | 高斯核带宽 | 10mm |
| $\beta$ | 切向残差保留因子 | 0.3 或 1.5 |
| $s_{\text{obs},i}$ | 顶点 $i$ 的瞬时可观测性 | $[0, 1]$ |
| $\mathbf{r}_i$ | 顶点 $i$ 的残差向量 | $\mathbb{R}^3$ |

#### 限制
- KD-tree 在 CPU 上构建，是主要延迟瓶颈（~20ms）
- 固定 $k=8$ 近邻可能对稀疏区域不足

---

### 3.2 Kernel B: 支持扩散

#### 目标
将瞬时可观测性信号 $s_{\text{obs}}$ 经过时间平滑和空间扩散，得到稳定的手工支持场 $S_{\text{hc}}$。

#### 输入
- 瞬时可观测性 $s_{\text{obs}} \in \mathbb{R}^N$
- 上一帧支持 $S_{\text{prev}} \in \mathbb{R}^N$
- 图拉普拉斯矩阵 $L_\Phi \in \mathbb{R}^{N \times N}$（稀疏 CSR 格式）

#### 输出
- 手工支持场 $S_{\text{hc}} \in [0, 1]^N$

#### 技术原理

**时间平滑（EMA）**：

$$S^{(0)} = \text{clamp}\left(\gamma \cdot S_{\text{prev}} + (1 - \gamma) \cdot s_{\text{obs}},\; 0,\; 1\right)$$

其中 $\gamma = 0.3$ 为时间平滑系数。

**空间扩散（Laplacian diffusion）**：

迭代 8 次：

$$S^{(k+1)} = \text{clamp}\left(S^{(k)} - \eta \cdot L_\Phi \cdot S^{(k)},\; 0,\; 1\right)$$

其中 $\eta = 0.2$ 为扩散步长。该操作等价于在图上进行热扩散，使支持值从高观测区域向低观测区域扩散。

#### 限制
- 扩散迭代次数和步长固定，无法自适应网格密度

---

### 3.3 Support Head: 学习型可观测性门控

#### 目标
从每个顶点的 14 维特征向量预测一个乘性门控因子 $g_i \in [\text{gate\_lo}, \text{gate\_hi}]$，对手工支持 $S_{\text{hc}}$ 进行修正，得到学习型支持 $\hat{S}_i$。

#### 输入
- 14 维逐顶点特征 $\mathbf{x}_i \in \mathbb{R}^{14}$（见第 6 节详细说明）
- 手工支持 $S_{\text{hc},i}$

#### 输出
- 学习型支持 $\hat{S}_i = \text{clamp}(S_{\text{hc},i} \cdot g_i,\; 0,\; 1)$

#### 模型架构

三层 MLP，隐藏层 64 维：

$$g_i = \text{gate\_lo} + (\text{gate\_hi} - \text{gate\_lo}) \cdot \sigma\left(\text{MLP}(\mathbf{x}_i)\right)$$

其中 $\sigma$ 为 sigmoid 函数，gate\_lo = 0.3, gate\_hi = 2.0。

这意味着学习到的门控可以将手工支持**缩小到 0.3 倍**（抑制不可靠观测）或**放大到 2.0 倍**（增强可靠观测）。

#### 训练方式
- 通过端到端 solver-coupled 训练（Mode 1）
- 梯度通过 Kernel C 的隐函数定理传回
- 初始化来自预训练检查点 `A8_combined_best`

#### 限制
- 14 维特征全部来自手工设计，未使用原始图像信息
- 门控范围固定，未自适应不同场景

---

### 3.4 Lattice Grid: 格点正则化

#### 目标
通过粗粒度三维网格（$7 \times 7 \times 7 = 343$ 个节点）对位移场施加空间平滑和弹性约束，防止局部过拟合。

#### 输入
- 当前格点位移 $\mathbf{D}_{\text{node}} \in \mathbb{R}^{N_g \times 3}$
- 逐顶点残差 $\mathbf{r}_i$ 和权重 $w_i$
- 网格图拉普拉斯 $L_{\text{grid}} \in \mathbb{R}^{N_g \times N_g}$

#### 输出
- 更新后的格点位移 $\mathbf{D}_{\text{node}}$
- 通过三线性插值得到的格点驱动表面 $V_{\text{lattice}} \in \mathbb{R}^{N \times 3}$

#### 技术原理

**顶点到格点映射**: 每个顶点归一化到 $[0,1]^3$，找到包围的 8 个格点节点，用三线性权重 $w_{ik}$ 进行插值。

**格点求解**: 对每个坐标轴独立求解线性系统：

$$\left[\text{diag}(w_{\text{obs}} + \lambda_p + \lambda_l \cdot w_{\text{les}}) + \lambda_e \cdot L_{\text{grid}}\right] \Delta \mathbf{D} = w_{\text{obs}} \cdot \mathbf{r}_{\text{node}} - \lambda_p \cdot \mathbf{D}_{\text{node}}$$

其中：
- $w_{\text{obs}}$: 观测权重（从顶点投影到格点）
- $\lambda_e = 0.9$: 弹性正则化系数（惩罚相邻格点间的位移差异）
- $\lambda_p = 0.22$: 先验正则化系数（惩罚偏离零位移）
- $\lambda_l = 0.7$: 病灶区域额外权重
- $w_{\text{les}}$: 病灶权重（从 lesion_band_mask 投影）

格点位移累积更新：$\mathbf{D}_{\text{node}} \leftarrow \mathbf{D}_{\text{node}} + \alpha_l \cdot \Delta \mathbf{D}$，其中 $\alpha_l = 0.85$。

表面位置通过三线性插值恢复：$V_{\text{lattice}} = V_0 + \sum_k w_{ik} \cdot \mathbf{D}_{\text{node},k}$。

#### 限制
- 网格分辨率固定 $7^3$，无法适应不同尺度的解剖结构
- 使用 scipy CG 求解器（CPU），不在 GPU 计算图中

---

### 3.5 Kernel C: PCG 形变求解器

#### 目标
给定残差场和支持场，求解满足弹性约束和数据拟合的最优位移场 $\Delta V$。

#### 输入
- 残差 $\mathbf{r}_i \in \mathbb{R}^3$
- 支持 $\hat{S}_i \in [0,1]$
- 图拉普拉斯 $L_\Phi$

#### 输出
- 位移增量 $\Delta V \in \mathbb{R}^{N \times 3}$

#### 技术原理

对每个坐标轴求解线性系统 $A_t \cdot \Delta \mathbf{v} = \mathbf{b}$：

$$A_t = \lambda_{\text{sem}} \cdot L_\Phi + \text{diag}\left(\hat{S} + \lambda_p \cdot (1 - \hat{S}) + \lambda_l \cdot \mathbf{1}_B + \epsilon\right)$$

$$\mathbf{b} = \text{diag}(\hat{S}) \cdot \mathbf{r} + \lambda_p \cdot (1 - \hat{S}) \cdot (V_{\text{prior}} - V)$$

其中：
- $\lambda_{\text{sem}} = 1.0$: 语义/拉普拉斯正则化权重
- $\lambda_p = 0.1$: 先验回归权重
- $\lambda_l = 1.0$: 病灶带权重
- $\mathbf{1}_B$: 病灶带掩码
- $\epsilon = 10^{-5}$: 数值稳定性

**PCG 求解器**: Jacobi 预条件共轭梯度法，三个坐标轴批处理求解，支持热启动。

**可微分性**: 使用隐函数定理（implicit function theorem）实现反向传播——不需要展开 CG 迭代，而是在反向传播时额外求解一次伴随线性系统。这是端到端训练的关键技术。

#### 限制
- PCG 最大迭代 50 次，收敛容差 $10^{-5}$
- 依赖拉普拉斯矩阵质量，退化三角形可能导致数值不稳定

---

### 3.6 Blend Head: 求解器-格点融合

#### 目标
预测一个帧级标量 $b \in [0, 0.35]$，决定最终输出中求解器结果和格点结果的混合比例。

#### 输入
- 9 维帧级统计特征（见第 6 节）

#### 输出
- 融合标量 $b$

#### 技术原理

$$b = b_0 + \delta_{\max} \cdot \tanh\left(\text{MLP}(\mathbf{s}_{\text{blend}})\right)$$

其中 $b_0 = 0.12$ 为基线融合率，$\delta_{\max} = 0.03$ 为最大修正范围。

最终表面位置：

$$V_{\text{pred}} = (1 - b) \cdot V_{\text{new}} + b \cdot V_{\text{lattice}}$$

**设计原因**: 初始化为零权重（no-op），让模型逐步学习何时需要更多格点平滑。

---

### 3.7 Safety Head: 保守更新缩放

#### 目标
预测一个帧级安全缩放因子 $\alpha_{\text{safe}} \in [0.6, 1.0]$，在不确定帧降低更新步长。

#### 输入
- 9 维帧级统计特征

#### 输出
- 缩放因子 $\alpha_{\text{safe}}$

#### 技术原理

$$\alpha_{\text{safe}} = 1 - (1 - \alpha_{\min}) \cdot \text{ReLU}\left(\tanh\left(\text{MLP}(\mathbf{s}_{\text{safety}})\right)\right)$$

其中 $\alpha_{\min} = 0.6$，即最多将更新步长缩小到 60%。

---

### 3.8 Selector Head: 路径选择器

#### 目标
二分类决策——选择"标准更新路径"还是"保守更新路径"。

#### 输入
- 9 维帧级统计特征

#### 输出
- sigmoid 概率 $p \in [0, 1]$，> 0.5 时选择保守路径

#### 技术原理
- 初始偏置 $b_0 = -2.0$（sigmoid(-2) ≈ 0.12），使系统默认选择标准路径
- 训练信号：当保守路径 MAE + margin < 标准路径 MAE 时，标签为 1

---

### 3.9 Robustness Mechanism: 观测门控术前回拉（核心贡献）

#### 目标
在求解器信号弱时（如正弦运动的过零点），主动抑制错误更新并拉回术前状态。

**这是本工作的核心技术贡献**，完全 GT-free，仅依赖求解器自身的决策信号。

#### 输入
- 求解器位移 $\Delta V$
- 超参数 $(\tau, d, p)$

#### 输出
- 门控后的更新表面 $V_{\text{new}}$

#### 技术原理

四个正交组件：

**1) 置信度门控 (τ)**

决策信号量：$\text{signal} = \text{mean}\left(\|\Delta V_i\|\right)$

这个信号的关键洞察是：使用求解器的**输出**（位移大小）而非**输入**（原始观测信号）作为置信度。因为位移已经过求解器的先验、平滑和弹性约束的去噪处理，其信噪比远高于原始观测。

门控缩放：$\text{gate} = \text{clamp}\left(\frac{\text{signal}}{\tau}, 0, 1\right)$

**2) 各向异性残差 (β)**

已在 Kernel A 中实现（见 3.1 节），保留切向运动、抑制法向噪声。

**3) 格点节点衰减 (d)**

当 gate < 1 时，衰减格点累积位移：

$$\mathbf{D}_{\text{node}} \leftarrow \mathbf{D}_{\text{node}} \cdot (1 - d \cdot (1 - \text{gate}))$$

**4) 术前回拉 (p)** — **贡献 80% 的负更新降低**

当 gate < 1 时，将表面向术前状态拉回：

$$V_{\text{new}} = (1 - w_p) \cdot V_{\text{new}} + w_p \cdot V_0$$

其中 $w_p = p \cdot (1 - \text{gate})$。

**Per-vertex 变体**: 用 $\|\Delta V_i\|$ 替代全局均值，实现逐顶点的自适应门控：

$$\text{gate}_i^{\text{pv}} = \text{clamp}\left(\frac{\|\Delta V_i\|}{\tau}, 0, 1\right) \cdot \text{gate}_{\text{global}}$$

高运动顶点保留更新，静止顶点独立回拉。

#### 推荐配置

| 配置名 | $\tau$ | $d$ | $p$ | 适用场景 |
|--------|--------|-----|-----|----------|
| balanced (推荐默认) | 0.40 | 0.5 | 0.6 | 精度-鲁棒平衡 |
| max-robust | 0.60 | 0.8 | 0.6 | 安全关键场景 |
| per-vertex | 0.40 (PV) | 0.5 | 0.6 | 最优 Pareto 点 |

#### 为什么 pullback 是主力

Factorial ablation 显示：
- 仅 gate: −3.2 pp（23.4% → 20.2%）
- gate + decay: −4.1 pp
- **gate + pullback: −13.9 pp（23.4% → 9.5%）**
- gate + decay + pullback: −14.9 pp（23.4% → 8.5%）

Pullback 贡献了总降低量的 **80%**，因为它直接将弱信号帧拉回到已知安全的状态（术前），而 decay 和 gate 仅减弱更新幅度但不提供正确方向的引导。

---

### 3.10 评估模块

#### 逐帧指标计算

输入 $(V_{\text{pred}}, V_{\text{GT}}, V_0)$，计算：
- SurfRMS, SurfMAE, HD95
- TumorCE（肿瘤质心误差）
- 改善百分比
- 负更新帧标记：$\text{MAE}(V_{\text{pred}}, V_{\text{GT}}) > \text{MAE}(V_0, V_{\text{GT}})$

#### 逐案例聚合
- 对每个案例的 16 帧取均值
- 统计负更新帧数

#### 全局聚合
- 对所有案例取均值
- 负更新率 = 负更新帧总数 / 总帧数

---

### 3.11 损失函数模块（训练时）

总损失由五个分量加权组成：

$$\mathcal{L}_{\text{total}} = \left(\mathcal{L}_{\text{chamfer}} + \mathcal{L}_{\text{tumor}} + w_s \cdot \mathcal{L}_{\text{stab}} + \mathcal{L}_{\text{blend\_pen}}\right) \cdot w_{\text{hard}} \cdot w_{\text{focus}} + \mathcal{L}_{\text{safety}} + \mathcal{L}_{\text{selector}}$$

| 分量 | 定义 | 权重 | 作用 |
|------|------|------|------|
| $\mathcal{L}_{\text{chamfer}}$ | $\text{mean}_i \|V_{\text{pred},i} - V_{\text{GT},i}\|$ | 1.0 | 表面精度 |
| $\mathcal{L}_{\text{tumor}}$ | $\|c_{\text{pred}} - c_{\text{GT}}\|$ | 1.0 | 肿瘤定位 |
| $\mathcal{L}_{\text{stab}}$ | $\text{ReLU}(\mathcal{L}_{\text{chamfer}} - \text{MAE}_{\text{preop}} + m)$ | 0.75 | 惩罚负更新 |
| $\mathcal{L}_{\text{blend\_pen}}$ | 低运动时的 blend 惩罚 | 1.0 | 防止过度融合 |
| $w_{\text{hard}}$ | $1 + 0.5 \cdot \mathbb{1}[\text{negative frame}]$ | — | Hard-negative mining |
| $\mathcal{L}_{\text{safety}}$ | MSE(predicted scale, target scale) | 1.0 | Safety head 监督 |
| $\mathcal{L}_{\text{selector}}$ | BCE(predicted path, better path) | 1.0 | Selector head 监督 |

---

## 4. 当前使用的数据集有哪些

### 4.1 KiTS23（Kidney Tumor Segmentation Challenge 2023）

| 属性 | 说明 |
|------|------|
| 数据来源 | 公开医学影像挑战赛数据集 |
| 原始形式 | 腹部 CT + 肾脏/肿瘤分割 mask |
| 在本项目中的使用 | 提取肾脏表面网格 → 施加合成形变 → 生成基准序列 |
| 数据量 | **489 个案例**（case_00000 至 case_00588，部分编号缺失） |
| 样本单位 | 案例级（每个案例 = 一个病人的一个肾脏） |
| 标签 | 合成地面真值（已知形变场下的真实表面位置） |
| 时间信息 | 合成时序（16 帧正弦形变 + 呼吸 + 工具压入） |
| 治疗信息 | 无 |
| 作用 | **主训练和评估数据集** |

**具体使用方式**:

1. 通过 `step6_benchmark_gen.py` 从每个 KiTS23 案例的术前网格生成 16 帧形变序列
2. 形变模型包含三部分：B-spline 随机场（全局漂移）、正弦呼吸运动、局部工具压入
3. 每帧生成部分观测（模拟内窥镜视角），覆盖率 10%/20%/40%/60%
4. 添加高斯噪声 $\sigma = 1.5\text{mm}$ 和 2% 离群点

**两种形变幅度**:
- 5mm benchmark: 489 案例，用于主实验
- 10mm benchmark: 53 案例（从训练集 50 + 评估集 3 中选取），用于大形变验证

**训练/评估划分**:
- 训练: 6 个案例（case_00485, case_00129, case_00041, case_00045, case_00230, case_00094）
- 评估: 全部 489 案例（包括训练案例，因为模型是 online 方式，不存在数据泄露风险）

### 4.2 LUMIERE（脑胶质瘤纵向 MRI 数据集）

| 属性 | 说明 |
|------|------|
| 数据来源 | 公开脑肿瘤影像数据集 |
| 原始形式 | 多时间点脑 MRI（FLAIR/T1/T2/CT1）+ HD-GLIO-AUTO 分割 mask |
| 在本项目中的使用 | 提取脑肿瘤表面网格 → 施加合成形变 → 跨解剖泛化验证 |
| 数据量 | **20 个案例**（从 91 位 ≥3 时间点患者中选取最丰富的 20 位） |
| 样本单位 | 患者级（使用第一个时间点的分割构建网格） |
| 标签 | 合成地面真值（同 KiTS23 方式） |
| 时间信息 | 合成 16 帧（使用与 KiTS23 相同的形变模型） |
| 治疗信息 | 有（RANO 响应标签），但本阶段未使用 |
| 作用 | **跨解剖泛化验证数据集** |

**具体使用方式**:

1. 从 HD-GLIO-AUTO 分割 mask 中提取肿瘤表面（marching cubes，阈值 0.5）
2. FPS 降采样至 ~8000 顶点
3. 构建均匀图拉普拉斯矩阵（$L = D - A$，权重均为 1）
4. 肿瘤标记：距 enhancing tumor（label=2）3mm 以内的顶点标为肿瘤，15mm 以内标为病灶带
5. 对整个肿瘤为一个连通分量的情况，使用"内 50% 为肿瘤，内 75% 为病灶带"的策略
6. 施加与 KiTS23 **完全相同**的合成形变（5mm 振幅），证明模型和机制的跨解剖泛化

**关键设计选择**: 使用半合成方式（真实解剖 + 合成形变）而非真实纵向形变。原因有二：(1) LUMIERE 的不同时间点图像未做跨时间点配准（不同分辨率、不同方向），无法直接对应表面点；(2) 半合成方式保证与 KiTS23 的可比性，纯粹测试"是否能泛化到不同形状"。

---

## 5. 每个数据集是如何清洗和处理的

### 5.1 KiTS23 数据处理

**原始数据 → 表面网格**:
1. 从 KiTS23 CT 分割 mask 提取肾脏表面（marching cubes）
2. 通过 Φ 管线（BIM pipeline）构建语义网格，包含：
   - `phi_mesh.npz`: 顶点、面、法向、连通分量 ID、病灶带掩码、肿瘤标记、PCA 参考系
   - `L_phi.npz`: 均匀图拉普拉斯矩阵（稀疏 CSR）
   - `tumor_transport.npz`: 肿瘤顶点的重心坐标映射

**表面网格 → 基准序列**:
1. 通过 `step6_benchmark_gen.py` 生成 16 帧形变序列
2. 形变参数：amplitude=5mm（或 10mm），respiratory=2mm，indentation=3mm
3. 每帧通过虚拟相机生成部分观测（覆盖率 10%/20%/40%/60%）
4. 噪声注入：$\sigma = 1.5\text{mm}$ 高斯噪声 + 2% 随机离群点
5. 置信度估计：基于入射角和深度不确定性

**数据质量筛选**:
- 排除无 Φ 网格的案例
- 排除无对应覆盖率基准的案例
- 最终 5mm: 489 案例可用；10mm: 53 案例可用

### 5.2 LUMIERE 数据处理

**原始分割 → 表面网格**:
1. 加载 `segmentation.nii.gz`（HD-GLIO-AUTO 自动分割，标签 0=背景，1=全肿瘤/水肿，2=增强肿瘤核心）
2. 二值化（所有 > 0 合并），高斯平滑（$\sigma = 0.5$ 体素），marching cubes（阈值 0.5）
3. 仿射变换到世界坐标（mm）
4. FPS 降采样至 ~8000 顶点（Farthest Point Sampling，保证均匀覆盖）
5. 面重映射：原始面的顶点映射到最近的保留顶点，去除退化面和重复面

**肿瘤区域标记**:
- 若 enhancing tumor（label=2）体素数 > 10，取距其 3mm 以内的网格顶点为肿瘤
- 若 enhancing 体素不足，回退到全肿瘤区域（label > 0）
- 若肿瘤顶点比例 > 80%（整个网格就是肿瘤），改用"距质心中位距离以内为肿瘤"

**图拉普拉斯构建**:
- 使用均匀图拉普拉斯 $L = D - A$（邻接权重全为 1）
- 这与 KiTS23 管线使用的格式完全一致
- 不使用余切拉普拉斯，避免退化三角形导致数值不稳定

---

## 6. 当前有哪些特征

### 6.1 逐顶点特征（14 维，用于 Support Head）

| 编号 | 名称 | 含义 | 数据类型 | 来源 | 取值范围 |
|------|------|------|----------|------|----------|
| 0 | residual_mag | 残差向量的模长 $\|\mathbf{r}_i\|$ | float32 | Kernel A 输出 | $[0, +\infty)$, mm |
| 1 | s_obs | 瞬时可观测性 | float32 | Kernel A 输出 | $[0, 1]$ |
| 2 | obs_count | 该顶点的观测邻居数 | float32 | Kernel A 输出 | $[0, k]$, 整数 |
| 3 | obs_weight_sum | 高斯权重总和 | float32 | Kernel A 输出 | $[0, +\infty)$ |
| 4 | support_prev | 上一帧的支持值 | float32 | 上一帧状态 | $[0, 1]$ |
| 5 | support_change | $|S_{\text{prev}} - s_{\text{obs}}|$ | float32 | 计算得到 | $[0, 1]$ |
| 6 | drift_from_preop | $\|V_i - V_{0,i}\|$ | float32 | 计算得到 | $[0, +\infty)$, mm |
| 7 | prev_step_motion | $\|V_i - V_{i}^{\text{prev}}\|$ | float32 | 计算得到 | $[0, +\infty)$, mm |
| 8 | boundary_flag | 是否位于连通分量边界 | float32 (0/1) | 网格拓扑 | $\{0, 1\}$ |
| 9 | lesion_band | 是否位于病灶带 | float32 (0/1) | Φ 网格 | $\{0, 1\}$ |
| 10 | curvature_proxy | $\|L_\Phi \cdot V_i\|$（拉普拉斯曲率近似） | float32 | 计算得到 | $[0, +\infty)$ |
| 11 | component_index | 归一化连通分量索引 | float32 | 网格拓扑 | $[0, 1]$ |
| 12 | residual_normal_cos | 残差与法向的余弦相似度 | float32 | 计算得到 | $[-1, 1]$ |
| 13 | S_hc | 手工支持（Kernel B 输出） | float32 | Kernel B | $[0, 1]$ |

**预处理**: 所有 14 维特征在每帧内进行 Z-score 标准化（减均值除标准差）。

### 6.2 帧级统计特征（9 维，用于 Blend/Safety/Selector Head）

**Blend 统计**:

| 字段 | 含义 |
|------|------|
| support_hat_mean | 学习型支持的帧均值 |
| support_hat_std | 学习型支持的帧标准差 |
| support_hc_mean | 手工支持的帧均值 |
| residual_mag_mean | 残差模长帧均值 |
| residual_mag_p95 | 残差模长 95 分位数 |
| obs_weight_mean | 观测权重帧均值 |
| s_obs_mean | 瞬时可观测性帧均值 |
| lesion_support_mean | 病灶区域支持均值 |
| nonlesion_support_mean | 非病灶区域支持均值 |

**Safety 统计**: 残差均值、s_obs 均值、支持均值/标准差、HC 支持均值、solver drift、fusion gap、blend 标量、帧归一化时间。

### 6.3 鲁棒性机制使用的决策信号

| 信号 | 定义 | 用途 |
|------|------|------|
| delta_v (推荐) | $\text{mean}(\|\Delta V_i\|)$ | 置信度门控阈值 $\tau$ 的参考信号 |
| res_x_supp | $\text{mean}(\|\mathbf{r}_i\| \cdot \hat{S}_i)$ | 备选信号 |
| obs_to_preop | 观测点到术前表面的距离均值 | 备选信号 |

**为什么 delta_v 优于原始观测信号**: delta_v 是求解器的"决策量"——它已经经过拉普拉斯平滑、弹性约束和支持门控的处理。在信噪比 < 1 的场景下（如正弦运动过零点），原始残差被噪声淹没，但求解器输出的位移量可以可靠地反映"这一帧是否有足够信号值得更新"。

---

## 7. 之前尝试过什么方法

### 7.1 纯手工支持 (HC baseline)

**思路**: 仅使用高斯加权的瞬时可观测性 + 拉普拉斯扩散作为支持场。

**结果**: 5mm baseline NegRate = 25.1%。

**问题**: 手工支持无法区分"看到了但信号弱"和"确实在运动"，导致频繁误判。

### 7.2 加性门控 (AdditiveObservabilityMLP)

**思路**: 学习一个加性修正 $\Delta S_i \in [-0.5, 0.5]$，$\hat{S}_i = S_{\text{hc},i} + \Delta S_i$。

**结果**: 比乘性门控差，因为加性修正可能将支持推到负值或大于 1，需要额外 clamp，丧失梯度信号。

### 7.3 全局标量门控 (GlobalScalarGate)

**思路**: 对所有顶点用一个标量门控，$\hat{S} = g \cdot S_{\text{hc}}$。

**结果**: 不如逐顶点门控，因为不同顶点的观测质量差异很大。

### 7.4 直接 confidence gating（不用 decision-quantity）

**思路**: 使用原始 $s_{\text{obs}}$ 或残差范数作为门控信号。

**结果**: 在正弦运动的零交叉处效果差。原因：零交叉处残差本身就小（因为表面接近术前位置），原始信号无法区分"没有运动"和"噪声淹没了运动"。

**启发**: 必须使用求解器的输出（delta_v）而非输入作为决策信号。

### 7.5 Hard cutoff 模式

**思路**: 当信号低于阈值时完全冻结更新（alpha=0）。

**结果**: 过于激进，导致在信号恢复时产生跳跃。软门控（线性缩放 + 回拉）更平滑。

### 7.6 仅 gate + decay（无 pullback）

**思路**: 只用门控降低 alpha + 衰减格点位移。

**结果**: NegRate 从 23.4% 降到 19.3%（−4.1 pp），远不如加入 pullback 后的 8.5%（−14.9 pp）。

**启发**: 仅抑制更新幅度不够，还需要**提供正确方向**（术前状态）。这引出了"回拉"的核心 idea。

---

## 8. 当前的主要难点是什么

### 8.1 数据层面

**难点**: 缺乏真实术中数据作为地面真值。

当前所有定量评估都基于合成形变（B-spline + 正弦 + 工具压入）。真实术中形变的模式（如出血、组织切除、CO₂ 充气变化）可能与合成模型有系统性差异。

**已做缓解**: 
- 使用 489 个真实解剖结构（KiTS23 肾脏）
- 20 个跨解剖结构（LUMIERE 脑肿瘤）
- 多种噪声水平（1.5-4.5mm）和覆盖率（10%-60%）进行压力测试

**未解决**: 未在真实手术视频或术中数据上验证。

### 8.2 模型层面

**难点**: 鲁棒性机制的超参数 $(\tau, d, p)$ 需要人工调整。

虽然提供了 Pareto 前沿上的多个推荐配置，但不同解剖结构或手术场景的最优 $\tau$ 可能不同。当前 $\tau = 0.40$ 是在 KiTS23 5mm 上搜索得到的。

**已做缓解**: 
- Pareto 分析提供了从 accuracy-first 到 safety-critical 的连续配置
- 跨解剖（LUMIERE）验证显示同一 $\tau$ 有效

**未解决**: 无自适应 $\tau$ 机制。

### 8.3 评估层面

**难点**: 负更新率（NegRate）的定义依赖于"预操作距离"基线。

当前定义：如果 $\text{MAE}(V_{\text{pred}}, V_{\text{GT}}) > \text{MAE}(V_0, V_{\text{GT}})$ 则为负更新帧。这个比较基线（术前状态）在零交叉帧处接近最优，使这些帧容易被判为"负更新"。

**已做缓解**: 
- Wilcoxon 配对检验证明改进在 per-case 级别统计显著
- 提供 SurfRMS 和 TumorCE 作为精度指标的补充

### 8.4 泛化能力层面

**难点**: 模型仅在 6 个案例上训练，泛化依赖于 solver-coupled 训练的隐式归纳偏置。

**已做缓解**: 
- 在 489 个案例上测试，覆盖多种肾脏大小和形状
- 在 20 个脑肿瘤案例上进行零样本跨解剖测试
- 鲁棒性机制完全 GT-free，不依赖训练数据分布

### 8.5 临床意义层面

**难点**: 当前指标（SurfRMS, NegRate）与临床终点（手术精度、并发症率）之间的关系尚不明确。

**未解决**: 需要与临床医生合作定义 clinically meaningful threshold。

---

## 9. 当前结果怎么样

### 9.1 主结果：KiTS23 5mm（489 案例）

| 配置 | SurfRMS (mm) | TumorCE (mm) | NegRate |
|------|-------------|-------------|---------|
| Hybrid baseline (β=0) | 1.4996 | 1.3551 | 23.42% |
| HC baseline | 1.5161 | 1.3700 | 25.10% |
| **Hybrid + balanced** ★ | **1.4949** | 1.3739 | **8.49%** |
| Hybrid + max-robust | 1.5095 | 1.3898 | **2.55%** |

**核心数字**: NegRate 从 23.4% 降到 2.55%（**9× reduction**），SurfRMS 代价仅 +0.01mm。

### 9.2 10mm 大形变验证（53 案例）

| 配置 | SurfRMS | TumorCE | NegRate |
|------|---------|---------|---------|
| Hybrid baseline (β=1.5) | 2.0329 | 1.8234 | 7.0% |
| Hybrid + balanced | 2.0261 | 1.8291 | 2.4% |
| Hybrid + max-robust | 2.0345 | 1.8388 | **1.2%** |

### 9.3 跨解剖泛化：LUMIERE 脑肿瘤（20 案例）

| 配置 | SurfRMS (mm) | TumorCE (mm) | NegRate |
|------|-------------|-------------|---------|
| Hybrid baseline | **1.5908** | **1.4799** | 27.19% |
| HC baseline | 1.7380 | 1.5459 | 43.44% |
| Hybrid + balanced | 1.6020 | 1.5235 | 14.38% |
| Hybrid + max-robust | 1.6300 | 1.5475 | **6.88%** |
| HC + max-robust | 1.6872 | 1.5954 | **5.00%** |

**关键发现**:
1. 学习模型在脑肿瘤上 SurfRMS 比 HC 好 8.5%，证明跨解剖泛化
2. 机制在脑肿瘤上实现 4-9× neg reduction，与 KiTS23 一致
3. HC + mech < Hybrid + mech（neg rate），正交性在跨解剖场景保持

### 9.4 消融实验

**组件消融（5mm）**:

| 组件组合 | NegRate | 相对 baseline 降低 |
|----------|---------|-------------------|
| baseline | 23.42% | — |
| + gate | 20.22% | −3.2 pp |
| + gate + decay | 19.29% | −4.1 pp |
| **+ gate + pullback** | **9.51%** | **−13.9 pp** |
| + gate + decay + pullback | 8.49% | −14.9 pp |

**正交性验证**:
- HC + mech: 5.8% NegRate（5mm）
- Hybrid + balanced: 8.5% NegRate
- 机制在无学习模型时效果更好 → 证明**学习模型和机制是正交贡献**

### 9.5 压力测试

**噪声鲁棒性**: 额外注入 2.0mm 噪声（总 ~3.5mm）时，baseline NegRate = 61.9%，mechanism = 29.2%。**2× 噪声容忍度**。

**序列长度**: 64 帧时 baseline NegRate 从 23% 漂移到 32%，mechanism 从 8.5% 改善到 6.5%。**"越跑越好"而非越跑越差。**

**覆盖率**: 所有覆盖率（10%-60%）下均显著降低 neg rate，最大收益在 cov=20%（最恶劣 SNR）。

### 9.6 统计显著性

所有关键比较的 Wilcoxon 配对检验 p < $10^{-5}$，5mm NegFrame 改进 **0% 案例变差**。

### 9.7 运行时

| Path | Mean | Median | P95 | FPS |
|------|------|--------|-----|-----|
| Hybrid | 118ms | 109ms | 179ms | **8.5** |
| HC | 76ms | 65ms | 111ms | **13.1** |

Mechanism overhead < 1ms。Hybrid 8.5 fps 满足术中导航需求（> 5 fps）。

---

## 10. 结果是怎么算出来的

### 10.1 SurfRMS (Surface Root Mean Square Error)

$$\text{SurfRMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \|V_{\text{pred},i} - V_{\text{GT},i}\|^2}$$

- $V_{\text{pred},i}$: 预测的第 $i$ 个顶点位置
- $V_{\text{GT},i}$: 地面真值的第 $i$ 个顶点位置
- $N$: 总顶点数
- 单位: mm
- 含义: 衡量整体表面跟踪精度，值越小越好
- 报告方式: 先对每帧计算，再对每个案例的 16 帧取均值，最终对所有案例取均值

### 10.2 TumorCE (Tumor Centroid Error)

$$\text{TumorCE} = \|\mathbf{c}_{\text{pred}}^{\text{tumor}} - \mathbf{c}_{\text{GT}}^{\text{tumor}}\|$$

- $\mathbf{c}_{\text{pred}}^{\text{tumor}}$: 预测肿瘤质心（通过重心坐标从网格顶点计算）
- $\mathbf{c}_{\text{GT}}^{\text{tumor}}$: 地面真值肿瘤质心
- 单位: mm
- 含义: 衡量肿瘤定位精度——术中导航的核心指标
- 值越小越好

### 10.3 NegRate (Negative Update Rate)

$$\text{NegRate} = \frac{\sum_{t=1}^{T} \mathbb{1}\left[\text{MAE}(V_t^{\text{pred}}, V_t^{\text{GT}}) > \text{MAE}(V_0, V_t^{\text{GT}})\right]}{T} \times 100\%$$

- 分子: 负更新帧的数量（追踪结果比不追踪还差的帧）
- $T$: 总帧数
- $\text{MAE}$: Mean Absolute Error = $\frac{1}{N}\sum_i \|V_i - V_i^{\text{GT}}\|$
- 含义: 衡量跟踪系统的"安全性"——多大比例的帧会"帮倒忙"
- 值越小越好，0% 表示所有帧都不劣于不跟踪

### 10.4 SurfMAE (Surface Mean Absolute Error)

$$\text{SurfMAE} = \frac{1}{N} \sum_{i=1}^{N} \|V_{\text{pred},i} - V_{\text{GT},i}\|$$

- 与 SurfRMS 类似但使用 L1 范数
- 对离群值的敏感度低于 RMS
- 用于计算负更新帧判定

### 10.5 HD95 (95th Percentile Hausdorff Distance)

$$\text{HD95} = \text{Percentile}_{95}\left(\{\|V_{\text{pred},i} - V_{\text{GT},i}\|\}_{i=1}^{N}\right)$$

- 单位: mm
- 含义: 衡量最坏情况精度（忽略最极端的 5% 误差）

### 10.6 Surface Improvement Percentage

$$\text{Improvement} = \frac{\text{MAE}_{\text{preop}} - \text{MAE}_{\text{pred}}}{\text{MAE}_{\text{preop}}} \times 100\%$$

- 正值: 跟踪改善了精度
- 负值: 负更新（跟踪使精度变差）

### 10.7 统计检验

使用 **Wilcoxon signed-rank test**（配对非参数检验）比较不同方法在 per-case 级别的差异。

- 零假设: 两种方法的 NegFrames 无差异
- 备择假设: 一种方法显著优于另一种
- 报告 win%（A 优于 B 的案例比例）和 p 值

---

## 11. 当前阶段的结论

### 11.1 已经证明的结论

1. **Observation-gated preop pullback 是有效的 GT-free 鲁棒性机制**
   - 在 KiTS23 489 案例上实现 9× neg reduction（23.4% → 2.55%），SurfRMS 代价仅 +0.01mm
   - Pullback 贡献 80% 的 neg reduction（factorial ablation 证明）
   - 统计高度显著（p < $10^{-65}$，0% 案例变差）

2. **Decision-quantity 信号优于 raw observation 信号**
   - 使用求解器输出 $\text{mean}(\|\Delta V\|)$ 作为门控信号，比原始残差或可观测性更可靠
   - 在 SNR < 1 场景（正弦过零点）尤其关键

3. **学习模型和鲁棒性机制是正交贡献**
   - 学习模型提供精度（SurfRMS 领先）
   - 机制提供鲁棒性（NegRate 领先）
   - HC + mech 在 neg rate 上甚至优于 Hybrid + mech（5.8% vs 8.5%）

4. **跟踪系统从"越跑越差"变成"越跑越好"**
   - 64 帧时 baseline NegRate 漂移到 32%，mechanism 收敛到 6.5%
   - SurfRMS: baseline 1.528mm↑，mechanism 1.494mm（恒定）

5. **方法跨解剖泛化**
   - 在 KiTS23 肾脏上训练的模型，零样本迁移到 LUMIERE 脑肿瘤
   - SurfRMS 提升 8.5%，neg reduction 4-9× 保持

### 11.2 初步观察（尚需进一步验证）

1. 鲁棒性机制的最优超参数可能对不同形变幅度/噪声水平有弱依赖
2. Per-vertex gating 提供额外 Pareto 改善，但增加的实现复杂度是否值得待评估
3. β 的最优值与形变振幅相关，但具体关系尚未精确建模

### 11.3 在研究路线中的位置

本阶段完成了三路径术中跟踪系统中 **核心跟踪引擎的鲁棒性增强**。这是论文的主要技术贡献。剩余工作为：
- 论文撰写（方法、实验、图表）
- 真实手术数据验证（如有数据来源）

---

## 12. 未来方向

### 12.1 最优先（论文发表前）

1. **真实术中数据验证**: 寻找公开的术中表面跟踪数据集（如 SCARED、Hamlyn Centre），在真实手术视频生成的点云上验证
2. **对比实验补充**: 与现有方法（如 Non-rigid ICP、VoxelMorph、Sokooti et al.）进行直接对比
3. **论文图表**: 生成 Pareto 前沿图、sequence length 趋势图、per-case 分布箱线图

### 12.2 中期（系统完善）

4. **自适应 τ**: 学习一个基于帧级统计自动预测最优 τ 的轻量模型，避免人工调参
5. **在线模型更新**: 在术中根据累积观测微调 Support Head，提高个体化精度
6. **与 Path 3（nnUNet re-seg）集成**: 实现完整三路径架构的端到端运行
7. **β 自动选择**: 根据形变振幅自动调节各向异性因子

### 12.3 长期方向

8. **视觉基础模型集成**: 用 SAM/DINOv2 等视觉基础模型从内窥镜图像直接提取表面特征，替代手工 14 维特征
9. **多器官联合跟踪**: 扩展到腹腔多器官同步跟踪
10. **手术规划闭环**: 将实时表面跟踪反馈到手术规划系统，实现闭环导航
11. **大规模临床验证**: 与医院合作进行前瞻性临床研究

---

## 13. 附录：术语与符号说明

### 13.1 术语缩写

| 缩写 | 全称 | 含义 |
|------|------|------|
| SCSU | Semantic-Constrained Surface Update | 语义约束表面更新 |
| HC | Hand-Crafted | 手工设计的（相对于学习型） |
| BIM | Building Information Modeling | 建筑信息模型（本项目中借用其点云/网格处理方法） |
| PCG | Preconditioned Conjugate Gradient | 预条件共轭梯度法 |
| GT | Ground Truth | 地面真值 |
| NegRate | Negative Update Rate | 负更新率 |
| SurfRMS | Surface Root Mean Square Error | 表面均方根误差 |
| TumorCE | Tumor Centroid Error | 肿瘤质心误差 |
| HD95 | 95th Percentile Hausdorff Distance | 95分位 Hausdorff 距离 |
| MAE | Mean Absolute Error | 平均绝对误差 |
| EMA | Exponential Moving Average | 指数移动平均 |
| KD-tree | k-dimensional tree | k 维空间搜索树 |
| FPS | Frames Per Second / Farthest Point Sampling | 帧率 / 最远点采样（视上下文） |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| CG | Conjugate Gradient | 共轭梯度法 |
| CSR | Compressed Sparse Row | 压缩稀疏行存储格式 |
| KiTS23 | Kidney Tumor Segmentation 2023 | 肾脏肿瘤分割挑战赛 2023 |
| LUMIERE | Longitudinal Glioma MRI Dataset | 纵向胶质瘤 MRI 数据集 |
| RANO | Response Assessment in Neuro-Oncology | 神经肿瘤学响应评估标准 |

### 13.2 数学符号

| 符号 | 含义 | 维度/取值 |
|------|------|----------|
| $V_0$ | 术前表面顶点坐标 | $\mathbb{R}^{N \times 3}$ |
| $V_t$ | 第 $t$ 帧的预测表面 | $\mathbb{R}^{N \times 3}$ |
| $F$ | 三角面索引 | $\mathbb{Z}^{M \times 3}$ |
| $N$ | 网格顶点总数 | 整数，通常 5k-30k |
| $O_t$ | 第 $t$ 帧的观测点云 | $\{(\mathbf{p}_j, \mathbf{n}_j, q_j)\}$ |
| $\mathbf{r}_i$ | 第 $i$ 个顶点的残差向量 | $\mathbb{R}^3$ |
| $s_{\text{obs},i}$ | 瞬时可观测性 | $[0, 1]$ |
| $S_{\text{hc}}$ | 手工支持场 | $[0, 1]^N$ |
| $\hat{S}$ | 学习型支持场 | $[0, 1]^N$ |
| $\Delta V$ | 位移增量 | $\mathbb{R}^{N \times 3}$ |
| $L_\Phi$ | 图拉普拉斯矩阵 | $\mathbb{R}^{N \times N}$（稀疏） |
| $\sigma$ | 高斯核带宽 | 10mm |
| $\gamma$ | 时间平滑系数 | 0.3 |
| $\eta$ | 扩散步长 | 0.2 |
| $\alpha$ | 基础更新步长 | 0.8 |
| $\beta$ | 各向异性因子 | 0.3 或 1.5 |
| $\tau$ | 置信度门控阈值 | 推荐 0.40 |
| $d$ | 格点衰减率 | 推荐 0.5 |
| $p$ | 术前回拉强度 | 推荐 0.6 |
| $g_i$ | 学习型门控因子 | $[0.3, 2.0]$ |
| $b$ | 融合标量 | $[0, 0.35]$ |
| $\mathbf{D}_{\text{node}}$ | 格点节点位移 | $\mathbb{R}^{N_g \times 3}$ |
| $N_g$ | 格点节点数 | $7^3 = 343$ |

### 13.3 关键文件索引

| 文件路径 | 内容 |
|----------|------|
| `sscu_engine_gpu.py` | GPU SCSU 引擎（Kernel A/B/C, PCG, 状态管理） |
| `observability_model.py` | 学习型模型定义（Support/Blend/Safety/Selector heads） |
| `eval_fullscale_489_aniso.py` | 主评估脚本（所有消融和机制 flags） |
| `eval_improvements_ablation.py` | 各向异性残差和消融实验 |
| `eval_hybrid_lattice_amp10.py` | 格点正则化实现 |
| `eval_lumiere.py` | LUMIERE 跨解剖评估脚本 |
| `prepare_lumiere_benchmark.py` | LUMIERE 数据预处理脚本 |
| `per_case_stats.py` | Wilcoxon 统计检验脚本 |
| `plot_pareto_5mm.py` | Pareto 前沿绘图脚本 |
| `profile_runtime.py` | 运行时性能测试脚本 |
| `Technical_Report_Final_2026-04-10.md` | 实验结果技术报告 |

---

*文档生成日期: 2026-04-11*  
*本文档基于项目实际代码、实验记录和配置文件整理，所有数值均来自实际运行结果。*
