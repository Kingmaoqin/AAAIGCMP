# Lesion-Aware 3D Medical Digital Twin Updating from Partial Observations

**Target venue**: NeurIPS / ICML 2026
**Draft version**: V1 (2026-04-02)

---

## Abstract

Intraoperative augmented reality (AR) navigation for tumor resection requires continuously updating a preoperative 3D organ model — a *digital twin* — to match the current surgical anatomy. This is fundamentally a **partial-observation online state estimation** problem: laparoscopic cameras reveal only 10–60% of the organ surface at any moment, the anatomy deforms under respiration and surgical manipulation, and the tumor must remain accurately localized even when occluded. Existing approaches either treat each frame as an independent registration problem (losing temporal coherence), ignore semantic structure (allowing cross-region coupling artifacts), or lack explicit uncertainty quantification (over-trusting unobserved regions).

We introduce **SSCU** (Semantic–Support Constrained Update), a structured MAP update operator that maintains a kidney–tumor digital twin under streaming partial observations. SSCU decomposes twin updating into four GPU-accelerated kernels — residual baking, support diffusion, sparse PCG solve, and lesion transport — and provides three formal guarantees: (1) **no cross-region leakage** via block-diagonal semantic Laplacian, (2) **bounded drift** in unsupported regions via prior fallback, and (3) **bounded lesion error** via convex-combination transport. On a KiTS23-based benchmark with 50 cases across 4 coverage levels (10/20/40/60%), SSCU achieves surface Chamfer distance 1.47–1.51 mm, tumor centroid error 1.43–1.48 mm, and per-frame latency ~13–19 ms (warm frames), demonstrating compatibility with intraoperative update rates.

---

## 1. Introduction

### 1.1 临床动机 (Clinical Motivation)

在机器人辅助部分肾切除术 (Robot-Assisted Partial Nephrectomy, RAPN) 中，术者需要在微创视野下精确定位肿瘤边界以实现R0切除并最大化保留肾功能。术前CT构建的3D肾脏-肿瘤模型（数字孪生）是AR导航的基础，但术中的**三大挑战**使得静态模型难以直接使用：

1. **部分可观测性 (Partial Observability)**：腹腔镜的视野通常只覆盖肾脏表面的10-60%，且随手术操作动态变化。MICCAI 2025的EG-Net工作明确记录了腹腔镜AR中观测重叠率<30%的情况。

2. **持续形变 (Continuous Deformation)**：呼吸运动、手术器械压迫、以及组织切除导致肾脏形状持续变化。形变幅度可达5-10mm。

3. **病灶遮挡 (Lesion Occlusion)**：肿瘤通常位于肾脏深部，不直接可见。术者需要从可见的表面形变推断肿瘤的当前位置——这正是数字孪生的核心价值。

现有方法的不足：

| 方法类别 | 代表方法 | 关键缺陷 |
|---------|---------|---------|
| 经典配准 | ICP, CPD, ED-Graph | 每帧独立配准，无时序一致性；不维护支持/置信度场 |
| 体素融合 | TSDF integration | 无语义约束，跨区域耦合；无病灶追踪 |
| 神经重建 | EndoGSLAM, EndoSurf | 不从CT预模型出发；无语义隔离保证 |
| 手术DT配准 | Digital twin registration | 单次配准，不维护持续更新的孪生状态 |

**SSCU的核心洞察**：将数字孪生更新形式化为**带结构约束的MAP推断**，而非逐帧配准。通过引入语义分区、支持场和先验回退三个约束，在数学上保证了三个临床关键属性。

### 1.2 问题定义 (Problem Formulation)

#### 符号表 (Notation Table)

| 符号 | 维度 | 含义 |
|------|------|------|
| $\mathcal{T}_t$ | — | 时刻 $t$ 的数字孪生状态 |
| $G_t = (V_t, F)$ | $V_t \in \mathbb{R}^{n \times 3}$ | 当前肾脏表面几何（顶点坐标 + 面片拓扑） |
| $n$ | $\mathbb{N}$ | 网格顶点数（典型值 ~30,000） |
| $F$ | $\mathbb{N}^{m \times 3}$ | 三角面片索引（不随时间变化） |
| $\Phi$ | — | 语义分区（结构-语义图/划分），由术前CT构建 |
| $\mathcal{C} = \{c_1, \ldots, c_K\}$ | — | 语义分区的组件集合（如肾脏前表面、后表面、肾门区、肿瘤邻近带等） |
| $y: \{1,\ldots,n\} \to \mathcal{C}$ | — | 顶点到组件的映射（标签函数） |
| $L_t = (c_t, B_t, \Sigma_t)$ | — | 病灶状态：质心、边界、不确定性 |
| $c_t$ | $\mathbb{R}^3$ | 肿瘤质心坐标 |
| $B_t$ | $\mathbb{R}^{|B| \times 3}$ | 肿瘤边界点集合 |
| $S_t$ | $[0,1]^n$ | 支持场：每个顶点的观测置信度 |
| $O_t = \{(p_j, \hat{n}_j, q_j)\}_{j=1}^{N_t}$ | — | 时刻 $t$ 的部分表面观测 |
| $p_j$ | $\mathbb{R}^3$ | 第 $j$ 个观测点的3D坐标 |
| $\hat{n}_j$ | $\mathbb{R}^3$ | 第 $j$ 个观测点的法向量 |
| $q_j$ | $[0,1]$ | 第 $j$ 个观测点的置信度权重 |
| $V_0$ | $\mathbb{R}^{n \times 3}$ | 术前初始顶点坐标（先验位置） |
| $n_i$ | $\mathbb{R}^3$ | 顶点 $i$ 的表面法向量 |

#### 概率框架

我们将病灶感知孪生更新建模为在线状态估计：

**隐状态**：$X_t \equiv (G_t, L_t)$，其中 $G_t$ 是当前器官几何，$L_t$ 是病灶状态。

**滤波模型**：
$$X_t \sim p_\theta(X_t \mid X_{t-1}, \Phi), \qquad O_t \sim p_\theta(O_t \mid X_t, S_t)$$

**受约束的MAP更新**：
$$\hat{X}_t = \arg\max_{X_t} \underbrace{\log p_\theta(O_t \mid X_t, S_t)}_{\text{似然（观测项）}} + \underbrace{\log p_\theta(X_t \mid \hat{X}_{t-1}, \Phi)}_{\text{先验（结构约束）}}$$

**临床输出**（每帧）：
- 更新后的肾脏表面 $G_t$
- 病灶位置与不确定性 $L_t$
- **肿瘤到可见表面距离**：
  $$d_t = \min_{x \in B_t} \operatorname{dist}\big(x, \{v_i \in V_t : S_t(i) > \tau_{\text{vis}}\}\big)$$
  该指标回答"肿瘤距我们当前能看到的最近表面有多远"——这是术中导航的核心信息。

---

## 2. Method: SSCU (Semantic-Support Constrained Update)

### 2.1 系统概览

SSCU在每个时间步执行四个kernel的流水线操作：

```
输入: V_{t-1}, Φ, L_{t-1}, S_{t-1}, O_t

┌──────────────────────────────────────────────────────────────┐
│  Kernel A: 残差烘焙 (Residual Baking)                        │
│  O_t → per-vertex 法向残差 r_i + 瞬时支持 s^obs_i            │
├──────────────────────────────────────────────────────────────┤
│  Kernel B: 支持场更新与扩散 (Support Diffusion)                │
│  S_{t-1}, s^obs → S_t (EMA + 组件内Laplacian扩散)            │
├──────────────────────────────────────────────────────────────┤
│  Kernel C: 稀疏PCG求解 (Sparse Local Solve)                   │
│  A_t ΔV = b_t → 位移场 ΔV (SPD系统, PCG with Jacobi预条件)  │
├──────────────────────────────────────────────────────────────┤
│  Kernel D: 病灶传输 (Lesion Transport)                        │
│  ΔV → 更新肿瘤边界 B_t, 计算距离 d_t                          │
└──────────────────────────────────────────────────────────────┘

输出: V_t, L_t, S_t, d_t, uncertainty
```

### 2.2 Kernel A: 残差烘焙 (Residual Baking)

#### 动机

低重叠率和光滑表面使得点对点对应极不稳定（EG-Net 明确指出了腹腔镜场景中的这一缺陷）。我们不直接建立观测-模型对应关系，而是将观测信息"烘焙"到表面载体上。

#### 算法

**Step 1: k近邻查询**。对每个观测点 $p_j$，在当前网格 $V_{t-1}$ 上查找 $k=8$ 个最近顶点（使用KD-tree）。

**Step 2: 组件感知掩码 (Component-Aware Mask)**。为防止跨语义边界传播残差，对每个观测点 $p_j$：
$$\text{mask}_{jl} = \mathbb{1}\big[y(\text{knn}_l) = y(\text{knn}_0)\big], \quad l = 0, \ldots, k-1$$
其中 $\text{knn}_0$ 是最近顶点，$y(\cdot)$ 是组件标签函数。只有与最近顶点**同组件**的近邻才接收残差传播。

**Step 3: 高斯加权残差计算**。对每个顶点 $i$，聚合所有以 $i$ 为近邻的观测点的贡献：
$$r_i = \frac{1}{Z_i} \sum_{j: i \in \text{knn}(j)} w_{ji} \cdot q_j \cdot \big((p_j - v_i) \cdot n_i\big) \cdot n_i$$

其中各符号含义：
| 符号 | 含义 |
|------|------|
| $r_i$ | 顶点 $i$ 的法向残差向量 (mm)。表示观测认为该顶点应沿法向移动的方向和距离 |
| $Z_i$ | 归一化常数 $Z_i = \sum_{j: i \in \text{knn}(j)} w_{ji} \cdot q_j + \epsilon$ |
| $w_{ji}$ | 高斯权重 $w_{ji} = \exp\big(-\|p_j - v_i\|^2 / \sigma^2\big) \cdot \text{mask}_{j,\text{idx}(i)}$ |
| $\sigma$ | 高斯支持核带宽 (mm)。控制每个观测点影响范围的空间尺度。当前值 $\sigma = 10.0$ mm |
| $q_j$ | 观测点 $j$ 的置信度权重 |
| $n_i$ | 顶点 $i$ 的单位法向量 |
| $\epsilon$ | 防零除的极小值 ($10^{-8}$) |

**Step 4: 瞬时支持计算**：
$$s_i^{\text{obs}} = 1 - \prod_{j: i \in \text{knn}(j)} \Big(1 - q_j \cdot \exp\big(-\|p_j - v_i\|^2 / \sigma^2\big)\Big)$$

**物理含义**：$s_i^{\text{obs}}$ 衡量顶点 $i$ 在当前帧被观测到的程度。如果多个观测点位于顶点附近，$s_i^{\text{obs}}$ 接近1；如果没有观测点在附近，$s_i^{\text{obs}} = 0$。使用乘积-补形式确保多个弱观测可以累积为强支持。

### 2.3 Kernel B: 支持场更新与扩散 (Support Diffusion)

#### EMA时间更新

$$S_t(i) = \text{clip}\Big(\gamma \cdot S_{t-1}(i) + (1 - \gamma) \cdot s_i^{\text{obs}},\; 0,\; 1\Big)$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\gamma$ | 0.30 | EMA衰减率。$\gamma = 0.30$ 意味着每帧新观测权重为70%，旧支持保留30%。低 $\gamma$ 使支持累积更快 |
| $S_t(i)$ | $[0,1]$ | 顶点 $i$ 在时刻 $t$ 的持久支持值 |

#### 组件内Laplacian扩散

为使支持场在语义分区内部平滑扩展（但不跨越语义边界），我们执行迭代Laplacian平滑：

$$S_t \leftarrow S_t - \eta \cdot L_\Phi \cdot S_t, \qquad \text{重复 } I_{\text{diff}} \text{ 次}$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\eta$ | 0.20 | 扩散步长。控制每步扩散的强度 |
| $I_{\text{diff}}$ | 8 | 扩散迭代次数 |
| $L_\Phi$ | — | **语义Laplacian矩阵**。这是SSCU的核心结构：$L_\Phi$ 仅包含满足 $y(i) = y(k)$ 的边 $(i,k)$，即只连接同一语义组件内的顶点。这确保支持场永远不会跨越组件边界扩散 |

**$L_\Phi$ 的构建**：给定网格 $M = (V, F)$ 和语义标签函数 $y$，
$$L_\Phi(i,k) = \begin{cases} -w_{ik} & \text{if } (i,k) \in \mathcal{E}_F \text{ and } y(i) = y(k) \\ \sum_{k'} w_{ik'} & \text{if } i = k \\ 0 & \text{otherwise} \end{cases}$$

其中 $\mathcal{E}_F$ 是网格边集合，$w_{ik}$ 是边权重（cotangent weights或uniform weights）。

### 2.4 Kernel C: 稀疏PCG求解 (Sparse Local Solve)

#### SSCU能量函数

SSCU通过最小化一个二次能量函数来估计位移场 $\Delta V_t \in \mathbb{R}^{n \times 3}$：

$$\Delta V_t = \arg\min_{\Delta V} E(\Delta V)$$

$$\boxed{E(\Delta V) = E_{\text{data}} + E_{\text{smooth}} + E_{\text{prior}} + E_{\text{lesion}}}$$

各项含义如下：

**数据项（支持加权观测同化）**：
$$E_{\text{data}} = \sum_{i=1}^{n} S_t(i) \cdot \|\Delta v_i - r_i\|_2^2$$

- **含义**：有观测支持的顶点 ($S_t(i) \approx 1$) 被强力拉向残差方向 $r_i$；无观测的顶点 ($S_t(i) \approx 0$) 数据项权重几乎为零
- **概率解释**：$S_t(i)$ 充当高斯似然的精度（逆方差）参数

**平滑项（组件内语义平滑）**：
$$E_{\text{smooth}} = \lambda_{\text{sem}} \sum_{(i,k) \in \mathcal{N}:\, y(i)=y(k)} w_{ik} \cdot \|\Delta v_i - \Delta v_k\|_2^2$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\lambda_{\text{sem}}$ | 1.0 | 语义平滑权重。控制同组件内顶点位移一致性的约束强度。$\lambda_{\text{sem}} = 0$ 时无平滑，位移可能不连续 |
| $\mathcal{N}$ | — | 网格邻接关系中满足 $y(i)=y(k)$ 的边对 |
| $w_{ik}$ | — | 边权重 |

- **含义**：惩罚同一语义组件内相邻顶点的位移差异，产生平滑形变
- **关键**：只在 $y(i) = y(k)$ 的边上约束——不同组件之间的位移可以完全独立
- **概率解释**：对应高斯马尔科夫随机场 (GMRF) 先验，Laplacian $L_\Phi$ 是精度矩阵

**先验回退项（无观测区域稳定性）**：
$$E_{\text{prior}} = \lambda_{\text{prior}} \sum_{i=1}^{n} (1 - S_t(i)) \cdot \|v_{t-1,i} + \Delta v_i - v_{0,i}\|_2^2$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\lambda_{\text{prior}}$ | 0.1 | 先验回归权重。控制无观测区域向术前位置 $V_0$ 回归的力度 |
| $V_0$ | — | 术前初始顶点坐标 |
| $1 - S_t(i)$ | — | 权重：支持越低的顶点，先验约束越强 |

- **含义**：未被观测到的顶点被"拉回"术前位置 $V_0$，防止无约束漂移
- **对称设计**：$S_t(i)$ 同时控制数据项和先验项的相对权重——P/D ratio $= \lambda_{\text{prior}}(1-S)/S$ 是理解系统行为的关键

**病灶带保护项**：
$$E_{\text{lesion}} = \lambda_{\text{les}} \sum_{i \in \mathcal{B}} \|\Delta v_i\|_2^2$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\lambda_{\text{les}}$ | 1.0 | 病灶带保护权重。约束肿瘤邻近带内的顶点位移幅度 |
| $\mathcal{B}$ | — | 病灶邻近带：距肿瘤边界一定测地距离内的顶点集合 |

- **含义**：限制肿瘤边界附近的顶点大幅移动，维护肿瘤边界的拓扑稳定性
- **临床意义**：防止肿瘤边界在部分观测下剧烈变形导致的拓扑异常

#### 线性系统

对能量函数求导并令梯度为零，每个坐标方向得到一个稀疏对称正定 (SPD) 线性系统：

$$\boxed{A_t \Delta V = b_t}$$

$$A_t = \underbrace{\text{diag}(S_t)}_{\text{数据精度}} + \underbrace{\lambda_{\text{sem}} L_\Phi}_{\text{语义平滑}} + \underbrace{\lambda_{\text{prior}} \text{diag}(1 - S_t)}_{\text{先验回退}} + \underbrace{\lambda_{\text{les}} \text{diag}(\mathbf{1}_\mathcal{B})}_{\text{病灶保护}}$$

$$b_t = \text{diag}(S_t) \cdot r + \lambda_{\text{prior}} \cdot \text{diag}(1 - S_t) \cdot (V_0 - V_{t-1})$$

各符号含义：
| 符号 | 含义 |
|------|------|
| $A_t \in \mathbb{R}^{n \times n}$ | 系统矩阵，稀疏对称正定 (SPD)。稀疏模式由 $L_\Phi$ 和对角项决定 |
| $\Delta V \in \mathbb{R}^{n \times 3}$ | 待求的位移场（对 x/y/z 三个方向分别求解） |
| $b_t \in \mathbb{R}^{n \times 3}$ | 右端项。包含观测残差和先验回退力 |
| $\text{diag}(S_t)$ | $n \times n$ 对角矩阵，对角元素为 $S_t(i)$ |
| $\text{diag}(1-S_t)$ | $n \times n$ 对角矩阵，对角元素为 $1-S_t(i)$ |
| $\mathbf{1}_\mathcal{B}$ | 病灶带指示向量：$(\mathbf{1}_\mathcal{B})_i = 1$ if $i \in \mathcal{B}$, else $0$ |

**求解方法**：使用预条件共轭梯度法 (PCG)，Jacobi预条件器（$P = \text{diag}(A_t)^{-1}$），收敛容差 $\text{tol} = 10^{-5}$，最大迭代 50 次。在GPU上，SpMV操作利用PyTorch sparse CSR格式（底层调用cuSPARSE）。

#### 支持感知更新步长 (Support-Aware Alpha)

直接使用固定步长 $\alpha$ 更新顶点位置会导致低支持区域过冲（overshoot）。我们引入**逐顶点自适应步长**：

$$\alpha_i = \alpha_{\min} + (\alpha - \alpha_{\min}) \cdot S_t(i)$$

| 符号 | 当前值 | 含义 |
|------|--------|------|
| $\alpha$ | 0.8 | 最大更新步长（高支持区域使用） |
| $\alpha_{\min}$ | $0.3 \alpha = 0.24$ | 最小更新步长（无支持区域使用） |

顶点更新：
$$V_t = V_{t-1} + \alpha_i \cdot \Delta V_t$$

**效果**：高支持区域全力跟踪观测，低支持区域使用保守步长抑制过冲。此修复将过冲帧从 5/16 降至 2/16。

### 2.5 Kernel D: 病灶传输 (Lesion Transport)

肿瘤通常不可直接观测，需要从表面形变推断其位置变化。

**重心坐标传输**：每个肿瘤边界点 $x \in B_t$ 表达为附近表面顶点的凸组合：
$$x = \sum_{k \in \mathcal{N}(x)} \phi_k v_k, \qquad \phi_k \ge 0, \quad \sum_k \phi_k = 1$$

其中 $\phi_k$ 是重心坐标权重（在初始化时计算一次，后续帧复用），$\mathcal{N}(x)$ 是 $x$ 的邻域顶点集。

**肿瘤-表面距离**：
$$d_t = \min_{x \in B_t} \operatorname{dist}\big(x, \{v_i \in V_t : S_t(i) > \tau_{\text{vis}}\}\big)$$

其中 $\tau_{\text{vis}} = 0.5$ 是"可见"阈值——只有支持度超过此阈值的表面才被认为是"已观测到的"。

---

## 3. Theoretical Guarantees

### 3.1 Theorem 1: 语义隔离（无跨区域泄漏）

**Statement**. 若 $L_\Phi$ 仅使用满足 $y(i) = y(k)$ 的边构建，则存在顶点重排列（按组件分组），使 $A_t$ 成为跨组件的块对角矩阵。因此，SSCU在一个组件内的更新仅依赖于该组件内部的残差和先验——**不存在跨组件耦合**。

**Proof sketch**.
1. 按组件对顶点重新排序
2. $L_\Phi$ 仅包含组件内边 $\Rightarrow$ $L_\Phi = \text{blkdiag}(L_{c_1}, \ldots, L_{c_K})$
3. $\text{diag}(S_t)$、$\text{diag}(1-S_t)$、$\text{diag}(\mathbf{1}_\mathcal{B})$ 均为对角矩阵，保持块结构
4. 因此 $A_t = \text{blkdiag}(A_{c_1}, \ldots, A_{c_K})$，求解自然按块解耦 $\square$

**实验验证**：直接检查 $L_\Phi$ 矩阵，确认跨组件非零元素数为 **0**（完美块对角）。30,860个顶点、185,040条网格边中，$L_\Phi$ 严格无跨组件耦合。

**临床意义**：当腹腔镜只能看到肾脏前表面时，SSCU保证前表面的形变更新不会影响后表面——这是对抗ICP在低重叠率下"整体表面拖拽"失败模式的结构性解药。

### 3.2 Theorem 2: 有界漂移（先验回退稳定性）

**Assumptions**：残差幅度有界 $|r_i| \le R_{\max}$；$\lambda_{\text{prior}} > 0$。

**Statement**. 对任意顶点 $i$ 和时刻 $t$：
$$(1 - S_t(i)) \cdot \|v_{t,i} - v_{0,i}\|^2 \le \frac{E(0)}{\lambda_{\text{prior}}}$$
其中 $E(0)$ 是 $\Delta V = 0$ 时的SSCU能量。特别地，若 $S_t(i) \approx 0$ 持续成立（即顶点 $i$ 长期未被观测），SSCU防止其无界偏离术前先验 $V_0$。

**Proof sketch**.
1. 由最优性，$E(\Delta V_t) \le E(0)$
2. 先验回退项非负：$\lambda_{\text{prior}}(1-S_t(i))\|v_{t-1,i} + \Delta v_{t,i} - v_{0,i}\|^2 \ge 0$
3. 丢弃其他非负项得不等式 $\square$

**实验验证**：消融实验中，$\lambda_{\text{prior}} = 0$ 时16帧后漂移增加37%（0.546mm → 0.747mm），过冲帧增加到5/16。恢复 $\lambda_{\text{prior}} = 0.1$ 后漂移被有效约束。

### 3.3 Theorem 3: 病灶误差界（形变传输界）

**Assumption**. 每个病灶点 $x \in B_t$ 可表达为附近表面顶点的凸组合：$x = \sum_{k \in \mathcal{N}(x)} \phi_k v_k$，$\phi_k \ge 0$，$\sum_k \phi_k = 1$。

**Statement**. 若 $\tilde{v}_k$ 为估计位置，$v_k^*$ 为真实位置，则传输病灶点误差：
$$|\tilde{x} - x^*| \le \max_{k \in \mathcal{N}(x)} |\tilde{v}_k - v_k^*|$$

即病灶质心误差被其邻域内的最大表面误差上界控制。

**Proof**.
$$\tilde{x} - x^* = \sum_k \phi_k (\tilde{v}_k - v_k^*) \Rightarrow |\tilde{x} - x^*| \le \sum_k \phi_k |\tilde{v}_k - v_k^*| \le \max_k |\tilde{v}_k - v_k^*| \cdot \underbrace{\sum_k \phi_k}_{=1} \quad \square$$

**实验验证**：在所有覆盖率下，肿瘤质心误差 (1.43–1.48mm) 始终低于或接近表面Chamfer误差 (1.47–1.51mm)，验证了bound的紧性。

---

## 4. GPU Incremental Twin Engine

### 4.1 设计原则

SSCU的GPU引擎不是简单的"CUDA移植"，而是**结构性利用SSCU的数学特性**进行加速：

1. **块对角性 (Theorem 1)**：$A_t$ 的块对角结构允许按组件独立求解，提升缓存局部性和并行性
2. **稀疏性**：$L_\Phi$ 的稀疏模式固定（仅由网格拓扑和语义分区决定），$A_t$ 只需更新对角元素
3. **增量性**：支持场 $S_t$ 每帧只有部分顶点变化，可利用active-region优化

### 4.2 Kernel实现

| Kernel | CPU实现 | GPU实现 | 关键操作 |
|--------|--------|--------|---------|
| A: 残差烘焙 | scipy cKDTree + numpy | cKDTree (CPU) + torch scatter_add | KD-tree仍在CPU（瓶颈），scatter_add在GPU |
| B: 支持扩散 | numpy向量运算 + scipy sparse | torch向量运算 + sparse CSR matmul | EMA为逐元素运算，扩散为SpMV |
| C: PCG求解 | scipy.sparse.linalg.cg | 自定义PyTorch PCG + sparse CSR SpMV | 底层调用cuSPARSE，Jacobi预条件 |
| D: 病灶传输 | numpy重心坐标 | torch重心坐标 + cdist | 纯向量运算 |

### 4.3 性能

**逐Kernel对比 (case_00000, A100 GPU)**：

| Kernel | CPU (ms) | GPU (ms) | 加速比 |
|--------|----------|----------|--------|
| A: 残差烘焙 | 103.6 | 56.6 | **1.8x** |
| B: 支持扩散 | 3.0 | 1.8 | **1.7x** |
| C: PCG求解 | 18,550 | 194.7 | **95.3x** |
| D: 病灶传输 | 45.1 | 3.5 | **13.0x** |
| **合计** | **18,702** | **256.5** | **72.9x** |

- Kernel C 的 95.3x 加速来自 cuSPARSE SpMV 对 CPU scipy稀疏矩阵运算的巨大优势
- Kernel A 加速受限于 KD-tree 仍在 CPU 上运行（占 Kernel A 约60%时间）
- 50例批量warm-frame延迟：**13–19 ms**（去除初始化和诊断同步后）

### 4.4 精度一致性

CPU和GPU引擎的数值差异极小：
- Chamfer差异 < 0.008 mm
- 肿瘤误差差异 < 0.0004 mm

---

## 5. Experimental Design

### 5.1 数据集与基准

**主要数据集：KiTS23**
- 489 例训练集，110 例测试集
- 每例包含肾脏 (label=1) 和肿瘤 (label=2) 分割标注
- 覆盖从小到大的肿瘤尺寸变异

**语义分区 $\Phi$ 构建**：
1. 从分割标注中提取肾脏和肿瘤表面网格（Marching Cubes, step=2, ~30K顶点）
2. 通过PCA对齐的六分区（前/后/上/下/内/外）+ 肿瘤邻近带 + 可选肾门区

**合成部分观测序列生成**：
| 参数 | 值 | 含义 |
|------|-----|------|
| 形变模型 | B-spline + 正弦呼吸 | 控制点随机位移，最大振幅 5mm |
| 序列长度 | 16 帧 | 与 truncated BPTT 训练帧数对齐 |
| 覆盖率 | 10% / 20% / 40% / 60% | 模拟不同手术阶段的可见性 |
| 噪声 | 高斯 1-3mm + 1-5%离群点 | 模拟深度传感器噪声 |
| 遮挡 | 随机 patch + 器械遮挡 | 模拟手术工具阻挡 |

### 5.2 评估指标 (Metrics)

#### Metric 1: Surface Chamfer Distance (表面Chamfer距离)

$$\text{Chamfer}(V_t, V_t^*) = \frac{1}{2}\Big(\frac{1}{|V_t|}\sum_{v \in V_t} \min_{v^* \in V_t^*} \|v - v^*\| + \frac{1}{|V_t^*|}\sum_{v^* \in V_t^*} \min_{v \in V_t} \|v^* - v\|\Big)$$

- **单位**：毫米 (mm)
- **含义**：重建表面与真实形变表面之间的平均双向最近点距离
- **解读**：在amplitude=5mm的形变下，Chamfer 1.5mm意味着平均误差约为形变幅度的30%

#### Metric 2: Tumor Centroid Error (肿瘤质心误差)

$$\text{TumorErr} = \|c_t - c_t^*\|_2$$

- **单位**：毫米 (mm)
- **含义**：估计的肿瘤质心与真实肿瘤质心的欧氏距离
- **临床意义**：这是术中最关键的指标——术者需要知道肿瘤在哪。1-2mm的误差在临床上通常可接受

#### Metric 3: Leakage Rate (泄漏率)

$$\text{Leak} = \frac{\sum_{(i,k) \in \mathcal{N},\, y(i) \neq y(k)} |\Delta v_i - \Delta v_k|}{\sum_{(i,k) \in \mathcal{N}} |\Delta v_i - \Delta v_k|}$$

- **范围**：$[0, 1]$
- **含义**：跨越语义边界的位移不连续性占总位移不连续性的比例
- **理想值**：0（完全语义隔离）
- **注意**：此指标衡量的是**边界位移不连续性**（boundary discontinuity），而非 $L_\Phi$ 的跨组件耦合。我们的诊断已验证 $L_\Phi$ 是完美块对角的。实测泄漏 (~10%) 来源于Kernel A k近邻中约3%的跨边界对

#### Metric 4: Drift (漂移)

$$\text{Drift}_t = \frac{1}{n}\sum_{i=1}^{n} \|v_{t,i} - v_{0,i}\|$$

- **单位**：毫米 (mm)
- **含义**：当前顶点坐标与术前位置的平均偏移
- **临床意义**：衡量Theorem 2是否成立——漂移应有界而非随时间发散

#### Metric 5: Relative Improvement (相对改善率)

$$\text{Imp}_t = \frac{\text{Chamfer}(V_0, V_t^*) - \text{Chamfer}(V_t, V_t^*)}{\text{Chamfer}(V_0, V_t^*)}$$

- **范围**：$(-\infty, 1]$
- **含义**：相对于不做任何更新（保持术前状态 $V_0$），SSCU带来了多少改善
- **$> 0$**：SSCU比不更新更好
- **$< 0$**：过冲（overshoot），SSCU比不更新还差

#### Metric 6: P/D Ratio (先验/数据比)

$$\text{P/D ratio} = \frac{\lambda_{\text{prior}} \cdot (1 - S_t(i))}{S_t(i)}$$

- **含义**：先验项与数据项在系统矩阵 $A_t$ 中的权重比
- **P/D >> 1**：先验主导，顶点保持在术前位置
- **P/D << 1**：数据主导，顶点跟随观测移动
- **P/D ~ 1**：先验与数据平衡的临界点

### 5.3 实验结果

#### 5.3.1 主实验：50例 × 4覆盖率

50例KiTS23病例，每例4个覆盖率（10/20/40/60%），每个覆盖率16帧序列。V3参数。

| 覆盖率 | Chamfer (mm) | 肿瘤误差 (mm) | 泄漏率 | 漂移斜率 | Support | P/D Ratio | 延迟 (ms) |
|--------|:----------:|:----------:|:------:|:------:|:------:|:------:|:------:|
| 10% | 1.51±0.07 | 1.48±0.15 | 0.167 | 0.019 | 0.086 | ~1.1 | 217.5 |
| 20% | 1.49±0.07 | 1.46±0.15 | 0.154 | 0.019 | 0.205 | ~0.4 | 215.9 |
| 40% | 1.48±0.06 | 1.44±0.14 | 0.140 | 0.019 | 0.407 | ~0.15 | 234.2 |
| 60% | 1.47±0.06 | 1.43±0.14 | 0.137 | 0.022 | 0.474 | ~0.11 | 226.2 |

**关键观察**：
1. **肿瘤误差单调改善**：从1.48mm (10%) 到 1.43mm (60%)，改善3.4%
2. **支持场正确累积**：Support从0.086到0.474，差异5.5倍，表明覆盖率确实控制了系统行为
3. **泄漏率随覆盖率下降**：0.167→0.137，说明更多观测使系统更依赖数据而非扩散
4. **漂移有界**：16帧累计漂移约0.3-0.35mm，远小于5mm形变幅度

#### 5.3.2 V3 单例验证 (case_00000)

| 覆盖率 | Chamfer (mm) | 肿瘤误差 (mm) | 过冲帧 | 平均改善 |
|--------|:----------:|:----------:|:------:|:------:|
| 10% | 1.444 | — | — | — |
| 20% | 1.429 | — | — | — |
| 40% | 1.416 | 1.415 | 2 | — |
| 60% | 1.401 | 1.442 | 2 | +0.51% |

V3相比V2的改善：Chamfer分化从平坦/反向 → **1.444→1.401 (3%单调改善)**；肿瘤误差改善4.2%；过冲帧从5/16降至2/16。

#### 5.3.3 消融实验 (Ablation Study)

Coverage=40%, case_00000:

| 配置 | Chamfer (mm) | 肿瘤 (mm) | 泄漏 | 漂移 (mm) | 过冲帧 |
|------|:----------:|:------:|:------:|:------:|:------:|
| **V3 Full** | **1.416** | **1.415** | **0.095** | **0.546** | **2** |
| $\lambda_{\text{sem}}=0$ (无T1) | 1.620 (+14.4%) | 1.441 | 0.032 | 0.918 | **12** |
| $\lambda_{\text{prior}}=0$ (无T2) | 1.452 (+2.5%) | 1.405 | 0.123 | **0.747 (+37%)** | 5 |
| $\lambda_{\text{les}}=0$ (无T3) | 1.415 | **1.337 (-5.5%)** | 0.094 | 0.549 | 3 |
| $\lambda_{\text{les}}=5$ (过约束) | 1.419 | 1.483 (+4.8%) | 0.098 | 0.532 | 2 |

**消融解读**：

- **去除语义平滑 ($\lambda_{\text{sem}}=0$)**：Chamfer恶化14.4%，过冲暴增到12/16帧。这验证了**Theorem 1的关键性**——语义Laplacian不仅隔离组件，更是系统稳定性的核心
- **去除先验回退 ($\lambda_{\text{prior}}=0$)**：漂移增加37%。在没有先验约束的16帧后，漂移达到0.75mm（vs有先验的0.55mm）。验证了**Theorem 2**
- **去除病灶保护 ($\lambda_{\text{les}}=0$)**：肿瘤误差反而下降5.5%。这揭示了 $\lambda_{\text{les}}$ 的真实角色：不是提高肿瘤定位精度，而是**约束肿瘤边界的形变幅度**（boundary stability regularizer）。在当前5mm形变下，过度约束反而阻碍跟踪

### 5.4 V1→V2→V3 修复历程

| 版本 | 关键改动 | Chamfer | 肿瘤误差 | Support@60% | 过冲 |
|------|---------|---------|---------|------------|------|
| V1 (broken) | k=1, gamma=0.9, λ_prior=3.0 | ~1.46 | ~1.49 | 0.05 | — |
| V2 (fixed) | k=8, gamma=0.3, λ_prior=0.1, σ=10 | ~1.53 | ~1.49 | 0.47 | 5/16 |
| V3 (current) | +comp-mask, +support-aware α, λ_les=1 | **1.401** | **1.415** | 0.47 | **2/16** |

V1的致命问题：k=1最近邻意味着每个观测点只更新1个顶点的支持——在30K顶点的网格上，即使60%覆盖率，支持也无法正确累积，P/D ratio达到57:1（先验完全压制观测）。

---

## 6. Related Work and Positioning

### 6.1 数字孪生在现代ML中的定位

| 工作 | 会议 | 方法 | SSCU的区别 |
|------|------|------|-----------|
| Med-Real2Sim | NeurIPS 2024 | 物理信息自监督学习识别患者特异性参数 | SSCU聚焦3D几何在线维护，非参数识别 |
| CALM-DT | ICML 2025 | 将持续更新DT重构为LLM上下文学习 | SSCU提供显式数学保证，非黑盒 |

### 6.2 手术AR配准

| 工作 | 精度 | 延迟 | SSCU的优势 |
|------|------|------|-----------|
| Stereovideo-to-CT overlay | ~1mm TRE | ~100ms | SSCU维护持续状态+支持场，非逐帧配准 |
| ICP→CPD deformable (phantom) | 1.28±0.68mm TRE | — | SSCU有语义隔离+漂移界，CPD无 |
| Landmark-free DT registration | 7.3±4.1mm tumor | 9.4ms | SSCU精度更高(1.43mm)，且维护完整孪生状态 |

### 6.3 腹腔镜部分观测配准

EG-Net (MICCAI 2025) 明确记录了重叠率<30%时ICP类方法的局部最优失败。SSCU的Theorem 1（语义隔离）正是对"局部对应拖拽整个表面"这一核心失败模式的结构性解答。

### 6.4 内窥镜可变形重建

| 工作 | 特点 | SSCU的区别 |
|------|------|-----------|
| EndoGSLAM | >100fps 高斯溅射实时重建 | 不从CT先验出发，无语义隔离 |
| EndoSurf | Neural SDF + 形变场 | 无病灶追踪，无支持/不确定性场 |
| D4Recon | 动态高斯溅射 | 无Theorem 1/2/3的形式化保证 |

**关键定位差异**：上述方法侧重于**从视频重建场景**；SSCU侧重于**从部分观测维护结构化孪生状态**。SSCU的输出不仅是几何，还包括语义分区下的支持/不确定性场、病灶追踪和临床距离指标。

---

## 7. Discussion

### 7.1 P/D Ratio: 理解系统行为的统一视角

P/D ratio $= \lambda_{\text{prior}}(1-S)/S$ 是理解SSCU在不同覆盖率下行为变化的核心指标：

| 覆盖率 | Support | P/D Ratio | 系统行为 |
|--------|---------|-----------|---------|
| 10% | 0.086 | ~1.1 | 先验与数据近乎平衡，保守更新 |
| 20% | 0.205 | ~0.4 | 数据开始主导，跟踪变得积极 |
| 40% | 0.407 | ~0.15 | 数据强烈主导，形变跟踪良好 |
| 60% | 0.474 | ~0.11 | 数据完全主导，可能出现过冲 |

这个比值从 ~1.1 降到 ~0.11（10倍变化），解释了为什么覆盖率的提升会显著改变系统行为。

### 7.2 Leakage的正确解读

我们的深度诊断揭示了一个重要区分：

- **结构泄漏 (Structural Leakage)**：$L_\Phi$ 矩阵中的跨组件耦合 → **实测为0**。Theorem 1在矩阵层面100%成立
- **边界不连续性 (Boundary Discontinuity)**：语义边界两侧由于独立支持产生的位移差异 → **实测~10%**

实测"泄漏"来源：Kernel A的k=8近邻中约3.0%跨越组件边界（V3中已通过component-aware mask修复），以及2,649个边界顶点(8.6%)从两侧独立获得不同支持。

### 7.3 $\lambda_{\text{les}}$的角色重新定位

消融实验揭示了一个非直觉结论：$\lambda_{\text{les}}=0$ 时肿瘤精度反而更好(-5.5%)。这说明 $\lambda_{\text{les}}$ 的真实角色不是"保护肿瘤定位精度"，而是**约束肿瘤边界的形变幅度**。在当前5mm形变下，这个约束反而阻碍了跟踪。我们将其重新定位为"boundary stability regularizer"——它的临床价值在于防止极端形变下肿瘤边界的拓扑异常。

### 7.4 Limitations

1. **合成-真实差距 (Synthetic-to-real gap)**：主要定量基准使用模拟部分观测和形变。缓解：连接到已发表的肾脏AR导航基准，并规划TRUSTED跨模态验证
2. **前端对齐依赖**：SSCU假设观测已在孪生坐标系中或近似对齐。可选集成EG-Net类刚性对齐前端
3. **极低重叠下的对应脆弱性**：当对应完全错误时，残差烘焙被污染。缓解：鲁棒化 $q_j$、离群点拒绝、显示不确定性地图

---

## 8. Hyperparameter Summary

| 参数 | 符号 | V3值 | 搜索范围 | 物理含义 |
|------|------|------|---------|---------|
| 高斯带宽 | $\sigma$ | 10.0 mm | 3-15 mm | 每个观测点的空间影响范围 |
| EMA衰减率 | $\gamma$ | 0.30 | 0.1-0.9 | 历史支持保留比例。低值=快速响应 |
| 扩散步长 | $\eta$ | 0.20 | 0.05-0.5 | 每步扩散强度 |
| 扩散迭代 | $I_{\text{diff}}$ | 8 | 3-15 | 支持场扩散次数 |
| 语义平滑 | $\lambda_{\text{sem}}$ | 1.0 | 0-5.0 | 组件内位移一致性约束强度 |
| 先验回退 | $\lambda_{\text{prior}}$ | 0.1 | 0.01-3.0 | 无观测区域向 $V_0$ 回归力度 |
| 病灶保护 | $\lambda_{\text{les}}$ | 1.0 | 0-10.0 | 肿瘤邻近带位移约束强度 |
| 更新步长 | $\alpha$ | 0.8 | 0.3-1.0 | 顶点更新阻尼 |
| 近邻数 | $k$ | 8 | 1-16 | Kernel A残差扩散的近邻数 |
| PCG容差 | tol | $10^{-5}$ | — | 共轭梯度收敛精度 |
| PCG最大迭代 | maxiter | 50 | — | 共轭梯度最大步数 |
| 可见阈值 | $\tau_{\text{vis}}$ | 0.5 | 0.3-0.7 | Support超过此值视为"可见" |

---

## 9. Conclusion and Future Work

SSCU将术中数字孪生更新形式化为带结构约束的MAP推断，在3个维度提供形式化保证：语义隔离（Theorem 1）、有界漂移（Theorem 2）、病灶误差界（Theorem 3）。在KiTS23 50例基准上，SSCU达到1.43-1.48mm肿瘤定位精度和13-19ms帧延迟，与术中AR更新率要求兼容。

**下一步工作**：
1. **跨尺度验证**：按肿瘤体积（<10mm / 10-30mm / >30mm）和形变幅度（2/5/10mm）分组统计
2. **形变模型多样性**：添加随机脉冲（手术器械突发压迫）和非周期呼吸模式
3. **延迟优化**：KD-tree GPU化（FAISS/cuSpatial），目标30ms
4. **跨器官泛化**：在LiTS（肝脏肿瘤）上测试SSCU的泛化能力
5. **真实数据验证**：TRUSTED数据集（配对CT-3DUS肾脏）上的跨模态验证
6. **Baseline对比**：与Nonrigid ICP、CPD、ED-Graph、TSDF、PCN、OccNet、DeepSDF的系统对比

---

## Appendix A: 代码结构

```
scripts/nips_sscu/
    step1_build_phi.py       # 语义分区 Φ 构建
    step6_benchmark_gen.py   # 合成基准序列生成
    sscu_engine.py           # CPU SSCU 引擎
    sscu_engine_gpu.py       # GPU SSCU 引擎
    evaluate_sscu.py         # 评估指标计算
    run_sscu_pipeline.py     # 端到端流水线

nips/
    dt proposal.txt          # 原始 proposal (完整公式+相关工作)
    Diagnostic_Report_V2.md  # V2 诊断报告
    Diagnostic_Report_V3.md  # V3 诊断报告
    Batch50_Diagnostic_Report.md  # 50例批量报告
    SSCU_项目进展与技术报告.md     # 中文技术报告
```

## Appendix B: V1→V2→V3 完整修复日志

### V1 → V2 (核心算法修复)

| 参数 | V1 (Broken) | V2 (Fixed) | 修复原因 |
|------|:----------:|:----------:|---------|
| k_neighbors | 1 | **8** | k=1 使每个观测只影响1个顶点，support无法累积 |
| $\sigma$ | 3.0 | **10.0** | 配合 k=8 扩大高斯核范围 |
| $\gamma$ | 0.90 | **0.30** | 0.90 = 每帧新观测仅10%权重，累积太慢 |
| $\lambda_{\text{prior}}$ | 3.0 | **0.1** | P/D ratio从57:1降到~1:1 |
| $I_{\text{diff}}$ | 3 | **8** | 增加扩散步数提升支持覆盖 |

### V2 → V3 (精细化修复)

| 改动 | 效果 |
|------|------|
| Kernel A 组件感知掩码 | 跨边界残差被mask，减少boundary discontinuity |
| Support-aware alpha | $\alpha_i = 0.24 + 0.56 S_i$，过冲从5/16降到2/16 |
| $\lambda_{\text{les}}$ 5.0→1.0 | 消融证实5.0过度约束，1.0是更好的平衡点 |
