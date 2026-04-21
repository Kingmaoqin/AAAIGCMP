# ZIF-8 数据建模与交互式网页系统完整报告

## 1. 项目概述

本项目基于两份 ZIF-8 合成数据表和一篇关于甲醇/水混合溶剂调控 ZIF-8 粒径、形貌、结晶性、孔结构与产率的论文，完成了两条主线工作：

1. 建立面向 `TEM_SEM_Diameter` 的机器学习回归建模流程。
2. 在最佳模型基础上构建一个可交互网页，用于实时调整合成参数、预测粒径，并联动显示颗粒级形貌与 canonical 分子结构示意。

本报告只描述当前代码和当前输出目录中已经真实实现的内容，不对尚未实现的功能作虚构性表述。

---

## 2. 输入文件与原始约束

项目输入文件位于 `ZIF8/` 目录下：

- `ZIF8_data_NanoTrends.xlsx`
- `Additiaonl data points.xlsx`
- `Solvent-Mediated Control of ZIF-8 Morphology, Crystallinity, and Yield - Insights from Methanol-Water Synthesis.pdf`

执行时遵守的关键约束是：

- 不修改任何原始输入文件。
- 所有新代码、结果表、图、模型、点云、日志、网页文件都只写入 `newtry_zif8_modeling/`。
- 目标变量固定为 `TEM_SEM_Diameter`。
- 同时比较原始直径回归与 `log(TEM_SEM_Diameter)` 回归。
- 结构可视化必须明确区分：
  - 颗粒级外部形貌示意
  - canonical 内部框架说明
  - 不能把粒径预测错误地解释为逐原子尺寸缩放

---

## 3. 数据读取与字段标准化

主脚本为：

- `newtry_zif8_modeling/run_pipeline.py`

其数据处理流程如下。

### 3.1 自动读取 Excel

脚本会自动检查工作簿中的 sheet，并读取首个非空 sheet。

本次实际读取结果为：

- `ZIF8_data_NanoTrends.xlsx` -> `Sheet1`
- `Additiaonl data points.xlsx` -> `Sheet1`

### 3.2 列名清洗与映射

程序会对列名做统一清洗：

- 去除空格和特殊字符
- 统一命名风格
- 映射到标准字段名

重点标准字段包括：

- `Zinc`
- `HmIm`
- `Solvent_D`
- `Solvent`
- `Temperature`
- `Reaction_time`
- `TEM_SEM_Diameter`
- `DOI`

实际映射日志保存在：

- `logs/column_mapping_log.csv`

例如：

- `Hmim` 被映射为 `HmIm`
- `Temp` 被映射为 `Temperature`
- `Time` 被映射为 `Reaction_time`
- `Methanol / Water (vol %)` 被映射为 `Methanol_Water_Ratio`

### 3.3 合并数据

两张表合并为统一数据表，并增加：

- `source_file`
- `source_sheet`
- `sample_id`

合并后的原始整理表保存在：

- `data/combined_raw.csv`

### 3.4 DOI 分组处理

`DOI` 被保留为 grouped cross validation 的分组字段。

对于补充数据表中缺失 DOI 的样本，程序使用：

- `source_file + sample_id`

构造临时分组标识，避免 grouped CV 因空值而失效。

---

## 4. 从 PDF 提取的科学先验

程序使用 `pdftotext` 自动抽取论文文本，并将文本和结构化先验保存到：

- `data/article_extracted.txt`
- `data/article_prior.json`

当前提取并写入系统的核心先验包括：

- 纯水条件颗粒更大、更圆钝，平均粒径约 123 nm。
- 纯甲醇条件颗粒更小、更 faceted，平均粒径约 50 nm。
- 随甲醇比例上升，粒径整体减小，形貌从 rounded 向 faceted rhombic dodecahedron 过渡。
- 50/50 甲醇/水条件的 BET 表面积最高，约 1465 m2/g。
- 50/50 条件是结构、性质、产率之间较平衡的折中点。
- 水富集条件产率高，甲醇富集条件更易得到更小、更锐利的颗粒。

这些先验在项目里主要用于两件事：

1. 在报告中解释建模背景。
2. 在网页和点云系统中做 article-guided 的形貌模板映射。

---

## 5. 数据统计与质量检查

程序自动输出了数据统计摘要：

- `tables/data_summary.csv`
- `tables/data_summary.md`
- `tables/numeric_describe.csv`
- `tables/doi_counts.csv`

统计内容包括：

- 总样本数
- 每列缺失值
- 数值列描述统计
- DOI 唯一数量
- DOI 样本数分布

同时输出了基础数据图：

- `figures/target_distribution.png`
- `figures/log_target_distribution.png`
- `figures/missingness.png`
- `figures/feature_correlation_heatmap.png`
- `figures/doi_counts.png`

---

## 6. 特征工程

处理后的完整特征表保存于：

- `processed/features_full.csv`

### 6.1 保留的基础特征

作为建模输入基础的原始变量包括：

- `Zinc`
- `HmIm`
- `Solvent_D`
- `Solvent`
- `Temperature`
- `Reaction_time`

### 6.2 构造的核心派生特征

程序真实构造并用于训练的派生特征包括：

- `methanol_fraction`
- `water_fraction`
- `HmIm_to_Zn`
- `Zn_to_Solvent`
- `HmIm_to_Solvent`
- `log_reaction_time`

### 6.3 增强特征

还进一步加入了：

- 各基础变量平方项，例如 `Zinc_sq`
- 各基础变量 `log1p` 项，例如 `log1p_HmIm`
- 交互项，例如：
  - `Zinc_x_HmIm`
  - `Zinc_x_Solvent`
  - `HmIm_x_Solvent`
  - `Solvent_D_x_Temperature`
  - `Solvent_D_x_Reaction_time`
  - `Temperature_x_Reaction_time`

### 6.4 显式排除的字段

训练中未把以下字段作为输入特征：

- `Paper_number`
- `DOI`
- `source_file`
- `source_sheet`
- `sample_id`
- `TEM_SEM_Diameter`

这一步的目的是真正避免数据泄漏，而不是仅靠列名习惯推断。

---

## 7. 模型训练与验证设计

### 7.1 训练目标

本项目同时比较了两种目标空间：

- `raw`：直接预测 `TEM_SEM_Diameter`
- `log`：预测 `log(TEM_SEM_Diameter)`，最后再反变换回 nm

最终最佳结果来自 `log` 目标空间。

### 7.2 验证策略

外层使用：

- `GroupKFold(n_splits=5)`

分组变量为：

- `DOI`

内层调参使用：

- `GroupShuffleSplit(n_splits=1, test_size=0.25)`

同样保持 DOI 分组约束。

这意味着：

- 同一 DOI 来源的样本不会在同一轮调参与验证中泄漏到不同侧。
- 评价更接近“跨文献泛化”的情境，而不是随机打散后的乐观估计。

### 7.3 实际训练的模型

本轮实际纳入比较的成功模型共 22 个，失败 1 个：

成功模型：

- `LinearRegression`
- `Ridge`
- `Lasso`
- `ElasticNet`
- `BayesianRidge`
- `ARDRegression`
- `HuberRegressor`
- `RANSACRegressor`
- `PassiveAggressiveRegressor`
- `OrthogonalMatchingPursuit`
- `KNeighborsRegressor`
- `DecisionTreeRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `BaggingRegressor`
- `AdaBoostRegressor`
- `GradientBoostingRegressor`
- `HistGradientBoostingRegressor`
- `MLPRegressor`
- `XGBoostRegressor`
- `XGBoostRFRegressor`
- `LightGBMRegressor`

失败并被自动记录但不中断流程的模型：

- `RadiusNeighborsRegressor`

失败原因是部分外层 fold 内无法找到有效邻域配置。

### 7.4 调参说明

本项目使用的是参数随机采样搜索，而不是仅跑默认参数。

为了保证本地环境下整个流程能完整跑通，当前预算按模型复杂度自适应收敛到约 10 到 18 组参数样本。这个实现是保守版搜索，不是穷尽搜索，也不是 Optuna 全量搜索。

这点已经在代码和报告中明确说明，不做夸大表述。

---

## 8. 模型结果与最佳模型

结果表如下：

- `tables/cv_results_group.csv`
- `tables/cv_results_group_detailed.csv`
- `tables/cv_results_reference_kfold.csv`
- `tables/model_ranking.csv`

### 8.1 最佳模型

当前最佳模型为：

- `BaggingRegressor`

目标空间为：

- `log`

### 8.2 最佳模型 grouped nested CV 结果

当前排名第一模型的主要指标为：

- 平均 RMSE：`309.15 nm`
- 平均 MAE：`108.81 nm`
- 平均 R²：`0.556`
- 平均 log-RMSE：`0.570`

这表明：

- log 空间建模确实优于当前 raw 空间最佳结果
- 但样本中极大粒径点对误差稳定性影响较强，RMSE 标准差偏大

### 8.3 全样本拟合输出

最佳模型的全样本预测结果保存在：

- `tables/predictions_all.csv`

包含字段：

- 真实直径
- 预测直径
- 残差
- 绝对误差
- 相对误差

### 8.4 最佳模型保存

模型保存于：

- `models/best_model.pkl`
- `models/best_model_params.json`

---

## 9. 模型解释与图表产物

### 9.1 特征解释

已输出：

- `tables/feature_importance.csv`
- `figures/feature_importance.png`

说明：

- 当前最佳模型采用的是 BaggingRegressor
- 对这类模型，项目优先输出特征重要性和 permutation importance 风格解释
- SHAP 本轮尝试过，但未稳定成功，因此未在最终主结果中伪造 SHAP 图

### 9.2 模型表现图

已生成：

- `figures/group_cv_rmse.png`
- `figures/group_cv_mae.png`
- `figures/group_cv_r2.png`
- `figures/best_true_vs_pred.png`
- `figures/best_residuals.png`
- `figures/best_error_distribution.png`

这些图用于横向比较各模型和观察最佳模型误差结构。

---

## 10. 点云与颗粒级可视化实现

### 10.1 设计原则

颗粒级可视化遵守以下原则：

- 可视化对象是颗粒外形，不是逐原子重建。
- 预测值决定的是颗粒外尺度，而不是 Zn-N 键长。
- 形貌模板来自 article-guided 规则，不是 TEM 图像端到端重建。

### 10.2 三类颗粒模板

实际实现了三种模板：

- `rounded_particle`
- `truncated_rhombic_dodecahedron`
- `rhombic_dodecahedron`

### 10.3 模板映射逻辑

模板根据：

- `methanol_fraction`
- `Solvent_D`

进行 article-guided 选择：

- 水富集 -> 更 rounded
- 中间混合 -> 截断 rhombic dodecahedron
- 甲醇富集 -> 更 faceted 的 rhombic dodecahedron

### 10.4 输出内容

点云输出目录：

- `pointclouds/`

总览图：

- `figures/pointcloud_gallery.png`

清单表：

- `tables/pointcloud_manifest.csv`

每个样本包含：

- `.ply`
- `.png`
- `.json`

---

## 11. 交互式网页系统概述

网页目录为：

- `newtry_zif8_modeling/crystal_viewer/`

关键文件：

- `app.py`
- `templates/index.html`
- `static/viewer.js`
- `static/style.css`
- `run_crystal_viewer.sh`

### 11.1 后端职责

网页后端使用 Flask。

后端真实职责包括：

1. 加载最佳模型 `best_model.pkl`
2. 根据当前输入重新计算工程特征
3. 实时调用模型输出预测粒径
4. 返回颗粒级几何数据
5. 返回 canonical 分子结构几何数据
6. 返回页面元信息、控件范围、参数说明、预设样本

### 11.2 后端 API

当前实现了以下接口：

#### `/api/meta`

返回：

- 最佳模型名称
- 目标空间
- 控件数值范围
- 默认值
- 参数说明
- 文章先验
- 预设样本列表

#### `/api/predict`

输入：

- `Zinc`
- `HmIm`
- `Solvent_D`
- `Solvent`
- `Temperature`
- `Reaction_time`

返回：

- 预测粒径
- `methanol_fraction`
- 形貌模板
- 点云
- mesh
- canonical 分子结构
- 结果解释文案

#### `/api/preset/<sample_id>`

作用：

- 载入某个代表性样本的真实参数
- 直接驱动前端更新图形和结果面板

---

## 12. 网页页面结构与每个按钮/控件功能

本节描述的是当前页面中真实存在的控件和行为。

### 12.1 顶部信息区

页面顶部显示：

- 项目标题
- 页面说明
- 三个摘要卡片

三个摘要卡片分别显示：

- `Best model`
- `Target space`
- `Shape template`

其中 `Shape template` 会随着当前预测实时更新。

### 12.2 Controls 区

#### `Preset` 下拉框

作用：

- 从代表性样本中选择一个预设参数组合

当前下拉框的文本会显示类似：

- `S162 | true=123.0 nm | pred=135.6 nm`

本次已修复：

- 过长 preset 文本导致控件溢出和按钮挤压的问题

#### `Load Preset` 按钮

作用：

- 将当前选中的预设样本参数读入所有控件
- 同时更新预测结果与图形

#### `3D Mode` 下拉框

当前包含 4 个模式：

- `Point Cloud`
- `Mesh`
- `Molecular Structure`
- `Split View`

各模式功能如下：

##### `Point Cloud`

- 显示颗粒级点云外形
- 适合看颗粒尺度与外部轮廓变化

##### `Mesh`

- 显示颗粒级 mesh 外壳
- 适合看 faceted 程度和几何轮廓

##### `Molecular Structure`

- 显示 canonical ZIF-8 风格球棍结构
- 带 unit-cell 线框
- 带元素图例
- 带角落里的 `a / b / c` 三轴坐标标识

##### `Split View`

- 左侧显示颗粒级形貌
- 右侧显示 canonical 分子结构
- 两侧由同一组输入参数联动刷新

#### `Reset to Median` 按钮

作用：

- 将全部参数恢复为训练数据中的中位数默认值
- 然后重新触发预测和图形更新

### 12.3 参数卡片

每个参数卡片都包含：

- 参数名称
- 数值范围
- 单位或含义提示
- 滑条
- 数值输入框
- 问号提示

#### 参数旁的 `?`

作用：

- 鼠标悬停后弹出说明框
- 解释该参数是什么、单位是什么、在合成中大致影响什么

当前已写入说明的参数包括：

- `Zinc`
- `HmIm`
- `Solvent_D`
- `Solvent`
- `Temperature`
- `Reaction_time`

#### 滑条

作用：

- 快速连续调整参数
- 调整后约经过短暂 debounce，再自动刷新预测结果

#### 数值框

作用：

- 精确输入参数值
- 同样会同步到滑条，并自动刷新预测

### 12.4 右侧 Prediction 面板

显示：

- `Pred diameter`
- `Shape template`
- `Point count`
- `Methanol fraction`
- `Water fraction`
- `Extent X/Y/Z`

这些值都来自当前后端预测 payload，不是前端虚构文本。

### 12.5 Result Explanation 面板

这里给每个结果项附带文字解释，说明其科学含义，例如：

- `Pred diameter` 是模型预测的颗粒直径
- `Shape template` 是 article-guided 形貌类别，不是 TEM 图像直接重建
- `Methanol fraction` 是依据数据中的 `Solvent_D` 推断出的甲醇比例代理量
- `Extent X/Y/Z` 是当前可视化对象在三轴方向上的包围盒尺寸

### 12.6 Current Inputs 面板

显示当前六个基础输入的数值和单位，方便对照当前预测状态。

### 12.7 Scientific Guardrail 面板

这里明确声明：

- 颗粒级视图只是粒子尺度示意
- 分子结构视图只是 canonical 内部框架说明
- 预测结果不会驱动原子键长按粒径比例缩放

这部分是避免网页展示造成科学误读的关键保护说明。

---

## 13. 分子结构视图当前实现状态

网页中的分子结构视图不是从外部在线 CIF 下载来的，而是当前项目中程序化构造的 canonical ZIF-8 风格球棍示意。

当前已实现：

- Zn、N、C 三类元素分色
- 原子点
- 键连接线
- 论文风格的 unit-cell 边框
- 右上角元素图例
- 右下角 `a / b / c` 三轴角标

当前未实现：

- 基于真实 CIF 的严格晶体学重建
- 与真实空间群完全一致的高保真坐标体系

因此该模式的正确用途是：

- 说明“内部框架是什么样”

而不是：

- 声称这是自动从预测粒径生成的真实原子级结构

---

## 14. 本次页面问题修复说明

针对用户可见问题，本次已做两项修复：

### 14.1 `a / b / c` 标识重叠

问题来源：

- 早期版本把三轴箭头放在分子结构内部，容易与结构主体重叠

修复方式：

- 将三轴从 3D 结构内部移除
- 改为页面角落的外部 overlay 标识
- 与元素图例分开定位

### 14.2 `Preset` 文本过长导致挤压

问题来源：

- 预设样本文本较长
- 控件栏宽度有限

修复方式：

- 调整 `preset` 行的栅格布局
- 让按钮换到第二行
- 给选择框加上宽度约束与溢出隐藏逻辑

---

## 15. 运行方式

### 15.1 建模主流程

```bash
python /home/xqin5/ZIF8/newtry_zif8_modeling/run_pipeline.py
```

### 15.2 交互式网页

```bash
/home/xqin5/ZIF8/newtry_zif8_modeling/crystal_viewer/run_crystal_viewer.sh
```

默认地址：

- `http://127.0.0.1:8794`

---

## 16. 当前实现的局限性

以下局限性是当前系统真实存在的，应明确保留：

- 当前颗粒级图是几何示意，不是真实逐原子 reconstruction。
- 当前分子结构图是 canonical ball-stick inset，不是在线下载的真实 CIF 严格重建。
- 当前形貌模板映射是 article-guided 的规则映射，不是从 TEM 图像直接分割或反演得到。
- 当前最优模型虽然在 grouped nested CV 下排名第一，但误差仍受极端大粒径样本显著影响。
- 本轮调参是可运行优先的随机采样搜索，不是大规模超参数优化。

---

## 17. 后续建议

如果继续推进，建议优先做以下方向：

1. 引入真实 CIF 并建立更严格的 canonical framework rendering。
2. 对超大粒径样本做分层建模或稳健回归增强。
3. 补充 BET、yield、XRD crystallite size、morphology label 后做多任务学习。
4. 在网页中加入真实样本对照、误差对照和更多预设场景。
5. 进一步优化分子结构布局，使其更接近论文配图的几何对称性和视角表现。

---

## 18. 结论

本项目已经完成：

- 原始数据自动读取与字段标准化
- 特征工程与严格 grouped nested CV 建模
- 最佳模型训练与全样本预测
- 颗粒级点云/mesh 可视化
- 基于最佳模型的交互式网页系统
- 颗粒级与 canonical 分子结构双层展示

同时，系统在科学表述上保持了边界清晰：

- 粒径预测对应颗粒尺度
- 分子结构视图对应 canonical 框架说明
- 二者联动展示，但不混淆为“粒径驱动原子键长缩放”

这也是当前实现最重要的可靠性基础。
