# RobotBridge

**Probabilistic Ray–Based Human Surface Alignment & Micro-Skin Optimization**

## 1. 项目目标（What & Why）

本项目实现了一套**以深度图为唯一观测**的人体皮肤几何对齐与局部微表面优化框架，目标是：

* 在**不依赖显式 3D 监督**的情况下
* 利用**单视角深度图（ROI）**
* 通过**概率射线似然（Prob-Ray Likelihood）**
* 分阶段（Stage1 → Stage2 → Stage3）

  * 对齐人体整体姿态
  * 并在 ROI 内恢复高频皮肤微结构（Micro-Skin）

该框架主要用于**仿真验证与方法论探索**，当前版本中：

* 深度图来自仿真生成（固定数据源）
* 人体几何由 SKEL 模型生成
* Stage2 / Stage3 使用同一份 GT SKEL 参数作为“几何真值参考”

---

## 2. 项目整体结构（Project Structure）

```
RobotBridge/
├── pipeline/                  # Pipeline 总控
│   ├── config_pipeline.py
│   └── pipeline_runner.py
│
├── stage1/                     # Stage1：相机 & ROI & 深度生成（仿真）
│   ├── camera.py
│   ├── roi.py
│   ├── render_depth.py
│   ├── optim_root.py
│   ├── run_sim_stage1.py
│   └── config_stage1.py
│
├── stage2/                     # Stage2：姿态 θ 优化（Prob-Ray）
│   ├── config_stage2.py
│   ├── io_stage1.py
│   ├── skel_adapter.py
│   ├── optim_theta.py
│   └── run_stage2.py
│
├── stage3/                     # Stage3：ROI 微表面优化（Micro-Skin）
│   ├── gaussian_adapter.py
│   ├── mock_generator.py
│   ├── renderer.py
│   ├── optim_gaussian.py       #（历史/备用）
│   ├── run_stage3_pipeline.py
│   └── run_synthetic_test.py   #（实验脚本）
│
├── utils/                      # 公共工具
│   ├── ray_likelihood.py       # Prob-Ray 核心实现
│   ├── metrics.py              # 误差指标（PA-MPJPE, MeanDev 等）
│   ├── vis.py
│   ├── vis_roi.py
│   └── ply_vis.py
│
├── experiments/
│   └── synthetic_case_001/
│       ├── skeleton/           # GT SKEL 参数（poses / betas）
│       ├── mesh/               # GT mesh（obj）
│       └── depth.npy           #（可选）仿真深度
│
├── output/                     # Pipeline 输出目录（自动生成）
│   └── YYYYMMDD_xxxxxx_mode1/
│       └── source_data/
│           └── depth_obs.npy   # 固定使用的观测深度
│
└── README.md
```

---

## 3. 核心数据与参数来源（Very Important）

### 3.1 SKEL 参数的**唯一源头**

所有 Stage2 / Stage3 中使用的 **人体几何真值** 都来自：

```
experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz
```

该文件包含：

* `poses`：SKEL 姿态参数（对齐到 46 维）
* `betas`：SKEL 形状参数（10 维）

这些参数通过 `SkelAdapter` 送入 SKEL 模型，生成 GT 网格。

> 注意：
>
> * Stage1 **不使用** SKEL 参数
> * Stage2 / Stage3 **不会从 Stage1 继承 pose/beta**

---

### 3.2 深度图的来源（Depth Observation）

Pipeline 在 `--use_fixed_data` 模式下，**始终使用同一张固定深度图**：

```
output/20251218_201849_mode1/source_data/depth_obs.npy
```

这是一张 **ROI 深度图（单位：米）**，用于：

* Stage2 的姿态优化
* Stage3 的微表面优化

该深度图被视为**观测数据**，并不要求与当前 GT 网格完全同源。

---

## 4. Pipeline 运行方式（How to Run）

### 4.1 Stage2：姿态 θ 优化

```bash
python -m pipeline.pipeline_runner \
  --mode 1 \
  --steps stage2 \
  --use_fixed_data \
  --force_gt_init \
  --perturb_theta
```

**Stage2 的目标：**

* 固定形状 β
* 仅优化姿态 θ
* 通过 Prob-Ray 深度似然，使预测人体在 ROI 内对齐观测深度

**输出关键指标：**

```
[Stage2][GlobalMeanDev] start=...m end=...m
```

表示 **全局皮肤顶点** 相对于 GT 网格的平均偏差（仅用于诊断）。

---

### 4.2 Stage3：ROI 微表面优化（Micro-Skin）

```bash
python -m pipeline.pipeline_runner \
  --mode 1 \
  --steps stage3 \
  --use_fixed_data \
  --force_gt_init \
  --perturb_vertices
```

**Stage3 的目标：**

* 固定全局姿态
* 仅在 ROI 内优化皮肤微结构
* 使用“位移基元 + 各向异性椭球”表示微表面形变
* 椭球中心必须由**变形后的网格**计算，不能直接学习

**输出关键指标：**

```
[Stage3][ROIMeanDev] start=...m end=...m
```

表示 **ROI 区域皮肤顶点** 相对于 GT 网格的平均偏差（仅用于诊断）。

---

## 5. 方法论概要（High-Level Logic）

### 5.1 为什么使用 Probabilistic Ray Likelihood

* 深度是 2.5D 观测，天然不完备
* 直接点对点回归不稳定
* Prob-Ray 将“深度一致性”表述为：

  * 沿像素射线
  * 在观测深度附近的窄带内
  * 预测几何密度的概率重叠

这使得优化：

* 对噪声更鲁棒
* 对遮挡更稳定
* 可自然扩展到混合几何表示（椭球）

---

### 5.2 为什么 Stage2 / Stage3 分开

* **Stage2**：

  * 解决中低频问题（整体姿态）
  * 只允许低维参数（θ）变化
* **Stage3**：

  * 解决高频问题（皮肤微结构）
  * 允许高维自由度，但只在 ROI 内

这是典型的 **Coarse-to-Fine Optimization**。

---

## 6. 关于误差指标的说明（Important Caveat）

项目中输出的：

* `GlobalMeanDev`
* `ROIMeanDev`

**并不是训练目标，仅用于诊断。**

原因：

* 训练目标是 **深度似然（NLL）**
* MeanDev 是 **对 GT 网格的 3D 点到点距离**
* 深度约束是 2.5D，存在多解性

因此：

> 深度似然下降 ≠ 顶点一定更接近 GT

这是**设计使然，不是 bug**。

---

## 7. 当前状态与已知限制

### 已完成

* 完整的 Stage1 / Stage2 / Stage3 pipeline
* Prob-Ray Likelihood 的稳定实现
* Micro-Skin 位移基元模型
* 指标输出与可视化

### 已知限制

* Stage3 尚未使用 Stage2 的 θ 输出（当前仍以 GT 作为起点）
* 深度观测与 GT 网格可能不同源
* Stage3 对切向形变约束较弱（可能导致几何漂移）

---

## 8. 项目定位

> **RobotBridge 是一个以深度图为唯一观测、通过概率射线似然实现人体皮肤几何分阶段对齐与微表面优化的研究型工程框架。**


