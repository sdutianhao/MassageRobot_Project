# RobotBridge

**Probabilistic GMM–Based Human Surface Alignment & Micro-Skin Optimization**

---

## 1. 项目目标（What & Why）

本项目实现了一套**以单视角深度图为唯一观测**的人体表面几何优化框架，当前版本聚焦 **Stage3：ROI 内皮肤微表面（Micro-Skin）优化**，目标是在固定全局姿态与相机参数的前提下，对局部皮肤几何进行高自由度、概率一致的优化。

核心目标包括：

* 不依赖显式 3D 点云或多视角重建
* 仅使用 ROI 深度图作为观测
* 通过概率模型刻画深度与表面的一致性
* 在 ROI 内恢复高频皮肤微结构

---

## 2. 项目整体结构（Project Structure）

```
RobotBridge/
├── pipeline/
│   ├── config_pipeline.py
│   └── pipeline_runner.py
│
├── stage1/
│   ├── camera.py
│   ├── roi.py
│   ├── render_depth.py
│   └── optim_root.py
│
├── stage2/
│   ├── skel_adapter.py
│   └── optim_theta.py
│
├── stage3/
│   ├── gaussian_adapter.py
│   └── run_stage3_pipeline.py
│
├── utils/
│   ├── gmm_likelihood.py
│   ├── metrics.py
│   ├── vis.py
│   ├── vis_roi.py
│   └── ply_vis.py
│
├── experiments/
│   └── synthetic_case_001/
│
├── output/
│
└── README.md
```

---

## 3. 核心数据与参数来源（Very Important）

### 3.1 人体几何真值

Stage3 使用 SKEL 模型生成的人体网格作为几何参考，其参数来源于：

```
experiments/synthetic_case_001/skeleton/HSMR-ballerina.png.npz
```

该文件包含姿态与形状参数，用于生成 GT 网格，仅作为诊断参考。

---

### 3.2 深度观测

在 `--use_fixed_data` 模式下，Stage3 使用固定的 ROI 深度图：

```
output/.../source_data/depth_obs.npy
```

深度单位为米，仅作为观测数据参与优化。

---

## 4. Pipeline 运行方式（How to Run）

### 4.1 Stage3：ROI 微表面优化（GMM）

```bash
cd ~/MassageRobot_Project/RobotBridge && \
python -m pipeline.pipeline_runner \
  --mode 1 \
  --steps stage3 \
  --use_fixed_data \
  --force_gt_init \
  --perturb_vertices \
  --stage3_data_term gmm \
  --stage3_update_rot \
  --stage3_gmm_center_mode verts \
  --stage3_gmm_sigma_start 0.05 \
  --stage3_gmm_sigma_end 0.005 \
  --stage3_vis_depth_gate 0.06 \
  --stage3_debug \
  --stage3_debug_every 40 \
  --stage3_debug_dump \
  --stage3_lambda_tan 100 \
  --stage3_lambda_tan_anchor 100
```

Stage3 固定全局姿态，仅在 ROI 内对皮肤顶点进行微表面优化。

参数说明：

--mode 1
使用仿真闭环模式，读取固定的数据与相机配置。

--steps stage3
仅执行 Stage3（ROI 微表面优化），不运行 Stage1 / Stage2。

--use_fixed_data
使用已有的固定深度观测（depth_obs.npy），不重新生成数据。

--force_gt_init
以 GT 姿态与形状作为 Stage3 的初始几何状态，用于隔离 Stage3 行为。

--perturb_vertices
在优化前对顶点施加小扰动，用于验证优化是否能够收敛。

--stage3_data_term gmm
使用 GMM Surface Likelihood 作为唯一的数据项。

--stage3_update_rot
启用椭球旋转参数的联合优化。

--stage3_gmm_center_mode verts
将 ROI 顶点本身作为 GMM 分量中心。

--stage3_gmm_sigma_start 0.05
GMM 协方差初始尺度（粗阶段），单位为米。

--stage3_gmm_sigma_end 0.005
GMM 协方差最终尺度（细阶段），单位为米。

--stage3_vis_depth_gate 0.06
静态深度可见性筛选阈值，用于初始化阶段剔除明显不可见顶点。

--stage3_debug
启用调试统计与中间状态记录。

--stage3_debug_every 40
每 40 次迭代输出一次调试信息。

--stage3_debug_dump
保存调试用的中间结果（NPZ / CSV）。

--stage3_lambda_tan 100
切向正则项权重。

--stage3_lambda_tan_anchor 100
切向锚定正则项权重。

---

## 5. 方法论概要（High-Level Logic）

Stage3 将 ROI 内的皮肤顶点视为 **高斯混合模型（GMM）分量中心**，并通过最大化预测 GMM 与观测深度在像素反投影空间中的概率一致性，实现对局部几何的优化。

每个 ROI 顶点对应一个各向异性高斯椭球，椭球的中心、尺度与旋转共同描述局部表面的不确定性结构。

---

## 6. 目标函数（Objective Function）

Stage3 的总损失函数定义为：

$$
\mathcal L
==========

\mathcal L_{\text{GMM}}
+
\lambda_{\text{disp}} \mathcal L_{\text{disp}}
+
\lambda_{\text{shape}} \mathcal L_{\text{shape}}
+
\lambda_{\text{tan}} \mathcal L_{\text{tan}}
+
\lambda_{\text{tan-anchor}} \mathcal L_{\text{tan-anchor}}
$$

---

### 6.1 GMM 表面深度似然（Data Term）

$$
\mathcal L_{\text{GMM}}
=======================

-\frac{1}{|\mathcal P|}
\sum_{(u,v)\in\mathcal P}
\log
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\exp!\left(
-\tfrac12
,
|x(u,v)-\mu_k|^2_{\Sigma_k^{-1}}
\right)
\right)
$$

**符号说明：**

* ( \mathcal P )：ROI 深度图中采样的有效像素集合
* ( x(u,v) )：由像素 ((u,v)) 与观测深度反投影得到的 3D 点
* ( K )：GMM 分量数量（等于 ROI 顶点数）
* ( \mu_k )：第 (k) 个 GMM 分量中心（ROI 顶点）
* ( \Sigma_k )：对应的各向异性协方差矩阵

该项是唯一的数据一致性约束。

---

### 6.2 正则项

#### 顶点位移正则

$$
\mathcal L_{\text{disp}}
========================

\frac{1}{N}
\sum_{i=1}^{N}
|d_i|^2
$$

目的：抑制过大的局部位移。

---

#### 椭球形状正则（体积保持）

$$
\tilde\ell_i
============

## \ell_i

\frac{1}{3}
\sum_{j=1}^{3}
\ell_{ij}
$$

$$
\mathcal L_{\text{shape}}
=========================

\frac{1}{N}
\sum_{i=1}^{N}
|\tilde\ell_i|^2
$$

目的：允许各向异性，同时保持总体体积稳定。

---

#### 切向正则

$$
\mathcal L_{\text{tan}}
=======================

\frac{1}{|\mathcal E|}
\sum_{(i,j)\in\mathcal E}
\left|
(d_i-d_j)
---------

\big((d_i-d_j)^\top n_i\big)n_i
\right|^2
$$

目的：抑制切向高频漂移。

---

#### 切向锚定项

$$
\mathcal L_{\text{tan-anchor}}
==============================

\left|
\frac{1}{N}
\sum_{i=1}^{N}
\big(d_i-(d_i^\top n_i)n_i\big)
\right|^2
$$

目的：防止 ROI 整体沿切向发生一致性平移。

---

## 7. 关于误差指标的说明（Important Caveat）

Stage3 输出的：

```
[Stage3][ROIMeanDev] start=...cm end=...cm
```

表示 ROI 顶点相对于 GT 网格的平均欧氏距离。
该指标 **不参与优化，仅用于诊断分析**。

---

## 8. 当前状态与已知限制

* Stage3 使用 GT 姿态作为初始条件
* 深度观测与 GT 网格可能不同源
* 深度约束为 2.5D，存在多解性

---

## 9. 项目定位

> **RobotBridge（Stage3 · GMM）是一个以单视角深度为唯一观测、通过顶点级高斯混合模型实现人体局部微表面优化的研究型工程框架。**

