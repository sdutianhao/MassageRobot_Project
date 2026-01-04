下面给你**一份完整、可直接覆盖当前 README.md 的版本**。
仅包含 **Stage3 · GMM（verts）**，**目标函数写到可直接进论文**；
**正则项只给公式 + 一句话目的，不做展开解释**；
内容与当前代码与命令行 **严格一致**。

---

# RobotBridge

## 基于单视角深度的 GMM 人体局部微表面优化（Stage3）

---

## 1. 项目目标

RobotBridge 是一个以**单视角深度图**为唯一观测的人体表面几何优化框架。
当前版本聚焦 **Stage3：ROI 内皮肤微表面（Micro-Skin）优化**，目标是在固定全局姿态的条件下，对局部皮肤几何进行高自由度、可微分的概率建模与优化。

核心特征：

* 仅使用 2.5D 深度观测
* ROI 局部优化
* 概率建模（GMM Surface Likelihood）
* 顶点级自由度 + 各向异性不确定性

---

## 2. 几何表示

### 2.1 椭球中心（GMM center）

* 采用 **`verts` 模式**
* ROI 内每一个皮肤顶点对应一个 GMM 分量
* 第 (k) 个分量中心：
  $$
  \mu_k \in \mathbb R^3
  $$
  直接由当前迭代下的 ROI 顶点位置给出

### 2.2 协方差参数化

每个 GMM 分量使用各向异性高斯椭球表示，其协方差写为：

$$
\Sigma_k
========

R_k
\operatorname{diag}(s_{k1}^2,s_{k2}^2,s_{k3}^2)
R_k^\top
$$

其中：

* ( R_k )：椭球旋转矩阵（可学习）
* ( s_{k\cdot} )：主轴尺度（以 log-scale 形式优化）
* 总体体积通过约束保持不变

---

## 3. 优化变量

在 Stage3 中优化的变量包括：

* ROI 顶点位移
  $$
  d_i \in \mathbb R^3
  $$
* 椭球 log-scale（各向异性）
* 椭球旋转（可选，显式开启）

全局姿态、相机参数保持固定。

---

## 4. 目标函数（完整形式）

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

## 5. GMM 表面深度似然（核心数据项）

### 5.1 定义

$$
\mathcal L_{\text{GMM}}
=======================

-\frac{1}{|\mathcal P|}
\sum_{(u,v)\in\mathcal P}
\log
\left(
\frac{1}{K}
\sum_{k=1}^{K}
\exp
!\left(
-\tfrac12
,
|x(u,v)-\mu_k|^2_{\Sigma_k^{-1}}
\right)
\right)
$$

### 5.2 符号说明

* ( \mathcal P )：从 ROI 深度图中采样的有效像素集合
* ( x(u,v) )：由像素 ((u,v)) 与观测深度反投影得到的 3D 点
* ( K )：ROI 内 GMM 分量数量（等于 ROI 顶点数）
* ( \mu_k )：第 (k) 个 GMM 分量中心
* ( \Sigma_k )：第 (k) 个分量的协方差矩阵
* ( |x-\mu|^2_{\Sigma^{-1}}=(x-\mu)^\top\Sigma^{-1}(x-\mu) )

该项是唯一的数据一致性约束。

---

## 6. 正则项

### 6.1 顶点位移正则

$$
\mathcal L_{\text{disp}}
========================

\frac{1}{N}
\sum_{i=1}^{N}
|d_i|^2
$$

目的：抑制局部顶点产生过大位移。

---

### 6.2 椭球形状正则（体积保持）

$$
\tilde\ell_i
============

## \ell_i

\frac{1}{3}
\sum_{j=1}^{3}\ell_{ij}
$$

$$
\mathcal L_{\text{shape}}
=========================

\frac{1}{N}
\sum_{i=1}^{N}
|\tilde\ell_i|^2
$$

目的：允许各向异性，同时保持椭球总体积稳定。

---

### 6.3 切向正则

$$
\Delta d_{ij}^{\perp}
=====================

## (d_i-d_j)

\big((d_i-d_j)^\top n_i\big)n_i
$$

$$
\mathcal L_{\text{tan}}
=======================

\frac{1}{|\mathcal E|}
\sum_{(i,j)\in\mathcal E}
|\Delta d_{ij}^{\perp}|^2
$$

目的：抑制切向方向的高频漂移。

---

### 6.4 切向锚定项

$$
\mathcal L_{\text{tan-anchor}}
==============================

\left|
\frac{1}{N}
\sum_{i=1}^{N}
\left(
d_i-(d_i^\top n_i)n_i
\right)
\right|^2
$$

目的：防止 ROI 整体沿切向发生一致性平移。

---

## 7. 深度可见性处理

* 在优化开始前进行一次 **静态可见性筛选**
* 条件包括：

  * 投影落入 ROI
  * 深度一致性满足阈值
* 优化过程中不再进行逐轮动态剔除

---

## 8. 运行方式（Stage3 · GMM）

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

---

## 9. 输出与指标

* ROI 网格与椭球的 PLY 可视化
* 深度预测结果
* 调试日志与中间状态

诊断指标：

```text
[Stage3][ROIMeanDev] start=...cm end=...cm
```

该指标不参与优化，仅用于分析。

---

## 10. 方法定位

Stage3（GMM）是一个**以深度似然为核心、通过顶点级 GMM 建模局部几何不确定性的人体微表面优化模块**，适用于研究深度约束下的高自由度局部形变问题。

---

如果你需要，我可以**下一步直接帮你压缩成论文 Method 一节版本（2–3 页）或补充梯度/复杂度分析版**。
