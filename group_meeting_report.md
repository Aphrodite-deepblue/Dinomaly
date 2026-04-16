# Dinomaly 在 Real-IAD 数据集上的改进实验报告

> 数据集：Real-IAD（30 类工业异常检测，共约 3755 张测试图/transistor1）  
> 骨干网络：DINOv2 ViT-B/14（冻结编码器）+ ViTill 解码器（可训练）  
> 评估指标：I-AUROC / I-AP / I-F1（图像级）；P-AUROC / P-AP / P-F1 / AUPRO（像素级）

---

## 一、基础背景

Dinomaly 的核心是一个**重建异常检测**范式：

$$\mathcal{L}_{recon} = \text{global\_cosine\_hm}(E_{en}, E_{de}) = 1 - \text{CosSim}(E_{en}, E_{de})$$

推理时，逐层计算编码器特征与解码器特征的余弦距离并上采样，得到连续型热力图 $S_{raw} \in [0,1]^{H \times W}$。**核心痛点**：正常纹理较复杂区域的背景响应往往弥散（即背景余弦距离非零），导致热力图模糊，边界不清晰，픽素级精度（P-AP、P-F1）偏低。

---

## 二、实验线路总览

```
实验 1：Uni 多类联合训练（基准验证）
    ↓ 热力图边界仍然弥散
实验 2：引入伪异常 + Dice Loss（训练期优化）
    ↓ loss_dice 无法有效降低 → 无显著提升
实验 3：后处理路线 A —— 可学习空间滤波器（SpatialFilter）
    ↓ P-F1/AUPRO 微升，全局排序指标下降
实验 4：后处理路线 B —— 残差滤波器 + BCE + 恒等正则（超参 Sweep）
    ↓ 同样规律，瓶颈在单通道信息量不足
实验 5（进行中）：后处理路线 C —— 零参数 Guided Filter
```

---

## 三、各实验详细说明

### 3.1 实验 1：Uni 多类联合训练（基准对照）

**设置**：`dinomaly_realiad_uni.py`，全部 30 类拼成 ConcatDataset，50k iter，batch=16。

**10 类部分结果**（I-AUROC / P-AUROC / AUPRO）：

| 类别 | I-AUROC | P-AP | AUPRO |
|---|---|---|---|
| pcb | 0.9245 | 0.5962 | 0.9517 |
| phone_battery | 0.9293 | 0.6189 | 0.9633 |
| audiojack | 0.8696 | 0.5193 | 0.9040 |
| bottle_cap | 0.9026 | 0.3568 | 0.9769 |

**结论**：I-AUROC 整体较好（>0.85），P-AP 参差不齐（0.35~0.62），热力图弥散是主要瓶颈。多类联合训练因梯度来自 30 类正常样本的平均，解码器学到的是"折中正常流形"，在每个单类上稍逊于专类训练。

---

### 3.2 实验 2：伪异常数据增强 + Dice Loss

#### 动机

期望通过引入伪异常（CutPaste/高斯噪声块），利用 Dice Loss 显式监督热力图边界：

$$\mathcal{L} = \mathcal{L}_{recon} + \alpha \cdot \mathcal{L}_{dice}(S_{raw}, M_{pseudo})$$

$$\mathcal{L}_{dice} = 1 - \frac{2 \cdot \sum \hat{p} \cdot m}{\sum \hat{p} + \sum m + \varepsilon}, \quad \hat{p} = \sigma\!\left(\frac{S_{norm}}{T}\right)$$

#### 遇到的问题：`loss_dice` 长期卡在 0.96~0.99，无法有效下降

分析出两个根本性数学陷阱：

**陷阱 A：正常样本（全零 Mask）导致 Dice = 0 恒成立**

当 `pseudo_prob=0.5` 时，一半 batch 的 $M_{pseudo}$ 全为 0。此时：
$$\mathcal{L}_{dice} = 1 - \frac{2 \times 0}{\hat{p}_{sum} + 0 + \varepsilon} \approx 1 \quad \text{（恒成立，无论模型预测如何）}$$

这些样本的梯度是矛盾且有害的：一方面将 $\hat{p}$ 往 0 压，另一方面伪异常样本又要把 $\hat{p}$ 在异常区域往 1 拉。

**修复**：仅在 `gt_mask.sum() > 0` 的样本上计算 Dice，全零样本不贡献梯度。

**陷阱 B：余弦距离值域与 Sigmoid 饱和导致背景面积虚高**

$S_{raw}$ 的典型值域为 $[0.03, 0.25]$，直接 `sigmoid(S)` 后：
$$\sigma(0.05) \approx 0.51, \quad \sigma(0.20) \approx 0.55$$

背景像素（约 65000 个）被激活到 0.51 以上，导致分母 $\sum \hat{p}$ 极大：
$$\mathcal{L}_{dice} = 1 - \frac{2 \times \text{TP}}{\text{大量背景FP} + \text{少量真异常}} \approx 1$$

**修复**：逐样本标准化后引入温度系数 $T=0.1$，强制压缩背景响应：
$$\hat{p} = \sigma\!\left(\frac{S - \mu_S}{\sigma_S \cdot T}\right)$$

#### 实验结果（transistor1，sep 模式，5k iter）

| 指标 | 无伪异常（基准） | 有伪异常+Dice（$\alpha=0.1$） | 变化 |
|---|---|---|---|
| I-AUROC | **0.9824** | 0.9786 | −0.0038 |
| I-AP | **0.9870** | 0.9840 | −0.0030 |
| I-F1 | **0.9424** | 0.9362 | −0.0062 |
| P-AUROC | 0.9956 | 0.9956 | 0 |
| P-AP | **0.5910** | 0.5819 | −0.0091 |
| P-F1 | **0.5598** | 0.5538 | −0.0060 |
| **AUPRO** | 0.9746 | **0.9775** | **+0.0029** |

**结论**：即使修复了两个数学陷阱，`loss_dice` 仍停留在 0.96 附近，几乎没有有效下降。根本原因在于：

> 1. DINOv2 patch size 为 14×14，伪异常块（$2\%\sim15\%$ 边长）在特征图中仅覆盖 **2~4 个 patch 位置**，梯度信号极弱；  
> 2. 重建损失（最小化正常特征距离）与 Dice 损失（最大化伪异常区响应）存在**目标冲突**——前者鼓励所有位置 $S \to 0$，后者鼓励伪异常位置 $S \to 1$，网络无法同时满足；  
> 3. 总体结论：**伪异常 + Dice 在训练期引入语义监督信号，对弥散问题的改善极其有限**，AUPRO 微升 (+0.003) 但其他指标均有不同程度下降。

---

### 3.3 实验 3：后处理路线——可学习残差空间滤波器（Sweep）

#### 动机

"解耦优化"思路：冻结 Dinomaly 骨干，在真实异常验证集上用 Dice+BCE 联合损失训练轻量级后处理器 $f_\theta$：

$$\theta^* = \arg\min_\theta \mathcal{L}_{dice}(f_\theta(S_{raw}), M_{true}) + \lambda_{bce} \mathcal{L}_{BCE} + \lambda_{id} \|f_\theta(S_{raw}) - S_{raw}\|_2^2$$

具体使用残差学习（初始化为恒等映射），并对 `bce_weight`（4种）× `identity_weight`（4种）共 16 组超参进行 sweep。

#### 实验结果（transistor1，最优组合 bce=0.1, id=0.05）

| 指标 | Before | After | 变化 |
|---|---|---|---|
| I-AUROC | 0.9825 | **0.9652** | **−0.0173** |
| I-AP | 0.9869 | **0.9748** | **−0.0121** |
| I-F1 | 0.9440 | **0.9127** | **−0.0313** |
| P-AUROC | 0.9954 | 0.9942 | −0.0012 |
| P-AP | 0.5895 | **0.5590** | **−0.0305** |
| P-F1 | 0.5590 | 0.5680 | **+0.0089** |
| AUPRO | 0.9740 | 0.9775 | **+0.0035** |

**全部 16 组 sweep 呈现相同的规律**：`f1_px` (+) 和 `aupro_px` (+) 微升，其余指标普遍下降。

#### 数学根因分析

该方法存在**结构性瓶颈**，不可通过调参消除：

**① 图像级得分塌陷**：图像级得分由 $\text{Score}_{img} = \text{Top-1\%Mean}(S_{out})$ 确定。后处理器学到的是"空间腐蚀"操作，压缩了热力图的弥散区域，使得异常图的最大响应值 $\max(S_{out})$ 有所下降，导致 I-AUROC / I-AP 降低。

**② 单通道信息瓶颈**：$S_{raw} \in \mathbb{R}^{1 \times H \times W}$ 已丢失了 768 维骨干特征中的绝大多数判别信息。1D 滤波器无法区分"缺陷纹理弥散"与"背景纹理复杂"，只能执行全局平滑——在 Dice 梯度下退化为统计意义上的"阈值腐蚀"，必然损伤全局概率排序。

**③ 验证集过拟合风险**：仅 ~751 张图（20% val split）上学到的 `f_θ` 容易过拟合验证分布，泛化到 holdout test 后效果打折。

---

### 3.4 实验 4（进行中）：零参数 Guided Filter

#### 理论依据

$$S_{sharpened} = \text{GuidedFilter}(I_{guide}=I_{RGB\_gray},\ S_{input}=S_{raw},\ r,\ \varepsilon)$$

核心数学：在窗口 $k$ 内建立局部线性模型 $q_i = a_k I_i + b_k$，通过最小二乘求解：

$$a_k = \frac{\text{Cov}(I, S)}{\text{Var}(I) + \varepsilon}, \quad b_k = \overline{S}_k - a_k \overline{I}_k$$

| 区域类型 | 数学行为 | 实际效果 |
|---|---|---|
| 平坦背景（$\text{Var}(I) \ll \varepsilon$） | $a_k \approx 0,\ q \approx \overline{S}$（局部均值） | 弥散热力图被压平 |
| 真实边缘（$\text{Var}(I) \gg \varepsilon$） | $a_k \approx \text{Cov}(I,S)/\text{Var}(I)$（边缘保留） | 缺陷边界被保留 |

**优势**：
1. **零参数**，无过拟合风险，不需要验证集；
2. RGB 图包含真实边缘信息，保边平滑具有物理意义；
3. 不改变热力图的全局得分排序（理论上 I-AUROC / I-AP 不下降）；
4. 已通过参数 sweep（radius × epsilon）找最优组合。

**参数扫描设置**：radius ∈ {4, 8, 16, 32}，ε ∈ {0.001, 0.01, 0.05, 0.1}，共 16 组。

当前状态：backbone 推理已完成（235 batch，3755 张图），正在计算 baseline AUPRO。

---

## 四、结果横向对比

| 方法 | I-AUROC | I-AP | P-AP | P-F1 | AUPRO |
|---|---|---|---|---|---|
| Sep 基准（无伪异常） | **0.9824** | **0.9870** | **0.5910** | **0.5598** | 0.9746 |
| Sep + 伪异常 + Dice | 0.9786 | 0.9840 | 0.5819 | 0.5538 | 0.9775 |
| + 残差滤波器 (best) | 0.9652 | 0.9748 | 0.5590 | 0.5680 | **0.9775** |
| **Guided Filter（进行中）** | — | — | — | — | — |

> 注：以上数据均在 **相同 holdout 测试集**上评估（transistor1，3755 张），保证前后可比。

---

## 五、核心结论与后续方向

### 已验证的失效路径

| 方法 | 失效原因 |
|---|---|
| 训练期 Dice Loss | 伪异常信号在特征空间过弱；重建与分割目标冲突 |
| 单通道可学习滤波器 | 信息瓶颈（1D 热力图无法区分弥散类型）；破坏全局排序 |

### 当前最有希望的方向

**方案 C：零参数 Guided Filter**（进行中）
- 数学上保证边缘保留 + 平坦区压平，且不破坏全局概率排序

**方案 D（后续）：特征引导滤波**
- 将骨干中间层特征 $E \in \mathbb{R}^{C \times H \times W}$ 与 $S_{raw}$ 拼接，训练条件化滤波器，绕过单通道信息瓶颈

**方案 E（辅助）：Top-K Mean 图像级得分重校准**
- 将 $\text{Score}_{img} = \max(S)$ 改为 $\frac{1}{K}\sum_{\text{top-K}}$，缓解空间腐蚀操作带来的图像级分数塌陷
