# AI-DPD：用神经网络做深度优化的数字预失真（工程实现）

一个**可训练、可量化、可部署**的 AI-DPD 工程：PyTorch 训练 → 定点量化 → C 参考推理 → Verilog 数据通路。
配套 `../dpd/`（上一轮交付的多项式 DPD 的 C + FPGA 全套）。

---

## 1. 网络最新进展摘要（2024–2025）

传统 DPD 用记忆多项式(GMP/MP)，假设 PA 行为静态；5G/6G 宽带 + 记忆效应 + I/Q 失衡下不够用 [^1](https://www.analog.com/en/resources/analog-dialogue/raqs/raq-issue-236.html)。近两年主流方向：

| 方向 | 代表工作 | 要点 |
|---|---|---|
| **稀疏神经网络 + FPGA 加速** | SparseDPD (Chang Gao 组, 2025) | 稀疏化 NN-DPD，FPGA 加速，实时可行 [^2](https://arxiv.org/abs/2506.16591) |
| **时序网络** | A-CNN-GRU 注意力增强 (2025) | 在 CNN-GRU 上加注意力，降低复杂度、提升线性化 [^3](https://www.sciencedirect.com/science/article/abs/pii/S0167926025000872) |
| **状态空间模型** | APN-Mamba (2025) | 相位归一化 + Mamba，长记忆宽带建模 [^4](https://www.sciencedirect.com/science/article/pii/S2215098626001345) |
| **Transformer** | Transformer 行为建模/预失真 (2024) | 长记忆、多载波场景 [^5](https://oulurepo.oulu.fi/bitstream/handle/10024/58775/nbnfioulu-202510136278.pdf) |
| **复数神经网络** | CVNN-DPD (MathWorks) | 复数线性层 + 复数激活，天然匹配 I/Q [^6](https://www.mathworks.com/help/comm/ug/complex-valued-neural-network-for-digital-predistortion-design-offline-training.html) |
| **实数时延网络** | RVTDNN / ARVTDNN (2025) | I/Q + 包络特征 + 时延，工程最常用、易量化 [^7](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13993/139935E/Digital-predistortion-linearization-for-power-amplifiers-based-on-a-crossed/10.1117/12.3094188.full) |
| **物理信息 / 稀疏脉冲** | PINN-GMP / SNN-DPD | 把多项式结构嵌入 NN，兼顾精度与可解释 [^8](https://www.diva-portal.org/smash/get/diva2:2090783/FULLTEXT01.pdf) |
| **工程化 / 基准** | NI ML-DPD 白皮书 | 用真实 PA 数据训练 + 对标的完整流程 [^9](https://www.ni.com/en/solutions/electronics/5g-6g-wireless-research-prototyping/prototyping-and-benchmarking-machine-learning-based-digital-pred.html) |
| **混合自适应 DPD** | FPGA 轻量自适应混合 DPD | 多项式 + NN 混合，低时延 [^10](https://ieeexplore.ieee.org/iel8/8782661/8901158/11660822.pdf) |

> 共识：NN-DPD 的价值在**复杂/宽带 PA、多模、PA 漂移**场景；对模型完全匹配的简单 PA，多项式仍是天花板。工程落地的关键不是"更大网络"，而是**残差学习 + 量化友好 + 低时延硬件映射**。

## 2. 本工程实现

```
ai-dpd/
├── dsp.py            # OFDM 激励 + ACLR/EVM/NMSE
├── pa.py             # MP / 广义 MP（GMP+交叉记忆）PA 模型
├── models.py         # RVTDNN / CVNN / TCN 三种神经 DPD
├── train.py          # 间接学习(ILA)训练 + 多项式 LS 基线
├── quant_export.py   # int16 定点量化 + 导出 weights.h / .mem / 参考向量
├── run.py            # 一键：训练→对比→量化→导出
└── deploy/
    ├── nn_dpd_infer.c        # int16 定点 C 推理（含自检）
    ├── nn_dpd_layer.v        # dense+ReLU+requant 层（int16 MAC 阵列）
    ├── nn_dpd_top.v          # 3 层 MLP + 残差
    ├── tb_nn_dpd_layer.v     # 对拍 Python 定点参考
    └── weights.h/*.mem       # 由 quant_export.py 生成
```

## 3. 快速开始

```bash
# 训练 + 量化 + 导出（需 torch，用你的 venv）
python run.py

# C 定点推理自检
cd deploy && gcc -O2 -I. -o nn_dpd_infer nn_dpd_infer.c -lm && ./nn_dpd_infer

# FPGA 对拍（iverilog）
cd deploy && iverilog -g2005 -o tb.vvp tb_nn_dpd_layer.v nn_dpd_layer.v && vvp tb.vvp
```

## 4. 实测结果（本机，训练/测试不同 seed，drive=0.25，4× 过采样 16-QAM OFDM）

**场景 1：记忆多项式 PA（多项式是"完美匹配"模型）**

| 方案 | ACLR_low | ACLR_up | EVM | NMSE |
|---|---|---|---|---|
| 无 DPD | −44.8 dB | −46.5 dB | 3.78% | −28.3 dB |
| PolyMP (LS) | −68.7 dB | −70.6 dB | 0.081% | −60.6 dB |
| **RVTDNN (relu+残差)** | **−53.7 dB** | **−54.1 dB** | **0.238%** | **−47.8 dB** |
| CVNN | −52.2 dB | −53.7 dB | 0.364% | −46.0 dB |
| TCN | −44.3 dB | −44.9 dB | 1.005% | −37.4 dB |

**场景 2：广义记忆多项式 PA（含交叉记忆，更难求逆）**

| 方案 | ACLR_low | ACLR_up | EVM | NMSE |
|---|---|---|---|---|
| 无 DPD | −51.3 dB | −49.3 dB | 1.07% | −38.7 dB |
| PolyMP (LS) | −66.1 dB | −71.3 dB | 0.051% | −62.4 dB |
| **RVTDNN** | **−53.5 dB** | **−54.2 dB** | **0.311%** | **−47.0 dB** |
| CVNN | −51.0 dB | −50.6 dB | 0.538% | −42.9 dB |
| TCN | −46.7 dB | −46.5 dB | 0.904% | −38.7 dB |

**量化/部署交叉验证**：int16 权重+激活 → SQNR **99.98 dB**；C 定点推理与参考最大误差 **5.1e-6**；Verilog 层与 Python 参考 **逐比特一致（0 mismatch）**。

> 结论（工程实话）：当 PA **恰好就是记忆多项式**时，多项式 DPD 是天花板，NN 追不上是对的；NN 的价值在非多项式/宽带/PA 漂移场景，以及**残差学习 + 量化友好**带来的部署便利。本工程把这条链路（训练→量化→C→RTL）全部打通并做了位真对拍。

## 5. 定点与硬件映射

- 权重/激活 **int16**，逐层标定 scale；累加 int32/64；`y = (acc·REQ)>>15`（REQ=Q15）。
- 3 层 MLP 全并行，**1 样本/时钟、3 拍延迟**；可平铺成 N 路并行降低时钟。
- 残差 `z = y_corr·scale + x(n)` 在硬件里只需一次加法。

## 6. 工程踩坑（值得记住）

1. **不要让延迟特征在乱序 mini-batch 上生成**：用 `roll` 造时延 tap 后，若按随机索引取 batch，tap 会取到"别的随机样本"，网络根本学不到记忆。必须**整段连续序列训练**。
2. **残差学习是 NN-DPD 的收敛关键**：`D(x)=x+NN(x)` 比直接学整映射收敛快、精度高。
3. **学习率不能太保守**：lr≈1e-2 才能跳出 ~−40 dB 的局部最优；lr=3e-4 会卡住。
4. **ILA 有域偏移**：后失真器在 `y/G` 上训练、却在 `x` 上使用；PA 深饱和时多项式外推会爆炸（详见 `../dpd/README.md`）。
