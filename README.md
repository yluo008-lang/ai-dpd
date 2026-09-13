# AI-DPD — 用神经网络做深度优化的数字预失真（工程实现）

一个**可训练、可量化、可部署**的 AI-DPD 工程：PyTorch 训练 → int16 定点量化 → C 参考推理 → Verilog 数据通路，关键模块**位真对拍**。
配套 `../dpd/`：通信 DPD 的 C + FPGA 全套（记忆多项式 / ILA / LS / LMS）。

> 技术报告见 [`REPORT.md`](REPORT.md) / [`REPORT.pdf`](REPORT.pdf)。

---

## 结果速览（本机实测）

| 方案 | ACLR | EVM |
|---|---|---|
| no-DPD | −46.4 dB | 3.80% |
| 多项式 LS (PolyMP) | −70.5 dB | 0.079% |
| RVTDNN (ILA / DLA) | −50…−57 dB | 0.2…0.45% |
| TCN-32×(1,2,4,8) | −54.8 dB | 0.218% |
| **Hybrid（MP 支路 + 神经残差）** | **−70.9 dB (float) / −70.43 dB (int16)** | **0.062 / 0.052%** |

- **int16 定点混合 DPD 与浮点只差 +0.04 dB**（−70.43 vs −70.46）→ 可部署。
- 失配 PA（I/Q 失衡）上：多项式 EVM 能力崩（6.33%→5.11%），**NN 压到 0.5%** —— NN 的价值在**多项式无法建模的失真**。

## 目录

```
ai-dpd/
├── REPORT.md / REPORT.pdf      # 技术报告
├── pa.py                       # PA 模型：MP / GmMP / IQMP(+I/Q失衡) / torch 可微版
├── dsp.py                      # OFDM 激励 + ACLR/EVM/NMSE
├── models.py                   # RVTDNN / CVNN / TCN / HybridDPD
├── train.py                    # ILA(train_nn) / DLA(train_dla) / 多项式 LS
├── quant_export.py             # MLP int16 量化 + C 导出（整数 requant）
├── quant_hybrid.py             # Hybrid int16 量化（含峰值余量标定）
├── quant_tcn.py / quant_tcn_export.py   # TCN 量化 + 导出
├── run*.py                     # 各实验入口（见下）
└── deploy/                     # C 推理 + Verilog RTL + testbench
    ├── nn_dpd_infer.c          # MLP int16 推理（自检）
    ├── hybrid_infer.c          # Hybrid（MP 基 + MLP）int16 推理
    ├── tcn_infer.c             # TCN 卷积 int16 推理
    ├── nn_dpd_layer.v          # 2 级流水 dense+ReLU+requant
    ├── nn_dpd_feat.v           # 延迟线 + 整数 sqrt 特征前端
    ├── nn_dpd_feat_par.v       # P 样本/时钟并行前端（共享历史）
    ├── nn_dpd_scale.v          # 特征→激活缩放
    ├── nn_dpd_par.v            # P-lane 并行 MLP 顶层
    └── tb_*.v                  # 各 testbench
```

## 快速开始

```bash
# 训练 + 量化 + 导出（需 torch；本机在 cxl-venv）
python run_hybrid.py            # Hybrid：ILA/DLA/Poly 对比
python run_hybrid_isolate.py    # 定点损失隔离（NN支路 vs 多项式支路）
python quant_hybrid.py          # Hybrid int16 量化（峰值余量标定）

# C 推理自检
cd deploy && make && ./nn_dpd_infer && ./hybrid_infer && ./tcn_infer

# Verilog 对拍（需 iverilog）
cd deploy && make rtl && vvp tb.vvp
```

## 训练要点（踩出来的）

1. **整段连续序列训练**：时延 tap 用 `roll`，若随机 batch 会让 tap 取到别的样本。
2. **残差零初始化**：`D(x)=x+NN(x)` → 单项 +≈6 dB。
3. **多项式支路 LS 初始化**：Hybrid 随机初始化 −46.9 → LS 初始化 −70.9 dB。
4. **lr≈1e-2** 才跳出局部最优。

## 量化规则（关键）

- int16 权重+激活，逐层 scale，整数 requant：`(acc·REQ + (BQ<<SHIFT) + rounding) >> SHIFT`。
- **峰值余量标定**：scale 必须覆盖**工作范围**，否则更大的峰值会削顶——而峰值正是 PA 压缩最狠、DPD 最需要发力的地方。`quant_hybrid.py` 里 `HEADROOM=2.0`。
- **校正保真度要测在校正量 `z−x` 上**；测整段输出会把保真度夸大。
- 双精度只用于**离线训练/标定**；数据通路是否定点取决于平台（FPGA→定点；GPU/SDR→浮点）。

## 硬件验证状态

| 模块 | 状态 |
|---|---|
| `nn_dpd_layer.v`（流水线 dense+requant） | 位真 PASS |
| `nn_dpd_feat.v`（延迟线 + 整数 sqrt） | 位真 PASS |
| `nn_dpd_feat_par.v`（P 样本/时钟） | P=2 位真 PASS |
| `nn_dpd_scale.v`（特征缩放） | 与参考 a0 逐值一致 |
| `nn_dpd_par.v`（P-lane 并行顶层） | 位真 PASS（feat→scale→L0，0/2048）|
| `nn_dpd_infer.c` / `hybrid_infer.c` / `tcn_infer.c` | PASS（≤8 LSB）|

> 状态：`nn_dpd_feat/feat_par/scale/L0`、各 C 推理均已位真对拍通过；`nn_dpd_par` 扩展到**完整 3 层 MLP + 残差**后整链对拍**尚未通过**（新增 L1/L2/残差通路后出现失配，正在定位）——未宣称完全通过。

## PA 漂移鲁棒性（train/test 的 I/Q 失衡不同）

| beta_train / beta_test | 方法 | ACLR | EVM |
|---|---|---|---|
| 0.05 / 0.05 (same) | Poly / NN | −53.9 / −56.1 | 5.13% / **0.22%** |
| 0.05 / 0.08 (drift) | Poly / NN | −53.9 / −54.7 | 8.10% / **3.05%** |
| 0.08 / 0.05 (drift) | Poly / NN | −49.1 / −53.7 | 5.29% / **2.98%** |
| 0.05 / 0.12 (drift) | Poly / NN | −53.9 / −54.7 | 12.08% / **7.07%** |

**结论**：PA 漂移时多项式的 EVM 能力几乎失效（跟随 no-DPD），**NN 的 EVM 仍显著更低**（漂移下 3% vs 8%、7% vs 12%），ACLR 也更稳 → **NN 对 PA 漂移更鲁棒**。脚本：`run_drift.py`。

## 关键结论

- 匹配 PA 上多项式是天花板；NN 的价值在**失配/漂移**场景。
- **Hybrid（多项式先验 + 神经残差 + 峰值余量定点）= 可在 int16 部署**，追平多项式。
- 见 [`REPORT.md`](REPORT.md) 的完整方法与数据。
