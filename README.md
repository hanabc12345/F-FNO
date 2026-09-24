# Error Propagation from Neural Near-Field Surrogates to Far-Field RCS

本仓库是论文 **"Error Propagation from Neural Near-Field Surrogates to Far-Field Radar Cross Section"**
（Shuai Han, Yaorui Guo — Qiyuan Laboratory）的配套代码。

研究对象：以 **3-D F-FNO（因子化傅里叶神经算子）** 作为散射近场代理，量化"**近场代理误差 → 近远场变换（NFFFT）→ 远场 RCS 误差**"的传播链。

核心内容：

- **NFFFT 独立校验**：在封闭盒面（closed-box）等效面上由真值场构造等效流，20 个入射角上相对全波解
  中位误差 **0.161 dB**、P90 **0.59 dB**、复相关 **0.998** ⇒ 解析算子本身无关几何/采样误差。
- **误差三路分解**：参考链路误差 `Eref`、可归因于代理的误差 `Esur`、端到端误差 `Eend`。
  三者均为**中位 |ΔRCS| (dB)** ⇒ **不可加减**，必须各自直接测量。
- **相干增益参考律**：`med|ΔRCS| ≈ 2.39·ε/√G`（G 中位 0.29 ⇒ 4.44 dB/ε），
  受控白噪声扫描实测斜率 **4.50 dB/ε**（R²=0.9989），两者相差 1.4%。
- **训练模型检查点单独报告**：全局体素级 H-block 相对 L2 误差与面元级扰动 ε 不同口径，
  故不作该定律的定量检验。

---

## 目录结构

```
F-FNO-paper/
├── 3d_fno_train/            # 论文核心：F-FNO 模型 / NFFFT / 实验 / 诊断
│   ├── fno_f16_3d.py            #   3D F-FNO 模型 + 入射角表 + 训练主流程（7→12 通道）
│   ├── fno_f16_3d_p3.py         #   生产配置（width 128，3.08 M 参数）
│   ├── fno_f16_3d_p4_nffft.py   #   Huygens 面 NFFFT（体素面提取 / 辐射积分 / RCS 归一化）
│   ├── exp_baselines.py         #   基线：U-Net / MLP / DeepONet
│   ├── exp_seeds.py             #   多 seed 训练 + 角度级 bootstrap（Table I 的 ± 与 95% CI）
│   ├── exp_nffft_anglesets.py   #   统一角度集下的 floor vs pred 分解（Eref/Esur/Eend）
│   ├── exp_errprop.py           #   相干增益定律：解析推导 + 白噪声扫描拟合
│   ├── _diag_nffft_boxpred.py   #   盒面等效原理路由（0.161 dB）vs 生产体素面路由（3.5 dB）
│   ├── _diag_errprop_exact.py   #   修正口径的 ε 扫描（Fig.3 数据源）
│   ├── _diag_averaging_bound.py #   平均化定律的适用边界（相关长度扫描）
│   ├── _diag_discretization_bound.py  # 求积/离散化误差界（F=1,2,4）
│   ├── fig_paper.py             #   Fig.1 workflow / Fig.2 近场切片
│   ├── fig_errprop_law.py       #   Fig.3 相干增益定律图
│   ├── incidence_table.npz      #   468 入射角表（θ/φ/e0/k̂/β，FEKO 约定）
│   └── results/                 #   论文配图 PNG + 指标 JSON + ONNX + 检查点 + RCS 方向图 npz
├── data/
│   └── f16_3d_rcs_truth_ff.h5   #   全波解真值远场 + RCS 子集（47 MB，供审稿复算）
├── feko_dataset/            # FEKO 批量求解与数据集构建/校验脚本
│   ├── gen_feko_batch.py        #   468 角批处理（MoM，3.0 GHz，~3.9e5 未知量）
│   ├── verify_dataset_v2.py     #   数据集完整性校验
│   └── probe*.lua               #   FEKO 求解器探针脚本
└── deploy_rcs_server/       # 生产链路（RCS 服务端 + NFFFT-lite + UE 客户端接口）
    ├── ue_rcs_service.py        #   TCP 服务：ONNX 推理 + 标定 + 方向图输出
    ├── nffft_lite.py            #   轻量 NFFFT（部署侧）
    ├── mock_ue_client.py        #   联调客户端
    ├── models/ data/            #   ONNX 模型与标定表
    └── 接口文档-飞机数据.md / 服务端启动文档.md
```

---

## 论文 ↔ 代码 对照

| 论文内容 | 对应实现 | 产出 |
|---|---|---|
| §II-A 数据与代理（7→12 通道，64×48×32，468 角，234/234 内插划分） | `3d_fno_train/fno_f16_3d.py`、`fno_f16_3d_p3.py` | `results/metrics_*.json` |
| §II-A 卷积基线（3-D U-Net，22.4 M 参数） | `3d_fno_train/exp_baselines.py` | `results/exp_baselines_full.json` |
| §II-B NFFFT 与封闭盒面独立校验（0.161 dB / 0.998） | `_diag_nffft_boxpred.py`、`fno_f16_3d_p4_nffft.py` | `results/_diag_nffft_boxpred.json` |
| §II-B 生产体素面路由残差 3.5 dB，八种电流构造 3.4–6.4 dB | `_diag_nffft_audit*.py`、`_diag_surface_fix.py`、`_diag_mesh_true_fix.py` | `results/_diag_nffft_audit*.json` |
| §II-C 直接误差分解（`Eref`/`Esur`/`Eend`，不可加减） | `exp_seeds.py`、`exp_nffft_anglesets.py` | `results/exp_seeds_*.json` |
| §III-A Table I（468 样本内 / 234 未见 / 正则化单次运行） | `exp_seeds.py`（三 seed 均值±标准差）、`exp_nffft_anglesets.py` | 上述 JSON |
| §III-B 相干增益参考律（4.44 vs 4.50 dB/ε，R²=0.9989） | `exp_errprop.py`、`_diag_errprop_exact.py` | `results/_diag_errprop_exact.json` |
| §III-C 有效性边界（面元级 ε 与全局相对 L2 不同口径） | `_diag_averaging_bound.py`、`_diag_discretization_bound.py` | `results/_diag_*_bound.json` |
| Fig. 1 工作流 / Fig. 2 近场切片 | `fig_paper.py` | `results/fig1_overview.png`、`fig2_arch.png`、`fig3_nearfield.png` |
| Fig. 3 误差传播结果 | `fig_errprop_law.py` | `results/fig_errprop.png` |

---

## 数据与产物清单

### 随仓库发布（审稿人可直接使用）

| 路径 | 内容 | 体积 |
|---|---|---|
| `data/f16_3d_rcs_truth_ff.h5` | **全波解真值远场 + RCS**：`rcs`/`rcs_ortho` (468,37,73)、`E_ff`/`E_ff_ortho` (468,37,73,2) 复数、`angles` (468,2)、`eps_field` (64,48,32)、`ff_theta`/`ff_phi`、`grid_x/y/z`。属性内注明来源与口径。 | 47 MB |
| `3d_fno_train/results/exp_seeds_ffno_full_rcs.npz` | F-FNO 三 seed 的远场方向图 `rcs_true` / `floor` / `pred_s0..s2`，各 (468,37,73) | 21 MB |
| `3d_fno_train/results/exp_seeds_ffno_interp_rcs.npz` | 同上，234 个未见入射角 | 10 MB |
| `3d_fno_train/results/exp_seeds_unet_full_rcs.npz` | U-Net 基线三 seed（468 角） | 21 MB |
| `3d_fno_train/results/exp_nffft_anglesets_rcs.npz` | floor / pred / truth 在 full468、interp234、legacy40 三套角度上的方向图（Eref/Esur/Eend 分解的原始数据） | 22 MB |
| `3d_fno_train/results/_diag_po_fullmap.npz`、`_diag_fno_fullmap.npz` | PO 链 / FNO 链的 468 角 × 2701 方向误差图 | 25 MB |
| `3d_fno_train/results/ckpt_full_p3.pt` | F-FNO 生产配置（width 128，3.08 M 参数，468 角全量训练）权重 | 23 MB |
| `3d_fno_train/results/seed_ffno_full_s{0,1,2}.pt` | F-FNO 三个独立 seed 的权重（Table I 的 ± 由此而来） | 70 MB |
| `3d_fno_train/results/fno_f16_3d_p3.onnx`、`resid_mlp.onnx` | 可推理模型（ONNX） | 24 MB |
| `3d_fno_train/results/*.png` | **全部论文配图与实验图**（29 张，含 Fig.1–3） | 3.6 MB |
| `3d_fno_train/results/*.json` | 全部指标 JSON（论文每个数字的原始记录） | 0.6 MB |

> `exp_seeds_*_rcs.npz` 配合 `exp_seeds_*.json` 即可**逐位复算 Table I**；
> `exp_nffft_anglesets_rcs.npz` 配合 `_diag_errprop_exact.json` 即可复算 **Fig. 3**。

### 未随仓库发布（体积超 GitHub 单文件上限，可本地重生成）

| 文件 | 体积 | 说明 |
|---|---|---|
| `f16_3d_rcs_dataset_v2.h5` | 4.47 GB | 全波解**近场体素场**：`E_scat`/`H_scat`/`E_scat_ortho`/`H_scat_ortho` 各 (468,64,48,32,3) complex64 = 1.10 GB × 4。**仅 Fig.2 的近场切片与重训需要**。 |
| `f16_3d_rcs_dataset.h5` | 94.5 MB | 早期 39 角版本（仅主极化、无 H 场），已被 v2 取代。 |
| `_p3_X.npy` / `_p3_Y.npy` | 1.29 / 2.21 GB | 由 v2.h5 派生的训练张量：输入 (468,7,64,48,32)、输出 (468,12,64,48,32)。**可一行重建**，非原始数据。 |
| `seed_unet_full_s{0,1,2}.pt` 等 | 23–86 MB/个 | U-Net 基线及其余消融检查点。 |

### 重生成原始数据集

以 FEKO 对 `f16` 模型在 3.0 GHz 求 468 个入射角的近场与远场解，按 `feko_dataset/gen_feko_batch.py`
的批处理流程落盘为 HDF5（MoM，约 3.9×10⁵ 未知量，单角约 8.4 min，468 角约 65 h）。
字段定义见 `3d_fno_train/fno_f16_3d.py` 的读取段。

由 v2.h5 重建训练张量（无缓存时自动组装，首次约 1–2 分钟）：

```python
# 组装逻辑在 fno_f16_3d_p3.load_p3()；带磁盘缓存的外层是 exp_baselines.load_p3_cached()
import exp_baselines as B
B.load_p3_cached(use_cache=True)   # 直接把 _p3_X/_p3_Y/_p3_angles/_p3_eps 写回 results/
```

等价做法：任一次运行 `python exp_baselines.py` 都会自动生成这四个缓存文件。

> **注意**：脚本中的数据集路径为绝对路径常量
> `H5 = r"...\3d_feko_data\f16_3d_rcs_dataset_v2.h5"`（见 `fno_f16_3d.py`、`f16_rcs_demo*.py`、
> `_diag_angle.py`、`export_ue_onnx_assets.py`），运行前请改为本机实际路径。


---

## 环境

```bash
pip install -r requirements.txt
```

参考环境：Python 3.11 + PyTorch（CUDA）。CPU 亦可跑通诊断类脚本。

## 复现命令

```bash
cd 3d_fno_train

# ① 主模型训练（生产配置：width 128，300 epoch）与内插划分
python fno_f16_3d.py --run full
python fno_f16_3d.py --run interp

# ② 基线对比（F-FNO / MLP / U-Net / DeepONet）
python exp_baselines.py --model all --split full

# ③ 多 seed + 角度级 bootstrap（Table I 的 ± 与 CI）
python exp_seeds.py --stage train --arch ffno --split full --seeds 0,1,2 --epochs 300 --width 128
python exp_seeds.py --stage train --arch unet --split full --seeds 0,1,2 --epochs 300 --width 64
python exp_seeds.py --stage eval  --arch ffno --split full --seeds 0,1,2

# ④ 统一角度集下的误差分解（floor vs pred）
python exp_nffft_anglesets.py

# ⑤ 相干增益定律：解析 + 白噪声扫描
python exp_errprop.py
python _diag_errprop_exact.py

# ⑥ 封闭盒面独立校验 / 有效性边界
python _diag_nffft_boxpred.py --ckpt ckpt_full_p3.pt --nangles 40
python _diag_averaging_bound.py
python _diag_discretization_bound.py

# ⑦ 论文配图
python fig_paper.py
python fig_errprop_law.py
```

部署侧自检（无需数据集，仅 ONNX + 标定表）：

```bash
cd deploy_rcs_server
python ue_rcs_service.py            # 启动 RCS 服务端
python mock_ue_client.py            # 模拟 UE 客户端联调
```

---

## 引用口径注意事项

1. **中位 |ΔRCS| 不可加减**：`Eref`、`Esur`、`Eend` 各自直接测量；`Esur` 不得由中位相减推得。
2. **`results/resid_mlp.onnx` 为全角度训练（无留出）**，其样本内误差是乐观值，对外应引用交叉验证值
   （见 `results/exp_resid_model*.json`）。
3. **相干增益定律是条件性解析参考**，仅适用于**独立、同向同性复高斯面元扰动**；
   训练模型的全局近场相对误差与之**不是同一口径**，不可直接对照。
4. 生产路由 3.5 dB 是**实测参考基线**，不是严格的几何下界或管线地板。

## 引用

```bibtex
@article{han_errorprop_awpl,
  author  = {Han, Shuai and Guo, Yaorui},
  title   = {Error Propagation from Neural Near-Field Surrogates to Far-Field Radar Cross Section},
  journal = {IEEE Antennas and Wireless Propagation Letters},
  note    = {Qiyuan Laboratory}
}
```
