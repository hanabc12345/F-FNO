# -*- coding: utf-8 -*-
"""
fno_f16_3d.py — 3D F-FNO（因子化傅里叶神经算子）散射近场代理模型 P1 基线训练
============================================================================
任务：给定入射波（方向 θ/φ + 极化 e0）+ 目标体素场 eps_field → 散射近场 E_scat(r)
      （64×48×32 网格，主极化 θ-pol，输出 Re/Im × 3 分量 = 6 实通道）

策略（对齐 2D 教训 FNO-2D-RCS训练方法记录.md）：
  - 电大尺寸散射场对角度混沌 → 角度外推不可行，查表式兜底（全量训练+最近角查询）
  - 但 3D 姿态空间是 2D 球面，10° 网格 468 组已成本上限 → 先量化内插/外推泛化能力
  - 架构用 F-FNO（因子化谱权重 = 三轴和，避免 modes³ 参数爆炸）
  - 复数权重 + Adam wd=0（复参数 weight_decay 梯度错误）

输入通道（7）：
  0:  eps 二值掩膜（0=空气, 1=金属）
  1..3: E_inc Re（解析入射场 E0·exp(-jβ k̂·r) 实部）
  4..6: E_inc Im
输出通道（6）：E_scat Re/Im × 3 分量

三个实验（对应 3D-F-FNO训练方案.md 二.2）：
  interp : 训练去掉"奇数 φ"留出集 → 评估 10° 方位网格内插能力（234/234）
  extrap : 训练去掉 θ=30/150 边缘层 → 评估俯仰外推能力（396/72）
  full   : 全量 468 训练 → "查表兜底"误差（训练集内插值误差）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" fno_f16_3d.py --run all
  & "F:/miniconda3/envs/isaac311/python.exe" fno_f16_3d.py --run interp --epochs 150
产出（results/ 下）：
  metrics_<run>.json / ckpt_<run>.pt / fno_f16_3d_result.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import sys
import math
import time
import json
import re
import argparse

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"
RUN_DIR = r"f:\MyWorkSpace\UAVGame\3d_feko_run"
RESULT_DIR = os.path.join(BASE, "results")
INC_CACHE = os.path.join(BASE, "incidence_table.npz")

SEED = 0
CLIP = 15.0            # E_scat |E| 峰值钳制阈值（v2 数据 |E|max≈10.2）
BETA0 = 2 * np.pi / (2.99792458e8 / 3e9)   # 62.875 rad/m


# ============================================================
# 一、入射场表：从 .out 头解析 E0/k̂/β0（与 FEKO 约定严格一致）
# ============================================================

def parse_incidence_head(out_path):
    """读 .out 前 32KB，解析 occurrence=0（主极化 θ-pol）平面波头。
    返回 (theta, phi, e0[3]complex, khat[3], beta0)。"""
    with open(out_path, "rb") as f:
        head = f.read(32768).decode("utf-8", "ignore")
    m = re.search(r"Direction of incidence:\s+THETA =\s+([\d.]+)\s+PHI =\s+([\d.]+)", head)
    if not m:
        raise ValueError(f"{out_path}: 未找到 Direction of incidence")
    theta, phi = float(m.group(1)), float(m.group(2))
    e0 = np.zeros(3, dtype=np.complex128)
    for c in "XYZ":
        m2 = re.search(rf"\|E0{c}\| =\s*([\d.Ee+-]+)\s+ARG\(E0{c}\) =\s*([\d.Ee+-]+)", head)
        if not m2:
            raise ValueError(f"{out_path}: 未找到 E0{c}")
        e0["XYZ".index(c)] = float(m2.group(1)) * np.exp(1j * np.deg2rad(float(m2.group(2))))
    khat = np.zeros(3)
    for c in "XYZ":
        m3 = re.search(rf"BETA0{c}\s*=\s*([-\d.Ee+-]+)", head)
        if not m3:
            raise ValueError(f"{out_path}: 未找到 BETA0{c}")
        khat["XYZ".index(c)] = float(m3.group(1))
    m4 = re.search(r"Wave number:\s+BETA0\s+=\s*\(\s*([\d.Ee+-]+)", head)
    if not m4:
        raise ValueError(f"{out_path}: 未找到 Wave number")
    beta0 = float(m4.group(1))
    return theta, phi, e0, khat, beta0


def build_incidence_table(force=False):
    """468 角度 → (e0, k̂, β0) 表。解析结果缓存到 npz。"""
    if os.path.exists(INC_CACHE) and not force:
        d = np.load(INC_CACHE, allow_pickle=True)
        print(f"  入射场表(缓存): {INC_CACHE}")
        return d["angles"], d["e0"], d["khat"], d["beta0"]
    thetas = list(range(30, 151, 10))
    phis = list(range(0, 360, 10))
    angles, e0s, khats, betas = [], [], [], []
    t0 = time.time()
    for i, (t, p) in enumerate((t, p) for t in thetas for p in phis):
        out = os.path.join(RUN_DIR, f"case_{i:03d}.out")
        if not os.path.exists(out):
            raise FileNotFoundError(f"缺 {out}，无法解析入射场")
        _, _, e0, khat, beta = parse_incidence_head(out)
        angles.append([t, p]); e0s.append(e0); khats.append(khat); betas.append(beta)
        if (i + 1) % 100 == 0:
            print(f"  解析 {i+1}/468, {(time.time()-t0):.0f}s", flush=True)
    angles = np.asarray(angles, dtype=np.float32)
    e0 = np.asarray(e0s, dtype=np.complex64)
    khat = np.asarray(khats, dtype=np.float32)
    beta = np.asarray(betas, dtype=np.float32)
    np.savez(INC_CACHE, angles=angles, e0=e0, khat=khat, beta0=beta)
    print(f"  入射场表解析完成: 468 组, 缓存到 {INC_CACHE}")
    return angles, e0, khat, beta


# ============================================================
# 二、数据集构建
# ============================================================

def load_data():
    """载入 h5 并组装 X(468,7,64,48,32)/Y(468,6,64,48,32)，返回与角度索引。"""
    angles, e0, khat, beta = build_incidence_table()
    with h5py.File(H5, "r") as f:
        E_scat = f["E_scat"][:]            # (468,64,48,32,3) complex64
        eps = f["eps_field"][:]            # (64,48,32)
        h_ang = f["angles"][:]
        gx = f["grid_x"][:]; gy = f["grid_y"][:]; gz = f["grid_z"][:]
    assert np.allclose(angles, h_ang), "h5 angles 与 .out 解析表不一致"
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")     # (64,48,32)
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)

    n = len(angles)
    x_all = np.zeros((n, 7, *X.shape), dtype=np.float32)
    y_all = np.zeros((n, 6, *X.shape), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
        e_inc = e0[i][None, None, None, :] * phase[..., None]      # (64,48,32,3)
        x_all[i, 0] = (eps > 1.5).astype(np.float32)              # 二值掩膜
        x_all[i, 1:4] = e_inc.real.transpose(3, 0, 1, 2)
        x_all[i, 4:7] = e_inc.imag.transpose(3, 0, 1, 2)
        Es = E_scat[i]
        y_all[i, :3] = Es.real.transpose(3, 0, 1, 2)
        y_all[i, 3:] = Es.imag.transpose(3, 0, 1, 2)
        if (i + 1) % 100 == 0:
            print(f"  预处理 {i+1}/{n}, {(time.time()-t0):.0f}s", flush=True)
    print(f"  数据集组装完成: X{x_all.shape} Y{y_all.shape}")

    # 数据划分索引
    def idx_where(cond):
        return np.where(cond)[0]
    ang = angles
    interp_test = idx_where((ang[:, 1] % 20 == 10))                 # φ=10,30,...,350
    extrap_test = idx_where((ang[:, 0] == 30) | (ang[:, 0] == 150)) # θ 边缘两层
    idx = {
        "all": np.arange(n),
        "interp_test": interp_test,
        "interp_train": np.setdiff1d(np.arange(n), interp_test),
        "extrap_test": extrap_test,
        "extrap_train": np.setdiff1d(np.arange(n), extrap_test),
    }
    return x_all, y_all, idx, angles, (eps > 1.5).astype(np.float32)


def standardize(data, idx_tr):
    """按训练集索引计算每通道 mean/std，返回 (标准化数据, mean, std)"""
    mean = data[idx_tr].mean(axis=(0, 2, 3, 4), keepdims=True)
    std = data[idx_tr].std(axis=(0, 2, 3, 4), keepdims=True) + 1e-8
    return (data - mean) / std, mean, std


def clip_mag(data, clip=CLIP):
    """按样本钳制 |E|（峰值 ≤ clip），data: (n,6,...) 前三通道为 Re"""
    mag = np.sqrt(np.sum(data[:, :3] ** 2 + data[:, 3:] ** 2, axis=1, keepdims=True))
    s = np.minimum(1.0, clip / np.maximum(mag.max(axis=(2, 3, 4), keepdims=True), 1e-12))
    return data * s


# ============================================================
# 三、F-FNO 因子化网络
# ============================================================

class FactorizedSpectralConv3d(nn.Module):
    """3D 因子化谱卷积：权重 = Wx+Wy+Wz（三轴独立，参数 in*out*(mx+my+mz)）。
    rfftn 后保留正频率 modes 盒 [0:mx,0:my,0:mz]，零填充其余 → irfftn。
    初始化沿用 2D 配方 scale=1/(in*out) + 加性块（谱路径偏弱但 Adam 会逐步放大）。"""
    def __init__(self, in_ch, out_ch, modes, grid=(64, 48, 32), gain=None):
        super().__init__()
        mx, my, mz = modes
        self.modes = modes
        s = gain if gain is not None else 1.0 / (in_ch * out_ch)
        self.wx = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, 1, 1, dtype=torch.cfloat))
        self.wy = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, my, 1, dtype=torch.cfloat))
        self.wz = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, 1, mz, dtype=torch.cfloat))

    def forward(self, x):
        mx, my, mz = self.modes
        b, c = x.shape[:2]
        nx, ny, nz = x.shape[2:]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))                # (b,c,nx,ny,nz//2+1)
        x_hat = x_ft[:, :, :mx, :my, :mz]
        w = (self.wx.expand(c, self.out_ch, mx, my, mz)
             + self.wy.expand(c, self.out_ch, mx, my, mz)
             + self.wz.expand(c, self.out_ch, mx, my, mz))         # (c,out,mx,my,mz)
        out_hat = torch.einsum("bcijk,coijk->boijk", x_hat, w)
        out_ft = torch.zeros(b, self.out_ch, nx, ny, nz // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :mx, :my, :mz] = out_hat
        return torch.fft.irfftn(out_ft, s=(nx, ny, nz), dim=(-3, -2, -1))

    @property
    def out_ch(self):
        return self.wx.shape[1]


class FFNO3D(nn.Module):
    """3D F-FNO：lift → n×[谱卷积 + 1×1卷积(加性) + GELU] → project
    与 2D (FNO2D_RCS) 结构一致：x = gelu(spectral(x) + conv(x))，1×1 卷积路径防止谱路径小权重导致梯度消失。"""
    def __init__(self, modes=(16, 12, 8), width=32, in_ch=7, out_ch=6, n_layers=4,
                 grid=(64, 48, 32), gain=None):
        super().__init__()
        self.lift = nn.Conv3d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [FactorizedSpectralConv3d(width, width, modes, grid=grid, gain=gain)
             for _ in range(n_layers)])
        self.convs = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(n_layers)])
        self.project = nn.Conv3d(width, out_ch, 1)

    def forward(self, x):
        x = self.lift(x)
        for s, c in zip(self.spectral, self.convs):
            x = F.gelu(s(x) + c(x))
        return self.project(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# 四、评估
# ============================================================

def evaluate(model, X, Y, ymean, ystd, device, batch=8):
    """返回 (全局相对L2, 逐样本相对L2中位数, |E|结构相关中位数, 逐样本最大相对L2)
    X/Y 为标准化输入/目标；预测与目标均反标准化回原始量纲后比较。"""
    model.eval()
    sd = sn = 0.0
    rels, corrs = [], []
    ystd_t = torch.tensor(ystd, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ymean, device=device, dtype=torch.float32)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(device)
            yb = (Y[i:i + batch] * ystd + ymean)              # 反标准化目标
            pb = (model(xb) * ystd_t + ymean_t).cpu().numpy()  # 反标准化预测
            sd += float(np.sum((pb - yb) ** 2))
            sn += float(np.sum(yb ** 2))
            for j in range(pb.shape[0]):
                rels.append(float(np.linalg.norm(pb[j] - yb[j]) / (np.linalg.norm(yb[j]) + 1e-12)))
                pm = np.sqrt(np.sum(pb[j, :3] ** 2 + pb[j, 3:] ** 2, axis=0))
                tm = np.sqrt(np.sum(yb[j, :3] ** 2 + yb[j, 3:] ** 2, axis=0))
                corrs.append(float(np.corrcoef(pm.ravel(), tm.ravel())[0, 1]))
    rel_global = float((sd / sn) ** 0.5)
    return rel_global, float(np.median(rels)), float(np.median(corrs)), float(np.max(rels))


# ============================================================
# 五、训练
# ============================================================

def rel_mse_loss(pred, target, eps=1e-8):
    """逐样本归一化 MSE（相对 L2 的平方）：每个样本等权，避免大场样本/背景主导损失。
    与 2D 的全局 MSE 不同：3D 场能量跨度大，全局 MSE 被近零背景主导，
    而热区（高 |E|）能量占比小却主导相对误差指标。
    注意：维度自适应——全网格 5D 与采样点 3D 均可用（按批维外所有维求和）。"""
    dims = tuple(range(1, pred.dim()))
    num = (pred - target).pow(2).sum(dim=dims)
    den = target.pow(2).sum(dim=dims) + eps
    return (num / den).mean()


# ============================================================
# 五之二、P2 物理损失（Helmholtz 残差 + PEC 边界）
# ============================================================

def to_complex(re_im):
    """(B,6,...) → (B,3,...) cfloat：前三通道 Re，后三通道 Im"""
    return torch.complex(re_im[:, 0:3], re_im[:, 3:6])


def _ddx(f, h):
    out = torch.zeros_like(f)
    out[:, :, 1:-1, :, :] = (f[:, :, 2:, :, :] - f[:, :, :-2, :, :]) / (2.0 * h)
    return out


def _ddy(f, h):
    out = torch.zeros_like(f)
    out[:, :, :, 1:-1, :] = (f[:, :, :, 2:, :] - f[:, :, :, :-2, :]) / (2.0 * h)
    return out


def _ddz(f, h):
    out = torch.zeros_like(f)
    out[:, :, :, :, 1:-1] = (f[:, :, :, :, 2:] - f[:, :, :, :, :-2]) / (2.0 * h)
    return out


def curl3d(E, h):
    """∇×E，E:(B,3,nx,ny,nz) cfloat，中心差分（边界层置 0）"""
    Ex, Ey, Ez = E[:, 0:1], E[:, 1:2], E[:, 2:3]
    cx = _ddy(Ez, h) - _ddz(Ey, h)
    cy = _ddz(Ex, h) - _ddx(Ez, h)
    cz = _ddx(Ey, h) - _ddy(Ex, h)
    return torch.cat([cx, cy, cz], dim=1)


def phys_losses(pred_raw, x_raw, mask_metal, h=0.03125, beta0=BETA0):
    """P2 物理损失（相对形式，无量纲，0=完全满足物理）：
      L_helm = ||∇×∇×E_s − β0²E_s||²(空气) / ||β0²E_s||²(空气)  自由空间 Helmholtz 残差
      L_pec  = ||E_inc + E_s||²(金属) / ||E_inc||²(金属)         PEC 体素内总场 ≈ 0
    pred_raw/x_raw 均为反标准化后的原始物理量纲。"""
    Es = to_complex(pred_raw)                    # 预测散射场
    Einc = to_complex(x_raw[:, 1:7])             # 入射场
    # 内层空气掩膜：两次 curl 后有效区域缩 2 层（clone 避免修改原张量）
    air = (~mask_metal.bool()).clone()[None, None]
    air[..., :2, :, :] = False; air[..., -2:, :, :] = False
    air[..., :, :2, :] = False; air[..., :, -2:, :] = False
    air[..., :, :, :2] = False; air[..., :, :, -2:] = False
    R = curl3d(curl3d(Es, h), h) - beta0 ** 2 * Es
    denom = (beta0 ** 2 * Es).abs().pow(2) * air
    L_helm = (R.abs().pow(2) * air).sum() / (denom.sum() + 1e-12)
    Etot = Es + Einc
    m = mask_metal[None, None]
    L_pec = (Etot.abs().pow(2) * m).sum() / ((Einc.abs().pow(2) * m).sum() + 1e-12)
    return L_helm, L_pec


def train_run(name, x_all, y_all, idx_tr, idx_te, epochs, batch, width, modes, device,
              gain=None, lr=1e-3, phys=None, mask_metal=None, h=0.03125):
    """单个实验：标准化(训练集统计) → 训练 → 评估 train/test → 保存指标+checkpoint
    phys: (λ_helm, λ_pec) 或 None（关闭 P2 物理损失）"""
    y_all_c = clip_mag(y_all)                      # CLIP 峰值
    x_tr, x_inc_mean, x_inc_std = standardize(x_all, idx_tr)
    y_tr, y_mean, y_std = standardize(y_all_c, idx_tr)

    xtr = torch.from_numpy(x_tr[idx_tr]).to(device)
    ytr_raw = torch.from_numpy(y_all_c[idx_tr]).to(device)      # 原始物理量纲目标（CLIP 后）
    ystd_t = torch.tensor(y_std, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    xstd_t = torch.tensor(x_inc_std, device=device, dtype=torch.float32)
    xmean_t = torch.tensor(x_inc_mean, device=device, dtype=torch.float32)
    mask_t = mask_metal.to(device) if mask_metal is not None else None
    print(f"\n=== 实验 {name}: train={len(idx_tr)} test={len(idx_te)} lr={lr}"
          + (f" phys_λ=({phys[0]},{phys[1]})" if phys else " 无物理损失") + " ===")

    torch.manual_seed(SEED); np.random.seed(SEED)
    model = FFNO3D(modes=modes, width=width, in_ch=x_all.shape[1], out_ch=y_all.shape[1],
                   gain=gain).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)      # wd=0（复数权重必需）
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = len(xtr)
    print(f"  参数量: {model.count_params():,}, batch={batch}, epochs={epochs}")
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        ep_loss = 0.0
        ep_lh = ep_lp = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            pred_raw = model(xtr[b]) * ystd_t + ymean_t     # 反标准化回物理量纲
            loss = rel_mse_loss(pred_raw, ytr_raw[b])        # 原始空间逐样本相对损失
            if phys:
                x_raw = xtr[b] * xstd_t + xmean_t            # 反标准化输入（原始量纲）
                lh, lp = phys_losses(pred_raw, x_raw, mask_t, h=h)
                ep_lh += lh.item(); ep_lp += lp.item()
                loss = loss + phys[0] * lh + phys[1] * lp
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            rel_tr = evaluate(model, x_tr[idx_tr], y_tr[idx_tr], y_mean, y_std, device)
            rel_te = evaluate(model, x_tr[idx_te], y_tr[idx_te], y_mean, y_std, device) if len(idx_te) else None
            msg = (f"  ep{ep+1:4d}/{epochs} loss={ep_loss:.4f} "
                   f"train_rel={rel_tr[0]*100:.2f}% med={rel_tr[1]*100:.2f}% corr={rel_tr[2]:.3f}")
            if phys:
                nb = (n + batch - 1) // batch
                msg += f" [helm={ep_lh/nb:.3f} pec={ep_lp/nb:.3f}]"
            if rel_te:
                msg += f" | test_rel={rel_te[0]*100:.2f}% med={rel_te[1]*100:.2f}% corr={rel_te[2]:.3f}"
            print(msg + f"  ({time.time()-t0:.0f}s)", flush=True)

    # 最终评估（还原物理量纲）
    def ev(set_idx):
        Xn = (x_all[set_idx] - x_inc_mean) / x_inc_std
        Yn = (clip_mag(y_all[set_idx]) - y_mean) / y_std
        return evaluate(model, Xn, Yn, y_mean, y_std, device)

    res = {}
    for tag, si in (("train", idx_tr), ("test", idx_te)):
        if len(si) == 0:
            continue
        rel_g, rel_m, corr, rel_max = ev(si)
        res[tag] = {"rel_global": rel_g, "rel_median": rel_m, "corr_mag": corr, "rel_max": rel_max}
        print(f"  [{name}] {tag:5s}: rel_g={rel_g*100:.2f}% rel_m={rel_m*100:.2f}% "
              f"|E|corr={corr:.3f} max={rel_max*100:.1f}%")

    os.makedirs(RESULT_DIR, exist_ok=True)
    suffix = "_phys" if phys else ""
    ckpt = {"model_state": model.state_dict(),
            "config": {"modes": list(modes), "width": width, "phys": bool(phys)},
            "stats": {"y_mean": y_mean, "y_std": y_std, "x_inc_mean": x_inc_mean, "x_inc_std": x_inc_std,
                      "clip": CLIP}, "metrics": res}
    torch.save(ckpt, os.path.join(RESULT_DIR, f"ckpt_{name}{suffix}.pt"))
    with open(os.path.join(RESULT_DIR, f"metrics_{name}{suffix}.json"), "w") as f:
        json.dump({"name": name, "n_train": len(idx_tr), "n_test": len(idx_te),
                   "modes": list(modes), "width": width, "epochs": epochs, "phys": bool(phys),
                   "metrics": res}, f, indent=2)
    return res


# ============================================================
# 六、可视化（full 模型预测 vs 真实，z 中截面 |E|）
# ============================================================

def visualize(model, x_all, y_all, angles, sel, stats, device, out_path):
    model.eval()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x_mean, x_std, y_mean, y_std = stats
    x_mean_t = torch.tensor(x_mean, device=device, dtype=torch.float32)
    x_std_t = torch.tensor(x_std, device=device, dtype=torch.float32)
    y_mean_t = torch.tensor(y_mean, device=device, dtype=torch.float32)
    y_std_t = torch.tensor(y_std, device=device, dtype=torch.float32)
    fig, axes = plt.subplots(len(sel), 3, figsize=(15, 4 * len(sel)))
    with torch.no_grad():
        for row, i in enumerate(sel):
            Xn = (torch.from_numpy(x_all[i:i + 1]).to(device) - x_mean_t) / x_std_t
            yb = y_all[i]
            pb = (model(Xn) * y_std_t + y_mean_t).cpu().numpy()[0]
            z = yb.shape[1] // 2
            mags = [np.sqrt(np.sum(a[:3] ** 2 + a[3:] ** 2, axis=0))[z]
                    for a in (yb, pb, pb - yb)]
            for col, (mag, title) in enumerate(zip(mags, ("True |E|", "Pred |E|", "Error |E|"))):
                im = axes[row, col].imshow(mag.T, origin="lower", cmap="jet")
                axes[row, col].set_title(f"{title}  case{i} (θ={int(angles[i,0])},φ={int(angles[i,1])})")
                fig.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle("3D F-FNO P1 baseline (z 中截面 |E|)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  可视化已保存: {out_path}")


# ============================================================
# 主流程
# ============================================================

def main():
    global SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="all", choices=["all", "interp", "extrap", "full"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--modes", default="20,16,10", help="谱模式数（x,y,z），需覆盖入射波空间频率 k·L）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--gain", type=float, default=None,
                    help="谱权重初始化 scale（默认 1/(in*out)）")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--phys", default="",
                    help="P2 物理损失：'λhelm,λpec'（如 '0.5,0.5'），留空关闭")
    args = ap.parse_args()
    SEED = args.seed
    modes = tuple(int(m) for m in args.modes.split(","))
    phys = None
    if args.phys:
        lh, lp = [float(v) for v in args.phys.split(",")]
        phys = (lh, lp)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    x_all, y_all, idx, angles, eps_mask = load_data()
    runs = {
        "interp": (idx["interp_train"], idx["interp_test"]),
        "extrap": (idx["extrap_train"], idx["extrap_test"]),
        "full": (idx["all"], np.array([], dtype=int)),
    }
    if args.run == "all":
        sel = ["interp", "extrap", "full"]
    else:
        sel = [args.run]

    for name in sel:
        itr, ite = runs[name]
        train_run(name, x_all, y_all, itr, ite, args.epochs, args.batch, args.width, modes, device,
                  gain=args.gain, lr=args.lr, phys=phys,
                  mask_metal=torch.from_numpy(eps_mask) if phys else None)

    # full 模型可视化（取最后训练的 full 实验 checkpoint）
    if "full" in sel:
        ck = torch.load(os.path.join(RESULT_DIR, "ckpt_full.pt"), map_location="cpu",
                        weights_only=False)
        model = FFNO3D(modes=tuple(ck["config"]["modes"]), width=ck["config"]["width"],
                       in_ch=x_all.shape[1], out_ch=y_all.shape[1]).to(device)
        model.load_state_dict(ck["model_state"]); model.eval()
        st = ck["stats"]
        sel_vis = [0, idx["interp_test"][0]]       # case0 + 一个留出角
        visualize(model, x_all, y_all, angles, sel_vis,
                  (st["x_inc_mean"], st["x_inc_std"], st["y_mean"], st["y_std"]), device,
                  os.path.join(RESULT_DIR, "fno_f16_3d_result.png"))


if __name__ == "__main__":
    main()
