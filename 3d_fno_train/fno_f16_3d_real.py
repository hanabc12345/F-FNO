# -*- coding: utf-8 -*-
"""
fno_f16_3d_real.py — 3D F-FNO 纯实算版本（可导出 ONNX / UE5 部署）
====================================================================
与 fno_f16_3d.FFNO3D 数学等价，但所有算子均为 ONNX 标准算子：
  1. 3D FFT 用预计算 DFT 矩阵（每轴 Fr/Fi）通过 einsum 表达（FFT 是线性变换）
  2. 复数乘法 (a+bi)(c+di) = (ac-bd) + i(ad+bc)，实部/虚部分开算
  3. 因子化谱权重 wx+wy+wz（复数）拆 _r/_i，expand 后求和
  4. irfftn 通过共轭对称重构缺失列 + 逐轴 IFFT

用法：
  from fno_f16_3d_real import FFNO3D_Real, to_real_state_dict_3d
  model_real = FFNO3D_Real(modes=..., width=..., in_ch=7, out_ch=12, grid=(64,48,32))
  model_real.load_state_dict(to_real_state_dict_3d(model_fft))   # 复数权重 → 实/虚

注意：谱权重形状与 2D (fno_real.py) 不同——3D 因子化是三轴和（不取正负频两块），
rfftn 后保留正频率 modes 盒 [0:mx,0:my,0:mz]，其余零填充（与训练版 FFNO3D 完全一致）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 傅里叶矩阵（每轴一个，grid=(nx,ny,nz)）
# ============================================================

def precompute_dft(N):
    """N 点 DFT 矩阵 F[k,n] = exp(-2πi·k·n/N)，返回 (Fr, Fi)：F = Fr + i·Fi"""
    k = torch.arange(N, dtype=torch.float32)
    n = torch.arange(N, dtype=torch.float32)
    K, NN = torch.meshgrid(k, n, indexing='ij')     # (N,N)
    ang = -2.0 * np.pi * K * NN / float(N)
    Fr = torch.cos(ang)
    Fi = torch.sin(ang)                             # -sin(2πkn/N)
    return Fr, Fi


def cmm(eq, A_r, A_i, B_r, B_i):
    """复数张量乘法 C = A ⊗ B（einsum 等式 eq）：C_r = A_r·B_r − A_i·B_i；C_i = A_r·B_i + A_i·B_r"""
    Cr = torch.einsum(eq, A_r, B_r) - torch.einsum(eq, A_i, B_i)
    Ci = torch.einsum(eq, A_r, B_i) + torch.einsum(eq, A_i, B_r)
    return Cr, Ci


# ============================================================
# 3D 因子化谱卷积（纯实算）
# ============================================================

class SpectralConv3dReal(nn.Module):
    """3D 因子化谱卷积（实数实现，可导出 ONNX）。
    前向：rfftn → 保留 [0:mx,0:my,0:mz] → 谱乘 W=Wx+Wy+Wz → 零填充 → irfftn。
    FFT 全部用预计算 DFT 矩阵 einsum，与 torch.fft 数学等价。"""

    def __init__(self, in_ch, out_ch, modes, grid=(64, 48, 32), gain=None):
        super().__init__()
        mx, my, mz = modes
        self.modes = modes
        self.nx, self.ny, self.nz = grid
        self.half = self.nz // 2 + 1
        s = gain if gain is not None else 1.0 / (in_ch * out_ch)
        # 复数权重 W = Wr + i·Wi，三轴因子化（与训练版 FFNO3D 一致）
        self.wx_r = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, 1, 1))
        self.wx_i = nn.Parameter(s * torch.randn(in_ch, out_ch, mx, 1, 1))
        self.wy_r = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, my, 1))
        self.wy_i = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, my, 1))
        self.wz_r = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, 1, mz))
        self.wz_i = nn.Parameter(s * torch.randn(in_ch, out_ch, 1, 1, mz))
        # 每轴 DFT 矩阵（导出为常量 buffer，名字加 _b 避免与属性冲突）
        self.Frx, self.Fix = precompute_dft(self.nx)
        self.Fry, self.Fiy = precompute_dft(self.ny)
        self.Frz, self.Fiz = precompute_dft(self.nz)
        self.register_buffer("Frx_b", self.Frx, persistent=False)
        self.register_buffer("Fix_b", self.Fix, persistent=False)
        self.register_buffer("Fry_b", self.Fry, persistent=False)
        self.register_buffer("Fiy_b", self.Fiy, persistent=False)
        self.register_buffer("Frz_b", self.Frz, persistent=False)
        self.register_buffer("Fiz_b", self.Fiz, persistent=False)

    # ---------- 一维复 FFT / IFFT（沿指定轴，保持 (B,C,x,y,z) 轴序） ----------
    def _fft_axis(self, Xr, Xi, Fr, Fi, axis):
        """X:(B,C,nx,ny,nz) → 沿 axis(2/3/4) 复 FFT。X[k] = Σ_n F[k,n]·x[n]"""
        nd = Xr.ndim
        perm = list(range(nd)); perm.append(perm.pop(axis))      # 目标轴移到末尾
        Ar = Xr.permute(perm); Ai = Xi.permute(perm)             # (...,n)
        Br = torch.einsum("...n,mn->...m", Ar, Fr) - torch.einsum("...n,mn->...m", Ai, Fi)
        Bi = torch.einsum("...n,mn->...m", Ar, Fi) + torch.einsum("...n,mn->...m", Ai, Fr)
        inv = [0] * nd
        for i, p in enumerate(perm):
            inv[p] = i
        return Br.permute(inv), Bi.permute(inv)

    def _ifft_axis(self, Xr, Xi, Fr, Fi, axis):
        """沿 axis 复 IFFT：x[n] = Σ_k conj(F)[k,n]·X[k]（归一化在最后统一除）"""
        nd = Xr.ndim
        perm = list(range(nd)); perm.append(perm.pop(axis))
        Ar = Xr.permute(perm); Ai = Xi.permute(perm)
        Br = torch.einsum("...n,mn->...m", Ar, Fr) + torch.einsum("...n,mn->...m", Ai, Fi)
        Bi = torch.einsum("...n,mn->...m", Ai, Fr) - torch.einsum("...n,mn->...m", Ar, Fi)
        inv = [0] * nd
        for i, p in enumerate(perm):
            inv[p] = i
        return Br.permute(inv), Bi.permute(inv)

    # ---------- 前向 RFFT3 ----------
    def _rfft3(self, x):
        """实输入 x (B,C,nx,ny,nz) → RFFTN 谱 (Zr, Zi)，形状 (B,C,nx,ny,nz//2+1)"""
        Xr, Xi = x, torch.zeros_like(x)
        Xr, Xi = self._fft_axis(Xr, Xi, self.Frx_b, self.Fix_b, 2)   # x 轴 FFT
        Xr, Xi = self._fft_axis(Xr, Xi, self.Fry_b, self.Fiy_b, 3)   # y 轴 FFT
        Xr, Xi = self._fft_axis(Xr, Xi, self.Frz_b, self.Fiz_b, 4)   # z 轴 FFT
        # RFFT：z 轴频率只保留 0..half-1
        return Xr[..., :self.half], Xi[..., :self.half]

    # ---------- 逆 IRFFT3 ----------
    def _irfft3(self, Zr, Zi):
        """复谱 (B,C,nx,ny,half) → 实输出 (B,C,nx,ny,nz)
        实信号 3D 共轭对称：X[k1,k2,k3] = conj(X[N1-k1,N2-k2,N3-k3])。
        rfftn 只保留 k3∈[0,nz/2]，缺失列通过三轴反转+共轭重构。"""
        nx, ny, nz, half = self.nx, self.ny, self.nz, self.half
        Frx, Fix = self.Frx_b, self.Fix_b
        Fry, Fiy = self.Fry_b, self.Fiy_b
        Frz, Fiz = self.Frz_b, self.Fiz_b

        # 1) 重构缺失列：tail[k1,k2,m] = conj(Z[N1-k1, N2-k2, half-1-m])，m=0..half-2
        #    （对应 k3 = half+m，N3-k3 = nz-half-m = half-1-m，值域 1..half-2）
        rev_x = torch.cat([torch.zeros(1, dtype=torch.long, device=Zr.device),
                           torch.arange(nx - 1, 0, -1, device=Zr.device)])
        rev_y = torch.cat([torch.zeros(1, dtype=torch.long, device=Zr.device),
                           torch.arange(ny - 1, 0, -1, device=Zr.device)])
        rev_z = torch.arange(half - 3, -1, -1, device=Zr.device)    # 切片后索引 half-3..0（原下标 1..half-2 的反转）
        tail_r = Zr[:, :, :, :, 1:half - 1]                          # (B,C,nx,ny,half-2)
        tail_i = Zi[:, :, :, :, 1:half - 1]
        tail_r = tail_r.index_select(2, rev_x).index_select(3, rev_y).index_select(4, rev_z)
        tail_i = tail_i.index_select(2, rev_x).index_select(3, rev_y).index_select(4, rev_z)
        Z_r = torch.cat([Zr, tail_r], dim=-1)                        # (B,C,nx,ny,nz)
        Z_i = torch.cat([Zi, -tail_i], dim=-1)                       # conj → 虚部取负

        # 2) z 轴 IFFT
        Yr, Yi = self._ifft_axis(Z_r, Z_i, Frz, Fiz, 4)
        # 3) y 轴 IFFT
        Yr, Yi = self._ifft_axis(Yr, Yi, Fry, Fiy, 3)
        # 4) x 轴 IFFT（实部即结果）
        Xr, _ = self._ifft_axis(Yr, Yi, Frx, Fix, 2)
        # 5) 归一化 1/(nx·ny·nz)
        return Xr / float(nx * ny * nz)

    # ---------- 前向 ----------
    def forward(self, x):
        b = x.shape[0]
        c_in = x.shape[1]
        mx, my, mz = self.modes
        out_ch = self.wx_r.shape[1]
        Zr, Zi = self._rfft3(x)                        # (B,C,nx,ny,half)

        # 谱域低频盒
        x1_r = Zr[:, :, :mx, :my, :mz]
        x1_i = Zi[:, :, :mx, :my, :mz]
        # 因子化权重求和：W = Wx + Wy + Wz（expand 到 (c_in,out,mx,my,mz)）
        Wr = (self.wx_r.expand(c_in, out_ch, mx, my, mz)
              + self.wy_r.expand(c_in, out_ch, mx, my, mz)
              + self.wz_r.expand(c_in, out_ch, mx, my, mz))
        Wi = (self.wx_i.expand(c_in, out_ch, mx, my, mz)
              + self.wy_i.expand(c_in, out_ch, mx, my, mz)
              + self.wz_i.expand(c_in, out_ch, mx, my, mz))
        z1_r, z1_i = cmm("bcijk,coijk->boijk", x1_r, x1_i, Wr, Wi)

        # 组装 out_ft (B,out,nx,ny,half)，未选频段置零
        out_r = torch.zeros(b, out_ch, self.nx, self.ny, self.half, device=x.device)
        out_i = torch.zeros_like(out_r)
        out_r[:, :, :mx, :my, :mz] = z1_r
        out_i[:, :, :mx, :my, :mz] = z1_i
        return self._irfft3(out_r, out_i)


# ============================================================
# 3D F-FNO 模型（纯实算，可导出 ONNX）
# ============================================================

class FFNO3D_Real(nn.Module):
    """与 FFNO3D 结构一致：lift → 4×[谱卷积+1×1卷积(加性)+GELU] → project。
    全部算子为 ONNX 标准（Conv / Einsum / Add / Sub / Mul / Gather / Concat / Gelu）。"""

    def __init__(self, modes=(16, 12, 8), width=32, in_ch=7, out_ch=6, n_layers=4,
                 grid=(64, 48, 32), gain=None):
        super().__init__()
        self.nx, self.ny, self.nz = grid
        self.lift = nn.Conv3d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv3dReal(width, width, modes, grid=grid, gain=gain)
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
# 权重转换：torch.fft 原版模型 → 实算模型
# ============================================================

def to_real_state_dict_3d(model_fft):
    """把 FFNO3D（复数谱权重）的 state_dict 转成 FFNO3D_Real 可用格式。
    复数参数 → <name>_r / <name>_i；实数参数原样保留。"""
    sd = {}
    for k, v in model_fft.state_dict().items():
        if v.is_complex():
            sd[k + "_r"] = v.real
            sd[k + "_i"] = v.imag
        else:
            sd[k] = v
    return sd
