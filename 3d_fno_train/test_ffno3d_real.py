# -*- coding: utf-8 -*-
"""验证：FFNO3D_Real（纯实算）与 FFNO3D（torch.fft 原版）输出一致
验证项：
  1. SpectralConv3dReal 单层（含 rfftn/irfftn 数学等价）
  2. 完整 FFNO3D_Real 模型（随机权重）
  3. P3 训练 ckpt 加载后前向一致（真实权重）
  4. GPU 反向传播冒烟测试
用法：
  & "F:/miniconda3/envs/isaac311/python.exe" test_ffno3d_real.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import numpy as np
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_real as R

torch.manual_seed(42)
np.random.seed(42)
GRID = (64, 48, 32)
MODES = (20, 16, 10)


# ---- 1. 单层 SpectralConv3d 一致性 ----
print("=== 1. SpectralConv3dReal 单层一致性 ===")
sc_fft = M.FactorizedSpectralConv3d(32, 32, MODES, grid=GRID)
sc_real = R.SpectralConv3dReal(32, 32, MODES, grid=GRID)
for nm in ("wx", "wy", "wz"):
    W = getattr(sc_fft, nm).data
    getattr(sc_real, f"{nm}_r").data = W.real.contiguous()
    getattr(sc_real, f"{nm}_i").data = W.imag.contiguous()
x = torch.randn(2, 32, *GRID)
with torch.no_grad():
    y_fft = sc_fft(x)
    y_real = sc_real(x)
err = (y_fft - y_real).abs().max().item()
rel = (y_fft - y_real).norm().item() / y_fft.norm().item()
print(f"  max|diff|={err:.3e}  rel={rel:.3e}  y_fft.norm={y_fft.norm().item():.3f}")


# ---- 2. 完整模型一致性（随机权重） ----
print("=== 2. FFNO3D_Real 完整模型一致性（随机权重） ===")
f_fft = M.FFNO3D(modes=MODES, width=32, in_ch=7, out_ch=12, grid=GRID)
f_real = R.FFNO3D_Real(modes=MODES, width=32, in_ch=7, out_ch=12, grid=GRID)
f_real.load_state_dict(R.to_real_state_dict_3d(f_fft))
xin = torch.randn(2, 7, *GRID)
with torch.no_grad():
    o_fft = f_fft(xin)
    o_real = f_real(xin)
err = (o_fft - o_real).abs().max().item()
rel = (o_fft - o_real).norm().item() / o_fft.norm().item()
print(f"  max|diff|={err:.3e}  rel={rel:.3e}")
assert rel < 1e-4, "实算模型与 torch.fft 原版不一致！"


# ---- 3. P3 训练 ckpt 前向一致（真实权重，width=128） ----
print("=== 3. P3 ckpt 权重前向一致性（width=128, out=12） ===")
ckpt = torch.load(os.path.join(BASE, "results", "ckpt_full_p3.pt"),
                  map_location="cpu", weights_only=False)
cfg = ckpt["config"]
f_fft3 = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                  in_ch=7, out_ch=cfg["out_ch"], grid=GRID)
f_fft3.load_state_dict(ckpt["model_state"])
f_real3 = R.FFNO3D_Real(modes=tuple(cfg["modes"]), width=cfg["width"],
                        in_ch=7, out_ch=cfg["out_ch"], grid=GRID)
f_real3.load_state_dict(R.to_real_state_dict_3d(f_fft3))
xin3 = torch.randn(2, 7, *GRID)
with torch.no_grad():
    o3_fft = f_fft3(xin3)
    o3_real = f_real3(xin3)
err = (o3_fft - o3_real).abs().max().item()
rel = (o3_fft - o3_real).norm().item() / o3_fft.norm().item()
print(f"  参数量 fft={f_fft3.count_params():,} / real={f_real3.count_params():,}")
print(f"  max|diff|={err:.3e}  rel={rel:.3e}")
assert rel < 1e-4, "P3 权重实算前向不一致！"


# ---- 4. GPU 反向传播冒烟测试 ----
print("=== 4. GPU 反向传播冒烟测试 ===")
if torch.cuda.is_available():
    f_real_g = R.FFNO3D_Real(modes=MODES, width=32, in_ch=7, out_ch=12, grid=GRID).cuda()
    xg = torch.randn(2, 7, *GRID, device="cuda")
    yg = f_real_g(xg)
    loss = yg.pow(2).mean()
    loss.backward()
    nparam = sum(1 for p in f_real_g.parameters() if p.grad is not None)
    print(f"  输出 shape={tuple(yg.shape)} loss={loss.item():.4f} 可更新参数={nparam}")
    print("  GPU 前向+反向 OK")
else:
    print("  无 GPU，跳过")

print("\n全部通过" if rel < 1e-4 else "\n存在不一致！")
