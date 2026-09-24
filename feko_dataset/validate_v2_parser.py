# -*- coding: utf-8 -*-
"""验证 v2 解析器：case_216.out(双极化+E+H) vs v1 HDF5 row 13(θ=90,φ=0, 旧单源 E-only)

case_216 = 新网格 idx216 = θ=90, φ=0，与旧数据集 case_013 同角度同频率同网格，
因此主极化 E_scat / rcs 应与 v1 逐点一致（求解确定性）。
"""
import sys
import numpy as np
import h5py

sys.path.insert(0, r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data")
import gen_feko_batch as g

RUN = r"f:\MyWorkSpace\UAVGame\3d_feko_run"
V1_H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset.h5"


def read(name):
    with open(rf"{RUN}\{name}", "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


text = read("case_216.out")
theta, phi, e0, khat, beta0 = g.parse_plane_wave(text, 0)
_, _, e0_ortho, _, _ = g.parse_plane_wave(text, 1)
E = g.parse_nearfield(text, g.MARKER_E, 0)
H = g.parse_nearfield(text, g.MARKER_H, 0)
Eo = g.parse_nearfield(text, g.MARKER_E, 1)
Ho = g.parse_nearfield(text, g.MARKER_H, 1)
ff = g.parse_farfield(text, 0)
ffo = g.parse_farfield(text, 1)

print("=== case_216.out (θ=90,φ=0, 双极化+E+H) ===")
print(f"theta={theta} phi={phi} beta0={beta0:.4f}")
print(f"e0(主,θ-pol)      = {np.round(e0, 4)}")
print(f"e0(正交,φ-pol)    = {np.round(e0_ortho, 4)}")
print(f"khat={np.round(khat, 4)}")
print(f"E_scat      {E.shape} |max|={np.abs(E).max():.4f}")
print(f"H_scat      {H.shape} |max|={np.abs(H).max():.4f}")
print(f"E_scat_ortho {Eo.shape} |max|={np.abs(Eo).max():.4f}")
print(f"H_scat_ortho {Ho.shape} |max|={np.abs(Ho).max():.4f}")
print(f"rcs(主)     {ff['rcs'].shape} max={ff['rcs'].max():.4f} m²")
print(f"rcs(正交)   {ffo['rcs'].shape} max={ffo['rcs'].max():.4f} m²")

print("\n=== v1 HDF5 row 13 (θ=90,φ=0, 旧 E-only) ===")
with h5py.File(V1_H5, "r") as f:
    E13 = f["E_scat"][13]
    rcs13 = f["rcs"][13]
    ff13_th = f["ff_theta"][:]
    ff13_ph = f["ff_phi"][:]
print(f"E13 {E13.shape} |max|={np.abs(E13).max():.4f}")
print(f"rcs13 {rcs13.shape} max={rcs13.max():.4f} m²")

print("\n=== 交叉对照（主极化应与 v1 一致） ===")
dE = np.abs(E13 - E).max()
print(f"E_scat 主极化 vs v1:  max diff = {dE:.3e}  "
      f"(rel {dE/np.abs(E13).max():.3e})")
dR = np.abs(rcs13 - ff["rcs"]).max()
print(f"rcs 主极化 vs v1:     max diff = {dR:.3e} m²")
dEo = np.abs(Eo - E).max() / np.abs(E).max()
print(f"正交 vs 主极化 E 相对差异 = {dEo:.4f}（应显著 >0 → 极化有效）")

# 近场 H 一致性：辐射区 |H|/|E| ≈ 1/η0，近场含驻波分量故仅参考
ratio = np.abs(H).max() / np.abs(E).max()
print(f"\n|H|max/|E|max = {ratio:.5f} (1/η0={1/376.73:.5f}, 近场含驻波/消逝分量故仅参考)")
# 远场区 H 与 E 之比应严格 ≈ 1/η0（用远场 Eθ/Eφ 间接验证）
print("全部解析通过 ✓" if E.shape == E13.shape else "形状不一致!")
