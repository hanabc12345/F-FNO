# -*- coding: utf-8 -*-
"""_diag_surface_fix.py — 降低"PEC 阶梯表面"路由离散化 floor 的候选修正对照

背景（_diag_nffft_audit4.py）：
  · 盒面等效原理路由（无插值/无阶梯化）对真值场可达 0.16 dB / corr 0.999
    ⇒ NFFFT 公式、相位约定、归一化全部正确；
  · 现有"PEC 阶梯面 + 空气侧体素场"路由对同一真值场为 3.50 dB
    ⇒ 3.55 dB floor 100% 来自**表面提取 + 离面场采样**的离散化。
  · 采样步长 h/λ=0.312，k·h/2=0.982 rad(56°)：面元中心与场采样点差 h/2，
    且体素间相位步长 k·h=1.96 rad(112°)，导致三线性插值反而更差（audit1 V2=4.67）。

本脚本只做一件事：在**不重跑 FEKO**的前提下，逐个试"表面电流构造"的廉价修正，
量化每一项目能把 floor 从 3.50 dB 压到多少。

变体（J = n̂ × H_*，远场用 (jk/4π)η₀N⊥，RCS 对整体符号不敏感）：
  A air_vox         H_tot 取空气侧体素                （现有管线基线）
  B inc_exact       H_inc 解析求值于**面元中心** + H_scat 取空气侧体素
  C inc_exact_interp H_inc 解析求值于面元中心 + H_scat 三线性插值到面元中心
  D inc_exact+k        B + H_scat 沿入射方向 k̂ 做平面波相位修正
  E inc_exact+kr       B + H_scat 沿镜面反射方向 k̂r=k̂−2(k̂·n̂)n̂ 做相位修正
  F metal_vox       H_tot 取金属侧体素
  G two_side_avg    0.5(H_air+H_met)
  H PO_lit          J = 2 n̂ × H_inc(面元中心)，仅取 k̂·n̂<0 的受照面
  I PO_all          J = 2 n̂ × H_inc(面元中心)，全部面元

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_surface_fix.py --nangles 20
产出：results/_diag_surface_fix.json + results/_diag_surface_fix.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import direction_grid, nffft, build_tot_fields
from _diag_nffft_audit import surface_parts_full, tri_interp, rcs_stats

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi

VARIANTS = ["A_air_vox", "B_inc_exact", "C_inc_exact_interp", "D_inc_exact_kph",
            "E_inc_exact_krph", "F_metal_vox", "G_two_side_avg", "H_PO_lit", "I_PO_all"]


def rcs_from_J(J, dS, rsurf, rhat, th, ph, k, e0mag, shape):
    E_ff = nffft(J, np.zeros_like(J), rsurf, dS, rhat, k)
    E_t = (E_ff * th).sum(axis=1)
    E_p = (E_ff * ph).sum(axis=1)
    rcs = 4.0 * np.pi * (np.abs(E_t) ** 2 + np.abs(E_p) ** 2) / e0mag ** 2
    return rcs.reshape(shape)


def inc_field_at(pts, e0v, khatv, beta, eta0):
    """解析入射磁场 H_inc(r) = (1/η₀) k̂ × E0 · exp(−jβ k̂·r)（与数据集约定一致）"""
    ph = np.exp(-1j * beta * (pts @ khatv))
    return (ph[:, None] * (np.cross(khatv, e0v) / eta0)[None, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=20)
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]
        H_scat = f["H_scat"][:]
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0]); h = float(gx[1] - gx[0])
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    idx_air, idx_met, rsurf, dS = surface_parts_full(eps, gx, gy, gz, h)
    dA = np.linalg.norm(dS, axis=1)
    nhat = dS / dA[:, None]
    print(f"面元 {len(rsurf)}  k·h/2={k*h/2:.4f} rad  k·h={k*h:.4f} rad  λ/h={2*np.pi/k/h:.2f}")

    idx_list = np.linspace(0, len(angles) - 1, args.nangles).astype(int)
    res = {v: [] for v in VARIANTS}
    t0 = time.time()

    for a, i in enumerate(idx_list):
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i)
        e0mag = float(np.linalg.norm(e0[i])); tgt = rcs_true[i]
        h_inc_face = inc_field_at(rsurf, e0[i], khat[i], beta[i], ETA0)   # (Nf,3)
        H_air = Htot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        H_met = Htot[idx_met[:, 0], idx_met[:, 1], idx_met[:, 2]]
        Hs_air = H_scat[i][idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        Hs_face = tri_interp(H_scat[i], rsurf, gx, gy, gz)

        # 面元中心相对空气侧体素的位移 Δr = −n̂·h/2
        dr = -nhat * (h / 2.0)
        ck = (khat[i] * nhat).sum(axis=1)                       # k̂·n̂
        k_r = khat[i][None, :] - 2.0 * ck[:, None] * nhat       # 镜面反射方向

        Hv = {}
        Hv["A_air_vox"] = H_air
        Hv["B_inc_exact"] = h_inc_face + Hs_air
        Hv["C_inc_exact_interp"] = h_inc_face + Hs_face
        Hv["D_inc_exact_kph"] = h_inc_face + Hs_air * np.exp(-1j * k * (dr @ khat[i]))[:, None]
        Hv["E_inc_exact_krph"] = h_inc_face + Hs_air * np.exp(
            -1j * k * (k_r * dr).sum(axis=1))[:, None]
        Hv["F_metal_vox"] = H_met
        Hv["G_two_side_avg"] = 0.5 * (H_air + H_met)

        for nm, H_s in Hv.items():
            J = np.cross(nhat, H_s)
            res[nm].append(rcs_stats(rcs_from_J(J, dS, rsurf, rhat, th, ph, k, e0mag, shape), tgt))

        # PO：J = 2 n̂ × H_inc（面元中心解析值）
        J_po = 2.0 * np.cross(nhat, h_inc_face)
        lit = ck < 0.0
        J_lit = np.zeros_like(J_po); J_lit[lit] = J_po[lit]
        res["H_PO_lit"].append(rcs_stats(rcs_from_J(J_lit, dS, rsurf, rhat, th, ph, k, e0mag, shape), tgt))
        res["I_PO_all"].append(rcs_stats(rcs_from_J(J_po, dS, rsurf, rhat, th, ph, k, e0mag, shape), tgt))

        if (a + 1) % 5 == 0:
            print(f"  [{a+1}/{len(idx_list)}] {time.time()-t0:.0f}s", flush=True)

    summary = {"k": k, "h_m": h, "kh2_rad": float(k * h / 2), "lambda_over_h": float(2 * np.pi / k / h),
               "n_faces": int(len(rsurf)), "n_angles": int(len(idx_list)),
               "angles": [int(x) for x in idx_list], "variants": {}}
    for nm in VARIANTS:
        arr = res[nm]
        summary["variants"][nm] = {
            "median_dB": float(np.median([x["masked"]["median_dB"] for x in arr])),
            "p90_dB": float(np.median([x["masked"]["p90_dB"] for x in arr])),
            "corr_dB": float(np.median([x["masked"]["corr_dB"] for x in arr])),
            "median_dB_std_over_angles": float(np.std([x["masked"]["median_dB"] for x in arr])),
        }
    jp = os.path.join(RESULT_DIR, "_diag_surface_fix.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(summary, fo, indent=2, ensure_ascii=False, default=float)

    base = summary["variants"]["A_air_vox"]["median_dB"]
    print("\n===== 表面电流构造变体（同一批角度，远场 RCS）=====")
    print(f"  {'变体':<22}{'中位 dB':>9}{'P90 dB':>9}{'corr':>8}{'vs 基线':>10}")
    for nm in VARIANTS:
        v = summary["variants"][nm]
        print(f"  {nm:<22}{v['median_dB']:>9.3f}{v['p90_dB']:>9.2f}{v['corr_dB']:>8.3f}"
              f"{v['median_dB']/base:>9.2f}x")
    print(f"\n已存: {jp}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4.6))
    meds = [summary["variants"][nm]["median_dB"] for nm in VARIANTS]
    cols = ["tab:gray"] + ["tab:blue"] * 4 + ["tab:orange"] * 2 + ["tab:green"] * 2
    ax.bar(range(len(VARIANTS)), meds, color=cols)
    ax.set_xticks(range(len(VARIANTS)))
    ax.set_xticklabels(VARIANTS, rotation=30, ha="right", fontsize=8)
    ax.axhline(0.161, ls="--", color="k", lw=1.0, label="box-equivalence 0.161 dB")
    ax.axhline(base, ls=":", color="r", lw=1.0, label=f"current baseline {base:.2f} dB")
    ax.set_ylabel("median |ΔRCS| (dB)"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, "_diag_surface_fix.png")
    fig.savefig(png, dpi=110)
    print(f"已存: {png}")


if __name__ == "__main__":
    main()
