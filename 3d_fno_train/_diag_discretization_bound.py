# -*- coding: utf-8 -*-
"""
_diag_discretization_bound.py — P5：离散化误差界（面元近似的收敛阶与 h 取值的合理性）
================================================================================
要回答的理论问题（P5）
--------------------------------------------------------------------------------
部署链路的远场是"体素表面面元 + 中点求积"的 NFFFT：

    E_ff(r̂) = Σ_{f} e^{jk r̂·r_f} (p̂·J_f) dA_f ,   J_f = n̂_f × H_tot(r_f)

两个独立误差源必须分开：
  (i)  **求积误差**：每个面元上 J 与相位 e^{jk r̂·r} 并非常量，中点求积只是近似；
  (ii) **几何阶梯误差**：体素表面是真实 STL 曲面的阶梯近似，面元位置偏差 |δr| ≤ h/2。
        这一项进相位 ⇒ 相位误差 ≤ k·h/2。本链路 h = λ/3.198 ⇒ **k h/2 = 0.982 rad ≈ 56°**
        —— 已经远大于"小相位"的任何合理界限（≲0.2 rad）。

本脚本在**几何模型完全不变**的前提下把求积加密（F = 1, 2, 4），从而：
  · 若误差随 F 显著下降 ⇒ 瓶颈是求积，加密有效；
  · 若误差不降 ⇒ 瓶颈是**几何阶梯**（模型误差），加密求积无效 ⇒ 必须减小 h 才能改善。
并给出"要拿到给定精度，h 需要多小"的定量要求。

做法（不重算任何场，全部复用既有 h5）
  · 把 eps / 场按块重复上采样到 h/F（eps 用 np.repeat ⇒ 阶梯几何**逐点不变**）；
  · 细面元的取场体素 = 粗体素索引 // F ⇒ 场在细面元上分段常量（相当于对同一个
    阶梯模型做纯求积加密）；
  · h_inc 用解析式在**细面元中心**精确求值（避免把采样误差混进来）；
  · NFFFT → 对 FEKO 真值取中位 |ΔRCS|，与 F=1 对照。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_discretization_bound.py
产出
  results/_diag_discretization_bound.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M                                              # noqa: E402
from fno_f16_3d_p4_nffft import surface_parts, direction_grid, rcs_from_surface  # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi


def dBs(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def med_err(a, ref, mask):
    return float(np.median(np.abs(dBs(a) - dBs(ref))[mask]))


def refined_grid(g, F, h):
    """在物理范围不变的前提下，把一维坐标加密 F 倍（子点落在原格点两侧 ±h/2F 内）。"""
    n = len(g)
    return g[0] - h * 0.5 + (np.arange(F * n) + 0.5) * (h / F)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="1,2,4")
    ap.add_argument("--nangles", type=int, default=12)
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    f = h5py.File(M.H5, "r")
    eps = f["eps_field"][:]
    ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
    gx = f["grid_x"][:].astype(np.float64)
    gy = f["grid_y"][:].astype(np.float64)
    gz = f["grid_z"][:].astype(np.float64)
    rcs_true = f["rcs"][:]
    h = float(gx[1] - gx[0])
    k = float(beta[0]); lam = 2 * np.pi / k
    print(f"h={h:.6f} m  λ={lam:.6f} m  λ/h={lam/h:.4f}  k·h/2={k*h/2:.4f} rad "
          f"（几何阶梯的最大相位误差）")
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    res = {"h_m": h, "lambda_over_h": float(lam / h), "kh2_rad": float(k * h / 2),
           "n_angles": len(idx_list), "angles": idx_list, "runs": {}}

    for F in [int(x) for x in args.factors.split(",")]:
        t0 = time.time()
        hF = h / F
        eps_f = np.repeat(np.repeat(np.repeat(eps, F, axis=0), F, axis=1), F, axis=2)
        gxf = refined_grid(gx, F, h)
        gyf = refined_grid(gy, F, h)
        gzf = refined_grid(gz, F, h)
        idxs_f, rsurf_f, dS_f = surface_parts(eps_f, gxf, gyf, gzf, h=hF)
        dA_f = np.linalg.norm(dS_f, axis=1)
        nvec_f = dS_f / dA_f[:, None]
        coarse = idxs_f // F                       # 细体素 → 粗体素（分段常量取场）
        print(f"\n[F={F}] 细面元 {len(dA_f)}  等效面元密度 ×{len(dA_f)/1346:.2f}  "
              f"({time.time()-t0:.0f}s)")

        errs, errs_inc, errs_zero = [], [], []
        for i in idx_list:
            Hs = f["H_scat"][i][coarse[:, 0], coarse[:, 1], coarse[:, 2]]     # (Ns,3)
            phase = np.exp(-1j * beta[i] * (rsurf_f @ khat[i]))
            h_inc = (1.0 / ETA0) * np.cross(np.broadcast_to(khat[i], rsurf_f.shape),
                                            np.broadcast_to(e0[i], rsurf_f.shape)) * phase[:, None]
            e0m = float(np.linalg.norm(e0[i]))
            m = rcs_true[i].reshape(shape) > 1e-6
            rt = rcs_true[i].reshape(shape)
            rcs_tot = rcs_from_surface(np.zeros_like(Hs), h_inc + Hs, dS_f, rsurf_f, rhat,
                                       th, ph, k, e0m).reshape(shape)
            rcs_inc = rcs_from_surface(np.zeros_like(Hs), h_inc, dS_f, rsurf_f, rhat,
                                       th, ph, k, e0m).reshape(shape)
            errs.append(med_err(rcs_tot, rt, m))
            errs_inc.append(med_err(rcs_inc, rt, m))
        res["runs"][f"F{F}"] = {
            "h_eff_m": hF, "lambda_over_h_eff": float(lam / hF), "n_faces": int(len(dA_f)),
            "med_dB_full": float(np.median(errs)),
            "med_dB_inc_only": float(np.median(errs_inc)),
            "per_angle_dB": [float(x) for x in errs]}
        r = res["runs"][f"F{F}"]
        print(f"  inc+H_scat 中位 {r['med_dB_full']:7.3f} dB   "
              f"（h_inc-only 基线 {r['med_dB_inc_only']:7.3f} dB）  ({time.time()-t0:.0f}s)")

    Fs = sorted(int(x) for x in args.factors.split(","))
    e0_ = res["runs"][f"F{Fs[0]}"]["med_dB_full"]
    e1_ = res["runs"][f"F{Fs[-1]}"]["med_dB_full"]
    res["quadrature_refinement_gain_dB"] = float(e0_ - e1_)
    print(f"\n求积加密（F={Fs[0]}→{Fs[-1]}，等效 h {h}→{h/Fs[-1]:.5f} m）"
          f"对中位误差的改变 = {e0_ - e1_:+.3f} dB")
    # 相位误差要求：若要求几何相位误差 ≤ φ_max，则 h ≤ 2φ_max/k
    res["h_requirement"] = {f"phi_max_{p}": float(2 * p / k)
                            for p in (0.1, 0.2, 0.5, 1.0)}
    print("要满足几何相位误差 ≤ φ_max，所需 h：")
    for pp, hh in res["h_requirement"].items():
        print(f"   {pp} rad  ⇒  h ≤ {hh*1000:6.2f} mm = λ/{lam/hh:5.1f}   "
              f"（现 h={h*1000:.2f} mm = λ/{lam/h:.1f}）")
    f.close()

    jp = os.path.join(RESULT_DIR, "_diag_discretization_bound.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(res, fo, indent=2, ensure_ascii=False, default=float)
    print(f"\n已存: {jp}")


if __name__ == "__main__":
    main()
