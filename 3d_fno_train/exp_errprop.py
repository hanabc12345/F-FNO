# -*- coding: utf-8 -*-
"""
exp_errprop.py — §6.1 误差传播解析模型：近场相对误差 -> 远场 RCS 误差
====================================================================
理论（一阶平均化）：
  远场主项 E_ff(rhat) = sum_s J_s e^{jk rhat·r_s} dA。
  若每面元独立相对复误差 δ_s (Var=σ²)，则 E_ff 相对误差 δ_ff 方差 = σ²/G(rhat)，
  其中 G(rhat) = |sum a_s e^{iψ_s}|² / sum a_s² 为相干增益（方向相关，主瓣大/谷点小）。
  一阶展开：|ΔRCS|dB ≈ 8.69·|Re δ_ff|，中位 = 8.69·0.6745·σ/√(2G_eff) ≈ 4.144·σ/√G_eff。

数值验证：
  对 truth 表面场 H_s 注入复高斯乘法噪声（相对误差 ≈ σ，与模型 H_rel 同口径）
  -> NFFFT -> 实测 ε→|ΔRCS| 曲线 -> 拟合斜率 K_meas，与理论 K_theory 对照。
  真实模型点对照（NFFFT 只用 H 场，ε 取各模型 H_rel）：
    P3 full        H_rel 53.6% -> 4.05 dB
    interp_plain   H_rel 85.7% -> 5.28 dB
    interp_augms   H_rel 96.3% -> 15.27 dB

产出：results/exp_errprop.json + results/exp_errprop.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import (surface_parts, rcs_from_surface, direction_grid,
                                 build_tot_fields, sample_surface_field)

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    np.random.seed(0)
    n_angles = 6
    eps_grid = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    n_rep = 3

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
    k = float(beta[0])

    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    idx_list = list(range(n_angles))
    tgt = rcs_true[idx_list]
    mask = tgt > 1e-6
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / (dA[:, None] + 1e-12)

    # ---------- 真值表面场 + 相干增益 G ----------
    H_s_true, G_all = [], []
    for a, i in enumerate(idx_list):
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i)
        H_s_true.append(sample_surface_field(Htot, idxs))
        J = np.cross(nvec, sample_surface_field(Htot, idxs))      # (Ns,3)
        N = np.exp(1j * k * (rsurf @ rhat.T)).T @ (J * dA[:, None])   # (Ndir,3)
        denom = float((np.abs(J * dA[:, None]) ** 2).sum())
        G = np.abs(N) ** 2 / denom
        G_all.append(G)
    H_s_true = np.asarray(H_s_true)
    G_all = np.concatenate(G_all)
    G_med = float(np.median(G_all)); G_p25 = float(np.percentile(G_all, 25))
    G_p90 = float(np.percentile(G_all, 90))
    K_theory = 8.69 * 0.6745 / np.sqrt(2.0 * G_med)
    print(f"\n=== 理论 ===\nG_eff 中位 {G_med:.1f} (P25 {G_p25:.1f}, P90 {G_p90:.1f})")
    print(f"K_theory = 4.144/sqrt(G_eff) = {K_theory:.4f} dB/ε")

    # ---------- 数值扫描：注入乘法噪声 -> NFFFT ----------
    def rcs_stats(rcs_est):
        dB_e = 10 * np.log10(np.maximum(rcs_est, 1e-9))
        dB_t = 10 * np.log10(np.maximum(tgt, 1e-9))
        diff = np.abs(dB_e - dB_t)
        return float(np.median(diff[mask]))

    print("\n=== 数值扫描（复高斯乘法噪声，NFFFT）===")
    t0 = time.time()
    med_grid, rep_std = [], []
    for eps_ in eps_grid:
        vals = []
        for r in range(n_rep):
            delta = (np.random.randn(*H_s_true.shape) + 1j * np.random.randn(*H_s_true.shape)) / np.sqrt(2)
            H_noisy = H_s_true * (1.0 + eps_ * delta)          # rel ≈ eps_
            est = []
            for a in range(n_angles):
                rcs = rcs_from_surface(np.zeros_like(H_noisy[a]), H_noisy[a],
                                       dS, rsurf, rhat, th, ph, k, np.linalg.norm(e0[idx_list[a]]))
                est.append(rcs.reshape(shape))
            vals.append(rcs_stats(np.asarray(est)))
        med = float(np.median(vals)); sd = float(np.std(vals))
        med_grid.append(med); rep_std.append(sd)
        print(f"  eps={eps_:.1f} -> |ΔRCS| 中位 {med:.2f} dB (±{sd:.2f}) "
              f"({time.time()-t0:.0f}s)", flush=True)

    # 线性拟合（带截距：管线基线 b + 噪声斜率 K）
    X = np.asarray(eps_grid, dtype=float)
    Y = np.asarray(med_grid, dtype=float)
    A = np.stack([X, np.ones_like(X)], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    K_meas, b = float(coef[0]), float(coef[1])
    print(f"\n=== 拟合 ===\n|ΔRCS|_med = {K_meas:.3f}·ε + {b:.2f} dB（b≈NFFFT 管线基线 3.55）")
    print(f"K_theory(一阶上界) = {K_theory:.4f} dB/ε")

    # ---------- 真实模型点对照 ----------
    model_pts = [
        {"name": "P3 full (H 53.6%)", "eps": 0.536, "rcs_dB": 4.05},
        {"name": "interp_plain (H 85.7%)", "eps": 0.857, "rcs_dB": 5.28},
        {"name": "interp_augms (H 96.3%)", "eps": 0.963, "rcs_dB": 15.27},
    ]
    print("\n=== 真实模型点 vs 理想平均化 ===")
    for p in model_pts:
        ideal = K_meas * p["eps"] + b
        ratio = p["rcs_dB"] / ideal
        print(f"  {p['name']}: 实测 {p['rcs_dB']:.2f} dB vs 理想 {ideal:.2f} dB "
              f"(偏离 {ratio:.1f}×)")

    out = {"theory": {"G_med": G_med, "G_p25": G_p25, "G_p90": G_p90,
                      "K_theory": K_theory},
           "sweep": {"eps": eps_grid, "rcs_med_db": med_grid, "rep_std": rep_std},
           "fit": {"K_meas": K_meas, "intercept_db": b},
           "model_points": model_pts}
    jpath = os.path.join(RESULT_DIR, "exp_errprop.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  已存: {jpath}")

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(eps_grid, med_grid, yerr=rep_std, fmt="o-", ms=5, capsize=3,
                label="random-noise sweep (NFFFT)")
    xs = np.linspace(0, 1, 50)
    ax.plot(xs, K_theory * xs, "--", color="gray", lw=1.2,
            label=f"theory 1st-order K={K_theory:.2f} (G_med={G_med:.1f})")
    ax.plot(xs, K_meas * xs + b, ":", color="tab:green", lw=1.4,
            label=f"fit |ΔRCS|={K_meas:.2f}ε+{b:.1f}")
    ax.axhline(3.55, color="tab:red", lw=0.8, ls="-.", alpha=0.6,
               label="NFFFT pipeline baseline 3.55 dB")
    for p in model_pts:
        ax.scatter([p["eps"]], [p["rcs_dB"]], marker="*", s=160, zorder=5)
        ax.annotate(p["name"], (p["eps"], p["rcs_dB"]),
                    textcoords="offset points", xytext=(8, 8), fontsize=8)
    ax.set_xlabel("near-field relative error ε (H_rel)")
    ax.set_ylabel("median |ΔRCS| (dB)")
    ax.set_title("Error propagation: near-field ε → far-field RCS")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, "exp_errprop.png")
    fig.savefig(png, dpi=110)
    print(f"  图已存: {png}")


if __name__ == "__main__":
    main()
