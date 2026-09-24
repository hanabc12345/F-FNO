# -*- coding: utf-8 -*-
"""_diag_errprop_exact.py — 修正 exp_errprop 的"相干增益定律"测量口径

问题诊断（对 exp_errprop.py 逐行核对后）：
  1. 扫描噪声是**逐面元 iid 复高斯乘法噪声**（第 103–104 行），是白噪声；
  2. 但误差统计是「**含噪估计 vs FEKO 真值**」的中位数（第 91–95 行），
     而这条链路本身带着 3.5 dB 的**系统性阶梯面离散化 floor**；
  3. 中位数对"共同模式下的大偏置 + 小随机量"极不敏感：当 floor(3.5 dB) ≫ 噪声项时，
     median|F+n| ≈ median|F|，噪声几乎不体现在中位数上；
     ⇒ 拟合出的 K_meas=1.45 主要是"floor 掩蔽"的产物，而非噪声传播斜率。
  4. 同时 K_theory=9.14 是在**同一条链路**上用 G 推的，两个量口径不一致。

本脚本用正确口径重测：**含噪估计 vs 干净估计（同一 NFFFT 链路）**，
让共同模式的 floor 相消，剩下纯噪声传播响应；再与一阶理论 K=4.144/√G 对照。

指标同时给：
  · median|ΔRCS|（dB）——可比原口径但已去 floor
  · rms(ΔRCS)（dB）——对小信号更敏感，斜率拟合更稳
  · 每 ε 的重复间标准误（--reps）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_errprop_exact.py --nangles 12 --reps 5
产出：results/_diag_errprop_exact.json + .png
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
from fno_f16_3d_p4_nffft import (direction_grid, surface_parts, rcs_from_surface,
                                 build_tot_fields, sample_surface_field)
from _diag_nffft_audit import rcs_stats

RESULT_DIR = os.path.join(BASE, "results")


def dBs(x):
    return 10.0 * np.log10(np.maximum(x, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=12)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--eps", default="0.02,0.05,0.1,0.2,0.3,0.4,0.6,0.8,1.0")
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]
        H_scat = f["H_scat"][:]
        eps_f = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0])
    idxs, rsurf, dS = surface_parts(eps_f, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    dA = np.linalg.norm(dS, axis=1); nvec = dS / dA[:, None]

    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    mask = rcs_true[idx_list] > 1e-6

    # ---------- 真值表面场 + 相干增益 G ----------
    H_true, G_all = [], []
    for i in idx_list:
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i)
        Hs = sample_surface_field(Htot, idxs)
        H_true.append(Hs)
        J = np.cross(nvec, Hs)
        N = np.exp(1j * k * (rsurf @ rhat.T)).T @ (J * dA[:, None])
        Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat      # 横场投影版本
        G_all.append((np.abs(Nperp) ** 2).sum(axis=1) / float((np.abs(J * dA[:, None]) ** 2).sum()))
    H_true = np.asarray(H_true); G_all = np.concatenate(G_all)
    G_med = float(np.median(G_all))
    K_theory = 4.144 / np.sqrt(G_med)
    print(f"G_med={G_med:.4f}  P25={np.percentile(G_all,25):.4f}  P90={np.percentile(G_all,90):.4f}")
    print(f"K_theory（一阶，白噪声）= 4.144/√G = {K_theory:.3f} dB/ε")

    def est(Hs, e0mag):
        return rcs_from_surface(np.zeros_like(Hs), Hs, dS, rsurf, rhat, th, ph, k, e0mag)

    e0mags = [float(np.linalg.norm(e0[i])) for i in idx_list]
    clean = np.asarray([est(H_true[a], e0mags[a]).reshape(shape) for a in range(len(idx_list))])
    # 口径对照：干净 NFFFT vs FEKO 真值（即论文所称 3.55 dB floor）
    floor_stats = {"vs_pipeline_floor_median_dB":
                   float(np.median([rcs_stats(clean[a], rcs_true[i])["masked"]["median_dB"]
                                    for a, i in enumerate(idx_list)]))}
    print(f"本批角度下「干净 NFFFT vs FEKO」中位 = {floor_stats['vs_pipeline_floor_median_dB']:.3f} dB")

    eps_grid = [float(x) for x in args.eps.split(",")]
    rng = np.random.default_rng(0)
    rows = []
    t0 = time.time()
    for eps_ in eps_grid:
        meds, rmss, meds_feko = [], [], []
        for r in range(args.reps):
            per_med, per_rms, per_med_feko = [], [], []
            for a in range(len(idx_list)):
                delta = (rng.standard_normal(H_true[a].shape)
                         + 1j * rng.standard_normal(H_true[a].shape)) / np.sqrt(2.0)
                Hs_n = H_true[a] * (1.0 + eps_ * delta)
                rcs_n = est(Hs_n, e0mags[a]).reshape(shape)
                per_med.append(rcs_stats(rcs_n, clean[a])["masked"]["median_dB"])
                d = (dBs(rcs_n) - dBs(clean[a]))[mask[a]]
                per_rms.append(float(np.sqrt(np.mean(d ** 2))))
                per_med_feko.append(rcs_stats(rcs_n, rcs_true[idx_list[a]])["masked"]["median_dB"])
            meds.append(np.median(per_med)); rmss.append(np.median(per_rms))
            meds_feko.append(np.median(per_med_feko))
        rows.append({"eps": eps_, "med_vs_clean": float(np.median(meds)),
                     "med_vs_clean_sem": float(np.std(meds) / np.sqrt(args.reps)),
                     "rms_vs_clean": float(np.median(rmss)),
                     "rms_vs_clean_sem": float(np.std(rmss) / np.sqrt(args.reps)),
                     "med_vs_FEKO": float(np.median(meds_feko))})
        print(f"  ε={eps_:.3f}  median|ΔRCS|(vs clean) {rows[-1]['med_vs_clean']:.3f}±"
              f"{rows[-1]['med_vs_clean_sem']:.3f} dB | rms {rows[-1]['rms_vs_clean']:.3f} dB | "
              f"median(vs FEKO) {rows[-1]['med_vs_FEKO']:.3f} dB | {time.time()-t0:.0f}s", flush=True)

    X = np.array([r["eps"] for r in rows])

    def fit(key):
        Y = np.array([r[key] for r in rows])
        A = np.stack([X, np.ones_like(X)], axis=1)
        coef, res, *_ = np.linalg.lstsq(A, Y, rcond=None)
        # 斜率标准误
        r_ = Y - A @ coef
        dof = max(1, len(X) - 2)
        s2 = float((r_ ** 2).sum() / dof)
        cov = s2 * np.linalg.inv(A.T @ A)
        return {"K": float(coef[0]), "b": float(coef[1]),
                "K_stderr": float(np.sqrt(cov[0, 0])), "r2": float(1 - (r_ ** 2).sum() / ((Y - Y.mean()) ** 2).sum())}

    fmed, frms = fit("med_vs_clean"), fit("rms_vs_clean")
    out = {"n_angles": len(idx_list), "reps": args.reps, "angles": [int(x) for x in idx_list],
           "G_med": G_med, "G_p25": float(np.percentile(G_all, 25)),
           "G_p90": float(np.percentile(G_all, 90)), "K_theory": K_theory,
           "floor_vs_FEKO_median_dB": floor_stats["vs_pipeline_floor_median_dB"],
           "sweep": rows, "fit_median_vs_clean": fmed, "fit_rms_vs_clean": frms}
    jp = os.path.join(RESULT_DIR, "_diag_errprop_exact.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False, default=float)

    print("\n===== 拟合（含截距 b，一阶应对 b≈0）=====")
    print(f"  K_theory（一阶，白噪声）= {K_theory:.3f} dB/ε")
    print(f"  median|ΔRCS| vs 干净估计 : K={fmed['K']:.3f} ± {fmed['K_stderr']:.3f}  "
          f"b={fmed['b']:.3f} dB  R²={fmed['r2']:.4f}")
    print(f"  rms(ΔRCS)    vs 干净估计 : K={frms['K']:.3f} ± {frms['K_stderr']:.3f}  "
          f"b={frms['b']:.3f} dB  R²={frms['r2']:.4f}")
    print(f"  （原 exp_errprop 口径 median|ΔRCS| vs FEKO：K=1.454, b=3.310）")
    print(f"\n已存: {jp}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(X, [r["med_vs_FEKO"] for r in rows], "o:", color="tab:red",
            label="median|ΔRCS| vs FEKO  (原口径, 被 floor 掩蔽)")
    ax.errorbar(X, [r["med_vs_clean"] for r in rows],
                yerr=[r["med_vs_clean_sem"] for r in rows], fmt="s-", color="tab:blue",
                capsize=3, label="median|ΔRCS| vs 干净估计 (修正口径)")
    ax.errorbar(X, [r["rms_vs_clean"] for r in rows],
                yerr=[r["rms_vs_clean_sem"] for r in rows], fmt="^-", color="tab:green",
                capsize=3, label="rms(ΔRCS) vs 干净估计")
    xs = np.linspace(0, max(X), 50)
    ax.plot(xs, K_theory * xs, "--", color="k", lw=1.2, label=f"theory 4.144/√G = {K_theory:.2f} dB/ε")
    ax.set_xlabel("near-field relative error ε"); ax.set_ylabel("|ΔRCS| (dB)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    png = os.path.join(RESULT_DIR, "_diag_errprop_exact.png")
    fig.savefig(png, dpi=110)
    print(f"已存: {png}")


if __name__ == "__main__":
    main()
