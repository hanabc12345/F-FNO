# -*- coding: utf-8 -*-
"""
_exp_common.py — 论文实验公共函数
=================================
NFFFT 远场 RCS 评估（truth 基线 + 代理模型预测），供 exp_improve / exp_extension 复用。
统计指标：|ΔRCS| 中位 / P90（dB）、远场方向图结构相关。
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import (surface_parts, rcs_from_surface, direction_grid,
                                 pred_to_surface, build_tot_fields, sample_surface_field)


def build_x_single(i, eps, e0, khat, beta, gx, gy, gz):
    """重建第 i 个角度的输入 x(1,7,64,48,32)（原始量纲，通道布局同 P3）"""
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)
    phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
    ei = e0[i][None, None, None, :] * phase[..., None]
    x = np.zeros((1, 7, 64, 48, 32), dtype=np.float32)
    x[0, 0] = (eps > 1.5).astype(np.float32)
    x[0, 1:4] = ei.real.transpose(3, 0, 1, 2)
    x[0, 4:7] = ei.imag.transpose(3, 0, 1, 2)
    return x


def _rcs_stats(rcs_est, rcs_true):
    mask = rcs_true > 1e-6
    dB_e = 10 * np.log10(np.maximum(rcs_est, 1e-9))
    dB_t = 10 * np.log10(np.maximum(rcs_true, 1e-9))
    diff = dB_e - dB_t
    corrs = []
    for a in range(len(rcs_est)):
        r1 = dB_e[a][mask[a]]; r2 = dB_t[a][mask[a]]
        if r1.std() > 1e-9 and r2.std() > 1e-9:
            corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
    return {"rcs_dB_err_median": float(np.median(np.abs(diff[mask]))),
            "rcs_dB_err_p90": float(np.percentile(np.abs(diff[mask]), 90)),
            "rcs_corr_median": float(np.median(corrs))}


def nffft_eval(models, angle_idx, xm, xs, ym, ys, device="cuda",
               tag="pred", save_png=None):
    """对给定角度子集评估 NFFFT 远场 RCS：truth 基线 + 模型预测。
    models: list[nn.Module]（ensemble 时取平均）或单个 nn.Module。
    返回 (stats, rcs_all)：stats={truth:{...}, pred:{...}}；rcs_all={'truth':(n,37,73), 'pred':(...)}"""
    if not isinstance(models, (list, tuple)):
        models = [models]
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
    tgt = rcs_true[angle_idx]

    def est_rcs(use_pred):
        out = []
        t0 = time.time()
        for a, i in enumerate(angle_idx):
            if use_pred:
                x = build_x_single(i, eps, e0, khat, beta, gx, gy, gz)
                x = (x - xm) / xs
                preds = []
                with torch.no_grad():
                    for m in models:
                        p = m(torch.from_numpy(x).to(device))[0].cpu().numpy()
                        preds.append(p * ys.reshape(12, 1, 1, 1) + ym.reshape(12, 1, 1, 1))
                pred = np.mean(preds, axis=0)
                E_s, H_s = pred_to_surface(pred, idxs)
            else:
                Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                              E_scat, H_scat, i)
                E_s = sample_surface_field(Etot, idxs)
                H_s = sample_surface_field(Htot, idxs)
            rcs = rcs_from_surface(E_s, H_s, dS, rsurf, rhat, th, ph, k,
                                   np.linalg.norm(e0[i]))
            out.append(rcs.reshape(shape))
            if (a + 1) % 20 == 0:
                print(f"    nffft[{tag}->{'pred' if use_pred else 'truth'}] {a+1}/{len(angle_idx)}"
                      f" ({time.time()-t0:.0f}s)", flush=True)
        return np.asarray(out)

    rcs_all = {}
    stats = {}
    for tag_, use_pred in (("truth", False), ("pred", True)):
        rcs_all[tag_] = est_rcs(use_pred)
        stats[tag_] = _rcs_stats(rcs_all[tag_], tgt)
        print(f"  [nffft-{tag_}] |ΔRCS| 中位 {stats[tag_]['rcs_dB_err_median']:.2f} dB, "
              f"P90 {stats[tag_]['rcs_dB_err_p90']:.2f} dB, corr {stats[tag_]['rcs_corr_median']:.3f}")

    if save_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = min(3, len(angle_idx))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
        axes = np.atleast_1d(axes)
        for a in range(n):
            i = angle_idx[a]
            t90 = np.argmin(np.abs(ff_theta - 90))
            axes[a].semilogy(ff_phi, rcs_all["truth"][a][t90], "-", lw=1.2, label="truth")
            axes[a].semilogy(ff_phi, rcs_all["pred"][a][t90], "--", lw=1.2, label=tag)
            axes[a].semilogy(ff_phi, tgt[a][t90], ":", lw=1.2, label="FEKO h5")
            axes[a].set_title(f"case_{i:03d} θ=90° φ-cut")
            axes[a].set_xlabel("φ"); axes[a].legend(fontsize=8)
        fig.suptitle(f"NFFFT vs FEKO (tag={tag})", fontsize=13)
        fig.tight_layout()
        fig.savefig(save_png, dpi=110)
        print(f"  图已存: {save_png}")
    return stats, rcs_all
