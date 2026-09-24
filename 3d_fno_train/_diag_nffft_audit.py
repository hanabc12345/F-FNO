# -*- coding: utf-8 -*-
"""_diag_nffft_audit.py — NFFFT 参考链路审计（F-16 真值场）

目的：把论文 3.55 dB 的"管线 floor"拆成可归因的分量，回答专家问题 1：
  在 NFFFT 解决前，4.05/5.28 dB 里有多少来自网络、多少来自管线？

三个对照实验（全部基于已有数据，不需要重跑 FEKO）：
  A) 约定/归一化：NFFFT 复场 vs h5 的 FEKO 远场 E_ff（含相位），
     报告每个方向的复数幅度比、相位差、复相关系数，以及最佳复数标定因子 c。
     - |c|≈1 且 arg(c)≈0 → 约定与归一化正确，误差来自离散化
     - |c| 系统性偏离或 arg(c) 明显 → 归一化/相位约定有 bug
  B) 场采样偏移：表面场三种取法（空气侧体素 / 面元中心三线性插值 / 金属-空气双侧平均）
     分别过同一 NFFFT，比较 |ΔRCS| 中位数。用于隔离代码里"面元中心与场采样点
     差 h/2（k·h/2≈0.98 rad）"这一项。
  C) 统计口径：同时给出「掩膜后（现有口径 rcs>1e-6）」与「全方向」两套中位/P90，
     并额外给出近零方向占比。

产出：results/_diag_nffft_audit.json + results/_diag_nffft_audit.png

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_audit.py --nangles 20
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
from fno_f16_3d_p4_nffft import (direction_grid, nffft, build_tot_fields)

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi   # 376.7303 Ω


# ============================================================
# 表面面元（与 fno_f16_3d_p4_nffft.surface_parts 逐位一致，另返回金属侧索引）
# ============================================================
def surface_parts_full(eps, gx, gy, gz, h=0.03125):
    """返回 (idx_air (Nf,3), idx_metal (Nf,3), rsurf (Nf,3), dS (Nf,3))
    idx_air 为场采样用空气侧体素；rsurf 为面元中心（两体素中点）；dS 为外法向·h²。"""
    metal = eps > 1.5
    Nx, Ny, Nz = metal.shape
    grid = [gx, gy, gz]
    neigh = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
             np.array([0, 1, 0]), np.array([0, -1, 0]),
             np.array([0, 0, 1]), np.array([0, 0, -1])]
    air_idx, met_idx, poss, ds = [], [], [], []
    for p in np.argwhere(metal):
        for d in neigh:
            q = p + d
            if (q < 0).any() or (q[0] >= Nx) or (q[1] >= Ny) or (q[2] >= Nz):
                continue
            if metal[tuple(q)]:
                continue
            ax = int(np.argmax(np.abs(d)))
            pos = [float(grid[a][p[a]]) for a in range(3)]
            pos[ax] += float(d[ax]) * (h / 2.0)
            dS_v = np.zeros(3); dS_v[ax] = float(d[ax]) * h * h
            air_idx.append(q); met_idx.append(p); poss.append(pos); ds.append(dS_v)
    return (np.asarray(air_idx), np.asarray(met_idx),
            np.asarray(poss), np.asarray(ds))


def tri_interp(field, pts, gx, gy, gz):
    """在体素网格上对 field(Nx,Ny,Nz,C) 做三线性插值到 pts(N,3)（物理坐标）。
    坐标轴序 (x,y,z)，与 meshgrid(indexing='ij') 一致。"""
    ax = [gx, gy, gz]
    N = pts.shape[0]
    idx0 = []
    wts = []
    for a in range(3):
        g = np.asarray(ax[a], dtype=np.float64)
        hh = g[1] - g[0]
        t = (pts[:, a] - g[0]) / hh
        t = np.clip(t, 0.0, len(g) - 1.0 - 1e-9)
        i0 = np.floor(t).astype(np.int64)
        w = t - i0
        idx0.append(i0); wts.append(w)
    out = np.zeros((N, field.shape[3]), dtype=field.dtype)
    for dx in (0, 1):
        wx = wts[0] if dx == 1 else (1.0 - wts[0])
        ix = np.clip(idx0[0] + dx, 0, field.shape[0] - 1)
        for dy in (0, 1):
            wy = wts[1] if dy == 1 else (1.0 - wts[1])
            iy = np.clip(idx0[1] + dy, 0, field.shape[1] - 1)
            for dz in (0, 1):
                wz = wts[2] if dz == 1 else (1.0 - wts[2])
                iz = np.clip(idx0[2] + dz, 0, field.shape[2] - 1)
                out += (wx * wy * wz)[:, None] * field[ix, iy, iz]
    return out


# ============================================================
# 由表面场算 RCS（与 fno_f16_3d_p4_nffft.rcs_from_surface 一致，M=0）
# ============================================================
def rcs_and_eff(H_s, dS, rsurf, rhat, th, ph, k, e0mag):
    dA = np.linalg.norm(dS, axis=1)
    n = dS / (dA[:, None] + 1e-12)
    J = np.cross(n, H_s)
    E_ff = nffft(J, np.zeros_like(J), rsurf, dS, rhat, k)
    E_t = (E_ff * th).sum(axis=1)
    E_p = (E_ff * ph).sum(axis=1)
    rcs = 4.0 * np.pi * (np.abs(E_t) ** 2 + np.abs(E_p) ** 2) / e0mag ** 2
    return rcs, E_t, E_p


def rcs_stats(rcs_est, rcs_true):
    """返回掩膜后 / 全方向两套统计"""
    dB_e = 10 * np.log10(np.maximum(rcs_est, 1e-9))
    dB_t = 10 * np.log10(np.maximum(rcs_true, 1e-9))
    diff = np.abs(dB_e - dB_t)
    m = rcs_true > 1e-6
    out = {}
    for nm, mm in (("masked", m), ("all", np.ones_like(m, dtype=bool))):
        corrs = []
        r1 = dB_e[mm]; r2 = dB_t[mm]
        out[nm] = {"median_dB": float(np.median(diff[mm])),
                   "p90_dB": float(np.percentile(diff[mm], 90)),
                   "corr_dB": float(np.corrcoef(r1, r2)[0, 1]),
                   "n": int(mm.sum())}
    out["null_frac"] = float(1.0 - m.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=20)
    ap.add_argument("--angle-stride", type=int, default=0,
                    help="0=linspace 均匀抽；>0=按固定步长抽")
    args = ap.parse_args()

    np.random.seed(0)
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]
        H_scat = f["H_scat"][:]
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        E_ff_h5 = f["E_ff"][:]                       # (468,37,73,2) complex
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0])
    h = float(gx[1] - gx[0])
    print(f"k={k:.4f} rad/m  h={h:.5f} m  k*h/2={k*h/2:.4f} rad  "
          f"(λ/h={2*np.pi/k/h:.2f})")

    idx_air, idx_met, rsurf, dS = surface_parts_full(eps, gx, gy, gz, h)
    print(f"表面面元 {len(rsurf)}（金属体素 {int((eps>1.5).sum())}）")
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    if args.angle_stride > 0:
        idx_list = list(range(0, len(angles), args.angle_stride))
    else:
        idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    print(f"审计角度 {len(idx_list)}: {idx_list[:8]}{'...' if len(idx_list)>8 else ''}")

    # ---------- A) 复场对比：NFFFT vs FEKO E_ff ----------
    A = {"n_dir_total": 0, "n_dir_used": 0,
         "ratio_theta": [], "ratio_phi": [],
         "dphase_theta_deg": [], "dphase_phi_deg": [],
         "rho_complex_theta": [], "rho_complex_phi": [],
         "c_theta": [], "c_phi": []}
    # ---------- B) 三种表面场取法 ----------
    V = {"V1_air": [], "V2_face_interp": [], "V3_two_side_avg": []}
    eff_dim = 0.0
    t0 = time.time()

    for a, i in enumerate(idx_list):
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                      E_scat, H_scat, i)
        e0mag = float(np.linalg.norm(e0[i]))
        H_air = Htot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        H_met = Htot[idx_met[:, 0], idx_met[:, 1], idx_met[:, 2]]
        H_face = tri_interp(Htot, rsurf, gx, gy, gz)

        variants = {"V1_air": H_air,
                    "V2_face_interp": H_face,
                    "V3_two_side_avg": 0.5 * (H_air + H_met)}
        tgt = rcs_true[i]
        for nm, H_s in variants.items():
            rcs, _, _ = rcs_and_eff(H_s, dS, rsurf, rhat, th, ph, k, e0mag)
            V[nm].append(rcs_stats(rcs.reshape(shape), tgt))

        # --- A) 与 FEKO 远场复场对比（用 V1，与现有管线口径一致）---
        _, Et_c, Ep_c = rcs_and_eff(H_air, dS, rsurf, rhat, th, ph, k, e0mag)
        Et_h = E_ff_h5[i][..., 0].ravel()
        Ep_h = E_ff_h5[i][..., 1].ravel()
        # 只在 FEKO 远场足够强的方向上比较（避免深零点处比值/相位无意义）
        thr = 0.05 * np.abs(E_ff_h5[i]).max()
        for (Ec, Eh, nm) in ((Et_c, Et_h, "theta"), (Ep_c, Ep_h, "phi")):
            m = np.abs(Eh) > thr
            A["n_dir_total"] += Eh.size
            A["n_dir_used"] += int(m.sum())
            ec = Ec[m]; eh = Eh[m]
            A[f"ratio_{nm}"].append(np.abs(ec) / np.abs(eh))
            dp = np.angle(ec * np.conj(eh))
            A[f"dphase_{nm}_deg"].append(np.rad2deg(dp))
            rho = np.abs(np.vdot(eh, ec)) / np.sqrt(
                np.vdot(ec, ec).real * np.vdot(eh, eh).real)
            A[f"rho_complex_{nm}"].append(float(rho))
            # 最佳复数标定因子 c = Σ ec·conj(eh) / Σ|eh|²
            A[f"c_{nm}"].append(np.vdot(eh, ec) / np.vdot(eh, eh))

        if (a + 1) % 5 == 0:
            print(f"  [{a+1}/{len(idx_list)}] {time.time()-t0:.0f}s", flush=True)

    # ---------- 汇总 ----------
    def cat(key):
        v = A[key]
        if isinstance(v, list):
            return np.concatenate([np.atleast_1d(np.asarray(x)).ravel() for x in v])
        return v

    summary = {
        "h_m": h, "kh2_rad": float(k * h / 2), "k": k,
        "n_faces": int(len(rsurf)), "n_angles": len(idx_list),
        "angles": [int(x) for x in idx_list],
        "dir_used_frac": float(A["n_dir_used"] / max(1, A["n_dir_total"])),
        "A_complex_vs_FEKO": {},
        "B_sampling_variants": {},
    }
    for nm in ("theta", "phi"):
        rt = cat(f"ratio_{nm}"); dp = cat(f"dphase_{nm}_deg"); cs = cat(f"c_{nm}")
        c_mag = np.abs(cs); c_ph = np.rad2deg(np.angle(cs))
        summary["A_complex_vs_FEKO"][nm] = {
            "mag_ratio_median": float(np.median(rt)),
            "mag_ratio_p10": float(np.percentile(rt, 10)),
            "mag_ratio_p90": float(np.percentile(rt, 90)),
            "dphase_med_deg": float(np.median(dp)),
            "dphase_std_deg": float(np.std(dp)),
            "dphase_circstd_deg": float(np.rad2deg(np.sqrt(
                -2 * np.log(max(1e-12, np.abs(np.mean(np.exp(1j*np.deg2rad(dp))))))))),
            "rho_complex_median": float(np.median(cat(f"rho_complex_{nm}"))),
            "c_overall_abs": float(np.abs(np.mean(cs))),
            "c_overall_arg_deg": float(np.rad2deg(np.angle(np.mean(cs)))),
            "c_perangle_abs_med": float(np.median(c_mag)),
            "c_perangle_abs_p10": float(np.percentile(c_mag, 10)),
            "c_perangle_abs_p90": float(np.percentile(c_mag, 90)),
            "c_perangle_arg_med_deg": float(np.median(c_ph)),
        }
    for nm in V:
        arr = V[nm]
        summary["B_sampling_variants"][nm] = {
            "masked_median_dB": float(np.median([x["masked"]["median_dB"] for x in arr])),
            "masked_p90_dB": float(np.median([x["masked"]["p90_dB"] for x in arr])),
            "all_median_dB": float(np.median([x["all"]["median_dB"] for x in arr])),
            "all_p90_dB": float(np.median([x["all"]["p90_dB"] for x in arr])),
            "corr_dB_masked": float(np.median([x["masked"]["corr_dB"] for x in arr])),
            "per_angle_masked_median_dB": [float(x["masked"]["median_dB"]) for x in arr],
        }
    summary["C_null_frac_median"] = float(np.median([x["null_frac"] for x in V["V1_air"]]))

    out_path = os.path.join(RESULT_DIR, "_diag_nffft_audit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    # ---------- 打印 ----------
    print("\n===== A) NFFFT 复场 vs FEKO E_ff（约定/归一化）=====")
    print(f"  参与比较方向占比 {summary['dir_used_frac']*100:.1f}%")
    for nm, lab in (("theta", "E_θ"), ("phi", "E_φ")):
        s = summary["A_complex_vs_FEKO"][nm]
        print(f"  {lab}: |比|中位 {s['mag_ratio_median']:.3f} "
              f"[P10 {s['mag_ratio_p10']:.3f}, P90 {s['mag_ratio_p90']:.3f}]  "
              f"Δφ中位 {s['dphase_med_deg']:+.2f}° (圆标准差 {s['dphase_circstd_deg']:.1f}°)  "
              f"复相关 {s['rho_complex_median']:.3f}")
        print(f"      标定 c: |c|总体 {s['c_overall_abs']:.3f} arg {s['c_overall_arg_deg']:+.2f}° | "
              f"逐角度 |c|中位 {s['c_perangle_abs_med']:.3f} "
              f"[{s['c_perangle_abs_p10']:.3f}, {s['c_perangle_abs_p90']:.3f}]")
    print("\n===== B) 表面场取法（同一 NFFFT）=====")
    for nm, lab in (("V1_air", "空气侧体素(现状)"),
                    ("V2_face_interp", "面元中心插值"),
                    ("V3_two_side_avg", "双侧平均")):
        s = summary["B_sampling_variants"][nm]
        print(f"  {lab:16s}: 掩膜后中位 {s['masked_median_dB']:.3f} dB / P90 {s['masked_p90_dB']:.2f} dB "
              f"/ corr {s['corr_dB_masked']:.3f}  | 全方向中位 {s['all_median_dB']:.3f} dB")
    print(f"\n===== C) 近零(RCS<1e-6)方向占比 中位 {summary['C_null_frac_median']*100:.1f}% =====")

    # ---------- 图 ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    dp_t = cat("dphase_theta_deg"); rt = cat("ratio_theta")
    axes[0].hist(dp_t, bins=60, color="tab:blue", alpha=0.8)
    axes[0].axvline(0, color="r", lw=1.0)
    axes[0].set_title("phase diff (E_θ): NFFFT vs FEKO [deg]")
    axes[1].hist(rt, bins=60, color="tab:orange", alpha=0.8)
    axes[1].axvline(1.0, color="r", lw=1.0)
    axes[1].set_title("magnitude ratio |E_θ|: NFFFT / FEKO")
    names = ["V1 air-side", "V2 face-interp", "V3 two-side avg"]
    meds = [summary["B_sampling_variants"][n]["masked_median_dB"] for n in V]
    axes[2].bar(names, meds, color=["tab:gray", "tab:green", "tab:purple"])
    axes[2].axhline(3.55, ls="--", color="r", lw=1.0, label="reported floor 3.55 dB")
    axes[2].set_ylabel("median |ΔRCS| (dB)"); axes[2].legend(fontsize=8)
    axes[2].set_title("surface-field sampling variants")
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, "_diag_nffft_audit.png")
    fig.savefig(png, dpi=110)
    print(f"\n已存: {out_path}\n已存: {png}")


if __name__ == "__main__":
    main()
