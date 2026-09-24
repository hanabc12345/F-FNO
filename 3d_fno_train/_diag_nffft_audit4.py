# -*- coding: utf-8 -*-
"""_diag_nffft_audit4.py — 盒面等效原理 NFFFT 对照（隔离"PEC 阶梯表面提取"这一项）

问题（专家问题 1 的核心分叉）：
  3.55 dB 的"pipeline floor"，究竟来自
    (a) 约定 / 归一化 / 远场积分公式实现的 bug，还是
    (b) PEC 阶梯表面提取 + 场在离面体素中心采样（k·h/2≈0.98 rad）的离散化误差？

方法（不需要重跑 FEKO，只用已有 h5 数据）：
  在近场网格外边界取一个**闭合盒面**（严格包住飞机），用 FEKO 的**散射场**做外域等效原理：
      J = n̂ × H_s ,  M = −n̂ × E_s      （n̂ 为盒面外法向）
      N = ∮ J e^{jkr̂·r'} dS ,  L = ∮ M e^{jkr̂·r'} dS
      E_ff = (jk/4π)[ −η₀ (N − (N·r̂)r̂) + r̂ × L ]     （Balanis 12-10，e^{jωt}）
  为什么这条链路是"干净的"：
    - 盒面上每个场样本都是**网格点原位值**：无需插值、无阶梯化、无法向估计、无 h/2 偏移；
    - 梯形求积对"带限场（Helmholtz ⇒ 空间谱落在 |ξ|=k 球面）× 振荡核 e^{jkr̂·r'}"
      是**谱精度**：混叠需 |2π/h − k| 落在场谱内，而 2π/h−k = 201.1−62.9 = 138.2 > k = 62.9，
      故无混叠，求积误差可忽略；
    - 盒面完整包覆散射源（PEC 电流），外域等效原理严格成立。
  ⇒ 这条链路剩下的误差**只可能来自约定 / 归一化 / 公式符号**。

判据：
  A) 盒面 |ΔRCS| 中位 ≈0.5 dB  → 约定/归一化/积分实现全对 ⇒ 3.55 dB 来自表面提取与采样（离散化）
  B) 盒面仍 ≈3.5 dB           → 约定/归一化有 bug ⇒ 再用双系数最小二乘定位

同时输出：
  1) 两种相对符号（+η₀N⊥+r̂×L 与 −η₀N⊥+r̂×L）的复相关 —— 定位 180° 相位反转
  2) 只取 J 项 / 只取 M 项的复相关与 RCS —— 说明盒面上 M 不可省（盒面不是 PEC）
  3) 双参数最小二乘：E_h5 ≈ (jk/4π)[a·η₀N⊥ + b·(r̂×L)]，理想 a=−1, b=+1
  4) h5 自带 E_ff 与自家 rcs 的一致性（独立核对 FEKO 远场归一化）
  5) 盒面内缩 0/1/2 体素 —— 求积与盒面位置的自检（精确应几乎不变）
  6) 同一批角度上重算"阶梯面管线"floor，与盒面结果并排

产出：results/_diag_nffft_audit4.json

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_audit4.py --nangles 20
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
from fno_f16_3d_p4_nffft import direction_grid, nffft, surface_parts, sample_surface_field
from _diag_nffft_audit import surface_parts_full, rcs_stats

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi   # 376.7303 Ω


# ============================================================
# 一、闭合盒面：网格点原位 + 梯形权重面积 + 外法向
# ============================================================
def box_surface(gx, gy, gz, h, inset=0):
    """在以 (gx,gy,gz) 为轴的网格上取闭合长方体表面。
    inset=k 表示各方向内缩 k 层体素（仍须包住目标）。
    返回 ijk(N,3) 网格索引, pts(N,3) 物理坐标, dS(N,3)=外法向·dA, face_id(N,)
    梯形权重：面内某轴上端点权重 0.5，内部 1.0（矩形域梯形求积，谱精度）。"""
    Nx, Ny, Nz = len(gx), len(gy), len(gz)
    jx = np.arange(inset, Nx - inset)
    jy = np.arange(inset, Ny - inset)
    jz = np.arange(inset, Nz - inset)

    def trap_w(idx, n):
        w = np.ones(len(idx))
        w[idx == 0] = 0.5
        w[idx == n - 1] = 0.5
        return w

    wy = trap_w(jy, Ny)
    wz = trap_w(jz, Nz)
    wx = trap_w(jx, Nx)

    ijk, pts, dS, fid = [], [], [], []
    axis_vals = [gx, gy, gz]

    # 6 个面：(固定轴 ax, 固定层 ix, 外法向符号 sg)
    faces = [(0, jx[0], -1), (0, jx[-1], +1),
             (1, jy[0], -1), (1, jy[-1], +1),
             (2, jz[0], -1), (2, jz[-1], +1)]
    for k, (ax, layer, sg) in enumerate(faces):
        u_ax, v_ax = [a for a in (0, 1, 2) if a != ax]
        u_idx = [jx, jy, jz][u_ax]
        v_idx = [jx, jy, jz][v_ax]
        w_u = [wx, wy, wz][u_ax]
        w_v = [wx, wy, wz][v_ax]
        U, V = np.meshgrid(np.arange(len(u_idx)), np.arange(len(v_idx)), indexing="ij")
        WU, WV = np.meshgrid(w_u, w_v, indexing="ij")
        idx = np.zeros((U.size, 3), dtype=np.int64)
        idx[:, ax] = layer
        idx[:, u_ax] = u_idx[U.ravel()]
        idx[:, v_ax] = v_idx[V.ravel()]
        p = np.stack([axis_vals[a][idx[:, a]] for a in range(3)], axis=1).astype(np.float64)
        n = np.zeros((idx.shape[0], 3)); n[:, ax] = float(sg)
        ds = n * (h * h * (WU * WV).ravel())[:, None]
        ijk.append(idx); pts.append(p); dS.append(ds)
        fid.append(np.full(idx.shape[0], k, dtype=np.int64))
    return (np.concatenate(ijk), np.concatenate(pts),
            np.concatenate(dS), np.concatenate(fid))


def nffft_NL(J, Mm, pts, dS, rhat, k, chunk=2048):
    """分块累加 N=∮J e^{jkr̂·r'}dS, L=∮M e^{jkr̂·r'}dS。返回 N⊥ (Ndir,3), r̂×L (Ndir,3)。"""
    dA = np.linalg.norm(dS, axis=1)
    N = np.zeros((rhat.shape[0], 3), dtype=np.complex128)
    L = np.zeros_like(N)
    for s in range(0, len(pts), chunk):
        e = min(s + chunk, len(pts))
        ph = np.exp(1j * k * (pts[s:e] @ rhat.T))          # (c,Ndir)
        N += ph.T @ (J[s:e] * dA[s:e, None])
        L += ph.T @ (Mm[s:e] * dA[s:e, None])
    Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
    return Nperp, np.cross(rhat, L)


# ============================================================
# 二、场对比统计（复相关 / 最佳标定因子 / RCS）
# ============================================================
def cmp_vs_h5(E_est, E_h5, rcs_true, e0mag, shape, thr_frac=0.05):
    """E_est,E_h5:(Ndir,2) complex（θ,φ 分量）。返回复相关、|c|、arg(c)、RCS 统计。"""
    out = {}
    mag = np.sqrt((np.abs(E_h5) ** 2).sum(axis=1))
    thr = thr_frac * mag.max()
    m = mag > thr
    out["n_dir_used"] = int(m.sum())
    ec = E_est[m].ravel()
    eh = E_h5[m].ravel()
    denom = np.sqrt(np.vdot(ec, ec).real * np.vdot(eh, eh).real)
    out["rho_complex"] = float(np.abs(np.vdot(eh, ec)) / max(denom, 1e-300))
    c = np.vdot(eh, ec) / max(np.vdot(eh, eh).real, 1e-300)
    out["c_abs"] = float(np.abs(c))
    out["c_arg_deg"] = float(np.rad2deg(np.angle(c)))
    rcs_est = 4.0 * np.pi * (np.abs(E_est) ** 2).sum(axis=1) / e0mag ** 2
    st = rcs_stats(rcs_est.reshape(shape), rcs_true)
    out["masked_median_dB"] = st["masked"]["median_dB"]
    out["masked_p90_dB"] = st["masked"]["p90_dB"]
    out["masked_corr_dB"] = st["masked"]["corr_dB"]
    out["all_median_dB"] = st["all"]["median_dB"]
    out["all_corr_dB"] = st["all"]["corr_dB"]
    out["null_frac"] = st["null_frac"]
    return out


def ls_two_basis(B1, B2, E_h5, mask):
    """最小二乘 E_h5 ≈ a·B1 + b·B2（复数 a,b）。<u,v>=Σu·conj(v)。
    B1,B2,E_h5:(Ndir,3)->取掩膜后展平。"""
    def flat(X):
        return X[mask].ravel()
    u1, u2, eh = flat(B1), flat(B2), flat(E_h5)
    A = np.array([[np.vdot(u1, u1), np.vdot(u1, u2)],
                  [np.vdot(u2, u1), np.vdot(u2, u2)]], dtype=np.complex128)
    rhs = np.array([np.vdot(eh, u1), np.vdot(eh, u2)], dtype=np.complex128)
    ab = np.linalg.solve(A, rhs)
    res = eh - ab[0] * u1 - ab[1] * u2
    rel = float(np.linalg.norm(res) / max(np.linalg.norm(eh), 1e-300))
    return ab, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=20)
    ap.add_argument("--insets", type=str, default="0,1,2")
    ap.add_argument("--also-surface", type=int, default=1,
                    help="1=同一批角度重算阶梯面管线 floor 作并排对照")
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        E_ff_h5 = f["E_ff"][:]                      # (468,37,73,2) complex
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0]); h = float(gx[1] - gx[0])
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    Ndir = rhat.shape[0]
    print(f"k={k:.4f} rad/m  h={h:.5f}  λ/h={2*np.pi/k/h:.2f}  "
          f"2π/h−k={2*np.pi/h-k:.1f} (须>k={k:.1f} 才无混叠)  Ndir={Ndir}")

    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    print(f"角度 {len(idx_list)}: {idx_list}")

    # ---------- 盒面构造 ----------
    insets = [int(x) for x in args.insets.split(",")]
    boxes = {}
    for ins in insets:
        ijk, bpts, bdS, fid = box_surface(gx, gy, gz, h, inset=ins)
        # 自检：盒面须严格包住 eps>1.5 的金属体素
        mi = np.argwhere(eps > 1.5)
        mc = np.stack([gx[mi[:, 0]], gy[mi[:, 1]], gz[mi[:, 2]]], axis=1)
        fmin = np.array([gx[ins], gy[ins], gz[ins]])
        fmax = np.array([gx[-1 - ins], gy[-1 - ins], gz[-1 - ins]])
        enclosed = bool((mc.min(axis=0) > fmin).all() and (mc.max(axis=0) < fmax).all())
        boxes[ins] = dict(ijk=ijk, pts=bpts, dS=bdS, fused=bdS.sum(), enclosed=enclosed)
        area = float(np.linalg.norm(bdS, axis=1).sum())   # 闭合面矢量和恒为 0，取 |dS| 之和
        print(f"  box inset={ins}: {len(bpts)} 面元  内包覆={enclosed}  "
              f"表面积={area:.4f} m²  盒界 [{fmin}]..[{fmax}]")
        n = bdS / np.linalg.norm(bdS, axis=1, keepdims=True)
        assert np.allclose(np.abs(n).sum(axis=1), 1.0), "盒面法向必须轴向"

    # ---------- 阶梯面管线（对照）----------
    if args.also_surface:
        idx_air, _, rsurf, dS_s = surface_parts_full(eps, gx, gy, gz, h)
        print(f"  阶梯面元 {len(rsurf)}")

    # ---------- 主循环 ----------
    A = {ins: {"pipe": [], "corr": [], "Jonly": [], "Monly": [], "fit": []}
         for ins in insets}
    surf = []
    h5self = []
    t0 = time.time()

    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(idx_list):
            Es_g = f["E_scat"][i]            # (64,48,32,3) complex64
            Hs_g = f["H_scat"][i]
            e0mag = float(np.linalg.norm(e0[i]))
            Eh2 = E_ff_h5[i].reshape(-1, 2)
            tgt = rcs_true[i]
            mag = np.sqrt((np.abs(Eh2) ** 2).sum(axis=1))
            mask = mag > 0.05 * mag.max()

            # h5 自检：E_ff 与自家 rcs 的一致性（E_est 直接取 E_ff，应得 0 dB / corr=1）
            h5self.append(cmp_vs_h5(Eh2, Eh2, tgt, e0mag, shape))

            for ins in insets:
                bx = boxes[ins]
                ik = bx["ijk"]
                Es = Es_g[ik[:, 0], ik[:, 1], ik[:, 2]]
                Hs = Hs_g[ik[:, 0], ik[:, 1], ik[:, 2]]
                n = bx["dS"] / np.linalg.norm(bx["dS"], axis=1, keepdims=True)
                J = np.cross(n, Hs)
                Mm = -np.cross(n, Es)
                Nperp, rcL = nffft_NL(J, Mm, bx["pts"], bx["dS"], rhat, k)
                coef = 1j * k / (4.0 * np.pi)
                E_pipe = coef * (ETA0 * Nperp + rcL)        # 现有管线相对符号
                E_corr = coef * (-ETA0 * Nperp + rcL)       # Balanis 正确相对符号
                E_J = coef * (ETA0 * Nperp)                 # 只 J
                E_M = coef * rcL                            # 只 M

                def to2(E):
                    return np.stack([(E * th).sum(axis=1), (E * ph).sum(axis=1)], axis=1)

                A[ins]["pipe"].append(cmp_vs_h5(to2(E_pipe), Eh2, tgt, e0mag, shape))
                A[ins]["corr"].append(cmp_vs_h5(to2(E_corr), Eh2, tgt, e0mag, shape))
                A[ins]["Jonly"].append(cmp_vs_h5(to2(E_J), Eh2, tgt, e0mag, shape))
                A[ins]["Monly"].append(cmp_vs_h5(to2(E_M), Eh2, tgt, e0mag, shape))
                ab, rel = ls_two_basis(coef * ETA0 * Nperp, coef * rcL,
                                       Eh2[:, 0:1] * th + Eh2[:, 1:2] * ph, mask)
                A[ins]["fit"].append({"a": complex(ab[0]), "b": complex(ab[1]), "rel_res": rel})

            # 阶梯面管线
            if args.also_surface:
                from fno_f16_3d_p4_nffft import build_tot_fields
                Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                              f["E_scat"], f["H_scat"], i)
                H_s = sample_surface_field(Htot, idx_air)
                n = dS_s / np.linalg.norm(dS_s, axis=1, keepdims=True)
                Js = np.cross(n, H_s)
                E_ff = nffft(Js, np.zeros_like(Js), rsurf, dS_s, rhat, k)
                E2 = np.stack([(E_ff * th).sum(axis=1), (E_ff * ph).sum(axis=1)], axis=1)
                surf.append(cmp_vs_h5(E2, Eh2, tgt, e0mag, shape))

            if (a + 1) % 5 == 0:
                print(f"  [{a+1}/{len(idx_list)}] {time.time()-t0:.0f}s", flush=True)

    # ---------- 汇总 ----------
    def agg(lst, keys):
        d = {kk: float(np.median([x[kk] for x in lst])) for kk in keys}
        ca = np.array([x["c_arg_deg"] for x in lst])
        d["c_arg_circ_deg"] = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(ca))))))
        d["null_frac"] = float(np.median([x["null_frac"] for x in lst]))
        return d

    KEYS = ["rho_complex", "c_abs", "masked_median_dB", "masked_p90_dB",
            "masked_corr_dB", "all_median_dB", "all_corr_dB"]
    summary = {"k": k, "h_m": h, "lambda_over_h": float(2 * np.pi / k / h),
               "n_dir": int(Ndir), "n_angles": len(idx_list),
               "angles": [int(x) for x in idx_list],
               "dir_used_frac": float(np.mean([x["n_dir_used"] for x in A[insets[0]]["pipe"]]) / Ndir),
               "h5_E_ff_vs_rcs": agg(h5self, ["masked_median_dB", "masked_p90_dB", "masked_corr_dB"]),
               "box": {}, "surface_pipeline": None, "ls_fit": {}}

    for ins in insets:
        d = A[ins]
        summary["box"][f"inset{ins}"] = {
            "n_faces": int(len(boxes[ins]["pts"])),
            "enclosed": boxes[ins]["enclosed"],
            "pipe_plusJ": agg(d["pipe"], KEYS),
            "correct_minusJ": agg(d["corr"], KEYS),
            "J_only": agg(d["Jonly"], KEYS),
            "M_only": agg(d["Monly"], KEYS),
        }
        ab_a = np.array([x["a"] for x in d["fit"]])
        ab_b = np.array([x["b"] for x in d["fit"]])
        summary["ls_fit"][f"inset{ins}"] = {
            "a_med_abs": float(np.median(np.abs(ab_a))),
            "a_med_arg_deg": float(np.median(np.rad2deg(np.angle(ab_a)))),
            "b_med_abs": float(np.median(np.abs(ab_b))),
            "b_med_arg_deg": float(np.median(np.rad2deg(np.angle(ab_b)))),
            "rel_res_med": float(np.median([x["rel_res"] for x in d["fit"]])),
        }
    if surf:
        summary["surface_pipeline"] = agg(surf, KEYS)

    out_path = os.path.join(RESULT_DIR, "_diag_nffft_audit4.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)

    # ---------- 打印 ----------
    print("\n===== 0) h5 自检：E_ff 与自家 rcs 是否自洽 =====")
    s = summary["h5_E_ff_vs_rcs"]
    print(f"  掩膜后中位 {s['masked_median_dB']:.3f} dB / P90 {s['masked_p90_dB']:.2f} dB / "
          f"corr {s['masked_corr_dB']:.4f}   （≈0 dB 且 corr≈1 才说明 FEKO 远场归一化自洽）")

    print("\n===== 1) 盒面等效原理（无插值/无阶梯化/无离面偏移）=====")
    print(f"  参与比较方向占比 {summary['dir_used_frac']*100:.1f}%")
    hdr = f"  {'inset':>5} {'公式':<16} {'复相关':>7} {'|c|':>6} {'arg c':>8} " \
          f"{'中位dB':>7} {'P90dB':>7} {'corr_dB':>8}"
    print(hdr)
    for ins in insets:
        b = summary["box"][f"inset{ins}"]
        for nm, lab in (("pipe_plusJ", "+η₀N⊥+r̂×L (管线)"),
                        ("correct_minusJ", "−η₀N⊥+r̂×L (Balanis)"),
                        ("J_only", "仅 J 项"),
                        ("M_only", "仅 M 项")):
            v = b[nm]
            print(f"  {ins:>5} {lab:<16} {v['rho_complex']:>7.3f} {v['c_abs']:>6.3f} "
                  f"{v['c_arg_circ_deg']:>8.2f} {v['masked_median_dB']:>7.3f} "
                  f"{v['masked_p90_dB']:>7.2f} {v['masked_corr_dB']:>8.3f}")

    print("\n===== 2) 双系数最小二乘 E_h5 ≈ (jk/4π)[a·η₀N⊥ + b·(r̂×L)]  (理想 a=−1, b=+1) =====")
    for ins in insets:
        s = summary["ls_fit"][f"inset{ins}"]
        print(f"  inset{ins}: a={s['a_med_abs']:.3f}∠{s['a_med_arg_deg']:+.1f}°  "
              f"b={s['b_med_abs']:.3f}∠{s['b_med_arg_deg']:+.1f}°  残差 {s['rel_res_med']:.4f}")

    if surf:
        s = summary["surface_pipeline"]
        print("\n===== 3) 并排对照：阶梯面管线（同一批角度）=====")
        print(f"  复相关 {s['rho_complex']:.3f} | |c| {s['c_abs']:.3f} ∠{s['c_arg_circ_deg']:+.1f}° | "
              f"中位 {s['masked_median_dB']:.3f} dB / P90 {s['masked_p90_dB']:.2f} dB / "
              f"corr {s['masked_corr_dB']:.3f}")

    print(f"\n已存: {out_path}")


if __name__ == "__main__":
    main()
