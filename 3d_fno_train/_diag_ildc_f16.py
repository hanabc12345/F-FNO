# -*- coding: utf-8 -*-
"""
_diag_ildc_f16.py — 在 F-16 真网格（STL 三角面）上接入 ILDC 边缘绕射
================================================================================
两个模式：

  dist （默认，秒级）：只做几何/风险量化，不跑远场
    · STL 绕向与外法向一致性自检（决定 l̂ / φ 的定向是否正确）
    · 棱边提取统计：总数 / 开放边 / 内劈半角 α 的分布（按条数与按长度加权）
    · 每个入射角下"几何可用（keep）+ 实际可见（DDA 遮挡）"的棱边占比，按 α 分带
    · 结论：F-16 上真正参与绕射的棱边落在哪个 α 区间 —— 用于对照 Gordon 闸门
      （α=0 通路已验证，α≠0 通路在参考公式层面缺独立基准）

  run --angles N：端到端 RCS 对比
    mesh_po               （复现 exp_po_mesh.py 的基线）
    mesh_po + ILDC fringe （相干叠加条纹波）
    并与 FEKO 真值比 med / P90 / ρ；同时对第 1 个角度做 α 分带消融，
    量化"未验证区间（α 大）"对结果的贡献。

物理约定（与全项目一致）：
  · ki = khat[i] 入射传播方向；ei = 入射电场矢量（canon_phase 后，幅值 e0m）
  · ILDC 模块约定 e_i = −ki（由目标指向源）、e_s = r̂（观察方向）
  · 条纹波远场 F 满足 E_s = F·e0m（e^{ikr}/r 已剥离），与 PO 的 NFFFT 同量纲，
    故 E_total = E_po + E_fringe 直接相干相加（σ = 4π|p̂·F|²，Gordon 闸门已验证）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_ildc_f16.py
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_ildc_f16.py run --angles 30
"""

import os
import sys
import json
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M                                  # noqa: E402
from fno_f16_3d_p4_nffft import direction_grid          # noqa: E402
from exp_po_mesh import (load_mesh, mesh_outside_voxel,  # noqa: E402
                         nffft_from, rcs_of, cstats, ETA0, H, STL)
from exp_po_locality import ray_occlusion                # noqa: E402
from _exp_common import _rcs_stats                      # noqa: E402
from po_patch_data import canon_phase                   # noqa: E402
from _diag_nffft_audit2 import read_stl                 # noqa: E402
import ildc_mesh as I                                   # noqa: E402

RESULT_DIR = M.RESULT_DIR
BANDS = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]

# 棱边筛选：α 上限（度）。α→90° 表示两面几乎共面（三角网格的离散化伪棱）。
ALPHA_MAX_DEG = 89.0

# ★ 2026-09 结论：α 筛选 **不是** 爆量的根本解法。真正的根因是 Mitzner 非一致
# （non-uniform）绕射系数 f/g 在 **几何光学边界**（V→-1，即 ψ→π：入射/散射掠过某一面）
# 上有未正则化的极点，cl-rcs 参考实现自己在 ildc.lisp 里就写着
#     ;; ((float-equal V-big -1) ;; f goes to infinity as V -> -1
#     ;;  FIXME what to do??
# 我们逐行忠实移植，因此同样发散。实测（_tmp_ildc_dir.py / _tmp_ildc_edge.py /
# _tmp_ildc_pole.py）：单条棱的 |c⊥| 可达 1e10，使整个方向图被 1~5 个方向支配。
# 按 α 收紧只降低"命中概率"而非消除：角 116/233 上 α<15° 子集仍给出
# |E_fr|/|E_po| = 2.1e6 / 1.1e4 ⇒ 锐棱同样会撞上极点。
# 端到端（5 角度，_diag_ildc_f16.py run --alphas 89,30,15）：
#   mesh_po        med  3.24  P90  9.42  ρ 0.925  |c| 0.952
#   ildc α<89      med 40.73  P90 57.58  ρ 0.013  |c| 575.6
#   ildc α<30      med 17.96  P90 33.62  ρ 0.084  |c| 3.38
#   ildc α<15      med 14.86  P90 29.96  ρ 0.198  |c| 2.70
# ⇒ 预注册判据（中位 ≤2.8、P90 ≤7、ρ 上升）全部 FAIL。要往下走得先补
#   **一致化（UTD 过渡函数）正则化**，而不是继续调棱边筛选阈值。


# ============================================================
# 一、几何
# ============================================================

def load_tri_consistent(stl):
    """读 STL 并保证"绕向 CCW-朝外"与 nvm 一致（build_edges 的半边定向依赖它）。"""
    tri, _ = read_stl(stl)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    av = 0.5 * np.cross(v1 - v0, v2 - v0)
    dA = np.linalg.norm(av, axis=1)
    cen = tri.mean(axis=1)
    svol = float((np.cross(v0, v1) * v2).sum() / 6.0)
    nvm = av / dA[:, None]
    if svol < 0:
        nvm = -nvm
    agree = float(((av / dA[:, None]) * nvm).sum(1).mean())
    flipped = agree < 0
    if flipped:
        tri = tri[:, [0, 2, 1], :]
    print(f"  STL {len(tri)} 面  面积 {dA.sum():.4f} m²  有符号体积 {svol:+.5f} m³")
    print(f"  绕向×外法向一致度 {agree:+.3f} → {'已反转绕向' if flipped else '无需反转'}")
    return tri, cen, nvm, dA


def build(E, ki, ei):
    """返回 (e_i, p_hat, P, vis)：ILDC 预计算 + 棱边可见性（DDA 遮挡）。
    另把"共面棱剔除"（α ≥ ALPHA_MAX_DEG）并入 P["keep"]，后续口径统一。"""
    e_i = -ki
    p_hat = ei / np.linalg.norm(ei)
    P = I.prepare_edges(E, e_i, p_hat)
    P["keep"] &= np.degrees(E["alpha"]) < ALPHA_MAX_DEG
    mid = E["r"] + 0.5 * E["Cvec"]
    q, _ = mesh_outside_voxel(mid, P["n"], G0, METAL)
    vis = ~ray_occlusion(q, -ki, METAL)
    return e_i, p_hat, P, vis


def rcs_pattern(Evec, e0m, th, ph, shape):
    return rcs_of(Evec, th, ph, e0m, shape)


# ============================================================
# 二、模式 dist：几何与风险量化
# ============================================================

def mode_dist(n_angle=4):
    tri, cen, nvm, dA = load_tri_consistent(STL)
    E = I.build_edges(tri, nvm)
    alpha = np.degrees(E["alpha"])
    Cn = E["Cn"]
    closed = ~E["open"]
    print(f"\n  棱边总数 {len(alpha)}  开放边 {int(E['open'].sum())}  "
          f"闭合（有对偶面）{int(closed.sum())}")
    print(f"  α 分位（闭合棱，度）：min {alpha[closed].min():.2f}  "
          f"p10 {np.percentile(alpha[closed], 10):.2f}  "
          f"中位 {np.median(alpha[closed]):.2f}  "
          f"p90 {np.percentile(alpha[closed], 90):.2f}  max {alpha[closed].max():.2f}")
    print(f"  ν 范围（闭合棱）：{E['nu'][closed].min():.3f} ~ {E['nu'][closed].max():.3f}"
          f"   （ν=0.5 对应 α=0 刀口，ν=1 对应 α=90° 平面）")
    print(f"\n  {'α 区间(°)':<14}{'条数':>9}{'占比':>8}{'长度占比':>10}{'Σ长度(m)':>11}")
    for lo, hi in BANDS:
        m = closed & (alpha >= lo) & (alpha < hi + (1e-9 if hi == 90 else 0))
        print(f"  {f'[{lo:g},{hi:g}]':<14}{int(m.sum()):>9}{m.mean()*100:>7.1f}%"
              f"{Cn[m].sum()/Cn[closed].sum()*100:>9.1f}%{Cn[m].sum():>11.2f}")
    print(f"\n  近 90° 细分（共面棱剔除阈值 {ALPHA_MAX_DEG:g}°）：")
    for lo, hi in [(85, 89), (89, 89.5), (89.5, 89.9), (89.9, 90.001)]:
        m = closed & (alpha >= lo) & (alpha < hi)
        print(f"    α∈[{lo:g},{hi:g})：{int(m.sum()):>6} 条，Σ长度 {Cn[m].sum():>8.2f} m")
    mcut = closed & (alpha >= ALPHA_MAX_DEG)
    print(f"  被剔除 {int(mcut.sum())} 条（占闭合棱 {mcut.sum()/closed.sum()*100:.1f}%，"
          f"长度占比 {Cn[mcut].sum()/Cn[closed].sum()*100:.1f}%）")

    angles, e0, khat, beta = M.build_incidence_table()
    aidx = np.linspace(0, len(angles) - 1, n_angle).astype(int)
    print(f"\n  {'角度 idx':>8}{'keep 条数':>11}{'可见 条数':>11}{'占比':>8}"
          f"   分带可见占比 [0-30|30-60|60-89]")
    per_band = []
    bands_kept = [(lo, hi, closed & (alpha >= lo) & (alpha < min(hi, ALPHA_MAX_DEG)))
                  for lo, hi in BANDS]
    for i in aidx:
        ki = khat[i].astype(np.float64)
        ei = e0[i].astype(np.complex128)
        ph0 = canon_phase(ei)
        ei = ei * ph0
        _e_i, _p, P, vis = build(E, ki, ei)
        act = P["keep"] & vis
        fr = [float(act[m].mean()) if m.any() else 0.0 for _lo, _hi, m in bands_kept]
        per_band.append(fr)
        print(f"  {i:>8}{int(P['keep'].sum()):>11}{int(act.sum()):>11}"
              f"{act.mean()*100:>7.1f}%   {fr[0]*100:5.1f}% | {fr[1]*100:5.1f}% | "
              f"{fr[2]*100:5.1f}%")
    return {"n_edges": int(len(alpha)), "n_open": int(E["open"].sum()),
            "alpha_closed_median_deg": float(np.median(alpha[closed])),
            "alpha_closed_p10": float(np.percentile(alpha[closed], 10)),
            "alpha_closed_p90": float(np.percentile(alpha[closed], 90)),
            "vis_frac_by_band": per_band}


# ============================================================
# 三、模式 run：端到端 RCS
# ============================================================

def mode_run(n_angle, alphas=(89.0,), dmaxs=()):
    tri, cen, nvm, dAm = load_tri_consistent(STL)
    E = I.build_edges(tri, nvm)
    alpha = np.degrees(E["alpha"])
    Cn = E["Cn"]
    print(f"  棱边 {len(alpha)}  开放 {int(E['open'].sum())}  "
          f"Σ长度 {Cn.sum():.2f} m")

    q_out, _ = mesh_outside_voxel(cen, nvm, G0, METAL)
    angles, e0, khat, beta = M.build_incidence_table()
    aidx = np.linspace(0, len(angles) - 1, n_angle).astype(int)
    with h5py.File(M.H5, "r") as f:
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]
        ff_phi = f["ff_phi"][:]
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    k = float(beta[0])

    # (变体名, α 上限, d_max)：d_max 为极点截断的诊断开关（见 ildc_mesh.fringe_far_field）
    variants = [(f"ildc_a{t:g}", t, None) for t in alphas] + \
               [(f"ildc_cap{d:g}", ALPHA_MAX_DEG, d) for d in dmaxs]
    acc = {v: {"rcs": [], "c": []} for v in ["mesh_po"] + [x[0] for x in variants]}
    t0 = time.time()
    for a, i in enumerate(aidx):
        ki = khat[i].astype(np.float64)
        ei = e0[i].astype(np.complex128)
        e0m = float(np.linalg.norm(ei))
        ph0 = canon_phase(ei)
        ei = ei * ph0
        e_i, p_hat, P, vis = build(E, ki, ei)

        # ---- PO（与 exp_po_mesh.py 完全同口径）----
        lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, METAL))
        psi_tri = np.exp(-1j * beta[i] * (cen @ ki))
        e0vec = np.cross(ki, ei)
        Jpo = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi_tri[:, None]
        Jpo[~lit_m] = 0.0
        E_po = nffft_from(Jpo, dAm, cen, rhat, k)

        # ---- ILDC 条纹波 ----
        t1 = time.time()
        Eset = {"mesh_po": E_po}
        amps = []
        for nm, thr, dm in variants:
            E_fr = I.fringe_far_field(E, P, k, e_i, rhat, p_hat,
                                      subset=vis & (alpha < thr), d_max=dm) * e0m
            Eset[nm] = E_po + E_fr
            amps.append(float(np.sqrt((np.abs(E_fr) ** 2).mean())
                              / max(np.sqrt((np.abs(E_po) ** 2).mean()), 1e-300)))
        tf = time.time() - t1

        Eh2 = f_E_ff(i)
        for v in acc:
            Ev = Eset[v]
            e2 = np.stack([(Ev * th).sum(1), (Ev * ph).sum(1)], axis=1)
            acc[v]["rcs"].append(rcs_pattern(Ev, e0m, th, ph, shape))
            acc[v]["c"].append(cstats(e2, Eh2))
        rat = "  ".join(f"{nm}: {x:.3g}" for (nm, _t, _d), x
                        in zip(variants, amps))
        print(f"    {a+1}/{len(aidx)}  角 {i}  fringe {tf:.1f}s  "
              f"幅值比 {rat}  (总 {time.time()-t0:.0f}s)", flush=True)

    print("\n[ILDC on F-16] 远场 vs FEKO 真值   med / P90 / corr | 复ρ / |c| ∠arg")
    res = {"n_angle": len(aidx), "angles": aidx.tolist(), "k": k,
           "alphas": list(alphas), "dmaxs": list(dmaxs),
           "n_edges": int(len(alpha)), "sum_edge_len_m": float(Cn.sum()),
           "variants": {}}
    for v in acc:
        st = _rcs_stats(np.stack(acc[v]["rcs"]), rcs_true[aidx])
        cc = acc[v]["c"]
        ca = np.array([x["c_arg_deg"] for x in cc])
        row = {**st,
               "rho_complex": float(np.median([x["rho_complex"] for x in cc])),
               "c_abs": float(np.median([x["c_abs"] for x in cc])),
               "c_arg_deg": float(np.rad2deg(np.angle(
                   np.mean(np.exp(1j * np.deg2rad(ca))))))}
        res["variants"][v] = row
        print(f"  {v:<14} med {row['rcs_dB_err_median']:6.2f}  "
              f"P90 {row['rcs_dB_err_p90']:6.2f}  corr {row['rcs_corr_median']:.3f}  | "
              f"ρ {row['rho_complex']:.3f}  |c| {row['c_abs']:10.3f} ∠{row['c_arg_deg']:+7.1f}°")

    # 逐角度（基线被极点毁掉的角会掩盖其余角的真实表现）
    print("\n  逐角度：med dB / ρ")
    print("    " + f"{'角度':>5}" + "".join(f"{v:>21}" for v in acc))
    per = {}
    for j, i in enumerate(aidx):
        cells, rec = [], {}
        for v in acc:
            r = _rcs_stats(np.stack([acc[v]["rcs"][j]]), rcs_true[[i]])
            rho = acc[v]["c"][j]["rho_complex"]
            rec[v] = {"med": float(r["rcs_dB_err_median"]),
                      "p90": float(r["rcs_dB_err_p90"]), "rho": float(rho)}
            cells.append(f"{r['rcs_dB_err_median']:>12.2f}/{rho:<8.3f}")
        per[int(i)] = rec
        print(f"    {i:>5}" + "".join(cells))
    res["per_angle"] = per

    m0 = res["variants"]["mesh_po"]["rcs_dB_err_median"]
    print("\n  Δ中位（正=ILDC 改善）：")
    for v in acc:
        if v == "mesh_po":
            continue
        m1 = res["variants"][v]["rcs_dB_err_median"]
        res[f"delta_{v}"] = float(m0 - m1)
        print(f"    {v:<14} {m0 - m1:+.2f} dB")

    jp = os.path.join(RESULT_DIR, "_diag_ildc_f16.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"已存 {jp}")


_F_H5 = {"f": None}


def f_E_ff(i):
    return _F_H5["f"]["E_ff"][i].reshape(-1, 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="dist", choices=["dist", "run"])
    ap.add_argument("--angles", type=int, default=30)
    ap.add_argument("--alphas", default="",
                    help="逗号分隔的内劈半角上限（度），逐个给一个 ILDC 变体；空串=跳过")
    ap.add_argument("--dmax", default="",
                    help="逗号分隔的极点截断档位（诊断用，见 ildc_mesh.fringe_far_field）")
    args = ap.parse_args()
    alphas = tuple(float(s) for s in args.alphas.split(",") if s.strip())
    dmaxs = tuple(float(s) for s in args.dmax.split(",") if s.strip())

    with h5py.File(M.H5, "r") as f:
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        eps = f["eps_field"][:]
        _F_H5["f"] = f if args.mode == "run" else None
        if args.mode == "run":
            print("=== ILDC on F-16 真网格：端到端 RCS ===")
            f["rcs"].shape        # 保持文件打开
        else:
            print("=== ILDC on F-16：几何与风险量化 ===")
        G0 = np.array([gx[0], gy[0], gz[0]])
        METAL = eps > 1.5
        out = (mode_dist() if args.mode == "dist"
               else mode_run(args.angles, alphas, dmaxs))
