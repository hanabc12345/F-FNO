# -*- coding: utf-8 -*-
"""
_diag_staircase_order.py — C1：几何阶梯的收敛阶（h/2、h/4）+ 局部细网格可行性评估
================================================================================
背景（item 11 + item 12 锁定）
  「阶梯体素面 → 真 STL 面」的几何升级只值 **0.58 dB**，且已证明
    · 面积维度彻底无关（A2-1，+0.033 dB vs 理论 +0.795）
    · 法向量化单修反而有害（A2-2，−0.092 dB，但 ρ/|c| 改善）
    · 主因 = **位置/曲面表示**（A2-3，0.707 dB 位置单因素差）
  ⇒ 悬而未决的唯一实质问题：**位置误差随 h 的幂次是多少？**

预注册判据（item 12 遗留①原文）
  · med 随 h 呈 **O(h)** 下降 ⇒ 位置相位是主控；
  · med 随 h 呈 **O(h²)** 下降 ⇒ 另有机制。

方法（**不重跑 FEKO**，关键简化）
  FEKO 真值远场 rcs_true / E_ff 是**几何无关**的（同一份 f16_refined.stl 跑的满波解）。
  而解析 PO 电流 J = 2n̂×H_inc 只需**表面几何 + 解析入射场**，不需要近场数据。
  ⇒ 把同一份 STL 在**不同 h** 上重新体素化，即可得到阶梯面的 h 族，
    对**同一份** FEKO 真值做受控比较。全程零 FEKO 成本。

  体素化口径：自写 z-parity（沿 x/y 网格列做奇偶填充），
    · STL watertight=True / winding=True ⇒ 奇偶填充合法；
    · 与 FEKO eps 网格（h0）逐体素对拍 IoU=0.776（零平移最优）⇒ 对齐正确，
      但两套规则在**表面层**不重合 ⇒ h0 锚点同时给 FEKO 规则与自写规则两条线。

  两个臂（口径逐位复用 `_diag_staircase_split.py` = item 12）
    A  vox_po        阶梯面 + 解析 PO（面积原样）           ← 部署链同类
    B  vox_nrm_true  真法向 + 面积对齐 + 位置仍阶梯         ← **纯位置收敛臂**（h→0 = mesh_po）
  参考 mesh_po（真曲面 + 解析 PO）在 FEKO 网格上算一次（h 无关）。

  位置维度的**独立对照**（γ 扫描，零体素化成本）
    在**固定**的 FEKO 阶梯面元上，把面元位置沿「阶梯 → 真曲面」缩放
        r(γ) = r_true + γ·(r_surf − r_true)，   γ=1 = 原阶梯，γ=0 = 投影到真曲面
    位置偏移 δ ∝ γ ∝ h ⇒ med(γ) 的幂次**直接**给出 p（面元集合、法向、面积全不变）。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_staircase_order.py --angles 12
产出
  results/_diag_staircase_order.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py
import trimesh
from scipy.spatial import cKDTree
from scipy.optimize import least_squares

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M                                          # noqa: E402
from fno_f16_3d_p4_nffft import surface_parts, direction_grid    # noqa: E402
from exp_po_mesh import (load_mesh, mesh_outside_voxel, nffft_from,  # noqa: E402
                         rcs_of, cstats, ETA0, H, STL, RESULT_DIR)
from exp_po_locality import ray_occlusion                        # noqa: E402
from po_patch_data import canon_phase                            # noqa: E402
from _diag_nffft_audit2 import read_stl                          # noqa: E402
from _exp_common import _rcs_stats                               # noqa: E402

GAMMAS = [0.0, 0.15, 0.3, 0.5, 0.7, 1.0]


# ============================================================
# 一、STL → 任意 h 的体素化（z-parity，向量化）
# ============================================================

def voxelize_stl(tri, h, g0, bbmax):
    """闭合 STL → 体素金属掩码（体素中心在实体内部者为金属）。
    沿 axis=2 奇偶填充：对每个 (i,j) 网格列统计曲面穿越数，前缀异或即内部。"""
    nu, nv, nw = [int(np.ceil((bbmax[a] - g0[a]) / h)) + 2 for a in range(3)]
    gu = g0[0] + np.arange(nu) * h
    gv = g0[1] + np.arange(nv) * h
    gw = g0[2] + np.arange(nw) * h
    cnt = np.zeros((nu, nv, nw), dtype=np.int16)
    P = tri[:, :, [0, 1]]
    W = tri[:, :, 2]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    for t in np.flatnonzero(np.abs(area2) > 1e-14):
        p0, p1, p2 = P[t]
        w0, w1, w2 = W[t]
        i0 = max(int(np.ceil((min(p0[0], p1[0], p2[0]) - gu[0]) / h)), 0)
        i1 = min(int(np.floor((max(p0[0], p1[0], p2[0]) - gu[0]) / h)), nu - 1)
        j0 = max(int(np.ceil((min(p0[1], p1[1], p2[1]) - gv[0]) / h)), 0)
        j1 = min(int(np.floor((max(p0[1], p1[1], p2[1]) - gv[0]) / h)), nv - 1)
        if i1 < i0 or j1 < j0:
            continue
        dx = gu[i0:i1 + 1][:, None] - p2[0]
        dy = gv[j0:j1 + 1][None, :] - p2[1]
        f = 1.0 / area2[t]
        a = ((p1[1] - p2[1]) * dx + (p2[0] - p1[0]) * dy) * f
        b = ((p2[1] - p0[1]) * dx + (p0[0] - p2[0]) * dy) * f
        m = (a >= 0) & (b >= 0) & (a + b <= 1)
        if not m.any():
            continue
        zc = a * w0 + b * w1 + (1.0 - a - b) * w2
        kk = np.floor((zc[m] - gw[0]) / h).astype(np.int64) + 1
        ii, jj = np.nonzero(m)
        ok = (kk >= 0) & (kk < nw)
        np.add.at(cnt, (ii[ok] + i0, jj[ok] + j0, kk[ok]), 1)
    return (np.cumsum(cnt, axis=2) % 2).astype(bool), (gu, gv, gw)


def facets_of(metal, axes, h):
    """阶梯体素面 → (owners, rsurf, dAv, nvv)。口径同 fno_f16_3d_p4_nffft.surface_parts。"""
    gu, gv, gw = axes
    owners, rsurf, dSv = surface_parts(metal.astype(np.float32) * 1e6, gu, gv, gw, h=h)
    dAv = np.linalg.norm(dSv, axis=1)
    return owners, rsurf, dAv, dSv / dAv[:, None]


def true_normals(cen, nvm, rsurf, k=8):
    """真曲面法向估计（K 近邻距离加权，口径同 item 12）。"""
    dk, ik = cKDTree(cen).query(rsurf, k=k)
    wk = 1.0 / (dk + 1e-6)
    ns = (wk[:, :, None] * nvm[ik]).sum(axis=1)
    return ns / np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)


# ============================================================
# 二、主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    aidx = (np.arange(len(angles)) if not args.angles
            else np.linspace(0, len(angles) - 1, args.angles).astype(int))
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]
        ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    g0 = np.array([gx[0], gy[0], gz[0]])
    k = float(beta[0])
    metal_feko = eps > 1.5
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    tri, _ = read_stl(STL)
    cen, nvm, dAm = load_mesh(STL)
    bbmax = tri.reshape(-1, 3).max(0)
    A_tri = float(dAm.sum())
    tm = trimesh.Trimesh(vertices=tri.reshape(-1, 3).copy(),
                         faces=np.arange(len(tri) * 3).reshape(-1, 3), process=True)
    print(f"  真 STL 面 {len(cen)} 面  面积 {A_tri:.5f} m²  watertight={tm.is_watertight}")

    # ---- 参考臂：FEKO 网格阶梯面（h0）+ 真曲面 mesh_po + γ 扫描 ----
    q_out, _ = mesh_outside_voxel(cen, nvm, g0, metal_feko)
    owners_f, rsurf_f, dAv_f, nvv_f = facets_of(metal_feko, (gx, gy, gz), H)
    n_true_f = true_normals(cen, nvm, rsurf_f)
    cos_f = np.abs((nvv_f * n_true_f).sum(axis=1))
    dAvf_locn = dAv_f * cos_f * (A_tri / (dAv_f * cos_f).sum())
    r_true_f, d_near_f, _ = tm.nearest.on_surface(rsurf_f)
    print(f"  FEKO 阶梯面元 {len(dAv_f)}  面积 {dAv_f.sum():.5f} m²"
          f"（STL/VOX {A_tri/dAv_f.sum():.5f}）")
    print(f"  FEKO 阶梯面元→真曲面距离 δ：中位 {np.median(d_near_f)*1e3:.2f} mm"
          f"  P90 {np.percentile(d_near_f, 90)*1e3:.2f} mm"
          f"  ⇒ k·δ 中位 {np.median(k*d_near_f):.3f} rad")

    # ---- h 族：自写体素化 ----
    levels = [("2h0", 2 * H), ("h0", H), ("h0/2", H / 2), ("h0/4", H / 4)]
    geo = {}
    for nm, h in levels:
        t0 = time.time()
        metal, axes = voxelize_stl(tri, h, g0, bbmax)
        owners, rsurf, dAv, nvv = facets_of(metal, axes, h)
        n_true = true_normals(cen, nvm, rsurf)
        cosv = np.abs((nvv * n_true).sum(axis=1))
        dloc = dAv * cosv
        cl, d_near, _ = tm.nearest.on_surface(rsurf)
        geo[nm] = {"h": h, "metal": metal, "owners": owners, "rsurf": rsurf,
                   "dAv": dAv, "nvv": nvv, "n_true": n_true,
                   "dAv_n": dloc * (A_tri / dloc.sum()), "d_near": d_near, "r_true": cl,
                   "ang_med": float(np.median(np.rad2deg(np.arccos(np.clip(cosv, -1, 1)))))}
        print(f"  [{nm}] h={h:.7f} 金属体素 {int(metal.sum()):>6} 面元 {len(dAv):>6}"
              f" 面积 {dAv.sum():.5f}  δ 中位 {np.median(d_near)*1e3:6.2f} mm"
              f" (k·δ {np.median(k*d_near):.3f} rad)"
              f"  ∠(n̂_v,n̂_true) {geo[nm]['ang_med']:5.1f}°  [{time.time()-t0:.0f}s]",
              flush=True)

    # ---- 前向：逐角计算所有臂 ----
    tags = list(geo) + ["FEKO"]
    arms = ["vox_po", "vox_nrm_true", "mesh_po"]
    acc = {t: {a: {"rcs": [], "c": []} for a in arms} for t in tags}
    acc["FEKO"]["vox_gamma"] = {f"{g:.2f}": {"rcs": [], "c": []} for g in GAMMAS}
    t0 = time.time()
    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei)
            ei = ei * ph0
            e0vec = np.cross(ki, ei)
            Eh2 = f["E_ff"][i].reshape(-1, 2)

            def record(tag, arm, E):
                acc[tag][arm]["rcs"].append(rcs_of(E, th, ph, e0m, shape))
                acc[tag][arm]["c"].append(cstats(
                    np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1), Eh2))

            def go(tag, owners, rsurf, dAv, dAv_n, nvv, n_true, metal, ms):
                psi = np.exp(-1j * beta[i] * (rsurf @ ki))
                lit_v = ((nvv @ ki) < 0) & (~ray_occlusion(owners, -ki, metal, max_steps=ms))
                Jv = 2.0 * np.cross(nvv, e0vec) / ETA0 * psi[:, None]
                Jv[~lit_v] = 0.0
                record(tag, "vox_po", nffft_from(Jv, dAv, rsurf, rhat, k, args.chunk))
                lit_n = ((n_true @ ki) < 0) & (~ray_occlusion(owners, -ki, metal, max_steps=ms))
                Jn = 2.0 * np.cross(n_true, e0vec) / ETA0 * psi[:, None]
                Jn[~lit_n] = 0.0
                record(tag, "vox_nrm_true", nffft_from(Jn, dAv_n, rsurf, rhat, k, args.chunk))

            # mesh_po（h 无关，只算一次）
            psi_t = np.exp(-1j * beta[i] * (cen @ ki))
            lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal_feko, max_steps=900))
            Jm = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi_t[:, None]
            Jm[~lit_m] = 0.0
            record("FEKO", "mesh_po", nffft_from(Jm, dAm, cen, rhat, k, args.chunk))

            # FEKO 网格阶梯面（h0 锚点）+ γ 扫描（固定面元集，位置偏移 ∝ γ ∝ h）
            go("FEKO", owners_f, rsurf_f, dAv_f, dAvf_locn, nvv_f, n_true_f, metal_feko, 900)
            lit_c = (nvv_f @ ki) < 0
            occ_c = ray_occlusion(owners_f, -ki, metal_feko, max_steps=900)
            for g in GAMMAS:
                rg = r_true_f + g * (rsurf_f - r_true_f)
                Jg = 2.0 * np.cross(nvv_f, e0vec) / ETA0 \
                    * np.exp(-1j * beta[i] * (rg @ ki))[:, None]
                Jg[~(lit_c & ~occ_c)] = 0.0
                Eg = nffft_from(Jg, dAv_f, rg, rhat, k, args.chunk)
                d = acc["FEKO"]["vox_gamma"][f"{g:.2f}"]
                d["rcs"].append(rcs_of(Eg, th, ph, e0m, shape))
                d["c"].append(cstats(np.stack([(Eg * th).sum(1), (Eg * ph).sum(1)], axis=1),
                                     Eh2))

            # h 族
            for nm, h in levels:
                G = geo[nm]
                go(nm, G["owners"], G["rsurf"], G["dAv"], G["dAv_n"], G["nvv"],
                   G["n_true"], G["metal"], int(2.5 * max(G["metal"].shape)) + 20)
            if (a + 1) % 3 == 0:
                print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)

    # ---- 汇总 ----
    # 注意：mesh_po 只在 FEKO 标签下算过（h 无关），不能按 (tag, arm) 全笛卡尔积取
    raw = {f"{t}|{a}": np.stack(acc[t][a]["rcs"])
           for t in tags for a in arms if acc[t][a]["rcs"]}
    raw.update({f"FEKO|vox_gamma|{g:.2f}": np.stack(acc["FEKO"]["vox_gamma"][f"{g:.2f}"]["rcs"])
                for g in GAMMAS})
    np.savez_compressed(os.path.join(RESULT_DIR, "_diag_staircase_order_raw.npz"), **raw)
    tgt = rcs_true[aidx]

    def pack(d):
        st = _rcs_stats(np.stack(d["rcs"]), tgt)
        cc = d["c"]
        ca = np.array([x["c_arg_deg"] for x in cc])
        return {**st,
                "rho_complex": float(np.median([x["rho_complex"] for x in cc])),
                "c_abs": float(np.median([x["c_abs"] for x in cc])),
                "c_arg_deg": float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(ca)))))),
                "dir_used_frac": float(np.median([x["dir_used_frac"] for x in cc]))}

    res = {"n_angle": int(len(aidx)), "angles": aidx.tolist(), "h0_m": H, "k": k,
           "gammas": GAMMAS, "levels": {}, "feko_h0": {}, "gamma_scan": {},
           "feko_geom": {"n_facet": int(len(dAv_f)), "area_vox_m2": float(dAv_f.sum()),
                         "area_tri_m2": A_tri,
                         "delta_med_mm": float(np.median(d_near_f) * 1e3),
                         "delta_p90_mm": float(np.percentile(d_near_f, 90) * 1e3),
                         "kdelta_med_rad": float(np.median(k * d_near_f))}}
    for nm, _ in levels:
        g = geo[nm]
        res["levels"][nm] = {
            "h": float(g["h"]), "h_over_h0": float(g["h"] / H),
            "n_metal": int(g["metal"].sum()), "n_facet": int(len(g["dAv"])),
            "area_vox_m2": float(g["dAv"].sum()),
            "delta_med_mm": float(np.median(g["d_near"]) * 1e3),
            "delta_p90_mm": float(np.percentile(g["d_near"], 90) * 1e3),
            "kdelta_med_rad": float(np.median(k * g["d_near"])),
            "normal_angle_med_deg": g["ang_med"],
            "arms": {a: pack(acc[nm][a]) for a in ("vox_po", "vox_nrm_true")}}
    res["feko_h0"] = {a: pack(acc["FEKO"][a]) for a in arms}
    m = res["feko_h0"]["mesh_po"]["rcs_dB_err_median"]
    res["mesh_po_dB"] = m
    for g in GAMMAS:
        res["gamma_scan"][f"{g:.2f}"] = pack(acc["FEKO"]["vox_gamma"][f"{g:.2f}"])

    print(f"\n[C1] 几何阶梯收敛（{len(aidx)} 角，真值 = FEKO 远场）")
    print(f"  {'级别':<6} {'h/h0':>6} {'面元':>7} {'面积 m²':>9} {'δ中位mm':>8} {'k·δ':>6} | "
          f"{'vox_po':>8} {'ρ':>6} {'|c|':>6} | {'nrm_true':>9} {'ρ':>6} {'|c|':>6}")
    for nm, _ in levels:
        q = res["levels"][nm]
        A, B = q["arms"]["vox_po"], q["arms"]["vox_nrm_true"]
        print(f"  {nm:<6} {q['h_over_h0']:6.3f} {q['n_facet']:7d} {q['area_vox_m2']:9.5f} "
              f"{q['delta_med_mm']:8.2f} {q['kdelta_med_rad']:6.3f} | "
              f"{A['rcs_dB_err_median']:8.3f} {A['rho_complex']:6.3f} {A['c_abs']:6.3f} | "
              f"{B['rcs_dB_err_median']:9.3f} {B['rho_complex']:6.3f} {B['c_abs']:6.3f}")
    A, B = res["feko_h0"]["vox_po"], res["feko_h0"]["vox_nrm_true"]
    print(f"  {'FEKOh0':<6} {1.0:6.3f} {len(dAv_f):7d} {dAv_f.sum():9.5f} "
          f"{np.median(d_near_f)*1e3:8.2f} {np.median(k*d_near_f):6.3f} | "
          f"{A['rcs_dB_err_median']:8.3f} {A['rho_complex']:6.3f} {A['c_abs']:6.3f} | "
          f"{B['rcs_dB_err_median']:9.3f} {B['rho_complex']:6.3f} {B['c_abs']:6.3f}")
    print(f"\n  参考 mesh_po（真曲面，h→0 极限）= {m:.3f} dB")

    # ---- 幂次拟合 ----
    def fit(pts, free_floor):
        x = np.log([p[0] for p in pts])
        y = np.log([p[1] for p in pts])
        if not free_floor:
            Aa = np.stack([x, np.ones_like(x)], axis=1)
            return float(np.linalg.lstsq(Aa, y, rcond=None)[0][0]), 0.0
        fun = lambda v: np.log(np.exp(v[1]) * np.exp(v[0] * x) + np.exp(v[2])) - y
        r = least_squares(fun, [1.0, np.log(max(pts[0][1], 1e-3)), -3.0])
        return float(r.x[0]), float(np.exp(r.x[2]))

    errv = lambda nm, arm: max(res["levels"][nm]["arms"][arm]["rcs_dB_err_median"] - m, 1e-4)
    fits = {}
    for arm in ("vox_po", "vox_nrm_true"):
        p3, _ = fit([(res["levels"][nm]["h"], errv(nm, arm))
                     for nm in ("h0", "h0/2", "h0/4")], False)
        p4, fl4 = fit([(res["levels"][nm]["h"], errv(nm, arm))
                       for nm, _ in levels], True)
        fits[arm] = {"p_3pt_nofloor": p3, "p_4pt_floor": p4, "floor_dB": fl4,
                     "pts": [[float(res["levels"][nm]["h"]), float(errv(nm, arm))]
                             for nm, _ in levels]}
    gp = [(g, max(res["gamma_scan"][f"{g:.2f}"]["rcs_dB_err_median"] - m, 1e-4))
          for g in GAMMAS if g > 0]
    p_gamma = float(np.linalg.lstsq(
        np.stack([np.log([p[0] for p in gp]), np.ones(len(gp))], axis=1),
        np.log([p[1] for p in gp]), rcond=None)[0][0])
    res["fits"] = {**fits, "gamma_p": p_gamma,
                   "gamma_pts": [[float(a), float(b)] for a, b in gp]}

    print(f"\n  幂次拟合（err(h) = med(h) − mesh_po）")
    for arm in ("vox_po", "vox_nrm_true"):
        q = fits[arm]
        print(f"    {arm:<14} 3 点(无底) p={q['p_3pt_nofloor']:+.2f}   "
              f"4 点(含底) p={q['p_4pt_floor']:+.2f}  底={q['floor_dB']:+.3f} dB")
        print("      " + "  ".join(f"h/h0={a/H:.2f}→{b:.3f}" for a, b in q["pts"]))
    # ---- 逐倍频斜率（比全域幂次拟合更诚实：不假设单一幂律）----
    def slope(a, b):
        return float(np.log(b[1] / a[1]) / np.log(b[0] / a[0]))

    oct_ = {}
    for arm in ("vox_po", "vox_nrm_true"):
        v = [(res["levels"][nm]["h"], errv(nm, arm))
             for nm in ("h0", "h0/2", "h0/4")]
        oct_[arm] = {"h0->h0/2": slope(v[0], v[1]), "h0/2->h0/4": slope(v[1], v[2])}
    print(f"    逐倍频斜率  vox_po      h0→h0/2 p={oct_['vox_po']['h0->h0/2']:+.2f}"
          f"   h0/2→h0/4 p={oct_['vox_po']['h0/2->h0/4']:+.2f}")
    print(f"    逐倍频斜率  vox_nrm_true h0→h0/2 p={oct_['vox_nrm_true']['h0->h0/2']:+.2f}"
          f"   h0/2→h0/4 p={oct_['vox_nrm_true']['h0/2->h0/4']:+.2f}")
    print(f"    γ 扫描（固定面元集，位置偏移 ∝ γ ∝ h）  p={p_gamma:+.2f}")
    print("       " + "  ".join(f"γ={a:.2f}→{b:.3f}" for a, b in gp))
    res["fits"]["octave"] = oct_

    # ---- 判决：以逐倍频斜率 + γ 扫描为准（全域幂次拟合会掩盖饱和）----
    s_fine = oct_["vox_nrm_true"]["h0->h0/2"]
    s_ultra = oct_["vox_nrm_true"]["h0/2->h0/4"]
    floor_emp = errv("h0/4", "vox_nrm_true")
    res["floor_empirical_dB"] = float(floor_emp)
    if s_fine >= 0.8 and s_ultra >= 0.8:
        vd = (f"O(h) 全程成立（h0→h0/2 p={s_fine:.2f}，h0/2→h0/4 p={s_ultra:.2f}）"
              f" ⇒ **位置相位是主控**，预注册判据命中")
    elif s_fine >= 0.8:
        vd = (f"O(h) 只在粗端成立（h0→h0/2 p={s_fine:.2f}），此后**饱和**"
              f"（h0/2→h0/4 p={s_ultra:.2f}，实测地板 {floor_emp:.3f} dB）"
              f" ⇒ 位置相位只在 h ≳ h0/2 段主导，剩余项与 h 无关")
    else:
        vd = f"亚线性（h0→h0/2 p={s_fine:.2f}）⇒ 位置相位非主控，另有机制"
    res["verdict"] = vd
    print(f"\n  判决 → {vd}")

    # ---- 局部细网格可行性 ----
    gap = float(A["rcs_dB_err_median"] - m)
    ratio = float(np.percentile(d_near_f, 90) / np.median(d_near_f))
    # δ 的空间集中度：按 δ 降序累加「面积×δ」，覆盖 50%/80% 所需的**面积占比**
    #   ⇒ 小 ⇒ 误差集中在少数面元（局部加密高效）；≈1 ⇒ 误差遍布全表面（局部加密无优势）
    od = np.argsort(-d_near_f)
    cw = np.cumsum(dAv_f[od] * d_near_f[od]) / float((dAv_f * d_near_f).sum())
    ca = np.cumsum(dAv_f[od]) / float(dAv_f.sum())
    f50 = float(ca[min(np.searchsorted(cw, 0.50), len(ca) - 1)])
    f80 = float(ca[min(np.searchsorted(cw, 0.80), len(ca) - 1)])
    need_half = H / (2.0 ** (1.0 / max(s_fine, 1e-3)))
    res["local_refine"] = {"gap_deploy_dB": gap, "delta_p90_over_med": ratio,
                           "area_frac_for_50pct_delta": f50,
                           "area_frac_for_80pct_delta": f80,
                           "h_for_half_gain_mm": float(need_half * 1e3),
                           "voxel_mult": float((H / need_half) ** 3)}
    print(f"\n  局部细网格可行性")
    print(f"    部署同类臂几何红利（FEKO h0 vox_po → mesh_po）= {gap:+.3f} dB")
    print(f"    δ 空间分布：P90/中位 = {ratio:.2f}；覆盖 50%/80% 的 δ 总量只需"
          f" {f50*100:.0f}%/{f80*100:.0f}% 面积"
          f" ⇒ {'误差遍布全表面，局部加密只能按面积占比分摊' if f80 > 0.8 else '误差集中，局部加密可能高效'}")
    print(f"    按粗端 p={s_fine:.2f}（乐观；细端已饱和）：要拿一半红利（{gap/2:.3f} dB）需 "
          f"h → {need_half*1e3:.2f} mm（h0 的 1/{H/need_half:.1f}），体素数 ×{(H/need_half)**3:.0f}")

    jp = os.path.join(RESULT_DIR, "_diag_staircase_order.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")


if __name__ == "__main__":
    main()
