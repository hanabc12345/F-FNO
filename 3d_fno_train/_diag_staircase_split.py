# -*- coding: utf-8 -*-
"""
_diag_staircase_split.py — A2：阶梯面 0.58 dB 红利的成分拆分（面积 vs 位置/相位）
================================================================================
背景（item 11 交叉验证锁定）
  「阶梯体素面 → 真 STL 面」这一几何升级，两条独立电流链给出**几乎相同的红利**：
      vox_po  3.745 → mesh_po 3.161   =  +0.58 dB   （解析 PO 链）
      vox_true 3.483 → C1_nn_air 2.915 = +0.57 dB   （真值近场链）
  ⇒ 该 0.58 dB 与电流来源无关，是**纯几何效应**。本脚本只拆它。

几何差在哪里（只有两处，都可独立操作）
  (G1) **面积过估**：阶梯面 1.31445 m² vs 真曲面 1.19955 m² ⇒ 比值 **1.0958（+9.6%）**。
       若纯全局标量 ⇒ RCS 偏置 20log₁₀(1.0958) = **+0.795 dB**。
  (G2) **位置/相位**：阶梯面元偏离真实曲面 ≤h/2，且法向被量化为轴对齐 ⇒ 相位 k·δ 与
       等效电流方向都有误差。

关键简化
  NFFFT 远场对 dA 是**线性的**（E = i k/4π · Σ J·dA·e^{jk r̂·r}），而解析 PO 电流 J
  本身不含 dA ⇒ **所有面元面积同乘 s 时 E(s) = s·E(1)、RCS(s) = s²·RCS(1)**。
  ⇒ 全局面积效应**不需要重跑 NFFFT**，纯后处理即可（本脚本的「面积扫描」）。

实验口径
  臂 A  vox_po      阶梯面 + 解析 PO（面积原样 1.31445）        ← 与 exp_po_mesh 同源
  臂 B  vox_area_loc  阶梯面 + 局部法向面积归一（见下）           ← 需 1 次额外 NFFFT
  臂 C  mesh_po     真曲面 + 解析 PO（面积原样 1.19955）        ← 与 exp_po_mesh 同源
  + 「面积扫描」：对 A/C 的 E(1) 乘 s ∈ SCALES 后重算 med（零额外成本）

  局部法向面积归一：每个阶梯面元的法向是轴对齐的（±x/±y/±z），而它替代的真实曲面是斜的。
  用 KD-tree 找该面元最近的 STL 三角面，取 |n̂_v · n̂_true| 缩放面积 ⇒ 把「把斜面摊成
  台阶」这一投影过估逐面元扣回去。两个子变体：
      vox_area_loc   缩放后**不**再校正总量（测局部投影本身够不够）
      vox_area_locn  缩放后**强制**总量 = 1.19955 m²（局部 + 总量都对齐）

拆分读数（交叉验证，两路径独立）
  总红利   = med(vox_po, s=1)        − med(mesh_po, s=1)
  面积项   = med(vox_po, s=1)        − med(vox_po, s=A_tri/A_vox)      [阶梯几何上]
  位置项   = med(vox_po, s=A_tri/A_vox) − med(mesh_po, s=1)            [面积已归一]
  交叉①   = med(mesh_po, s=A_vox/A_tri) − med(mesh_po, s=1)  应 ≈ −面积项
  交叉②   = med(vox_po, s=1)        − med(mesh_po, s=A_vox/A_tri) 应 ≈ +位置项

判读（预注册）
  · 面积项 ≈ +0.795 dB（理论标量值）且位置项 ≈ −0.2 dB ⇒ 「面积过估」是主因，
    且**位置量化反而略微帮忙** ⇒ 几何修复应优先做面积（省事：一个标量）。
  · 面积项 ≪ 0.795 ⇒ med 对整体平移不敏感（误差是散布型非偏置型）⇒ 面积不是主因，
    0.58 dB 主要来自位置/相位 ⇒ 必须动 h，与 item 10 判决合流。
  · 各口径 med 曲线若对 s 呈「V 形且谷底在 s=1 附近」 ⇒ 该臂幅度标定已最优。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_staircase_split.py --angles 12
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_staircase_split.py            # 全 468
产出
  results/_diag_staircase_split.json
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
import fno_f16_3d as M                                          # noqa: E402
from fno_f16_3d_p4_nffft import surface_parts, direction_grid    # noqa: E402
from exp_po_mesh import (load_mesh, mesh_outside_voxel, nffft_from,  # noqa: E402
                         rcs_of, cstats, ETA0, H, STL, RESULT_DIR)
from exp_po_locality import ray_occlusion                        # noqa: E402
from po_patch_data import canon_phase                            # noqa: E402
from _exp_common import _rcs_stats                               # noqa: E402

# 面积扫描点：以真曲面/阶梯面比值的邻域为中心
SCALES = [0.80, 0.85, 0.8800, 0.9126, 0.95, 1.00, 1.05, 1.0958, 1.15, 1.20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=0, help="抽样角数（0=全 468）")
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
    metal = eps > 1.5
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    # ---- 两套几何 ----
    cen, nvm, dAm = load_mesh(STL)                      # 真 STL 三角面
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    owners, rsurf, dSv = surface_parts(eps, gx, gy, gz)  # 阶梯体素面
    dAv = np.linalg.norm(dSv, axis=1)
    nvv = dSv / dAv[:, None]
    A_tri, A_vox = float(dAm.sum()), float(dAv.sum())
    s_area = A_tri / A_vox                                # 面积归一化因子（阶梯臂）
    bias_area_dB = 20.0 * np.log10(A_vox / A_tri)         # 阶梯面纯标量偏置（理论值）

    # ---- 局部法向面积归一：按最近 STL 三角面的法向做投影 ----
    from scipy.spatial import cKDTree
    tree = cKDTree(cen)
    _, nrst = tree.query(rsurf, k=1)
    cosproj = np.abs((nvv * nvm[nrst]).sum(axis=1))
    dAv_loc = dAv * cosproj
    dAv_locn = dAv_loc * (A_tri / dAv_loc.sum())          # 强制总量对齐
    # ---- 真曲面法向估计（K 近邻距离加权，稳健于单个最近三角面的噪声）----
    dk, ik = tree.query(rsurf, k=8)
    wk = 1.0 / (dk + 1e-6)
    nsum = (wk[:, :, None] * nvm[ik]).sum(axis=1)
    n_true = nsum / np.maximum(np.linalg.norm(nsum, axis=1, keepdims=True), 1e-12)
    # 法向量化误差角 / 位置偏移相位（两项诊断，零成本）
    ang_vt = np.rad2deg(np.arccos(np.clip(np.abs((nvv * n_true).sum(axis=1)), -1.0, 1.0)))
    d_near = dk[:, 0]
    ph_off = k * d_near

    print(f"  真 STL 面   {len(cen):>6} 面  面积 {A_tri:.5f} m²")
    print(f"  阶梯体素面 {len(dAv):>6} 面  面积 {A_vox:.5f} m²"
          f"   T/V 面积比 {A_tri/A_vox:.5f}（过估 {100*(A_vox/A_tri-1):+.2f}%）")
    print(f"  纯标量面积偏置理论值 20log₁₀(A_vox/A_tri) = {bias_area_dB:+.3f} dB")
    print(f"  局部法向归一后面积 {dAv_loc.sum():.5f} m²"
          f"（占原 {dAv_loc.sum()/A_vox*100:.1f}%） ⇒ 强制对齐得 {dAv_locn.sum():.5f} m²")
    print(f"  投影因子 |n̂_v·n̂_true|：中位 {np.median(cosproj):.3f}  均值 {cosproj.mean():.3f}")
    print(f"  法向量化夹角 |∠(n̂_v,n̂_true)|：中位 {np.median(ang_vt):.1f}°  P90 "
          f"{np.percentile(ang_vt, 90):.1f}°")
    print(f"  阶梯面元到真曲面最近三角面距离：中位 {np.median(d_near)*1e3:.2f} mm  P90 "
          f"{np.percentile(d_near, 90)*1e3:.2f} mm  ⇒ 相位 k·δ 中位 {np.median(ph_off):.3f} rad")

    arms = ["vox_po", "vox_area_loc", "vox_area_locn", "vox_nrm_true", "mesh_po"]
    acc = {a: {"rcs": [], "c": []} for a in arms}
    t0 = time.time()
    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei)
            ei = ei * ph0
            psi_air = np.exp(-1j * beta[i] * (rsurf @ ki))
            psi_tri = np.exp(-1j * beta[i] * (cen @ ki))
            e0vec = np.cross(ki, ei)

            # ---- 阶梯面：解析 PO + 遮蔽（同 exp_po_mesh）----
            lit_v = ((nvv @ ki) < 0) & (~ray_occlusion(owners, -ki, metal))
            Jv = 2.0 * np.cross(nvv, e0vec) / ETA0 * psi_air[:, None]
            Jv[~lit_v] = 0.0
            # ---- 真曲面：解析 PO + 遮蔽（同 exp_po_mesh）----
            lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
            Jm = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi_tri[:, None]
            Jm[~lit_m] = 0.0
            # ---- 法向量化单因素臂：位置/面积仍阶梯，只把电流法向换成真曲面法向 ----
            lit_n = ((n_true @ ki) < 0) & (~ray_occlusion(owners, -ki, metal))
            Jn = 2.0 * np.cross(n_true, e0vec) / ETA0 * psi_air[:, None]
            Jn[~lit_n] = 0.0

            Ev = nffft_from(Jv, dAv, rsurf, rhat, k, args.chunk)          # vox_po 基准
            Evl = nffft_from(Jv, dAv_loc, rsurf, rhat, k, args.chunk)     # 局部归一
            Evln = nffft_from(Jv, dAv_locn, rsurf, rhat, k, args.chunk)   # 局部+总量
            Evn = nffft_from(Jn, dAv_locn, rsurf, rhat, k, args.chunk)    # 真法向+面积对齐
            Em = nffft_from(Jm, dAm, cen, rhat, k, args.chunk)            # mesh_po 基准

            Eh2 = f["E_ff"][i].reshape(-1, 2)
            for nm, E in (("vox_po", Ev), ("vox_area_loc", Evl), ("vox_area_locn", Evln),
                          ("vox_nrm_true", Evn), ("mesh_po", Em)):
                acc[nm]["rcs"].append(rcs_of(E, th, ph, e0m, shape))
                acc[nm]["c"].append(cstats(np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1),
                                           Eh2))
            if (a + 1) % 10 == 0:
                print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)

    # ---- 面积扫描是纯后处理：`rcs_of` 是 |E|² 的线性函数、且 E(s)=s·E(1)（NFFFT 对 dA 线性）
    #      ⇒ RCS(s) = s²·RCS(1)，逐位等价，无需重跑 NFFFT ----
    res = {"n_angle": int(len(aidx)), "angles": aidx.tolist(), "h_m": float(H),
           "area_tri_m2": A_tri, "area_vox_m2": A_vox, "s_area_tri_over_vox": float(s_area),
           "bias_area_theory_dB": float(bias_area_dB),
           "area_loc_m2": float(dAv_loc.sum()), "area_locn_m2": float(dAv_locn.sum()),
           "proj_factor_med": float(np.median(cosproj)),
           "normal_angle_med_deg": float(np.median(ang_vt)),
           "pos_offset_med_mm": float(np.median(d_near) * 1e3),
           "phase_offset_med_rad": float(np.median(ph_off)),
           "scales": SCALES, "arms": {}, "scan": {}}
    tgt = rcs_true[aidx]

    def _pack(nm):
        R = np.stack(acc[nm]["rcs"])
        st = _rcs_stats(R, tgt)
        cc = acc[nm]["c"]
        ca = np.array([x["c_arg_deg"] for x in cc])
        return {**st,
                "rho_complex": float(np.median([x["rho_complex"] for x in cc])),
                "c_abs": float(np.median([x["c_abs"] for x in cc])),
                "c_arg_deg": float(np.rad2deg(np.angle(np.mean(np.exp(1j*np.deg2rad(ca)))))),
                "dir_used_frac": float(np.median([x["dir_used_frac"] for x in cc]))}

    print(f"\n[A2] 阶梯面红利成分拆分（{len(aidx)} 角，真值 = FEKO 远场）")
    print(f"  {'臂':<16} {'面积 m²':>10} {'med':>7} {'P90':>7} {'ρ':>6} {'|c|':>6} {'∠c':>7}")
    area_of = {"vox_po": A_vox, "vox_area_loc": float(dAv_loc.sum()),
               "vox_area_locn": float(dAv_locn.sum()), "vox_nrm_true": float(dAv_locn.sum()),
               "mesh_po": A_tri}
    for nm in arms:
        res["arms"][nm] = _pack(nm)
        q = res["arms"][nm]
        print(f"  {nm:<16} {area_of[nm]:10.5f} {q['rcs_dB_err_median']:7.3f} "
              f"{q['rcs_dB_err_p90']:7.3f} {q['rho_complex']:6.3f} {q['c_abs']:6.3f} "
              f"{q['c_arg_deg']:+7.1f}")

    # ---- 面积扫描表（零额外 NFFFT：RCS(s)=s²·RCS(1)）----
    print(f"\n  面积扫描（RCS(s) = s²·RCS(1)，解析等价）")
    print(f"  {'s':>8} {'20log₁₀s':>9} {'vox_po med':>11} {'mesh_po med':>12}")
    for s in SCALES:
        row = {}
        for nm in ("vox_po", "mesh_po"):
            R = np.stack(acc[nm]["rcs"]) * (s ** 2)
            row[nm] = _rcs_stats(R, tgt)["rcs_dB_err_median"]
        res["scan"][f"{s:.4f}"] = {kk: float(vv) for kk, vv in row.items()}
        print(f"  {s:8.4f} {20*np.log10(s):+9.3f} {row['vox_po']:11.3f} {row['mesh_po']:12.3f}")

    # 扫描表里挑出 s = s_area（阶梯面积归一到真曲面）与 s = 1/s_area（真曲面面积过估）
    def med_at(nm, s):
        return _rcs_stats(np.stack(acc[nm]["rcs"]) * (s ** 2), tgt)["rcs_dB_err_median"]

    q_vox, q_mesh = res["arms"]["vox_po"], res["arms"]["mesh_po"]
    total = q_vox["rcs_dB_err_median"] - q_mesh["rcs_dB_err_median"]
    a_step = q_vox["rcs_dB_err_median"] - med_at("vox_po", s_area)          # 面积项（阶梯几何）
    p_step = med_at("vox_po", s_area) - q_mesh["rcs_dB_err_median"]         # 位置项（面积已归一）
    a_mesh = med_at("mesh_po", 1.0 / s_area) - q_mesh["rcs_dB_err_median"]  # 交叉①（真几何）
    p_mesh = q_vox["rcs_dB_err_median"] - med_at("mesh_po", 1.0 / s_area)   # 交叉②
    loc = res["arms"]["vox_area_loc"]["rcs_dB_err_median"]
    locn = res["arms"]["vox_area_locn"]["rcs_dB_err_median"]
    res["split"] = {"total_dB": float(total), "area_step_dB": float(a_step),
                    "pos_step_dB": float(p_step), "area_mesh_dB": float(a_mesh),
                    "pos_mesh_dB": float(p_mesh),
                    "med_vox_at_s_area": float(med_at("vox_po", s_area)),
                    "med_mesh_at_s_inv": float(med_at("mesh_po", 1.0 / s_area)),
                    "vox_area_loc_dB": float(loc), "vox_area_locn_dB": float(locn),
                    "gain_loc_vs_vox": float(q_vox["rcs_dB_err_median"] - loc),
                    "gain_locn_vs_vox": float(q_vox["rcs_dB_err_median"] - locn),
                    "gain_locn_vs_mesh": float(locn - q_mesh["rcs_dB_err_median"])}

    print(f"\n  拆分读数（总红利 {total:+.3f} dB）")
    print(f"    面积项（阶梯几何上，s={s_area:.4f}）        = {a_step:+.3f} dB"
          f"   [理论标量值 {bias_area_dB:+.3f}]")
    print(f"    位置项（面积已归一后换真几何）              = {p_step:+.3f} dB")
    print(f"    交叉① 面积项（真几何上，s={1/s_area:.4f}）  = {a_mesh:+.3f} dB")
    print(f"    交叉② 位置项（面积都过估下换真几何）        = {p_mesh:+.3f} dB")
    print(f"    局部法向归一（面积 {dAv_loc.sum():.4f}）           = {loc:.3f} dB"
          f"（相对 vox_po {q_vox['rcs_dB_err_median']-loc:+.3f}）")
    print(f"    局部+总量归一（面积 {dAv_locn.sum():.4f}）         = {locn:.3f} dB"
          f"（相对 vox_po {q_vox['rcs_dB_err_median']-locn:+.3f}"
          f"，相对 mesh_po {locn-q_mesh['rcs_dB_err_median']:+.3f}）")

    # ---- 第二层拆分：位置偏移（L3）vs 法向量化（L2）----
    nrm = res["arms"]["vox_nrm_true"]["rcs_dB_err_median"]
    g_vox = q_vox["rcs_dB_err_median"] - locn      # 面积归一带来的（≈0）
    g_nrm = q_vox["rcs_dB_err_median"] - nrm       # 换真法向带来的
    res["split2"] = {"vox_nrm_true_dB": float(nrm),
                     "gain_normal_dB": float(g_nrm),
                     "gain_area_norm_dB": float(g_vox),
                     "residual_position_dB": float(nrm - q_mesh["rcs_dB_err_median"])}
    print(f"\n  第二层拆分：位置偏移（L3） vs 法向量化（L2）")
    print(f"    法向换成真曲面法向（位置/遮蔽仍阶梯，面积已对齐）= {nrm:.3f} dB"
          f"（相对 vox_po {g_nrm:+.3f}，相对 mesh_po {nrm-q_mesh['rcs_dB_err_median']:+.3f}）")
    if g_nrm > 0.4 * (q_vox["rcs_dB_err_median"] - q_mesh["rcs_dB_err_median"]):
        print(f"    ⇒ 判决：**法向量化是主因**（吃掉"
              f"{100*g_nrm/total:.0f}% 的红利）⇒ 便宜修复可行（重定向法向，不必加密 h）")
    elif g_nrm < 0.15:
        print(f"    ⇒ 判决：**法向量化几乎不起作用**（只 {g_nrm:+.3f} dB）⇒ 主因是"
              f"**位置阶梯**（面元偏离真实曲面）⇒ 只能动 h，与 item 10 判决合流")
    else:
        print(f"    ⇒ 判决：法向量化与位置阶梯**各占一部分**（法向 {g_nrm:+.3f} dB）")
    print(f"\n  参考 mesh_po = {q_mesh['rcs_dB_err_median']:.3f} dB   vox_po = "
          f"{q_vox['rcs_dB_err_median']:.3f} dB")

    jp = os.path.join(RESULT_DIR, "_diag_staircase_split.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")


if __name__ == "__main__":
    main()
