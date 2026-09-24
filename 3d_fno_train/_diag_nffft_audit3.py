# -*- coding: utf-8 -*-
"""_diag_nffft_audit3.py — NFFFT floor 归因（第三轮：用真实网格 f16_refined.stl 定量）

第二轮关键结果（results/_diag_nffft_audit2.json）：
  * 阶梯面元总面积 1.31 m²（用的是 f16.stl，2.15 m²）→ 面积比 0.61
  * 远场积分相位符号 e^{+jk r̂·r'} 正确（ρ=0.755 vs -jk 的 0.072）→ 约定无误
  * 最佳复标定因子 |c| ≈ 0.52，与面积亏缺同量级 → 强烈指向"表面提取漏面积"

本轮：
  1) 换用数据集真正使用的网格 f16_refined.stl（33850 三角面）核对面积/体积
  2) 对比"阶梯面面积 / 体素金属体积"与"真实面积 / 真实体积"
  3) 直接用 STL 三角面（法向取文件自带）做同一 NFFFT → 看 |ΔRCS| 能否显著下降
  4) 量化"漏掉的面积"：三角面到最近金属体素中心的距离分布

产出：results/_diag_nffft_audit3.json
用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_audit3.py --nangles 8
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
from fno_f16_3d_p4_nffft import direction_grid, build_tot_fields
from _diag_nffft_audit import surface_parts_full, tri_interp, rcs_stats
from _diag_nffft_audit2 import read_stl, nffft_chunked

RESULT_DIR = os.path.join(BASE, "results")
STL = r"f:\MyWorkSpace\UAVGame\3d_feko_run\f16_refined.stl"
ETA0 = 119.9169832 * np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=8000)
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]; H_scat = f["H_scat"][:]
        eps = f["eps_field"][:]; rcs_true = f["rcs"][:]
        E_ff_h5 = f["E_ff"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0]); h = float(gx[1] - gx[0])
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    out = {"stl": STL, "h_m": h, "k": k}

    # ---------- 1) 网格几何核对 ----------
    tris, nrm_file = read_stl(STL)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    av = 0.5 * np.cross(v1 - v0, v2 - v0)
    area_tri = np.linalg.norm(av, axis=1)
    cen = tris.mean(axis=1)
    # 闭合网格体积（散度定理）：V = Σ (1/6)(v0×v1)·v2
    vol_tri = float(np.abs((np.cross(v0, v1) * v2).sum() / 6.0))
    # 文件自带法向 vs 面元叉乘法向 一致性
    dotf = (nrm_file * av).sum(axis=1)
    agree = float((dotf > 0).mean())
    # 定向：以文件法向为准（不一致的翻转）
    av_o = np.where((dotf[:, None] >= 0), av, -av)
    area_tri_o = np.linalg.norm(av_o, axis=1)

    idx_air, idx_met, rsurf, dS = surface_parts_full(eps, gx, gy, gz, h)
    dAn = np.linalg.norm(dS, axis=1)
    area_stair = float(dAn.sum())
    vol_metal = float((eps > 1.5).sum()) * h ** 3

    out["geom"] = {
        "stl_n_tri": int(len(tris)), "stl_area_m2": float(area_tri.sum()),
        "stl_volume_m3": vol_tri, "stl_bbox_min": tris.reshape(-1, 3).min(0).tolist(),
        "stl_bbox_max": tris.reshape(-1, 3).max(0).tolist(),
        "normal_file_vs_cross_agree_frac": agree,
        "stair_n_faces": int(len(rsurf)), "stair_area_m2": area_stair,
        "voxel_metal_n": int((eps > 1.5).sum()), "voxel_metal_volume_m3": vol_metal,
        "ratio_area_stair_over_stl": area_stair / float(area_tri.sum()),
        "ratio_volume_voxel_over_stl": vol_metal / vol_tri,
        "h_over_lambda": h / (2 * np.pi / k),
    }
    print("[1] 网格 f16_refined.stl: %d 三角面  面积 %.4f m²  体积 %.5f m³"
          % (len(tris), area_tri.sum(), vol_tri))
    print("    bbox %s .. %s" % (np.round(tris.reshape(-1,3).min(0), 3),
                                 np.round(tris.reshape(-1,3).max(0), 3)))
    print("    文件法向与叉乘法向一致比例 %.3f" % agree)
    print("    阶梯面: %d 面元  面积 %.4f m²  → 面积比 %.3f"
          % (len(rsurf), area_stair, area_stair / area_tri.sum()))
    print("    体素金属: %d 个  体积 %.5f m³  → 体积比 %.3f"
          % (int((eps > 1.5).sum()), vol_metal, vol_metal / vol_tri))
    print("    h/λ = %.3f" % (h / (2 * np.pi / k)))

    # ---------- 4) 漏面积量化：三角面中心到最近金属体素中心的距离 ----------
    mi = np.argwhere(eps > 1.5)
    mc = np.stack([gx[mi[:, 0]], gy[mi[:, 1]], gz[mi[:, 2]]], axis=1)
    # 只取 2 个角度的 1/8 三角面做距离统计（省时）
    sel = np.arange(0, len(cen), 8)
    dmin = np.empty(len(sel))
    for s_i, s in enumerate(sel):
        dmin[s_i] = np.min(np.linalg.norm(mc - cen[s], axis=1))
    out["tri_dist_to_metal_voxel"] = {
        "median": float(np.median(dmin)), "p90": float(np.percentile(dmin, 90)),
        "frac_gt_half_h": float((dmin > h / 2).mean()),
        "frac_gt_h": float((dmin > h).mean()),
        "h": h,
    }
    print("[4] 三角面中心到最近金属体素中心距离: 中位 %.4f m (h=%.4f, h/2=%.4f)"
          % (np.median(dmin), h, h / 2))
    print("    距离 > h/2 的三角面占比 %.3f；> h 的占比 %.3f"
          % ((dmin > h / 2).mean(), (dmin > h).mean()))

    # ---------- 3) STL 三角面 NFFFT ----------
    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    res_stl, res_stair = [], []
    t0 = time.time()
    for a_i, i in enumerate(idx_list):
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                      E_scat, H_scat, i)
        e0mag = float(np.linalg.norm(e0[i]))
        # --- STL 三角面（文件法向定向，场在面心三线性插值）---
        H_tri = tri_interp(Htot, cen, gx, gy, gz)
        nv = av_o / (area_tri_o[:, None] + 1e-12)
        J = np.cross(nv, H_tri)
        E_stl = nffft_chunked(J, cen, av_o, rhat, k, sign=+1, chunk=args.chunk)
        rcs_stl = (4.0 * np.pi * (np.abs((E_stl * th).sum(1)) ** 2 +
                                  np.abs((E_stl * ph).sum(1)) ** 2)
                   / e0mag ** 2).reshape(shape)
        res_stl.append(rcs_stats(rcs_stl, rcs_true[i]))
        # --- 阶梯面（现状管线）---
        Ha = Htot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        nvs = dS / (dAn[:, None] + 1e-12)
        Js = np.cross(nvs, Ha)
        E_st = nffft_chunked(Js, rsurf, dS, rhat, k, sign=+1, chunk=args.chunk)
        rcs_st = (4.0 * np.pi * (np.abs((E_st * th).sum(1)) ** 2 +
                                 np.abs((E_st * ph).sum(1)) ** 2)
                  / e0mag ** 2).reshape(shape)
        res_stair.append(rcs_stats(rcs_st, rcs_true[i]))
        # --- 复场对比（用相同方向子集）---
        if a_i == 0:
            Et_h = E_ff_h5[i][..., 0].ravel(); Ep_h = E_ff_h5[i][..., 1].ravel()
            thr = 0.05 * np.abs(E_ff_h5[i]).max()
            m = np.abs(Et_h) > thr
            Eh = np.concatenate([Et_h[m], Ep_h[m]])

            def cc(E):
                Ec = np.concatenate([(E * th).sum(1)[m], (E * ph).sum(1)[m]])
                rho = np.abs(np.vdot(Eh, Ec)) / np.sqrt(
                    np.vdot(Ec, Ec).real * np.vdot(Eh, Eh).real)
                c = np.vdot(Eh, Ec) / np.vdot(Eh, Eh)
                return float(rho), float(np.abs(c)), float(np.rad2deg(np.angle(c)))
            out["complex_case0"] = {"stair": cc(E_st), "stl_tri": cc(E_stl)}
        if (a_i + 1) % 2 == 0:
            print("  [%d/%d] %.0fs" % (a_i + 1, len(idx_list), time.time() - t0), flush=True)

    def agg(rs):
        return {"masked_median_dB": float(np.median([x["masked"]["median_dB"] for x in rs])),
                "masked_p90_dB": float(np.median([x["masked"]["p90_dB"] for x in rs])),
                "all_median_dB": float(np.median([x["all"]["median_dB"] for x in rs])),
                "corr_dB_masked": float(np.median([x["masked"]["corr_dB"] for x in rs])),
                "per_angle": [float(x["masked"]["median_dB"]) for x in rs]}
    out["rcs_compare"] = {"stair_voxel_current": agg(res_stair),
                          "stl_triangles": agg(res_stl)}

    print("\n===== 3) 同一 NFFFT，两种表面参数化 =====")
    for kk, lab in (("stair_voxel_current", "阶梯体素面（现状）"),
                    ("stl_triangles", "STL 三角面")):
        s = out["rcs_compare"][kk]
        print("  %-16s: 掩膜后中位 %.3f dB / P90 %.2f dB / corr %.3f | 全方向中位 %.3f dB"
              % (lab, s["masked_median_dB"], s["masked_p90_dB"],
                 s["corr_dB_masked"], s["all_median_dB"]))
    if "complex_case0" in out:
        print("\n  case0 复场对比 FEKO E_ff（ρ / |c| / arg c）:")
        for kk in ("stair", "stl_tri"):
            v = out["complex_case0"][kk]
            print("    %-8s ρ=%.3f  |c|=%.3f  arg=%.1f°" % (kk, v[0], v[1], v[2]))

    with open(os.path.join(RESULT_DIR, "_diag_nffft_audit3.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print("\n已存: results/_diag_nffft_audit3.json")


if __name__ == "__main__":
    main()
