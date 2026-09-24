# -*- coding: utf-8 -*-
"""_diag_nffft_audit2.py — NFFFT floor 归因（第二轮：定位主要来源）

第一轮结论（results/_diag_nffft_audit.json）：
  * 复现了论文 floor：20 角度、掩膜后中位 3.50 dB（论文 3.55 dB）
  * NFFFT 远场 vs FEKO E_ff：|比|中位 0.74，复相关 0.75 → 幅度系统性偏低 + 相位散布大
  * 面元中心插值/双侧平均反而更差 → h/2 采样偏移不是主因

本轮逐个量级排查候选来源：
  1) STL 真实面积 vs 阶梯面元总面积（面积亏缺会按比例压低 |E_ff|）
  2) PEC 边界条件：空气侧体素处 |n̂×E_tot| 是否 << |E_tot|（校验入射场重建）
  3) 导体内侧场 |H_metal| 是否≈0（决定"双侧平均"是否有物理意义）
  4) 远场积分相位符号：e^{+jk r̂·r'} vs e^{-jk r̂·r'} 哪个与 FEKO 一致
  5) 主瓣/旁瓣分开看 |E_ff| 比值（区分整体面积亏缺 vs 方向图结构误差）
  6) 直接用 STL 三角面（替代阶梯面）做同一积分 → 隔离"阶梯化"这一项

产出：results/_diag_nffft_audit2.json
用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_audit2.py --nangles 8
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import struct
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import direction_grid, build_tot_fields
from _diag_nffft_audit import surface_parts_full, tri_interp

RESULT_DIR = os.path.join(BASE, "results")
STL = os.path.join(os.path.dirname(BASE), "3d_feko_data", "f16.stl")
ETA0 = 119.9169832 * np.pi


def read_stl(path):
    """二进制/ASCII STL → (tris (N,3,3), normals (N,3))"""
    with open(path, "rb") as f:
        head = f.read(84)
        if head[:5] == b"solid" and b"facet" in head:
            f.seek(0)
            txt = f.read().decode("ascii", errors="ignore")
            vs, ns = [], []
            cur = []
            for line in txt.splitlines():
                s = line.strip()
                if s.startswith("facet normal"):
                    ns.append([float(x) for x in s.split()[2:5]])
                elif s.startswith("vertex"):
                    cur.append([float(x) for x in s.split()[1:4]])
                    if len(cur) == 3:
                        vs.append(cur); cur = []
            return np.asarray(vs, dtype=np.float64), np.asarray(ns, dtype=np.float64)
        n = struct.unpack("<I", head[80:84])[0]
        data = f.read(n * 50)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, 50)
    fl = arr[:, :48].copy().view(np.float32).reshape(n, 12).astype(np.float64)
    return fl[:, 3:12].reshape(n, 3, 3), fl[:, 0:3]


def nffft_chunked(J, rsurf, dS, rhat, k, sign=+1, chunk=4000):
    """分块累加 N = Σ J dA e^{sign·j k r̂·r'}，避免 (Nf×Ndir) 内存爆炸"""
    dA = np.linalg.norm(dS, axis=1)
    N = np.zeros((rhat.shape[0], 3), dtype=np.complex128)
    for s in range(0, len(rsurf), chunk):
        e = min(s + chunk, len(rsurf))
        ph = np.exp(sign * 1j * k * (rsurf[s:e] @ rhat.T))     # (c,Ndir)
        N += ph.T @ (J[s:e] * dA[s:e, None])
    Nr = (N * rhat).sum(axis=1, keepdims=True) * rhat
    return (1j * k / (4.0 * np.pi)) * ETA0 * (N - Nr)


def eff_components(E_ff, th, ph):
    return (E_ff * th).sum(axis=1), (E_ff * ph).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=8)
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

    out = {"h_m": h, "k": k}

    # ---------- 1) 面积核对 ----------
    tris, nrm = read_stl(STL)
    a = tris[:, 1] - tris[:, 0]; b = tris[:, 2] - tris[:, 0]
    av = 0.5 * np.cross(a, b)
    area_tri = np.linalg.norm(av, axis=1)
    cen = tris.mean(axis=1)
    # 用几何中心定向（STL 自带法向可能不可靠）
    cmean = cen.mean(axis=0)
    outward = np.sign(((cen - cmean) * av).sum(axis=1))
    outward[outward == 0] = 1.0
    av = av * outward[:, None]
    idx_air, idx_met, rsurf, dS = surface_parts_full(eps, gx, gy, gz, h)
    area_stair = float(np.linalg.norm(dS, axis=1).sum())
    out["area"] = {"stl_n_tri": int(len(tris)), "stl_area_m2": float(area_tri.sum()),
                   "stair_n_faces": int(len(rsurf)), "stair_area_m2": area_stair,
                   "ratio_stair_over_stl": area_stair / float(area_tri.sum()),
                   "stl_bbox_min": tris.reshape(-1, 3).min(0).tolist(),
                   "stl_bbox_max": tris.reshape(-1, 3).max(0).tolist()}
    print(f"[1] STL {len(tris)} 三角面 面积 {area_tri.sum():.4f} m²")
    print(f"    阶梯面 {len(rsurf)} 面元 面积 {area_stair:.4f} m² "
          f"→ 比 {area_stair/area_tri.sum():.3f}")
    print(f"    STL bbox {np.round(tris.reshape(-1,3).min(0),3)} .. "
          f"{np.round(tris.reshape(-1,3).max(0),3)}")

    # ---------- 逐角度 ----------
    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    rec = []
    t0 = time.time()
    for a_i, i in enumerate(idx_list):
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                      E_scat, H_scat, i)
        e0mag = float(np.linalg.norm(e0[i]))
        # --- 2)/3) 边界条件与内侧场 ---
        Ea = Etot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        Ha = Htot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
        Hm = Htot[idx_met[:, 0], idx_met[:, 1], idx_met[:, 2]]
        dAn = np.linalg.norm(dS, axis=1)
        nv = dS / (dAn[:, None] + 1e-12)
        Et = np.cross(nv, Ea)                       # 切向总场（PEC 应≈0）
        En = (Ea * nv).sum(axis=1)                  # 法向总场
        bc = {"med_abs_Et": float(np.median(np.linalg.norm(Et, axis=1))),
              "med_abs_Etot": float(np.median(np.linalg.norm(Ea, axis=1))),
              "med_abs_En": float(np.median(np.abs(En))),
              "med_abs_Hair": float(np.median(np.linalg.norm(Ha, axis=1))),
              "med_abs_Hmetal": float(np.median(np.linalg.norm(Hm, axis=1)))}
        bc["Et_over_Etot"] = bc["med_abs_Et"] / bc["med_abs_Etot"]
        bc["En_over_Etot"] = bc["med_abs_En"] / bc["med_abs_Etot"]
        bc["Hmetal_over_Hair"] = bc["med_abs_Hmetal"] / bc["med_abs_Hair"]

        def eff_of(H_s, pts, dSv, sign=+1):
            n = dSv / (np.linalg.norm(dSv, axis=1, keepdims=True) + 1e-12)
            J = np.cross(n, H_s)
            E = nffft_chunked(J, pts, dSv, rhat, k, sign=sign)
            return E

        # --- 4) 相位符号 ---
        E_pos = eff_of(Ha, rsurf, dS, +1)
        E_neg = eff_of(Ha, rsurf, dS, -1)
        # --- 5)/6) STL 三角面 ---
        H_tri = tri_interp(Htot, cen, gx, gy, gz)
        E_tri = eff_of(H_tri, cen, av, +1)

        Et_h = E_ff_h5[i][..., 0].ravel(); Ep_h = E_ff_h5[i][..., 1].ravel()
        thr = 0.05 * np.abs(E_ff_h5[i]).max()
        m = np.abs(Et_h) > thr
        Eh = np.concatenate([Et_h[m], Ep_h[m]])

        def rho_c(Ec):
            t_, p_ = eff_components(Ec, th, ph)
            Ec_ = np.concatenate([t_[m], p_[m]])
            rho = np.abs(np.vdot(Eh, Ec_)) / np.sqrt(
                np.vdot(Ec_, Ec_).real * np.vdot(Eh, Eh).real)
            c = np.vdot(Eh, Ec_) / np.vdot(Eh, Eh)
            return float(rho), float(np.abs(c)), float(np.rad2deg(np.angle(c)))

        # 主瓣子集（|E_ff_h5| 前 10%）
        r_pos = rho_c(E_pos); r_neg = rho_c(E_neg); r_tri = rho_c(E_tri)
        d = {"i": int(i), "bc": bc,
             "nffft_stair_pos": {"rho": r_pos[0], "c_abs": r_pos[1], "c_arg_deg": r_pos[2]},
             "nffft_stair_neg": {"rho": r_neg[0], "c_abs": r_neg[1], "c_arg_deg": r_neg[2]},
             "nffft_stl_tri": {"rho": r_tri[0], "c_abs": r_tri[1], "c_arg_deg": r_tri[2]}}
        rec.append(d)
        if (a_i + 1) % 2 == 0:
            print(f"  [{a_i+1}/{len(idx_list)}] {time.time()-t0:.0f}s "
                  f"| Et/Etot={bc['Et_over_Etot']:.3f} Hm/Ha={bc['Hmetal_over_Hair']:.3f}",
                  flush=True)

    def med(kk, sub=None):
        vals = []
        for x in rec:
            v = x[kk] if sub is None else x[kk][sub]
            vals.append(v)
        return float(np.median(vals))

    out["bc"] = {kk: med("bc", kk) for kk in rec[0]["bc"]}
    out["complex_vs_FEKO"] = {
        "stair_pos_sign": {kk: med("nffft_stair_pos", kk) for kk in ("rho", "c_abs", "c_arg_deg")},
        "stair_neg_sign": {kk: med("nffft_stair_neg", kk) for kk in ("rho", "c_abs", "c_arg_deg")},
        "stl_triangles": {kk: med("nffft_stl_tri", kk) for kk in ("rho", "c_abs", "c_arg_deg")},
    }
    out["per_angle"] = rec

    print("\n===== 2) PEC 边界条件（空气侧体素）=====")
    print(f"  |n̂×E_tot| / |E_tot| 中位 = {out['bc']['Et_over_Etot']:.4f}（PEC 应 <<1）")
    print(f"  |n̂·E_tot| / |E_tot| 中位 = {out['bc']['En_over_Etot']:.4f}")
    print("\n===== 3) 导体内侧磁场 =====")
    print(f"  |H_metal| / |H_air| 中位 = {out['bc']['Hmetal_over_Hair']:.4f}")
    print("\n===== 4)/6) 复场对比 FEKO E_ff（ρ=复相关, c=最佳复标定因子）=====")
    for kk, lab in (("stair_pos_sign", "阶梯面 +jk"),
                    ("stair_neg_sign", "阶梯面 -jk"),
                    ("stl_triangles", "STL三角面 +jk")):
        s = out["complex_vs_FEKO"][kk]
        print(f"  {lab:14s}: ρ={s['rho']:.3f}  |c|={s['c_abs']:.3f}  arg(c)={s['c_arg_deg']:+.1f}°")

    with open(os.path.join(RESULT_DIR, "_diag_nffft_audit2.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f"\n已存: results/_diag_nffft_audit2.json")


if __name__ == "__main__":
    main()
