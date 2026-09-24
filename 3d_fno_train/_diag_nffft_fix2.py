# -*- coding: utf-8 -*-
"""_diag_nffft_fix2.py — 寻找"既严格又良态"的 NFFFT 路由

已知（_diag_nffft_audit4.py / _diag_nffft_boxpred.py）：
  · 近场盒（网格外边界）等效原理：真值 0.161 dB / corr 0.998 —— 严格、精确；
    但盒面离目标 2.5–2.8λ，散射场弱 → 求和剧烈相消 → 对模型误差**病态**
    （pred 时反而退化到 11.5 dB）。
  · PEC 阶梯面路由：真值 3.50 dB（离散化硬限，_diag_surface_fix.py 试了 9 种
    表面电流修正，最好也只到 3.405 dB）；但对模型误差**良态**（pred 5.28 dB）。
  ⇒ 需要第三条路：**贴体闭合面等效原理**——面严格包住目标（等效原理成立）、
    同时离目标足够近（场强、相消弱 → 良态）。

本脚本对照：
  1) 贴体/多层盒面扫描：以"金属体素包围盒 + m 体素"为界，m=1..10
     （m=1 即最贴近的网格对齐闭合面），全部用网格点原位值 + 梯形求积；
  2) 阶梯面闭合 Huygens（J 与 M 都用，不再假设 PEC）——检验"阶梯面是否为合法包围面"；
  3) 对最优路由，附加 ckpt 预测场结果（--ckpt）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_fix2.py --nangles 12
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_fix2.py --ckpt ckpt_imp_interp_plain_s0.pt
产出：results/_diag_nffft_fix2.json + .png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import (direction_grid, nffft, surface_parts,
                                 rcs_from_surface, build_tot_fields, sample_surface_field)
from _diag_nffft_audit import surface_parts_full, rcs_stats
from _diag_nffft_audit4 import nffft_NL

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi


def box_ijk(g, ranges, h):
    """ranges = ((x0,x1),(y0,y1),(z0,z1)) 索引闭区间 → 6 面网格点 + 梯形权重 dS。
    返回 ijk(N,3), pts(N,3), dS(N,3)。"""
    axis = [np.arange(a, b + 1) for (a, b) in ranges]
    W = []
    for ax in range(3):
        n = len(g[ax])
        w = np.ones(len(axis[ax]))
        w[axis[ax] == 0] = 0.5
        w[axis[ax] == n - 1] = 0.5
        W.append(w)
    ijk, pts, dS = [], [], []
    for fix_ax in range(3):
        for layer, sg in ((axis[fix_ax][0], -1), (axis[fix_ax][-1], +1)):
            u_ax, v_ax = [a for a in (0, 1, 2) if a != fix_ax]
            U, V = np.meshgrid(np.arange(len(axis[u_ax])), np.arange(len(axis[v_ax])), indexing="ij")
            WU, WV = np.meshgrid(W[u_ax], W[v_ax], indexing="ij")
            idx = np.zeros((U.size, 3), dtype=np.int64)
            idx[:, fix_ax] = layer
            idx[:, u_ax] = axis[u_ax][U.ravel()]
            idx[:, v_ax] = axis[v_ax][V.ravel()]
            p = np.stack([g[a][idx[:, a]] for a in range(3)], axis=1)
            n = np.zeros((idx.shape[0], 3)); n[:, fix_ax] = float(sg)
            ijk.append(idx); pts.append(p)
            dS.append(n * (h * h * (WU * WV).ravel())[:, None])
    return np.concatenate(ijk), np.concatenate(pts), np.concatenate(dS)


def surf_eq(J, Mm, pts, dS, rhat, k, th, ph, e0mag, shape):
    """E_ff = (jk/4π)[−η₀N⊥ + r̂×L]（Balanis 相对符号）→ RCS 图"""
    Nperp, rcL = nffft_NL(J, Mm, pts, dS, rhat, k)
    E_ff = (1j * k / (4.0 * np.pi)) * (-ETA0 * Nperp + rcL)
    rcs = 4.0 * np.pi * (np.abs((E_ff * th).sum(axis=1)) ** 2
                         + np.abs((E_ff * ph).sum(axis=1)) ** 2) / e0mag ** 2
    return rcs.reshape(shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=12)
    ap.add_argument("--margins", default="1,2,3,4,6,8,10")
    ap.add_argument("--ckpt", default="", help="给定则在最优路由上并报预测场")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0]); h = float(gx[1] - gx[0])
    g = [gx, gy, gz]
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    mi = np.argwhere(eps > 1.5)
    lo, hi = mi.min(axis=0), mi.max(axis=0)
    print(f"金属体素索引范围 {lo}..{hi}（{len(mi)} 个）")
    gshape = np.array([len(gx), len(gy), len(gz)])

    boxes = {}
    for m in [int(x) for x in args.margins.split(",")]:
        r = tuple((max(0, lo[a] - m), min(gshape[a] - 1, hi[a] + m)) for a in range(3))
        ijk, pts, dS = box_ijk(g, r, h)
        boxes[m] = dict(ijk=ijk, pts=pts, dS=dS, area=float(np.linalg.norm(dS, axis=1).sum()),
                        bound=r)
        print(f"  margin={m}: {len(pts)} 面元  面积 {boxes[m]['area']:.3f} m²  界 {r}")

    idx_list = np.linspace(0, len(angles) - 1, args.nangles).astype(int)
    res = {f"box_m{m}": [] for m in boxes}
    res["staircase_huygens"] = []
    # 阶梯面（对照）
    idx_air, idx_met, rsurf, dS_s = surface_parts_full(eps, gx, gy, gz, h)
    res["staircase_pec"] = []
    n_s = dS_s / np.linalg.norm(dS_s, axis=1, keepdims=True)

    t0 = time.time()
    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(idx_list):
            Es_g = f["E_scat"][i]; Hs_g = f["H_scat"][i]
            e0mag = float(np.linalg.norm(e0[i])); tgt = rcs_true[i]

            for m, bx in boxes.items():
                ik = bx["ijk"]
                Es = Es_g[ik[:, 0], ik[:, 1], ik[:, 2]]
                Hs = Hs_g[ik[:, 0], ik[:, 1], ik[:, 2]]
                n = bx["dS"] / np.linalg.norm(bx["dS"], axis=1, keepdims=True)
                res[f"box_m{m}"].append(rcs_stats(
                    surf_eq(np.cross(n, Hs), -np.cross(n, Es), bx["pts"], bx["dS"],
                            rhat, k, th, ph, e0mag, shape), tgt))

            # 阶梯面闭合 Huygens（J 与 M 都用）
            Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                          f["E_scat"], f["H_scat"], i)
            E_a = Etot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
            H_a = Htot[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
            res["staircase_huygens"].append(rcs_stats(
                surf_eq(np.cross(n_s, H_a), -np.cross(n_s, E_a), rsurf, dS_s,
                        rhat, k, th, ph, e0mag, shape), tgt))
            res["staircase_pec"].append(rcs_stats(
                rcs_from_surface(E_a, H_a, dS_s, rsurf, rhat, th, ph, k, e0mag).reshape(shape), tgt))

            if (a + 1) % 4 == 0:
                print(f"  [{a+1}/{len(idx_list)}] {time.time()-t0:.0f}s", flush=True)

    def agg(lst):
        return {"median_dB": float(np.median([x["masked"]["median_dB"] for x in lst])),
                "p90_dB": float(np.median([x["masked"]["p90_dB"] for x in lst])),
                "corr_dB": float(np.median([x["masked"]["corr_dB"] for x in lst]))}

    out = {"n_angles": int(len(idx_list)), "angles": [int(x) for x in idx_list],
           "kh2": float(k * h / 2), "results": {kk: agg(v) for kk, v in res.items()},
           "area_m2": {f"box_m{m}": boxes[m]["area"] for m in boxes},
           "bound_idx": {f"box_m{m}": boxes[m]["bound"] for m in boxes}}

    # 可选：预测场（对所有盒面 margin + 两种阶梯面路由都算，用于找"真值准×预测稳"的折中）
    if args.ckpt:
        ck = torch.load(os.path.join(RESULT_DIR, args.ckpt), map_location="cpu", weights_only=False)
        cfg = ck["config"]; st = ck["stats"]
        ym, ys = st["ym"], st["ys"]
        xm = st.get("xm", st.get("x_inc_mean")); xs = st.get("xs", st.get("x_inc_std"))
        model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                         in_ch=7, out_ch=12).to(device)
        model.load_state_dict(ck["model_state"]); model.eval()
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)
        epsm = (eps > 1.5).astype(np.float32)
        pred_rows = {f"pred_box_m{m}": [] for m in boxes}
        pred_rows["pred_staircase_huygens"] = []
        pred_rows["pred_staircase_pec"] = []
        with h5py.File(M.H5, "r") as f:
            for i in idx_list:
                phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
                ei = e0[i][None, None, None, :] * phase[..., None]
                x = np.zeros((1, 7, 64, 48, 32), dtype=np.float32)
                x[0, 0] = epsm
                x[0, 1:4] = ei.real.transpose(3, 0, 1, 2)
                x[0, 4:7] = ei.imag.transpose(3, 0, 1, 2)
                x = (x - xm) / xs
                with torch.no_grad():
                    p = model(torch.from_numpy(x).to(device))[0].cpu().numpy()
                p = p * ys.reshape(12, 1, 1, 1) + ym.reshape(12, 1, 1, 1)
                Ep = np.transpose(p[0:3] + 1j * p[3:6], (1, 2, 3, 0))
                Hp = np.transpose(p[6:9] + 1j * p[9:12], (1, 2, 3, 0))
                e0mag = float(np.linalg.norm(e0[i])); tgt = rcs_true[i]

                for m, bx in boxes.items():
                    ik = bx["ijk"]
                    n = bx["dS"] / np.linalg.norm(bx["dS"], axis=1, keepdims=True)
                    E_b = Ep[ik[:, 0], ik[:, 1], ik[:, 2]]
                    H_b = Hp[ik[:, 0], ik[:, 1], ik[:, 2]]
                    pred_rows[f"pred_box_m{m}"].append(rcs_stats(
                        surf_eq(np.cross(n, H_b), -np.cross(n, E_b), bx["pts"], bx["dS"],
                                rhat, k, th, ph, e0mag, shape), tgt))

                ph_i = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
                e_inc = e0[i][None, None, None, :] * ph_i[..., None]
                h_inc = (1.0 / ETA0) * np.cross(khat[i][None, None, None, :],
                                                np.broadcast_to(e_inc, (64, 48, 32, 3)))
                Ea = (e_inc + Ep)[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
                Ha = (h_inc + Hp)[idx_air[:, 0], idx_air[:, 1], idx_air[:, 2]]
                pred_rows["pred_staircase_huygens"].append(rcs_stats(
                    surf_eq(np.cross(n_s, Ha), -np.cross(n_s, Ea), rsurf, dS_s,
                            rhat, k, th, ph, e0mag, shape), tgt))
                pred_rows["pred_staircase_pec"].append(rcs_stats(
                    rcs_from_surface(Ea, Ha, dS_s, rsurf, rhat, th, ph, k, e0mag).reshape(shape), tgt))
        out["pred"] = {kk: agg(v) for kk, v in pred_rows.items()}

    jp = os.path.join(RESULT_DIR, "_diag_nffft_fix2.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False, default=float)

    print("\n===== 真值场：各闭合面路由 =====")
    print(f"  {'路由':<24}{'面积 m²':>9}{'中位 dB':>9}{'P90 dB':>9}{'corr':>8}")
    for kk in list(out["results"]):
        v = out["results"][kk]
        a_ = out["area_m2"].get(kk, float("nan"))
        print(f"  {kk:<24}{a_:>9.3f}{v['median_dB']:>9.3f}{v['p90_dB']:>9.2f}{v['corr_dB']:>8.3f}")
    if "pred" in out:
        print("\n===== 预测场 =====")
        print(f"  {'路由':<24}{'中位 dB':>9}{'P90 dB':>9}{'corr':>8}")
        for kk, v in out["pred"].items():
            print(f"  {kk:<24}{v['median_dB']:>9.3f}{v['p90_dB']:>9.2f}{v['corr_dB']:>8.3f}")
    print(f"\n已存: {jp}")


if __name__ == "__main__":
    main()
