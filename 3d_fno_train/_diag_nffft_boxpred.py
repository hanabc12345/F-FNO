# -*- coding: utf-8 -*-
"""_diag_nffft_boxpred.py — 盒面等效原理 NFFFT 作为**生产级**近→远场变换的验证

背景（_diag_nffft_audit4.py 结论）：
  · 现有"PEC 阶梯表面"路由：truth 场 → 中位 3.50 dB / P90 8.84 / corr 0.613
  · "近场盒面等效原理"路由：truth 场 → 中位 0.161 dB / P90 0.59 / corr 0.998
  ⇒ 3.55 dB 的 floor 全部来自 PEC 阶梯表面提取 + 离面场采样，与相位约定/归一化无关。

本脚本回答下一步：**能否把盒面路由直接用作生产 NFFFT**（用于代理模型预测场）？
  网络输出的就是整网格的散射场，盒面（网格外边界）上的样本它同样会预测，
  因此盒面路由只依赖 E_scat/H_scat 两通道，**不需要** eps 掩膜、不需要表面提取、
  不需要法向估计、不需要解析入射场重建、也不需要 PEC 假设。

对照（同一角度子集，同一 ckpt）：
  ① truth + 阶梯面路由   （= 论文里的 3.55 dB "pipeline floor"）
  ② truth + 盒面路由     （管线自身可达精度）
  ③ pred  + 阶梯面路由   （= 论文里的 5.28 dB）
  ④ pred  + 盒面路由     （是否显著下降 → 决定论文主线是否可改）

另报：模型在**盒面**上的近场相对误差（盒面路由的误差来源）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_boxpred.py --ckpt ckpt_imp_interp_plain_s0.pt
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_nffft_boxpred.py --ckpt ckpt_full_p3.pt --nangles 40
产出：results/_diag_nffft_boxpred.json
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
from fno_f16_3d_p4_nffft import (direction_grid, surface_parts, rcs_from_surface,
                                 build_tot_fields, sample_surface_field)
from _diag_nffft_audit import rcs_stats
from _diag_nffft_audit4 import box_surface, nffft_NL

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi


def box_rcs(E_s, H_s, bx, idx, rhat, th, ph, k, e0mag, shape):
    """盒面等效原理：J=n̂×H_s, M=−n̂×E_s（散射场，无需入射场）
    E_ff = (jk/4π)[−η₀(N−N·r̂ r̂) + r̂×L]（Balanis 12-10 相对符号）
    返回 (rcs (Ndir,), E2 (Ndir,2))"""
    es = E_s[idx[:, 0], idx[:, 1], idx[:, 2]]
    hs = H_s[idx[:, 0], idx[:, 1], idx[:, 2]]
    n = bx["dS"] / np.linalg.norm(bx["dS"], axis=1, keepdims=True)
    J = np.cross(n, hs)
    Mm = -np.cross(n, es)
    Nperp, rcL = nffft_NL(J, Mm, bx["pts"], bx["dS"], rhat, k)
    E_ff = (1j * k / (4.0 * np.pi)) * (-ETA0 * Nperp + rcL)
    E2 = np.stack([(E_ff * th).sum(axis=1), (E_ff * ph).sum(axis=1)], axis=1)
    rcs = 4.0 * np.pi * (np.abs(E2) ** 2).sum(axis=1) / e0mag ** 2
    return rcs, E2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_imp_interp_plain_s0.pt")
    ap.add_argument("--nangles", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1, help=">1 时对 ckpt_imp_<tag>_s*.pt 取平均")
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
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    ijk, bpts, bdS, _ = box_surface(gx, gy, gz, h, inset=0)
    bx = dict(ijk=ijk, pts=bpts, dS=bdS)
    print(f"盒面 {len(bpts)} 面元, k={k:.3f}, h={h:.5f}")

    # 阶梯面（对照）
    idxs, rsurf, dS_s = surface_parts(eps, gx, gy, gz)

    # 载入 ckpt
    tag = args.ckpt.replace("ckpt_imp_", "").replace("ckpt_", "").replace(".pt", "")
    ck = torch.load(os.path.join(RESULT_DIR, args.ckpt), map_location="cpu", weights_only=False)
    cfg = ck["config"]; st = ck["stats"]
    ym, ys = st["ym"], st["ys"]
    xm = st.get("xm", st.get("x_inc_mean")); xs = st.get("xs", st.get("x_inc_std"))
    model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=12).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"ckpt={args.ckpt} modes={cfg['modes']} width={cfg['width']}")

    n = min(args.nangles, len(angles))
    idx_list = np.linspace(0, len(angles) - 1, n).astype(int)
    print(f"角度 {len(idx_list)}: linspace(0,467,{n})")

    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)
    epsm = (eps > 1.5).astype(np.float32)

    rows = {"truth_surface": [], "truth_box": [], "pred_surface": [], "pred_box": []}
    nf = {"box_rel_E": [], "box_rel_H": [], "all_rel_E": [], "all_rel_H": [], "surf_rel_H": []}
    t0 = time.time()

    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(idx_list):
            # ---------- 预测 ----------
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
            Ep = p[0:3] + 1j * p[3:6]      # (3,64,48,32)
            Hp = p[6:9] + 1j * p[9:12]
            Ep_g = np.transpose(Ep, (1, 2, 3, 0))     # (64,48,32,3)
            Hp_g = np.transpose(Hp, (1, 2, 3, 0))

            # ---------- 真值散射场（只取该角度）----------
            Es_g = f["E_scat"][i]
            Hs_g = f["H_scat"][i]

            # ---------- 近场误差（全网格 / 盒面 / 表面）----------
            def relerr(P, T):
                return float(np.linalg.norm(P - T) / max(np.linalg.norm(T), 1e-30))
            nf["all_rel_E"].append(relerr(Ep_g, Es_g))
            nf["all_rel_H"].append(relerr(Hp_g, Hs_g))
            nf["box_rel_E"].append(relerr(Ep_g[ijk[:, 0], ijk[:, 1], ijk[:, 2]],
                                          Es_g[ijk[:, 0], ijk[:, 1], ijk[:, 2]]))
            nf["box_rel_H"].append(relerr(Hp_g[ijk[:, 0], ijk[:, 1], ijk[:, 2]],
                                          Hs_g[ijk[:, 0], ijk[:, 1], ijk[:, 2]]))

            e0mag = float(np.linalg.norm(e0[i])); tgt = rcs_true[i]

            # ① truth + 阶梯面
            Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz,
                                          f["E_scat"], f["H_scat"], i)
            E_s = sample_surface_field(Etot, idxs); H_s = sample_surface_field(Htot, idxs)
            nf["surf_rel_H"].append(relerr(
                Hp_g[idxs[:, 0], idxs[:, 1], idxs[:, 2]], H_s))
            r1 = rcs_from_surface(E_s, H_s, dS_s, rsurf, rhat, th, ph, k, e0mag)
            rows["truth_surface"].append(rcs_stats(r1.reshape(shape), tgt))

            # ② truth + 盒面
            r2, _ = box_rcs(Es_g, Hs_g, bx, ijk, rhat, th, ph, k, e0mag, shape)
            rows["truth_box"].append(rcs_stats(r2.reshape(shape), tgt))

            # ③ pred + 阶梯面（用预测场 + 解析入射场）
            Xa, Ya, Za = np.meshgrid(gx, gy, gz, indexing="ij")
            ph_i = np.exp(-1j * beta[i] * (khat[i, 0] * Xa + khat[i, 1] * Ya + khat[i, 2] * Za))
            e_inc = e0[i][None, None, None, :] * ph_i[..., None]
            h_inc = (1.0 / ETA0) * np.cross(khat[i][None, None, None, :],
                                            np.broadcast_to(e_inc, (64, 48, 32, 3)))
            E_sp = sample_surface_field(e_inc + Ep_g, idxs)
            H_sp = sample_surface_field(h_inc + Hp_g, idxs)
            r3 = rcs_from_surface(E_sp, H_sp, dS_s, rsurf, rhat, th, ph, k, e0mag)
            rows["pred_surface"].append(rcs_stats(r3.reshape(shape), tgt))

            # ④ pred + 盒面
            r4, _ = box_rcs(Ep_g, Hp_g, bx, ijk, rhat, th, ph, k, e0mag, shape)
            rows["pred_box"].append(rcs_stats(r4.reshape(shape), tgt))

            if (a + 1) % 10 == 0:
                print(f"  [{a+1}/{len(idx_list)}] {time.time()-t0:.0f}s", flush=True)

    def sum_rows(lst):
        return {"rcs_dB_err_median": float(np.median([x["masked"]["median_dB"] for x in lst])),
                "rcs_dB_err_p90": float(np.median([x["masked"]["p90_dB"] for x in lst])),
                "rcs_corr_median": float(np.median([x["masked"]["corr_dB"] for x in lst]))}

    out = {"ckpt": args.ckpt, "tag": tag, "n_angles": int(len(idx_list)),
           "angles": [int(v) for v in idx_list], "box_n_faces": int(len(bpts)),
           "nearfield": {kk: float(np.median(v)) for kk, v in nf.items()},
           **{kk: sum_rows(v) for kk, v in rows.items()}}
    jp = os.path.join(RESULT_DIR, "_diag_nffft_boxpred.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False)

    print("\n===== 远场 RCS：四种组合（同一批角度）=====")
    print(f"  {'组合':<28}{'中位 dB':>9}{'P90 dB':>9}{'corr':>8}{'vs 阶梯面':>11}")
    base3 = out["truth_surface"]["rcs_dB_err_median"]
    base5 = out["pred_surface"]["rcs_dB_err_median"]
    for kk, lab in (("truth_surface", "① truth + 阶梯面（论文 floor）"),
                    ("truth_box", "② truth + 盒面等效原理"),
                    ("pred_surface", "③ pred  + 阶梯面（论文 5.28）"),
                    ("pred_box", "④ pred  + 盒面等效原理")):
        v = out[kk]
        ref = base3 if kk.startswith("truth") else base5
        print(f"  {lab:<28}{v['rcs_dB_err_median']:>9.3f}{v['rcs_dB_err_p90']:>9.2f}"
              f"{v['rcs_corr_median']:>8.3f}{v['rcs_dB_err_median']/ref:>10.2f}x")

    print("\n===== 模型近场相对误差（盒面路由的误差来源）=====")
    print(f"  全网格  E {out['nearfield']['all_rel_E']*100:.1f}%  H {out['nearfield']['all_rel_H']*100:.1f}%")
    print(f"  盒面上  E {out['nearfield']['box_rel_E']*100:.1f}%  H {out['nearfield']['box_rel_H']*100:.1f}%")
    print(f"  表面处  H {out['nearfield']['surf_rel_H']*100:.1f}%")
    print(f"\n已存: {jp}")


if __name__ == "__main__":
    main()
