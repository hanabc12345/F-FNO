# -*- coding: utf-8 -*-
"""
_diag_mesh_true.py — `mesh_true` 臂的「三线性插值口径」定位与修复实验
================================================================================
背景（AGENTS.md 遗留①）：exp_po_mesh.py 的 2×2 对照里三个臂都符合预期
（mesh_po 3.181 < vox_po 3.74；vox_true 3.48 < vox_po），唯独
mesh_true（真网格 + 采样真值电流 n̂×H_tot）= 6.595 dB 反劣于 mesh_po。

先验怀疑：E_scat/H_scat 采样在 h = 0.03125 m = **λ/3.2**（每波长仅 3.2 个样本）的
体素网格上，而 mesh_true 必须把体素场**三线性插值**到 STL 三角面质心。对复场直接
做线性插值，在 3.2 样本/波长时中点幅度衰减 cos(π/3.2) = 0.556（最坏 −5.1 dB）——
这是插值伪影，不是物理误差。vox_true 不需要插值（电流就在网格点上取值），
所以两条臂的差异恰好落在插值上。

两步实验（先定位、再修）
--------------------------------------------------------------------------------
  Stage 1  **解析已知答案**：把入射场 H_inc 在网格上采样 → 三线性插值到质心，
           与解析 H_inc(质心) 逐点比幅度比 / 相位误差；并由此构造两条电流
           （J=2n̂×H_inc^exact vs J=2n̂×H_inc^interp）算出 RCS 差 —— 纯插值伪影的贡献。
  Stage 2  **口径修复对照**（同一批角、同一网格）：
           T0 = 现行：J = n̂ × tri_interp(H_inc + H_scat)
           T1 = 修复：J = n̂ × (H_inc^解析(质心) + tri_interp(H_scat))   ← 入射不插值
           各自与 FEKO 真值比 med|Δ|（统计口径与 exp_po_mesh 的 rcs_dB_err_median 一致）。

若 |T1−T0| 小 → 插值伪影不是元凶，该臂属数据分辨率受限不可修（结论同样有用）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_mesh_true.py --angles 12
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_mesh_true.py            # 全 468 角
产出：results/_diag_mesh_true.json
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
from fno_f16_3d_p4_nffft import direction_grid                  # noqa: E402
from exp_po_mesh import (load_mesh, mesh_outside_voxel, nffft_from,   # noqa: E402
                         ETA0, H, STL, RESULT_DIR)
from exp_po_locality import ray_occlusion                       # noqa: E402
from po_patch_data import canon_phase                           # noqa: E402
from _diag_nffft_audit import tri_interp                        # noqa: E402
from _exp_common import _rcs_stats                              # noqa: E402


def _sigma(E_ff, th, ph, e0m, shape):
    return (4 * np.pi * (np.abs((E_ff * th).sum(1)) ** 2
                         + np.abs((E_ff * ph).sum(1)) ** 2) / e0m ** 2).reshape(shape)


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
    lam = 2 * np.pi / k
    metal = eps > 1.5
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    print(f"  STL {len(cen)} 面  面积 {dAm.sum():.4f} m²；"
          f"空气侧起点成功 {int(air_ok.sum())}")
    print(f"  λ = {lam:.5f} m   网格步 h = {H:.5f} m = λ/{lam/H:.1f}"
          f"   ⇒ 每波长仅 {lam/H:.1f} 个样本")
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")

    # ---- Stage 1 统计量 ----
    s1 = {"amp_ratio_med": [], "amp_ratio_p10": [], "amp_ratio_p90": [],
          "phase_err_med_deg": [], "delta_med_dB": [], "delta_p90_dB": [],
          "delta_bias_dB": [], "delta_mono_dB": []}
    acc = {v: [] for v in ("T0_interp_all", "T1_analytic_inc", "T2_T1_rescale_scat")}
    kept = []
    rho_used = []
    t0 = time.time()
    with h5py.File(M.H5, "r") as f:
        for c, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei)
            ei = ei * ph0
            lit = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
            if not lit.any():
                continue
            kept.append(int(i))
            psi_tri = np.exp(-1j * beta[i] * (cen @ ki))

            # ---------- Stage 1：入射场插值伪影（解析已知答案）----------
            phase_g = np.exp(-1j * beta[i] * (ki[0] * X + ki[1] * Y + ki[2] * Z))
            e_g = ei[None, None, None, :] * phase_g[..., None]
            h_g = np.cross(ki[None, None, None, :],
                           np.broadcast_to(e_g, (64, 48, 32, 3))) / ETA0
            h_interp = tri_interp(h_g, cen, gx, gy, gz)          # 插值到质心
            e_exact = ei[None, :] * psi_tri[:, None]             # 解析（无插值）
            h_exact = np.cross(ki[None, :], e_exact) / ETA0
            mag_i = np.abs(h_interp[lit]).ravel()
            mag_e = np.abs(h_exact[lit]).ravel()
            ratio = mag_i / np.maximum(mag_e, 1e-30)
            s1["amp_ratio_med"].append(float(np.median(ratio)))
            s1["amp_ratio_p10"].append(float(np.percentile(ratio, 10)))
            s1["amp_ratio_p90"].append(float(np.percentile(ratio, 90)))
            cph = (h_interp * np.conj(h_exact)).sum(axis=1)
            s1["phase_err_med_deg"].append(float(np.median(np.abs(
                np.rad2deg(np.angle(cph[lit]))))))

            J_ex = 2.0 * np.cross(nvm, np.cross(ki, ei)) / ETA0 * psi_tri[:, None]
            J_in = 2.0 * np.cross(nvm, h_interp)
            J_ex[~lit] = 0.0
            J_in[~lit] = 0.0
            sg_ex = _sigma(nffft_from(J_ex, dAm, cen, rhat, k, args.chunk),
                           th, ph, e0m, shape)
            sg_in = _sigma(nffft_from(J_in, dAm, cen, rhat, k, args.chunk),
                           th, ph, e0m, shape)
            dd = 10 * np.log10(np.maximum(sg_in, 1e-30) / np.maximum(sg_ex, 1e-30))
            s1["delta_med_dB"].append(float(np.median(np.abs(dd))))
            s1["delta_p90_dB"].append(float(np.percentile(np.abs(dd), 90)))
            s1["delta_bias_dB"].append(float(np.median(dd)))
            it_i = int(np.rint(angles[i, 0] / (ff_theta[1] - ff_theta[0])))
            ip_i = int(np.rint(angles[i, 1] / (ff_phi[1] - ff_phi[0])))
            s1["delta_mono_dB"].append(float(dd[it_i, ip_i]))

            # ---------- Stage 2：口径修复对照 ----------
            Hsc = f["H_scat"][i].astype(np.complex128) * ph0
            Htot = h_g + Hsc
            J0 = np.cross(nvm, tri_interp(Htot, cen, gx, gy, gz))     # T0 现行
            Hs_tri = tri_interp(Hsc, cen, gx, gy, gz)
            J1 = np.cross(nvm, h_exact + Hs_tri)                     # T1 修复：入射解析
            # T2 诊断上界：把散射场的插值衰减按**入射场实测到的衰减因子**补回
            #  （非物理标定，仅用于定量分解"散射场插值伪影"占多少）
            rho = float(np.median(ratio))
            rho_used.append(rho)
            J2 = np.cross(nvm, h_exact + Hs_tri / max(rho, 1e-6))
            J0[~lit] = 0.0
            J1[~lit] = 0.0
            J2[~lit] = 0.0
            acc["T0_interp_all"].append(
                _sigma(nffft_from(J0, dAm, cen, rhat, k, args.chunk), th, ph, e0m, shape))
            acc["T1_analytic_inc"].append(
                _sigma(nffft_from(J1, dAm, cen, rhat, k, args.chunk), th, ph, e0m, shape))
            acc["T2_T1_rescale_scat"].append(
                _sigma(nffft_from(J2, dAm, cen, rhat, k, args.chunk), th, ph, e0m, shape))
            if (c + 1) % 10 == 0:
                print(f"    {c+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)

    med = {kk: float(np.median(v)) for kk, v in s1.items()}
    res = {
        "n_angle": int(len(aidx)), "h_m": float(H), "lambda_m": float(lam),
        "samples_per_lambda": float(lam / H), "wall_s": float(time.time() - t0),
        "stage1_incident_interp_artifact": {
            "amp_ratio_median": med["amp_ratio_med"],
            "amp_ratio_chord_p10": med["amp_ratio_p10"],
            "amp_ratio_chord_p90": med["amp_ratio_p90"],
            "phase_err_median_deg": med["phase_err_med_deg"],
            "rcs_delta_med_abs_dB": med["delta_med_dB"],
            "rcs_delta_p90_abs_dB": med["delta_p90_dB"],
            "rcs_delta_bias_dB": med["delta_bias_dB"],
            "rcs_delta_mono_bias_dB": med["delta_mono_dB"],
            "theory_amp_loss_midpoint_dB": float(
                20 * np.log10(abs(np.cos(np.pi * (lam / H) ** -1))))},
        "stage2_variants": {},
    }
    for v, lst in acc.items():
        res["stage2_variants"][v] = _rcs_stats(np.stack(lst), rcs_true[np.asarray(kept)])

    print("\n[Stage 1 入射场插值伪影（解析已知答案）]")
    a = res["stage1_incident_interp_artifact"]
    print(f"  |H_interp|/|H_exact| 中位 {a['amp_ratio_median']:.4f}  "
          f"（10–90% 分位 {a['amp_ratio_chord_p10']:.4f}–{a['amp_ratio_chord_p90']:.4f}）"
          f"  理论中点最坏 {a['theory_amp_loss_midpoint_dB']:.2f} dB")
    print(f"  相位误差中位 {a['phase_err_median_deg']:.2f}°")
    print(f"  纯插值造成的 RCS 偏差：med|Δ| {a['rcs_delta_med_abs_dB']:.3f} dB  "
          f"P90 {a['rcs_delta_p90_abs_dB']:.3f} dB  bias {a['rcs_delta_bias_dB']:+.3f} dB  "
          f"单站 bias {a['rcs_delta_mono_bias_dB']:+.3f} dB")

    print(f"\n[Stage 2 口径对照] {len(aidx)} 角，真值 = FEKO")
    for v, st_ in res["stage2_variants"].items():
        print(f"  {v:<20} med|Δ| {st_['rcs_dB_err_median']:6.3f} dB  "
              f"P90 {st_['rcs_dB_err_p90']:6.3f} dB  corr {st_['rcs_corr_median']:.4f}")
    d0 = res["stage2_variants"]["T0_interp_all"]["rcs_dB_err_median"]
    d1 = res["stage2_variants"]["T1_analytic_inc"]["rcs_dB_err_median"]
    d2 = res["stage2_variants"]["T2_T1_rescale_scat"]["rcs_dB_err_median"]
    print(f"  ⇒ T1 − T0 = {d1 - d0:+.3f} dB（入射场插值伪影的直接贡献）")
    print(f"  ⇒ T2 − T1 = {d2 - d1:+.3f} dB（散射场插值伪影的残余贡献，T2 用实测衰减因子 "
          f"{float(np.median(rho_used)):.4f} 作非物理标定）")
    res["T2_rho_used_median"] = float(np.median(rho_used))
    print(f"  参考：同口径无伪影的臂 —— vox_true 3.48 / mesh_po 3.18 / vox_po 3.74 dB")

    jp = os.path.join(RESULT_DIR, "_diag_mesh_true.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return res


if __name__ == "__main__":
    main()
