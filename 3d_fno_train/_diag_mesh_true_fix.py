# -*- coding: utf-8 -*-
"""
_diag_mesh_true_fix.py — A1：`mesh_true` 臂「场投射口径」的系统修复对照
================================================================================
背景
  `exp_po_mesh.py` 的 2×2 受控对照（全 468 角）：
      vox_po  3.745   阶梯面 + 解析 PO
      vox_true 3.483  阶梯面 + 真值近场（空气侧体素直接取值，**不插值**）
      mesh_po 3.161   真 STL 面 + 解析 PO        ← 现行封板基线
      mesh_true 5.776 真 STL 面 + 真值近场        ← |c| 0.946→0.314
  三条臂都符合"真值优于近似"，唯独 `mesh_true` **用真值反而差 2.62 dB**。
  这是自相矛盾信号 ⇒ 只能是**把体素场搬到 STL 三角面心的口径坏了**，不是物理结论。
  （即 AGENTS.md 2026-09-18 遗留①「mesh_true 臂三线性插值口径未修」，至今未修。）

已定位的两条机制（互相独立，都可解析预测）
  (M1) **三线性插值低通衰减**：复场在 h = λ/3.198 上采样，每体素相位步 kh = 1.965 rad
       三角核幅度响应 sinc²(ω/2)|_{ω=kh} = 0.7166；实测中位 0.712 ⇒ 吻合。
  (M2) **八点模板跨界**：三角面心正好落在金属/空气分界，三线性 8 点里有一半在金属内部
       （PEC 内 H_scat = −h_inc ⇒ 与外侧解**不连续**）⇒ 把两个不同的解混在一起。

本脚本在同一批角、同一 STL、同一真值场、同一 NFFFT 下做 **7 种投射口径**对照
（几何完全一致，只有"如何把体素场搬到三角面心"这一环不同）：

  C0 raw_tri       : tri_interp(H_scat) @ cen            （现行口径，复现 5.776）
  C1 nn_air        : 取 cen + h/2·n̂ 的**最近空气侧体素**（零阶，不跨界，无衰减）
  C2 nn_air_phase  : C1 + 用局部镜反载波把值**相位外推**回 cen
  C3 strip_kr      : 剥**局部镜反载波** k̂_r = k̂_i − 2(k̂_i·n̂)n̂ 后三线性插值包络，再乘回
  C4 strip_ki      : 剥**入射载波** k̂_i（对照：检验"局部镜反"是否必要）
  C5 one_sided     : 三线性只在**空气侧**邻点上取（权重截断后归一，单侧插值）
  C6 consist_nn    : 零阶，且**入射也取在同一体素**（消除 C1 里"入射在 cen、
                     散射在 r_idx"的内部不一致 —— C1 的 |c|=0.710 疑似由此而来）
  mesh_po          : STL + 解析 PO（同批参考，应对上 3.161）

另输出 `rescaled` 诊断表：把每个口径的 σ 统一除以 |c|²（剥掉逐角系统性幅度偏置），
只暴露**方向图形状**的质量 —— 对 C1 的 |c|=0.710 尤其实用。

（首轮 12 角已跑，结果：C0 5.880 / C1 **2.915** / C2 3.250 / C3 5.082 / C4 6.868 /
  C5 3.518 / mesh_po 3.067 ⇒ 主线机制是 **M2 跨界混合**（占 2.36 dB），M1 低通只占 0.60 dB。）

判读（预注册，与 `mesh_po` 比）
  · 最佳口径 **≪ 3.161** ⇒ 「真曲面 + 真近场」路线成立 ⇒ 部署链路可换成真 STL 积分面
    （不改 h、不重训、不重跑 FEKO，纯后处理）；
  · 最佳口径 **≈ 或 > 3.161** ⇒ 地板在**近场采样分辨率本身**（λ/3.2 不够），
    h 才是唯一杠杆，与 `exp_po_mesh` 的判决 C 一致（也补上 P5-2 被撤回后的归因空位）。

自检
  (a) C0 应与 `_diag_nffft_audit.tri_interp` 逐位一致（首角 assert allclose）；
  (b) C0 在 12 角应复现 `_diag_mesh_true.py` 的 T1 值 ≈5.10 dB。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_mesh_true_fix.py --angles 12
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_mesh_true_fix.py            # 全 468 角
产出
  results/_diag_mesh_true_fix.json
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
                         rcs_of, cstats, ETA0, H, STL, RESULT_DIR)
from exp_po_locality import ray_occlusion                       # noqa: E402
from po_patch_data import canon_phase                           # noqa: E402
from _diag_nffft_audit import tri_interp                        # noqa: E402
from _exp_common import _rcs_stats                              # noqa: E402

VARIANTS = ["C0_raw_tri", "C1_nn_air", "C2_nn_air_phase", "C3_strip_kr",
            "C4_strip_ki", "C5_one_sided", "C6_consist_nn", "mesh_po"]


def stencil(pts, gx, gy, gz):
    """8 邻点索引 idx8(N,8,3) 与权重 w8(N,8)，与 tri_interp 逐位同源。"""
    N = pts.shape[0]
    i0, wt = [], []
    for a, g in enumerate((gx, gy, gz)):
        g = np.asarray(g, dtype=np.float64)
        t = np.clip((pts[:, a] - g[0]) / (g[1] - g[0]), 0.0, len(g) - 1.0 - 1e-9)
        j = np.floor(t).astype(np.int64)
        i0.append(j)
        wt.append(t - j)
    idx = np.empty((N, 8, 3), dtype=np.int64)
    w = np.empty((N, 8), dtype=np.float64)
    c = 0
    for dx in (0, 1):
        wx = wt[0] if dx else (1.0 - wt[0])
        ix = np.clip(i0[0] + dx, 0, len(gx) - 1)
        for dy in (0, 1):
            wy = wt[1] if dy else (1.0 - wt[1])
            iy = np.clip(i0[1] + dy, 0, len(gy) - 1)
            for dz in (0, 1):
                wz = wt[2] if dz else (1.0 - wt[2])
                iz = np.clip(i0[2] + dz, 0, len(gz) - 1)
                idx[:, c] = np.stack([ix, iy, iz], axis=1)
                w[:, c] = wx * wy * wz
                c += 1
    return idx, w


def gather(field, idx8):
    return field[idx8[:, :, 0], idx8[:, :, 1], idx8[:, :, 2]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=0, help="抽样角数（0=全 468）")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--only", default="", help="逗号分隔，只跑这些口径（省时）")
    args = ap.parse_args()
    sel = ([v for v in VARIANTS if v in args.only.split(",")] if args.only else VARIANTS)

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
    r_idx = np.stack([gx[q_out[:, 0]], gy[q_out[:, 1]], gz[q_out[:, 2]]], axis=1)
    idx8, w8 = stencil(cen, gx, gy, gz)
    r8 = np.stack([gx[idx8[:, :, 0]], gy[idx8[:, :, 1]], gz[idx8[:, :, 2]]], axis=2)
    air8 = ~metal[idx8[:, :, 0], idx8[:, :, 1], idx8[:, :, 2]]
    w5 = w8 * air8
    s5 = w5.sum(axis=1)
    ok5 = s5 > 1e-9
    w5[ok5] /= s5[ok5, None]
    print(f"  STL {len(cen)} 面  面积 {dAm.sum():.4f} m²  空气侧起点成功 {int(air_ok.sum())}")
    print(f"  λ = {lam:.5f} m   h = {H:.5f} m = λ/{lam/H:.3f}   kh = {k*H:.4f} rad"
          f"   sinc²(kh/2) = {np.sinc(k*H/(2*np.pi))**2:.4f}（M1 预测的插值衰减）")
    print(f"  单侧插值可归一的面元 {int(ok5.sum())}/{len(cen)}"
          f"（{int((~ok5).sum())} 个 8 邻点全在金属内，回退零阶）")

    acc = {v: {"rcs": [], "c": []} for v in sel}
    diag = {}
    t0 = time.time()
    checked = False
    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei)
            ei = ei * ph0
            Hs = f["H_scat"][i].astype(np.complex128) * ph0
            L = float(np.linalg.norm(Hs))
            psi_tri = np.exp(-1j * beta[i] * (cen @ ki))
            h_exact = (np.cross(ki, ei) / ETA0)[None, :] * psi_tri[:, None]

            # ---- 局部镜反方向 k̂_r = k̂_i − 2(k̂_i·n̂)n̂ ----
            cni = nvm @ ki
            kr = ki[None, :] - 2.0 * cni[:, None] * nvm
            kr /= np.maximum(np.linalg.norm(kr, axis=1, keepdims=True), 1e-12)

            Hs8 = gather(Hs, idx8)                       # (N,8,3)
            # ---- C0 现行三线性 ----
            Hs_c0 = (w8[:, :, None] * Hs8).sum(axis=1)
            if not checked:
                ref = tri_interp(Hs, cen, gx, gy, gz)
                err = float(np.abs(Hs_c0 - ref).max() / max(L, 1e-30))
                print(f"  [自检] C0 与 tri_interp 最大相对差 {err:.3e}")
                assert err < 1e-10, "C0 与 tri_interp 不同源"
                checked = True
            # ---- C1 最近空气侧体素（零阶）----
            Hs_nn = Hs[q_out[:, 0], q_out[:, 1], q_out[:, 2]]
            # ---- C2 C1 + 相位外推回 cen ----
            d_c2 = ((cen - r_idx) * kr).sum(axis=1)
            Hs_c2 = Hs_nn * np.exp(-1j * k * d_c2)[:, None]
            # ---- C3 剥镜反载波后插值包络 ----
            dots = np.einsum('nd,ned->ne', kr, r8)
            A3 = (w8[:, :, None] * (Hs8 * np.exp(1j * k * dots)[:, :, None])).sum(axis=1)
            Hs_c3 = A3 * np.exp(-1j * k * (kr * cen).sum(axis=1))[:, None]
            # ---- C4 剥入射载波（对照）----
            dot4 = np.einsum('d,ned->ne', ki, r8)
            A4 = (w8[:, :, None] * (Hs8 * np.exp(1j * k * dot4)[:, :, None])).sum(axis=1)
            Hs_c4 = A4 * np.exp(-1j * k * (cen @ ki))[:, None]
            # ---- C5 单侧（只用空气侧邻点）----
            Hs_c5 = (w5[:, :, None] * Hs8).sum(axis=1)
            Hs_c5[~ok5] = Hs_nn[~ok5]
            # ---- C6 一致性零阶：入射也解析取在**同一体素** r_idx（消除 C1 的
            #      "入射在 cen、散射在 r_idx" 内部不一致，检验 |c|=0.710 是否源于此）
            Hs_c6 = Hs_nn
            h_inc_nn = ((np.cross(ki, ei) / ETA0)[None, :]
                        * np.exp(-1j * beta[i] * (r_idx @ ki))[:, None])

            # ---- 幅度诊断：各口径在面心处重建的 |H_scat| 相对 C0 的中位比 ----
            for nm, Hv in (("C0", Hs_c0), ("C1", Hs_nn), ("C2", Hs_c2),
                           ("C3", Hs_c3), ("C6", Hs_c6)):
                r = np.abs(Hv).sum(axis=1) / np.maximum(np.abs(Hs_c0).sum(axis=1), 1e-30)
                diag.setdefault(f"amp_ratio_{nm}_med", []).append(float(np.median(r)))

            # ---- 统一 NFFFT ----
            Js = {nm: np.cross(nvm, h_exact + Hv) for nm, Hv in
                  (("C0_raw_tri", Hs_c0), ("C1_nn_air", Hs_nn), ("C2_nn_air_phase", Hs_c2),
                   ("C3_strip_kr", Hs_c3), ("C4_strip_ki", Hs_c4), ("C5_one_sided", Hs_c5))}
            # C6 的入射项与散射项取自同一体素（一致性口径）
            Js["C6_consist_nn"] = np.cross(nvm, h_inc_nn + Hs_c6)
            lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
            e0vec = np.cross(ki, ei)
            Jpo = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi_tri[:, None]
            Jpo[~lit_m] = 0.0
            Js["mesh_po"] = Jpo

            Eh2 = f["E_ff"][i].reshape(-1, 2)
            for v in sel:
                E = nffft_from(Js[v], dAm, cen, rhat, k, args.chunk)
                acc[v]["rcs"].append(rcs_of(E, th, ph, e0m, shape))
                acc[v]["c"].append(cstats(np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1),
                                          Eh2))
            if (a + 1) % 10 == 0:
                print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)

    res = {"n_angle": int(len(aidx)), "angles": aidx.tolist(), "h_m": float(H),
           "lambda_m": float(lam), "lambda_over_h": float(lam / H), "kh_rad": float(k * H),
           "sinc2_kh_half": float(np.sinc(k * H / (2 * np.pi)) ** 2),
           "n_tri": int(len(cen)), "area_tri_m2": float(dAm.sum()),
           "wall_s": float(time.time() - t0), "variants": {},
           "amp_ratio_vs_C0": {kk: float(np.median(v)) for kk, v in diag.items()},
           "n_onesided_ok": int(ok5.sum())}

    print(f"\n[A1] 场投射口径对照（{len(aidx)} 角，真值 = FEKO 远场）")
    print(f"  {'口径':<16} {'med':>7} {'P90':>7} {'corr':>6} | {'ρ':>6} {'|c|':>6} {'∠c':>7}"
          f" | {'resc':>7}")
    tgt = rcs_true[aidx]
    for v in sel:
        R = np.stack(acc[v]["rcs"])
        st_ = _rcs_stats(R, tgt)
        cc = acc[v]["c"]
        rho = float(np.median([x["rho_complex"] for x in cc]))
        ca = np.array([x["c_arg_deg"] for x in cc])
        cab = float(np.median([x["c_abs"] for x in cc]))
        carg = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(ca))))))
        # 逐角按 |c|² 重标（剥掉系统性幅度偏置，只留方向图形状质量）
        sc = np.array([x["c_abs"] ** 2 for x in cc])[:, None, None]
        resc = _rcs_stats(R * sc, tgt)["rcs_dB_err_median"]
        res["variants"][v] = {**st_, "rho_complex": rho, "c_abs": cab, "c_arg_deg": carg,
                              "rcs_dB_err_median_rescaled": float(resc),
                              "dir_used_frac": float(np.median(
                                  [x["dir_used_frac"] for x in cc]))}
        print(f"  {v:<16} {st_['rcs_dB_err_median']:7.3f} {st_['rcs_dB_err_p90']:7.3f} "
              f"{st_['rcs_corr_median']:6.3f} | {rho:6.3f} {cab:6.3f} {carg:+7.1f}"
              f" | {resc:7.3f}")

    vm = res["variants"]
    cand = [v for v in sel if v != "mesh_po"]
    best = min((vm[v]["rcs_dB_err_median"], v) for v in cand)
    res["best_caliber"] = best[1]
    res["best_dB"] = float(best[0])
    res["best_caliber_rescaled"] = min((vm[v]["rcs_dB_err_median_rescaled"], v) for v in cand)[1]
    if "mesh_po" in vm:
        base = vm["mesh_po"]["rcs_dB_err_median"]
        res["ref_mesh_po_dB"] = float(base)
        print(f"\n  参考 mesh_po（解析 PO，同批）= {base:.3f} dB")
        print(f"  最佳投射口径 = {best[1]}  {best[0]:.3f} dB"
              f"（相对 mesh_po {best[0]-base:+.3f} dB）")
    print(f"  重标后最佳 = {res['best_caliber_rescaled']}"
          f"  {min(vm[v]['rcs_dB_err_median_rescaled'] for v in cand):.3f} dB")
    if "C0_raw_tri" in vm:
        print(f"  现行口径 C0 = {vm['C0_raw_tri']['rcs_dB_err_median']:.3f} dB"
              f"（自检：应与 _diag_mesh_true.py 的 T1 ≈5.10 一致）")

    jp = os.path.join(RESULT_DIR, "_diag_mesh_true_fix.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")


if __name__ == "__main__":
    main()
