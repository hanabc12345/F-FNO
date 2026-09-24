# -*- coding: utf-8 -*-
"""
exp_po_mesh.py — 闸门 A：真实网格三角面上的解析 PO 远场 vs FEKO
================================================================================
问题（第 2 步负面结论的归因裁决）：
  阶梯体素面链路下，PO 3.74 dB、真值电流 3.48 dB，收益上限仅 0.26 dB。
  但同一项目已有诊断表明：远场积分公式与相位约定是干净的（盒面 Huygens 等效原理
  0.16 dB / 复相关 0.9995，见 _diag_nffft_audit4.py）。⇒ 误差只可能来自
  **表面参数化**（h=λ/3.2 轴对齐阶梯面 + 场在离面 h/2 的空气体素中心采样）。

本脚本做 2×2 受控对照，把「表面参数化」与「电流来源」两个因素分离：

                解析 PO 电流 2n̂×H_inc        采样真值电流 n̂×H_tot
  阶梯体素面     vox_po  （已有 3.74 dB）      vox_true （已有 3.48 dB）
  STL 三角面     mesh_po （**本脚本主问**）     mesh_true（audit3 曾得 7.20 dB）

注：mesh_true 臂须把体素场搬到三角面心，而网格仅 λ/3.2（复场线性插值有幅度衰减
伪影，中位 0.712×、纯数值贡献 2.69 dB）。入射场在面心可解析求得，故该臂改为
**入射解析 + 散射场三线性插值**（`_diag_mesh_true.py` 的 T1 变体：同批 12 角
med|Δ| 5.893 → 5.101 dB、P90 12.37 → 10.77、复相关 0.578 → 0.664）。

判据（预注册）：
  A) mesh_po ≤ 2 dB  → 3.5 dB 的元凶是体素化；解析 PO 在真实网格上已经够好，
                       路线一（真网格 PO+PTD + ML 补高阶残差）成立 ⇒ 值得继续。
  B) mesh_po ≈ vox_po（≈3.5 dB）→ PO 本身在该电尺寸（15λ）就不够，必须先上 PTD，
                       学习路线要往后放。
  C) mesh_po < vox_po 但 mesh_true > mesh_po → 说明"电流估计"比"网格"更关键，
                       ML 的着力点应放在电流（或直接放远场残差），而非网格。

同时输出复相关 ρ / 最佳标定因子 |c|∠arg c（诊断相干幅度亏缺），以及线性域相对 L2。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_po_mesh.py --angles 40      # 快测
  & "F:/miniconda3/envs/isaac311/python.exe" exp_po_mesh.py                  # 全 468 角
产出：results/exp_po_mesh.json
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
from fno_f16_3d_p4_nffft import surface_parts, direction_grid
from exp_po_locality import ray_occlusion
from _exp_common import _rcs_stats
from po_patch_data import canon_phase
from _diag_nffft_audit2 import read_stl
from _diag_nffft_audit import tri_interp
from exp_po_patchnet import _extra_stats

ETA0 = 119.9169832 * np.pi
H = 0.03125
STL = os.path.join(r"f:\MyWorkSpace\UAVGame", "3d_feko_run", "f16_refined.stl")
RESULT_DIR = M.RESULT_DIR


# ============================================================
# 一、几何：STL 三角面（精确法向/面积/面心）+ 阶梯体素面（对照）
# ============================================================

def load_mesh(path):
    """返回 (cen, nv, dA)：面心、朝外单位法向、面积。朝外由有符号体积定向。"""
    tri, nrm_file = read_stl(path)
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    av = 0.5 * np.cross(v1 - v0, v2 - v0)
    dA = np.linalg.norm(av, axis=1)
    cen = tri.mean(axis=1)
    svol = float((np.cross(v0, v1) * v2).sum() / 6.0)          # 散度定理有符号体积
    if svol < 0:
        av = -av
    print(f"  STL: {len(tri)} 三角面  面积 {dA.sum():.4f} m²  "
          f"有符号体积 {svol:+.5f} m³（>0 表示法向已朝外）")
    return cen, av / dA[:, None], dA


def mesh_outside_voxel(cen, nv, g0, metal, max_layers=8):
    """从三角面心**沿外法向**找到最近的**空气**体素，作为 DDA 起点（与阶梯面元口径一致）。

    旧版取 cen − 0.5h·n̂（沿外法向退半格 → 落回台阶内部），DDA 起点落在金属里，
    首步即自遮挡 ⇒ 受照占比被系统性压低（实测 20.3% vs 阶梯面 35.1%）、
    |c| 偏低。现改为向外推进至空气侧，使 2×2 对照里"遮蔽判定"这一环对两臂完全相同。

    返回 (q, done)：q 为起点体素索引；done=False 表示连续 max_layers 层仍为金属
    （厚壁/内腔面元），已回退到旧的向内口径。
    """
    lim = np.array(metal.shape, dtype=np.int64)
    n = len(cen)
    q = np.zeros((n, 3), dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    for s in range(max_layers):
        if done.all():
            break
        p = cen + (s + 0.5) * H * nv                    # s=0 → 面外半格；s=1 → 面外 1.5 格 …
        idx = np.floor((p - g0[None, :]) / H + 0.5).astype(np.int64)
        oob = (idx < 0).any(1) | (idx >= lim).any(1)    # 网格外视为自由空间
        ii = np.clip(idx, 0, lim - 1)
        is_air = oob | (~metal[ii[:, 0], ii[:, 1], ii[:, 2]])
        take = (~done) & is_air
        q[take] = ii[take]
        done |= take
    if not done.all():
        rem = ~done
        p = cen[rem] - 0.5 * H * nv[rem]
        q[rem] = np.clip(np.floor((p - g0[None, :]) / H + 0.5).astype(np.int64), 0, lim - 1)
    return q, done


# ============================================================
# 二、NFFFT（与全项目同一约定：E_ff = (jk/4π)(−η₀ N⊥)）
# ============================================================

def nffft_from(J, dA, r, rhat, k, chunk=4096):
    N = np.zeros((rhat.shape[0], 3), dtype=np.complex128)
    for s in range(0, len(r), chunk):
        e = min(s + chunk, len(r))
        kphase = np.exp(1j * k * (r[s:e] @ rhat.T))
        N += kphase.T @ (J[s:e] * dA[s:e, None])
    Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
    return (1j * k / (4 * np.pi)) * (-ETA0 * Nperp)


def rcs_of(E_ff, th, ph, e0m, shape):
    return (4 * np.pi * (np.abs((E_ff * th).sum(1)) ** 2
                         + np.abs((E_ff * ph).sum(1)) ** 2) / e0m ** 2).reshape(shape)


def cstats(E_est, E_h5_2, thr_frac=0.05):
    """复相关 ρ、最佳标定因子 c（E_h5 ≈ c·E_est）、参与方向占比。"""
    mag = np.sqrt((np.abs(E_h5_2) ** 2).sum(axis=1))
    m = mag > thr_frac * mag.max()
    ec = E_est[m].ravel(); eh = E_h5_2[m].ravel()
    den = np.sqrt(np.vdot(ec, ec).real * np.vdot(eh, eh).real)
    c = np.vdot(eh, ec) / max(np.vdot(eh, eh).real, 1e-300)
    return {"rho_complex": float(np.abs(np.vdot(eh, ec)) / max(den, 1e-300)),
            "c_abs": float(np.abs(c)), "c_arg_deg": float(np.rad2deg(np.angle(c))),
            "dir_used_frac": float(m.mean())}


# ============================================================
# 三、主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--angles", type=int, default=0, help=">0 时均匀取该角度数，0=全 468")
    ap.add_argument("--chunk", type=int, default=4096)
    args = ap.parse_args()

    angles, e0, khat, beta = M.build_incidence_table()
    aidx = (np.arange(len(angles)) if not args.angles
            else np.linspace(0, len(angles) - 1, args.angles).astype(int))
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    g0 = np.array([gx[0], gy[0], gz[0]])
    k = float(beta[0])
    metal = eps > 1.5
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    print(f"  三角面空气侧起点: 成功 {int(air_ok.sum())}/{len(cen)}"
          f"（{int((~air_ok).sum())} 个厚壁/内腔面元回退向内口径）")

    owners, rsurf, dS = surface_parts(eps, gx, gy, gz)
    dAv = np.linalg.norm(dS, axis=1)
    nvv = dS / dAv[:, None]
    p_idx = owners - np.rint(dS / (H * H)).astype(np.int64)
    r_air = np.stack([gx[owners[:, 0]], gy[owners[:, 1]], gz[owners[:, 2]]], axis=1)
    print(f"  阶梯面元 {len(dAv)}  面积 {dAv.sum():.4f} m²  "
          f"STL/阶梯 面积比 {dAm.sum()/dAv.sum():.3f}")

    variants = ["vox_po", "vox_true", "mesh_po", "mesh_true"]
    acc = {v: {"rcs": [], "c": []} for v in variants}
    lit_info = []
    t0 = time.time()
    with h5py.File(M.H5, "r") as f:
        for a, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei); ei = ei * ph0
            u_inc = (np.cross(ki, ei) / e0m).real          # 实单位入射 H 方向
            Hs = f["H_scat"][i].astype(np.complex128) * ph0
            psi_air = np.exp(-1j * beta[i] * (r_air @ ki))
            psi_tri = np.exp(-1j * beta[i] * (cen @ ki))

            # ---- 遮蔽：体素阶梯面（起点=空气侧体素）/ 三角面（起点=离面半格体素）----
            lit_v = ((nvv @ ki) < 0) & (~ray_occlusion(owners, -ki, metal))
            lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
            e0vec = np.cross(ki, ei)                        # = e0m·û
            Jpo_v = 2.0 * np.cross(nvv, e0vec) / ETA0 * psi_air[:, None]
            Jpo_m = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi_tri[:, None]
            Jpo_v[~lit_v] = 0.0
            Jpo_m[~lit_m] = 0.0

            # ---- 采样真值电流 n̂×H_tot ----
            H_tot = np.cross(ki, ei)[None, :] / ETA0 * psi_air[:, None] \
                + Hs[owners[:, 0], owners[:, 1], owners[:, 2]]
            Jtr_v = np.cross(nvv, H_tot)
            # 入射场在三角面心可解析求得，不必插值（网格仅 λ/3.2，复场线性插值有
            # 幅度衰减伪影，实测中位 0.712×）；只有散射场必须走三线性插值。
            h_inc_tri = (np.cross(ki, ei) / ETA0)[None, :] * psi_tri[:, None]
            H_tri = h_inc_tri + tri_interp(Hs, cen, gx, gy, gz)
            Jtr_m = np.cross(nvm, H_tri)

            Eset = {"vox_po": nffft_from(Jpo_v, dAv, rsurf, rhat, k, args.chunk),
                    "vox_true": nffft_from(Jtr_v, dAv, rsurf, rhat, k, args.chunk),
                    "mesh_po": nffft_from(Jpo_m, dAm, cen, rhat, k, args.chunk),
                    "mesh_true": nffft_from(Jtr_m, dAm, cen, rhat, k, args.chunk)}
            Eh2 = f["E_ff"][i].reshape(-1, 2)
            for v in variants:
                E = Eset[v]
                e2 = np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1)
                acc[v]["rcs"].append(rcs_of(E, th, ph, e0m, shape))
                acc[v]["c"].append(cstats(e2, Eh2))
            lit_info.append((float(lit_v.mean()), float(lit_m.mean()),
                             float(np.linalg.norm(Jpo_m[lit_m]).item() if lit_m.any() else 0.0)))
            if (a + 1) % 50 == 0:
                print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)

    res = {"n_angle": len(aidx), "angles": aidx.tolist(), "k": k, "h_m": H,
           "stl": STL, "n_tri": int(len(cen)), "area_tri_m2": float(dAm.sum()),
           "n_facet_vox": int(len(dAv)), "area_vox_m2": float(dAv.sum()),
           "lit_frac": {"vox": float(np.mean([x[0] for x in lit_info])),
                        "mesh": float(np.mean([x[1] for x in lit_info]))},
           "wall_s": None, "variants": {}}

    print("\n[闸门 A] 远场 vs FEKO 真值   med / P90 / corr_dB / 线性relL2 | 复ρ / |c| ∠arg")
    for v in variants:
        st = _rcs_stats(np.stack(acc[v]["rcs"]), rcs_true[aidx])
        ex = _extra_stats(np.stack(acc[v]["rcs"]), rcs_true[aidx])
        cc = acc[v]["c"]
        rho = float(np.median([x["rho_complex"] for x in cc]))
        ca = np.array([x["c_arg_deg"] for x in cc])
        cab = float(np.median([x["c_abs"] for x in cc]))
        carg = float(np.rad2deg(np.angle(np.mean(np.exp(1j * np.deg2rad(ca))))))
        res["variants"][v] = {**st, "extra": ex,
                              "rho_complex": rho, "c_abs": cab, "c_arg_deg": carg,
                              "dir_used_frac": float(np.median([x["dir_used_frac"] for x in cc]))}
        print(f"  {v:<11} med {st['rcs_dB_err_median']:6.2f}  P90 {st['rcs_dB_err_p90']:6.2f}  "
              f"corr {st['rcs_corr_median']:.3f}  relL2 {ex['rel_lin']*100:5.1f}%  | "
              f"ρ {rho:.3f}  |c| {cab:5.3f} ∠{carg:+7.1f}°")
    res["wall_s"] = round(time.time() - t0, 1)
    print(f"\n  受照占比: 阶梯面 {res['lit_frac']['vox']*100:.1f}%  "
          f"三角面 {res['lit_frac']['mesh']*100:.1f}%   耗时 {res['wall_s']:.0f}s")

    # ---- 预注册判据 ----
    vm = res["variants"]
    verdict = ("A: 元凶是体素化，解析 PO 已够好 → 路线一成立"
               if vm["mesh_po"]["rcs_dB_err_median"] <= 2.0 else
               "C: 网格有改善但电流估计更关键" if
               vm["mesh_po"]["rcs_dB_err_median"] < vm["vox_po"]["rcs_dB_err_median"] - 0.3
               else "B: PO 本身在该电尺寸不够，须先上 PTD")
    res["verdict"] = verdict
    print(f"\n  判决 → {verdict}")

    jp = os.path.join(RESULT_DIR, "exp_po_mesh.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"已存 {jp}")


if __name__ == "__main__":
    main()
