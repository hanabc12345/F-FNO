# -*- coding: utf-8 -*-
"""
_diag_oracle_dJ.py — 动作 0/2 之间的"收益上限"（oracle）实验
================================================================================
动机：
  _diag_dJ_mom.py 已经证明残差高度局域化：
    · MoM 在"我们判为阴影"的区域有 23.9% 的电流能量（PO 结构上恒为 0）
    · 受照区里，距最近锐棱 <5 mm 的面元相对误差中位 94%（占受照能量 22%），
      而 >50 mm 的面元只有 39%
  那么在动手实现 ILDC（公式获取风险高、工作量大）之前，必须先回答：
  **把某个区域的电流换成真值，RCS 误差最多能降多少？** 这就是 oracle 上限。

做法（同一入射角 θ=30°, φ=0，同一个 STL 网格，同一个 NFFFT）：
  用 FEKO 的 MoM 逐面电流（cur_000.out）当作"真值电流"，构造一族混合电流：
    po            : 全用我们的解析 PO
    po+mom_sh     : 阴影区换成 MoM
    po+mom_near1  : 受照区且距锐棱<5mm 的换成 MoM
    po+mom_near2  : 同上，阈值 20mm
    po+mom_sh+n1  : 阴影区 ∪ 近棱(<5mm) 换成 MoM
    po+mom_sh+n2  : 阴影区 ∪ 近棱(<20mm) 换成 MoM
    mom_all       : 全用 MoM（= 真值，同时校验 NFFFT 与约定）
  每个变体做一次远场积分，与 FEKO 真值 RCS（H5 中该角度的 37×73 方向）比较。

判据（预注册）：
  O1) mom_all 的中位误差 ≲0.5 dB  → NFFFT / 相位约定 / 网格对齐全部可信，
      本 oracle 的结论可用；否则先修链路，不谈 ILDC。
  O2) po+mom_sh+n2 的中位误差 ≤ po 的一半 → 边缘+阴影电流是主要矛盾，
      ILDC 值得投入（真实 ILDC 能拿到其中一部分）
  O3) po+mom_sh+n2 与 po 差距 <0.3 dB → 即使电流完全正确也救不回远场误差，
      说明该电尺寸下误差另有来源（多次散射/腔体/爬行波），ILDC 收益有限 → 转 R5/R1

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_oracle_dJ.py
产出：
  results/_diag_oracle_dJ.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import direction_grid
from exp_po_mesh import (load_mesh, mesh_outside_voxel, nffft_from, rcs_of, cstats,
                         ETA0, STL)
from exp_po_locality import ray_occlusion
from po_patch_data import canon_phase
from _diag_nffft_audit2 import read_stl
from exp_po_patchnet import _extra_stats
from _exp_common import _rcs_stats
from _diag_dJ_mom import parse_current_table, sharp_edge_samples, OUT, THETA0, PHI0

NEAR1, NEAR2 = 0.005, 0.020


def main():
    angles, e0, khat, beta = M.build_incidence_table()
    ia = int(np.argmin((angles[:, 0] - THETA0) ** 2 + (angles[:, 1] - PHI0) ** 2))
    print(f"  入射角索引 {ia}: θ={angles[ia,0]:.0f} φ={angles[ia,1]:.0f}")

    idx_f, cen_f, J_mom_raw = parse_current_table(OUT)

    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        E_ff_h5 = f["E_ff"][ia].astype(np.complex128)
    g0 = np.array([gx[0], gy[0], gz[0]])
    k = float(beta[ia])
    metal = eps > 1.5
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    cen, nvm, dAm = load_mesh(STL)
    q_out, _ = mesh_outside_voxel(cen, nvm, g0, metal)
    from scipy.spatial import cKDTree
    dist, nn = cKDTree(cen).query(cen_f, k=1)
    assert len(np.unique(nn)) == len(nn), "FEKO/STL 面心不是一一对应"
    J_mom = np.zeros_like(J_mom_raw)
    J_mom[nn] = J_mom_raw

    # ---- 解析 PO（与 exp_po_mesh / _diag_dJ_mom 同口径）----
    ki = khat[ia].astype(np.float64)
    ei = e0[ia].astype(np.complex128)
    e0m = float(np.linalg.norm(ei))
    ph0 = canon_phase(ei); ei = ei * ph0
    psi = np.exp(-1j * beta[ia] * (cen @ ki))
    e0vec = np.cross(ki, ei)
    J_po = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi[:, None]
    lit = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
    J_po_r = J_po.copy(); J_po_r[~lit] = 0.0

    tri, _ = read_stl(STL)
    P, n_sharp, _, _ = sharp_edge_samples(tri, nvm)
    d_edge = cKDTree(P).query(cen, k=1)[0] if len(P) else np.full(len(cen), np.inf)
    near1 = lit & (d_edge < NEAR1)
    near2 = lit & (d_edge < NEAR2)
    sh = ~lit
    print(f"  受照 {lit.sum()}  阴影 {sh.sum()}  近棱5mm {near1.sum()}  近棱20mm {near2.sum()}")

    # ---- 约定自检：MoM 电流与 PO 电流各自的远场 vs FEKO 远场 ----
    E_po_raw = nffft_from(J_po_r, dAm, cen, rhat, k)
    E_mom_raw = nffft_from(J_mom, dAm, cen, rhat, k)
    conv = {}
    for nm, E in (("po_raw", E_po_raw), ("mom_raw", E_mom_raw)):
        e2 = np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1)
        st = cstats(e2, E_ff_h5.reshape(-1, 2))
        conv[nm] = st
        print(f"  约定自检 {nm:<8}: ρ={st['rho_complex']:.4f}  |c|={st['c_abs']:.4f}  "
              f"∠c={st['c_arg_deg']:+.1f}°")
    # 用 MoM 相对 PO 的相对相位把两者对齐（只补一个全局相位，不改变物理）
    num = np.vdot(J_mom[lit].ravel(), J_po_r[lit].ravel())
    rel = float(np.angle(num))
    J_mom_al = J_mom * np.exp(-1j * rel)      # ≈ J_po 的同号版本
    print(f"  MoM 相对 PO 的全局相位 = {np.rad2deg(rel):+.1f}°（已补偿，仅约定差）")

    # ---- 构造变体 ----
    def mix(mask):
        J = J_po_r.copy()
        J[mask] = J_mom_al[mask]
        return J

    variants = {
        "po":          J_po_r,
        "po+mom_sh":   mix(sh),
        "po+mom_near1": mix(near1),
        "po+mom_near2": mix(near2),
        "po+mom_sh+n1": mix(sh | near1),
        "po+mom_sh+n2": mix(sh | near2),
        "mom_all":     J_mom_al,
    }

    res = {"angle": [float(angles[ia, 0]), float(angles[ia, 1])],
           "n_tri": int(len(cen)), "n_sharp_edges": int(n_sharp),
           "convention_check": conv, "mom_vs_po_phase_deg": float(np.rad2deg(rel)),
           "masks": {"lit": int(lit.sum()), "shadow": int(sh.sum()),
                     "near_5mm": int(near1.sum()), "near_20mm": int(near2.sum())},
           "variants": {}}

    print("\n[oracle] θ=30 φ=0  单角度 × 37×73 方向")
    print("  变体             med dB   P90 dB   corr    relL2     ρ      |c|    ∠c")
    for name, J in variants.items():
        E = nffft_from(J, dAm, cen, rhat, k)
        rcs_e = rcs_of(E, th, ph, e0m, shape)
        st = _rcs_stats(rcs_e[None], rcs_true[ia:ia + 1])
        ex = _extra_stats(rcs_e[None], rcs_true[ia:ia + 1])
        e2 = np.stack([(E * th).sum(1), (E * ph).sum(1)], axis=1)
        cc = cstats(e2, E_ff_h5.reshape(-1, 2))
        res["variants"][name] = {**st, "extra": ex, **cc}
        print(f"  {name:<14} {st['rcs_dB_err_median']:6.2f}  {st['rcs_dB_err_p90']:6.2f}  "
              f"{st['rcs_corr_median']:.3f}  {ex['rel_lin']*100:5.1f}%  "
              f"{cc['rho_complex']:.3f}  {cc['c_abs']:.3f}  {cc['c_arg_deg']:+6.1f}°")

    vm = res["variants"]
    med = {k: vm[k]["rcs_dB_err_median"] for k in vm}
    rec = []
    if med["mom_all"] <= 0.5:
        rec.append(f"O1 成立: mom_all 中位 {med['mom_all']:.2f} dB ≤0.5 → "
                   "NFFFT / 相位约定 / 网格对齐全部可信，oracle 结论可用")
    else:
        rec.append(f"O1 **不成立**: mom_all 中位 {med['mom_all']:.2f} dB >0.5 → "
                   "远场链路本身还有问题，先修链路，不谈 ILDC")
    gain = med["po"] - med["po+mom_sh+n2"]
    if med["po+mom_sh+n2"] <= med["po"] / 2:
        rec.append(f"O2 成立: po+mom_sh+n2 中位 {med['po+mom_sh+n2']:.2f} dB ≤ "
                   f"po 的一半（{med['po']:.2f}）→ 阴影+边缘电流是主要矛盾，"
                   f"ILDC 值得投入（本次上限增益 {gain:.2f} dB）")
    elif gain < 0.3:
        rec.append(f"O3 成立: 即使把阴影区与近棱电流全部换成真值也只降 {gain:.2f} dB "
                   "<0.3 → 该电尺寸下远场误差另有来源，ILDC 收益有限 → 转 R5/R1")
    else:
        rec.append(f"O2/O3 之间: 上限增益 {gain:.2f} dB（po {med['po']:.2f} → "
                   f"{med['po+mom_sh+n2']:.2f}）")
    res["verdict"] = rec
    print("\n  判据 →")
    for x in rec:
        print("    · " + x)

    jp = os.path.join(M.RESULT_DIR, "_diag_oracle_dJ.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"已存 {jp}")


if __name__ == "__main__":
    main()
