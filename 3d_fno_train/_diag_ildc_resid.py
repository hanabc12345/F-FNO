# -*- coding: utf-8 -*-
"""判别实验（第 2 步）：极点压住之后，ILDC 条纹波是否指向 FEKO 与 PO 的**差值**？

已知（_diag_ildc_f16.py run --dmax 3,30）：
  极点截断后 |E_fr|/|E_po| 已被压到 7.7~36，但每个角度的匹配都崩了
  （角 0：med 3.18→20.41、ρ 0.934→0.567）。⇒ 除了极点，还有第二层问题。

本脚本把问题拆成两个互斥假设：
  H1「尺度/相位约定错」：E_fr 与残差 R = E_h5 − E_po 高度相关（ρ 大），
      只是幅度/相位常数不对 ⇒ 便宜可救（改 pref / 归一化）。
  H2「内容错」：ρ(E_fr, R) 低 ⇒ 条纹波根本不是缺的那块 ⇒ 该路线判死。

同时打印：基线 ρ(E_po, E_h5)、ρ(E_fr, E_h5)、残差能量占比。
"""
import os
import sys
import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import ildc_mesh as I                                    # noqa: E402
import _diag_ildc_f16 as D                               # noqa: E402
from _diag_ildc_f16 import load_tri_consistent, build, STL, f_E_ff   # noqa: E402
from exp_po_mesh import mesh_outside_voxel, nffft_from, cstats, ETA0  # noqa: E402
from exp_po_locality import ray_occlusion                # noqa: E402
from fno_f16_3d_p4_nffft import direction_grid           # noqa: E402
from po_patch_data import canon_phase                    # noqa: E402
import fno_f16_3d as M                                   # noqa: E402

CAP = 3.0


def main(angle_list):
    fh = h5py.File(M.H5, "r")
    gx = fh["grid_x"][:].astype(np.float64)
    gy = fh["grid_y"][:].astype(np.float64)
    gz = fh["grid_z"][:].astype(np.float64)
    eps = fh["eps_field"][:]
    ff_theta = fh["ff_theta"][:]
    ff_phi = fh["ff_phi"][:]
    D._F_H5["f"] = fh          # f_E_ff 依赖该模块级句柄（见 _diag_ildc_f16.py:290）
    D.G0 = np.array([gx[0], gy[0], gz[0]])
    D.METAL = eps > 1.5
    tri, cen, nvm, dA = load_tri_consistent(STL)
    E = I.build_edges(tri, nvm)
    alpha = np.degrees(E["alpha"])
    Cn = E["Cn"]
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    print(f"  棱边 {len(alpha)}  Σ长度 {Cn.sum():.2f} m   截断 |c| ≤ {CAP:g}")
    q_out, _ = mesh_outside_voxel(cen, nvm, D.G0, D.METAL)
    angles, e0, khat, beta = M.build_incidence_table()
    k = float(beta[0])

    for i in angle_list:
        ki = khat[i].astype(np.float64)
        ei = e0[i].astype(np.complex128)
        e0m = float(np.linalg.norm(ei))
        ei = ei * canon_phase(ei)
        e_i, p_hat, P, vis = build(E, ki, ei)

        lit_m = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, D.METAL))
        psi_tri = np.exp(-1j * beta[i] * (cen @ ki))
        Jpo = 2.0 * np.cross(nvm, np.cross(ki, ei)) / ETA0 * psi_tri[:, None]
        Jpo[~lit_m] = 0.0
        Epo = nffft_from(Jpo, dA, cen, rhat, k)
        Efr = I.fringe_far_field(E, P, k, e_i, rhat, p_hat, subset=vis,
                                 d_max=CAP) * e0m

        tw = lambda Ev: np.stack([(Ev * th).sum(1), (Ev * ph).sum(1)], axis=1)
        H2, Po2, Fr2 = f_E_ff(i), tw(Epo), tw(Efr)
        R2 = H2 - Po2
        n = lambda X: float(np.sqrt((np.abs(X) ** 2).mean()))

        print(f"\n  === 角 {i} ===")
        print(f"    rms |E_po| {n(Po2):.4e}  |E_fr| {n(Fr2):.4e}  "
              f"|E_h5| {n(H2):.4e}  残差|R| {n(R2):.4e}  "
              f"（残差/真值 {n(R2)/n(H2)*100:.1f}%，|E_fr|/|R| {n(Fr2)/n(R2):.3f}）")
        print(f"    {'配对':<28}{'ρ':>8}{'|c|':>10}{'∠c':>9}")
        for nm, a, b in (("E_po  ↔ 真值", Po2, H2),
                         ("E_fr  ↔ 真值", Fr2, H2),
                         ("E_fr  ↔ 残差 R", Fr2, R2),
                         ("E_po+E_fr ↔ 真值", Po2 + Fr2, H2)):
            cc = cstats(a, b)
            print(f"    {nm:<28}{cc['rho_complex']:>8.3f}{cc['c_abs']:>10.3f}"
                  f"{cc['c_arg_deg']:>8.1f}°")


if __name__ == "__main__":
    main([int(s) for s in (sys.argv[1:] or ["0", "233"])])
