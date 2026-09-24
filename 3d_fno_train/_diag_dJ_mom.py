# -*- coding: utf-8 -*-
"""
_diag_dJ_mom.py — 动作 0：FEKO MoM 表面电流 vs 我们解析 PO 电流的 ΔJ 空间归因
================================================================================
目的（关掉"理论 vs 工程"的归因盲区）：
  闸门 A 得到 mesh_po 中位 3.18 dB（P90 9.07）。这个残差到底是
  (a) PO 物理机制缺失（阴影过渡区 / 棱边绕射 / 爬行波），还是
  (b) 我们 PO 实现本身的 bug？
  只靠远场标量指标分不清。本诊断把两者放到**同一个表面**上直接比：

      J_MoM  ← FEKO 全波（MLFMM）表面电流，来自 cur_000.out 的
               "VALUES OF THE CURRENT DENSITY VECTOR ON TRIANGLES"（自带 x/y/z 坐标）
      J_PO   ← 我们自己的解析 PO：J = 2n̂×H_inc，仅在"受照"三角面上非零

  用同一个复标量 c 做全局对齐（吸收约定差/整体幅度），再看 ΔJ = J_MoM − c·J_PO
  的空间分布：

  · 能量口径：Σ|ΔJ|²dA / Σ|J_MoM|²dA —— PO 一共漏掉多少电流能量
  · 分区口径：漏掉的能量中，"我们判为阴影"的区域占多少 —— 这部分是 PO 结构上
    不可能给出的（爬行波/绕射），是纯机制缺失，与实现无关
  · 定位口径：把每个三角面的相对误差对 **到最近锐棱的距离** 与 **|n̂·k̂|（掠射程度）**
    分箱 —— 若误差集中在锐棱附近/掠射区，就直接指认 ILDC 该补在哪里

判据（预注册）：
  D1) 阴影区电流能量占比 ≲5%   → 缺的主要不是阴影区绕射，ILDC 收益有限，先怀疑别的
  D2) 占比 ≳15%               → 阴影区（含过渡区）是主要缺口，ILDC 正对准了 → 执行动作 2
  D3) 受照区相对误差对"到锐棱距离"强单调（远小近大）→ 锐棱绕射项是主要残差
  D4) 受照区 |c| 显著 <1（>20% 亏缺）→ PO 幅度系统性偏低，也存在非局部机制

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_dJ_mom.py
依赖：
  f:\\MyWorkSpace\\UAVGame\\3d_feko_run\\cur_000.out  —— 由 cur_000.lua 跑出（含电流表）
产出：
  results/_diag_dJ_mom.json        标量结论
  results/_diag_dJ_mom.npz         逐面数组（供后续 ILDC 复用）
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json

import numpy as np
import h5py
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from exp_po_mesh import load_mesh, mesh_outside_voxel, ETA0, STL
from exp_po_locality import ray_occlusion
from po_patch_data import canon_phase
from _diag_nffft_audit2 import read_stl

OUT = os.path.join(M.RUN_DIR, "cur_000.out")
MARKER = "CURRENT DENSITY VECTOR ON TRIANGLES"
THETA0, PHI0 = 30.0, 0.0
EDGE_ANGLE_DEG = 25.0          # 二面角大于此值视为"锐棱"（绕射中心候选）


# ============================================================
# 一、解析 cur_000.out 的电流密度表
# ============================================================
def parse_current_table(path):
    """返回 (idx, cen, J)：三角面序号、面心坐标 (n,3)、复电流矢量 (n,3)。

    FEKO 表头两行，数据行 13 列：
      idx  x  y  z   JXmag JXph JYmag JYph JZmag JZph   c1 c2 c3
    """
    rows = []
    n_head = 0
    seen = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not seen:
                if MARKER in line:
                    seen = True
                continue
            parts = line.replace("D", "E").split()
            if len(parts) < 10:
                n_head += 1
                if n_head > 6:
                    break
                continue
            try:
                idx = int(parts[0])
                v = [float(x) for x in parts[1:10]]
            except ValueError:
                n_head += 1
                if n_head > 6:
                    break
                continue
            rows.append((idx, v))
            n_head = 0
    if not rows:
        raise RuntimeError(f"{path}: 未解析到电流密度表（marker={MARKER}）")
    idx = np.array([r[0] for r in rows], dtype=np.int64)
    v = np.array([r[1] for r in rows], dtype=np.float64)
    cen = v[:, 0:3]
    J = np.stack([v[:, 3] * np.exp(1j * np.deg2rad(v[:, 4])),
                  v[:, 5] * np.exp(1j * np.deg2rad(v[:, 6])),
                  v[:, 7] * np.exp(1j * np.deg2rad(v[:, 8]))], axis=1)
    return idx, cen, J


# ============================================================
# 二、锐棱提取（二面角 > 阈值）+ 到最近锐棱的距离
# ============================================================
def sharp_edge_samples(tri, nvm, ang_thresh_deg=EDGE_ANGLE_DEG, n_per_edge=4):
    """返回 (P, n_sharp, n_open)：锐棱上的采样点 (m,3)、锐棱数、开放边数。"""
    V = tri.reshape(-1, 3)
    key = np.round(V / 1e-6).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    Q = uniq.astype(np.float64) * 1e-6

    emap = {}
    for t in range(len(F)):
        a, b, c = int(F[t, 0]), int(F[t, 1]), int(F[t, 2])
        for i, j in ((a, b), (b, c), (c, a)):
            k = (i, j) if i < j else (j, i)
            emap.setdefault(k, []).append(t)

    pts = []
    n_sharp = 0
    n_open = 0
    for k, ts in emap.items():
        if len(ts) == 1:
            n_open += 1
            continue
        if len(ts) != 2:
            continue
        cosang = float(np.clip(nvm[ts[0]] @ nvm[ts[1]], -1.0, 1.0))
        if np.degrees(np.arccos(cosang)) < ang_thresh_deg:
            continue
        n_sharp += 1
        p1, p2 = Q[k[0]], Q[k[1]]
        for s in range(n_per_edge):
            pts.append(p1 + (s + 1) / (n_per_edge + 1) * (p2 - p1))
    P = np.array(pts) if pts else np.zeros((0, 3))
    return P, n_sharp, n_open, len(emap)


def binned_table(metric, values, bins, weights=None, label=""):
    """按 metric 分箱统计 values（中位/均值）与权重占比。"""
    out = []
    tot_w = float(np.sum(weights)) if weights is not None else float(len(metric))
    for i in range(len(bins) - 1):
        m = (metric >= bins[i]) & (metric < bins[i + 1])
        n = int(m.sum())
        rec = {"bin_lo": bins[i], "bin_hi": bins[i + 1], "n": n}
        if n:
            rec["median"] = float(np.median(values[m]))
            rec["mean"] = float(np.mean(values[m]))
            if weights is not None:
                rec["w_share"] = float(np.sum(weights[m]) / max(tot_w, 1e-300))
        out.append(rec)
    return out


# ============================================================
# 三、主流程
# ============================================================
def main():
    angles, e0, khat, beta = M.build_incidence_table()
    ia = int(np.argmin((angles[:, 0] - THETA0) ** 2 + (angles[:, 1] - PHI0) ** 2))
    print(f"  入射角索引 {ia}: θ={angles[ia,0]:.0f} φ={angles[ia,1]:.0f}")

    idx_f, cen_f, J_mom = parse_current_table(OUT)
    print(f"  FEKO 电流表: {len(idx_f)} 个三角面（序号 {idx_f.min()}..{idx_f.max()}）")

    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    g0 = np.array([gx[0], gy[0], gz[0]])
    metal = eps > 1.5
    k = float(beta[ia])

    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)

    # ---- 面心匹配（FEKO 三角面 ↔ STL 三角面）----
    tree = cKDTree(cen)
    dist, nn = tree.query(cen_f, k=1)
    dup = len(np.unique(nn)) != len(nn)
    print(f"  面心匹配: 中位 {np.median(dist)*1e3:.4f} mm  最大 {dist.max()*1e3:.4f} mm"
          f"  唯一命中 {len(np.unique(nn))}/{len(nn)}"
          f"{'  [警告: 存在多对一]' if dup else ''}")

    # 把 FEKO 电流搬运到 STL 面序上
    Jm = np.zeros((len(cen), 3), dtype=np.complex128)
    got = np.zeros(len(cen), dtype=bool)
    Jm[nn] = J_mom
    got[nn] = True

    # ---- 解析 PO 电流（与 exp_po_mesh.py 完全同口径）----
    ki = khat[ia].astype(np.float64)
    ei = e0[ia].astype(np.complex128)
    e0m = float(np.linalg.norm(ei))
    ph0 = canon_phase(ei)
    ei = ei * ph0
    psi = np.exp(-1j * beta[ia] * (cen @ ki))
    e0vec = np.cross(ki, ei)
    Jpo = 2.0 * np.cross(nvm, e0vec) / ETA0 * psi[:, None]
    lit = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
    Jpo[~lit] = 0.0

    # ---- 全局复对齐 c：J_MoM ≈ c·J_PO（只在受照区估计）----
    m = lit & got
    num = np.vdot(Jm[m].ravel(), Jpo[m].ravel())        # Σ Jm · conj(Jpo)
    den = float(np.vdot(Jpo[m].ravel(), Jpo[m].ravel()).real)
    c = num / max(den, 1e-300)
    Jpo_a = Jpo * c
    dJ = Jm - Jpo_a

    # ---- 朝向一致性自检：受照区 Re(Jm·conj(Jpo_a)) 的符号 ----
    re_dot = (Jm[m] * np.conj(Jpo_a[m])).sum(axis=1).real
    same_sign = float((re_dot > 0).mean())

    # ---- 能量口径 ----
    E_mom = float((np.abs(Jm) ** 2).sum(axis=1) @ dAm)
    E_po = float((np.abs(Jpo_a) ** 2).sum(axis=1) @ dAm)
    E_dJ = float((np.abs(dJ) ** 2).sum(axis=1) @ dAm)
    E_mom_lit = float((np.abs(Jm[lit]) ** 2).sum(axis=1) @ dAm[lit])
    E_mom_sh = E_mom - E_mom_lit
    E_po_lit = float((np.abs(Jpo_a[lit]) ** 2).sum(axis=1) @ dAm[lit])
    amp_ratio_lit = float(np.sqrt(E_po_lit / max(E_mom_lit, 1e-300)))
    amp_ratio_sh = float(np.sqrt(
        float((np.abs(Jm[~lit]) ** 2).sum(axis=1) @ dAm[~lit]) / max(E_mom, 1e-300)))

    # ---- 锐棱距离与掠射 ----
    tri, _nrm_file = read_stl(STL)
    P, n_sharp, n_open, n_edges = sharp_edge_samples(tri, nvm)
    print(f"  棱边: 总 {n_edges}  开放 {n_open}  锐棱(>{EDGE_ANGLE_DEG}°) {n_sharp}  采样点 {len(P)}")
    if len(P):
        d_edge = cKDTree(P).query(cen, k=1)[0]
    else:
        d_edge = np.full(len(cen), np.inf)

    graze = np.abs(nvm @ ki)                    # |n̂·k̂|，越小越掠射
    rel = np.linalg.norm(dJ, axis=1) / np.maximum(np.linalg.norm(Jm, axis=1), 1e-30)
    rel_po = np.linalg.norm(dJ, axis=1) / np.maximum(np.linalg.norm(Jpo_a, axis=1), 1e-30)
    w = np.linalg.norm(Jm, axis=1) ** 2 * dAm     # 能量权重

    bins_e = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, np.inf]
    bins_g = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]

    res = {
        "n_tri_feko": int(len(idx_f)), "n_tri_stl": int(len(cen)),
        "match_median_mm": float(np.median(dist) * 1e3),
        "match_max_mm": float(dist.max() * 1e3),
        "match_unique": int(len(np.unique(nn))),
        "lit_frac": float(lit.mean()),
        "orient_same_sign_frac": same_sign,
        "calib_c_abs": float(np.abs(c)), "calib_c_arg_deg": float(np.rad2deg(np.angle(c))),
        "energy": {
            "E_mom": E_mom, "E_po_aligned": E_po, "E_dJ": E_dJ,
            "dJ_energy_frac": E_dJ / max(E_mom, 1e-300),
            "mom_shadow_energy_frac": E_mom_sh / max(E_mom, 1e-300),
            "po_amp_frac": np.sqrt(E_po / max(E_mom, 1e-300)),
            "po_amp_frac_lit": amp_ratio_lit,
            "mom_shadow_frac_of_mom": amp_ratio_sh,
        },
        "rel_err_median_lit": float(np.median(rel[m])),
        "n_sharp_edges": int(n_sharp), "n_open_edges": int(n_open),
        "by_edge_dist": binned_table(d_edge[lit & got], rel[lit & got], bins_e,
                                     weights=w[lit & got]),
        "by_grazing_lit": binned_table(graze[lit & got], rel[lit & got], bins_g,
                                       weights=w[lit & got]),
        "shadow_j_mom_median": float(np.median(np.linalg.norm(Jm[~lit], axis=1)))
        if (~lit).any() else None,
        "lit_j_po_median": float(np.median(np.linalg.norm(Jpo_a[m], axis=1))) if m.any() else None,
    }

    print("\n[动作 0] MoM 电流 vs 解析 PO 电流")
    print(f"  全局对齐 c = {np.abs(c):.3f} ∠{np.rad2deg(np.angle(c)):+.1f}°"
          f"   受照区朝向一致率 {same_sign*100:.2f}%")
    e = res["energy"]
    print(f"  PO 幅度占比 √(E_po/E_mom) = {e['po_amp_frac']*100:.1f}%"
          f"   （仅受照区 {e['po_amp_frac_lit']*100:.1f}%）")
    print(f"  ΔJ 能量占 MoM 总能量     = {e['dJ_energy_frac']*100:.1f}%")
    print(f"  其中 MoM 在'我们判为阴影'区域的能量 = {e['mom_shadow_energy_frac']*100:.1f}%"
          f"（占 MoM 总能量）")
    print(f"  受照区相对误差中位 = {res['rel_err_median_lit']*100:.1f}%")

    print("\n  按 到最近锐棱距离 分箱（受照区，权重=|J_MoM|²dA）")
    print("    d_edge[m]        n      rel误差中位   rel误差均值   能量占比")
    for r in res["by_edge_dist"]:
        lo = f"{r['bin_lo']:.3f}"
        hi = "inf" if not np.isfinite(r["bin_hi"]) else f"{r['bin_hi']:.3f}"
        if r["n"]:
            print(f"    [{lo:>6},{hi:>6})  {r['n']:6d}     {r['median']*100:6.1f}%"
                  f"       {r['mean']*100:6.1f}%     {r.get('w_share', 0)*100:5.1f}%")
        else:
            print(f"    [{lo:>6},{hi:>6})  {r['n']:6d}          -")

    print("\n  按 |n̂·k̂| 分箱（受照区，越接近 0 越掠射）")
    print("    |n.k|            n      rel误差中位   rel误差均值   能量占比")
    for r in res["by_grazing_lit"]:
        lo = f"{r['bin_lo']:.2f}"
        hi = f"{r['bin_hi']:.2f}"
        if r["n"]:
            print(f"    [{lo:>4},{hi:>4})  {r['n']:6d}     {r['median']*100:6.1f}%"
                  f"       {r['mean']*100:6.1f}%     {r.get('w_share', 0)*100:5.1f}%")
        else:
            print(f"    [{lo:>4},{hi:>4})  {r['n']:6d}          -")

    # ---- 预注册判据 ----
    sh_frac = e["mom_shadow_energy_frac"]
    ne = [r for r in res["by_edge_dist"] if r["n"] > 0]
    rec = []
    if sh_frac <= 0.05:
        rec.append("D1 成立: 阴影区电流能量占比 ≤5%，ILDC（补阴影区绕射）收益上限有限")
    elif sh_frac >= 0.15:
        rec.append(f"D2 成立: MoM 在阴影区的电流能量占比 {sh_frac*100:.1f}% ≥15%，"
                   "ILDC/PTD 正对准最大可识别缺口 → 执行动作 2")
    else:
        rec.append(f"D2 部分: 阴影区电流能量占比 {sh_frac*100:.1f}%（5%~15% 之间）")
    if len(ne) >= 2:
        near, far = ne[0], ne[-1]
        ratio = near["median"] / max(far["median"], 1e-30)
        if ratio > 1.3:
            rec.append(f"D3 成立: 受照区相对误差对'到锐棱距离'强单调 —— "
                       f"近棱 {near['median']*100:.0f}%（d<{near['bin_hi']*1e3:.0f}mm，"
                       f"占能量 {near.get('w_share',0)*100:.0f}%）"
                       f" vs 远离棱 {far['median']*100:.0f}%（d>{far['bin_lo']*1e3:.0f}mm），"
                       f"比值 {ratio:.2f} → 锐棱绕射项是主要残差")
        else:
            rec.append(f"D3 不成立: 误差对'到锐棱距离'不单调（近/远 = {ratio:.2f}）"
                       " → 残差不是单纯的锐棱绕射")
    else:
        rec.append("D3 无法判定: 分箱样本不足")
    if abs(e["po_amp_frac_lit"] - 1.0) > 0.20:
        rec.append(f"D4 成立: 受照区 PO 幅度比 {e['po_amp_frac_lit']:.3f}，"
                   "偏离 1 超过 20%")
    else:
        rec.append(f"D4 不成立: 受照区 PO 幅度比 {e['po_amp_frac_lit']:.3f}，偏离 <20%")
    rec.append("注: 全局对齐 c 的相位 %+.1f° 说明 FEKO 电流矢量定义与我们的 J_PO 反号"
               "（纯约定差，不影响 |E|），幅度 |c|=%.3f"
               % (np.rad2deg(np.angle(c)), np.abs(c)))
    res["verdict"] = rec
    print("\n  判据 →")
    for x in rec:
        print("    · " + x)

    os.makedirs(M.RESULT_DIR, exist_ok=True)
    jp = os.path.join(M.RESULT_DIR, "_diag_dJ_mom.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    np.savez_compressed(
        os.path.join(M.RESULT_DIR, "_diag_dJ_mom.npz"),
        cen=cen, nvm=nvm, dAm=dAm, J_mom=Jm, J_po=Jpo_a, dJ=dJ,
        lit=lit, d_edge=d_edge, graze=graze, c=np.array([c]))
    print(f"已存 {jp}")


if __name__ == "__main__":
    main()
