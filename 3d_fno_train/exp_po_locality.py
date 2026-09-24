# -*- coding: utf-8 -*-
"""
exp_po_locality.py — 检验「物理粗解(PO) + 局部算子修正」路线的核心假设
=====================================================================
假设 H：PEC 表面电流相对 PO 粗解 2n̂×H_inc 的残差 ΔJ = J_num − J_PO
        只依赖**局部**信息（局部法向、局部入射方向、局部几何），
        与面元在物体上的绝对位置无关。
  · 若 H 成立 → 可用「局部 patch 训练集」训练局部修正算子，再拼装到任意物体上
    （几何泛化的成本从"穷举形状"降为"覆盖局部几何族"）；
  · 若 H 不成立（残差被全局多次散射/边缘绕射支配）→ 整条路线需重新设计。

判据（先定后测）：
  R²_spatial ≥ 0.5·R²_random 且 R²_spatial > 0.3   → 成立
  R²_random ≫ R²_spatial 或 R²_spatial ≈ 0        → 不成立

四个关键处理（缺一不可）：
 1. 去快相位（de-ramp）Ĵ = J/ψ，ψ = e^{−jβk̂·r}：平面波的整体相位必须剥掉，
    否则残差相位随位置剧烈变化，任何回归都不可能（即"解析化"原则）。
 2. 归一化一致：Ĵ 统一按 η₀/|e0| 无量纲化，使 PO 与数据可比（否则残差被常数项淹没）。
 3. 绝对坐标控制组：加 (x,y,z) 后 R²_random 会因"记住邻域"虚高——
    这正是区分"局部物理"与"空间记忆"的判据。
 4. 只统计受照面：全样本 R² 会被"明暗分布是否可预测"这种平凡结论抬高，
    真正决定成败的是**受照面残差**的可预测性（单独报 r2_lit）。

同时产出：
  · 面元级 PO 误差 |ΔJ|/|J_num|（必须被学习的修正量级）；
  · PO→NFFFT 远场基线 vs FEKO 真值（判断 PO+修正是否值得投入）；
  · 阶梯面法向 vs 光滑法向两种 PO，用于分离离散化贡献。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_po_locality.py
产出：results/exp_po_locality.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import surface_parts, direction_grid
from _exp_common import _rcs_stats

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi
H = 0.03125


# ============================================================
# 一、几何描述子（角度无关，逐面元，只用局部邻域）
# ============================================================

def facet_geometry(owners, dvec, metal):
    """面元 = (空气体素 q，法向 d̂=±轴)；金属体素 p = q − d̂。
    返回:
      planar_deg 切向 4 邻面中"同一平面延续"的个数（<4 即位于棱/角）
      edge_dist  到最近"非平面面元"的面元图 BFS 距离（单位 h，封顶 12）
      n_concave  切向邻接处表面向空气侧转折的个数（凹）
      n_convex   朝空气侧"缺面"的个数（凸棱）
      nb_metal26 金属体素 26 邻域金属计数
    """
    Nf = len(owners)
    key = {}
    for s in range(Nf):
        key[(int(owners[s, 0]), int(owners[s, 1]), int(owners[s, 2]),
             int(dvec[s, 0]), int(dvec[s, 1]), int(dvec[s, 2]))] = s
    tangent = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
               np.array([0, 1, 0]), np.array([0, -1, 0]),
               np.array([0, 0, 1]), np.array([0, 0, -1])]
    Nx, Ny, Nz = metal.shape
    planar_deg = np.zeros(Nf, dtype=np.int32)
    n_concave = np.zeros(Nf, dtype=np.int32)
    n_convex = np.zeros(Nf, dtype=np.int32)
    nbrs = [[] for _ in range(Nf)]
    hit_p = owners - dvec
    for s in range(Nf):
        d = dvec[s]; q = owners[s]
        for t in tangent:
            if abs(float(t @ d)) > 0:
                continue
            q2 = q + t; p2 = hit_p[s] + t
            in_q = (0 <= q2[0] < Nx) and (0 <= q2[1] < Ny) and (0 <= q2[2] < Nz)
            in_p = (0 <= p2[0] < Nx) and (0 <= p2[1] < Ny) and (0 <= p2[2] < Nz)
            if not in_q:
                n_convex[s] += 1; continue
            is_metal_next = in_p and metal[p2[0], p2[1], p2[2]]
            is_air_next = not metal[q2[0], q2[1], q2[2]]
            if is_metal_next and is_air_next:
                planar_deg[s] += 1
                j = key.get((int(q2[0]), int(q2[1]), int(q2[2]),
                             int(d[0]), int(d[1]), int(d[2])))
                if j is not None:
                    nbrs[s].append(j)
            elif is_metal_next:
                n_concave[s] += 1
            else:
                n_convex[s] += 1

    INF = 10 ** 6
    dist = np.where(planar_deg < 4, 0, INF).astype(np.int64)
    adj = [[] for _ in range(Nf)]
    for s in range(Nf):
        for j in nbrs[s]:
            adj[s].append(j); adj[j].append(s)
    frontier = list(np.where(dist == 0)[0])
    dcur = 0
    while frontier and dcur < 12:
        dcur += 1
        nxt = []
        for s in frontier:
            for j in adj[s]:
                if dist[j] > dcur:
                    dist[j] = dcur; nxt.append(j)
        frontier = nxt
    edge_dist = np.minimum(dist, 12).astype(np.float32)

    nb = np.zeros(Nf, dtype=np.float32)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            for c in (-1, 0, 1):
                idx = hit_p + np.array([a, b, c])
                ok = ((idx[:, 0] >= 0) & (idx[:, 0] < Nx) & (idx[:, 1] >= 0) & (idx[:, 1] < Ny)
                      & (idx[:, 2] >= 0) & (idx[:, 2] < Nz))
                nb += np.where(ok, metal[idx[:, 0].clip(0, Nx - 1), idx[:, 1].clip(0, Ny - 1),
                                         idx[:, 2].clip(0, Nz - 1)], 0.0)
    return {"planar_deg": planar_deg.astype(np.float32), "edge_dist": edge_dist,
            "n_concave": n_concave.astype(np.float32), "n_convex": n_convex.astype(np.float32),
            "nb_metal26": nb}


def ray_occlusion(q, d, metal, max_steps=300):
    """向量化 DDA：从空气体素 q 沿 d = −k̂（朝源方向）行进，进入金属 → 遮挡。"""
    Nx, Ny, Nz = metal.shape
    n = len(q)
    step = np.zeros(3, dtype=np.int64)
    tmax = np.full((n, 3), np.inf)
    tdelta = np.full((n, 3), np.inf)
    p = q.astype(np.float64)
    for a in range(3):
        if d[a] > 1e-12:
            step[a] = 1
            tmax[:, a] = (np.floor(p[:, a]) + 1.0 - p[:, a]) / d[a]
            tdelta[:, a] = 1.0 / d[a]
        elif d[a] < -1e-12:
            step[a] = -1
            tmax[:, a] = (np.ceil(p[:, a]) - 1.0 - p[:, a]) / d[a]
            tdelta[:, a] = -1.0 / d[a]
    cur = q.copy()
    occl = np.zeros(n, dtype=bool)
    alive = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        ax = np.argmin(tmax, axis=1)
        for a in range(3):
            m = (ax == a) & alive
            if m.any():
                cur[m, a] += step[a]
                tmax[m, a] += tdelta[m, a]
        oob = ((cur < 0).any(1) | (cur[:, 0] >= Nx) | (cur[:, 1] >= Ny) | (cur[:, 2] >= Nz))
        alive &= ~oob
        c = cur.clip(min=0)
        c[:, 0] = c[:, 0].clip(0, Nx - 1); c[:, 1] = c[:, 1].clip(0, Ny - 1)
        c[:, 2] = c[:, 2].clip(0, Nz - 1)
        hit = alive & metal[c[:, 0], c[:, 1], c[:, 2]]
        occl |= hit
        alive &= ~hit
        if not alive.any():
            break
    return occl


def smooth3(a, kk=(1.0, 2.0, 1.0)):
    kk = np.asarray(kk) / np.sum(kk)
    for ax in range(3):
        a = np.apply_along_axis(lambda v: np.convolve(v, kk, mode="same"), ax, a)
    return a


# ============================================================
# 二、回归与划分
# ============================================================

def eval_splits(X, Y, fold, mask=None):
    """leave-one-fold-out 最小二乘交叉验证，返回 R²（以训练均值为基准）。
    mask：只在这些样本上评估（用于"仅受照面"口径）。"""
    if mask is not None:
        X = X[mask]; Y = Y[mask]; fold = fold[mask]
    P = np.zeros_like(Y)
    for f in np.unique(fold):
        tr = fold != f; te = ~tr
        if te.sum() < 30 or tr.sum() < 30:
            continue
        mu = X[tr].mean(0); sd = X[tr].std(0); sd[sd < 1e-12] = 1.0
        A = np.hstack([(X[tr] - mu) / sd, np.ones((int(tr.sum()), 1))])
        B = np.hstack([(X[te] - mu) / sd, np.ones((int(te.sum()), 1))])
        W, *_ = np.linalg.lstsq(A, Y[tr], rcond=None)
        P[te] = B @ W
    den = np.sum((Y - Y.mean(0)) ** 2)
    return float(1.0 - np.sum((P - Y) ** 2) / den) if den > 0 else float("nan")


def kmeans_np(X, K, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), K, replace=False)].copy()
    for _ in range(iters):
        d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        for k in range(K):
            m = lab == k
            if m.any():
                C[k] = X[m].mean(0)
    return lab


def main():
    t0 = time.time()
    print("=" * 78)
    print("检验核心假设：PO 残差 ΔJ 是否只依赖局部信息（与绝对位置无关）")
    print("=" * 78)
    angles, e0, khat, beta = M.build_incidence_table()
    k = float(beta[0])
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        rcs_true_all = f["rcs"][:]
        assert np.allclose(angles, f["angles"][:])
    metal = eps > 1.5
    e0mag = np.linalg.norm(e0, axis=1).real

    owners, rsurf, dS = surface_parts(eps, gx, gy, gz)
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / dA[:, None]
    dvec = np.rint(dS / (H * H)).astype(np.int64)
    r_air = np.stack([gx[owners[:, 0]], gy[owners[:, 1]], gz[owners[:, 2]]], axis=1)
    Ns, Nang = len(dA), len(angles)
    print(f"面元 {Ns}（金属体素 {int(metal.sum())}）  角度 {Nang}  样本 {Ns*Nang:,}")
    print(f"  |e0| 范围 {e0mag.min():.4f}~{e0mag.max():.4f}（应≈常数，做 RCS 归一化用）")

    geo = facet_geometry(owners, dvec, metal)
    print("几何描述子均值/最大:", {kk: (round(float(vv.mean()), 2), int(vv.max()))
                                  for kk, vv in geo.items()})
    X_geo = np.stack([geo["planar_deg"], geo["edge_dist"], geo["n_concave"],
                      geo["n_convex"], geo["nb_metal26"]], axis=1).astype(np.float32)
    X_pos = r_air.astype(np.float32)
    print(f"  局部几何特征 {X_geo.shape[1]} 维；控制组再加绝对坐标 3 维  ({time.time()-t0:.0f}s)")

    nfeat = 7
    Xa = np.zeros((Nang, Ns, nfeat), dtype=np.float32)
    dJr = np.zeros((Nang, Ns, 4), dtype=np.float32)      # ΔĴ 在局部基下的 4 实分量
    Jn_norm = np.zeros((Nang, Ns), dtype=np.float32)
    Jnum_phys = np.zeros((Nang, Ns, 3), dtype=np.complex128)   # 物理电流（远场用，复现 floor）
    Jpo_surf = np.zeros((Nang, Ns, 3), dtype=np.complex128)    # PO@rsurf（物理相位）
    occl_all = np.zeros((Nang, Ns), dtype=bool)
    fh = h5py.File(M.H5, "r")
    for i in range(Nang):
        Hs = fh["H_scat"][i].astype(np.complex128)
        ki = khat[i].astype(np.float64)
        ei = e0[i].astype(np.complex128)
        u_inc = np.cross(ki, ei) / e0mag[i]                    # 单位入射 H 方向（k̂×ê0）
        psi_air = np.exp(-1j * beta[i] * (r_air @ ki))
        e_air = ei[None, :] * psi_air[:, None]
        h_inc = np.cross(ki[None, :], e_air) / ETA0
        H_tot = h_inc + Hs[owners[:, 0], owners[:, 1], owners[:, 2]]
        J_num = np.cross(nvec, H_tot)
        Jnum_phys[i] = J_num
        # 去相位 + 无量纲化：Ĵ = J/ψ · η₀/|e0|，使入射项恰好等于 û
        scale = ETA0 / e0mag[i]
        Jn = J_num * (np.conj(psi_air) * scale)[:, None]
        cosi = -(nvec @ ki)
        occl = ray_occlusion(owners, -ki, metal)
        occl_all[i] = occl
        lit = (cosi > 0) & (~occl)
        Jpo = 2.0 * np.cross(nvec, np.broadcast_to(u_inc, (Ns, 3))) * lit[:, None]
        dJ = Jn - Jpo

        # 局部切向基：e1 = k̂ 的面内投影方向，e2 = n̂×e1
        t1 = ki[None, :] + cosi[:, None] * nvec
        n1 = np.linalg.norm(t1, axis=1)
        bad = n1 < 1e-9
        if bad.any():
            fb = np.zeros(3); fb[int(np.argmin(np.abs(ki)))] = 1.0
            t1[bad] = fb[None, :] - (nvec[bad] @ fb)[:, None] * nvec[bad]
            n1 = np.linalg.norm(t1, axis=1)
        n1[n1 < 1e-12] = 1.0
        e1 = t1 / n1[:, None]
        e2 = np.cross(nvec, e1)

        dJr[i, :, 0] = (dJ * e1).sum(1).real
        dJr[i, :, 1] = (dJ * e1).sum(1).imag
        dJr[i, :, 2] = (dJ * e2).sum(1).real
        dJr[i, :, 3] = (dJ * e2).sum(1).imag
        Jn_norm[i] = np.linalg.norm(Jn, axis=1)

        psi_s = np.exp(-1j * beta[i] * (rsurf @ ki))
        Jpo_surf[i] = 2.0 * np.cross(nvec, np.broadcast_to(np.cross(ki, ei) / ETA0, (Ns, 3))) \
            * psi_s[:, None] * lit[:, None] / e0mag[i]

        Xa[i, :, 0] = cosi
        Xa[i, :, 1] = lit
        Xa[i, :, 2] = occl
        Xa[i, :, 3] = (np.broadcast_to(u_inc, (Ns, 3)) * e1).sum(1).real
        Xa[i, :, 4] = (np.broadcast_to(u_inc, (Ns, 3)) * e2).sum(1).real
        Xa[i, :, 5] = (np.broadcast_to(u_inc, (Ns, 3)) * nvec).sum(1).real
        Xa[i, :, 6] = np.sqrt(np.maximum(1.0 - cosi ** 2, 0.0))
        if (i + 1) % 120 == 0:
            print(f"  角度 {i+1}/{Nang}  ({time.time()-t0:.0f}s)", flush=True)
    fh.close()

    litmask = (Xa[:, :, 1] > 0.5)
    rel_po = np.linalg.norm(dJr, axis=2) / np.maximum(Jn_norm, 1e-30)
    print(f"\n[面元级] 受照面占比 {litmask.mean()*100:.1f}%  遮挡 {Xa[:,:,2].mean()*100:.1f}%")
    print(f"  |ΔJ|/|J_num| 中位：全部 {np.median(rel_po)*100:.1f}%  "
          f"仅受照 {np.median(rel_po[litmask])*100:.1f}%")

    fidx_dim = Ns
    X_ang = Xa.reshape(-1, nfeat)
    Y = dJr.reshape(-1, 4)
    lit_flat = litmask.reshape(-1)
    Xfull = np.hstack([X_ang, np.tile(X_geo, (Nang, 1))])
    Xctrl = np.hstack([Xfull, np.tile(X_pos, (Nang, 1))])

    rng = np.random.default_rng(0)
    f_rand = rng.integers(0, 5, size=Ns)
    lab = kmeans_np(r_air.astype(np.float64), 8, seed=0)
    f_spat = lab
    f_mirr = (r_air[:, 1] > 0).astype(np.int64)
    print(f"\n[划分] 随机5折 / 空间8簇 {np.bincount(lab).tolist()} / "
          f"镜像 y>0 {int((r_air[:,1]>0).sum())} 面元, y≤0 {int((r_air[:,1]<=0).sum())} 面元")

    out = {"n_facet": int(Ns), "n_angle": int(Nang), "n_sample": int(Ns * Nang),
           "lit_frac": float(litmask.mean()),
           "po_facet_rel_median_all": float(np.median(rel_po)),
           "po_facet_rel_median_lit": float(np.median(rel_po[litmask])),
           "r2": {}, "r2_lit": {}}
    folds = {"random": np.tile(f_rand, Nang), "spatial": np.tile(f_spat, Nang),
             "mirror": np.tile(f_mirr, Nang)}
    print("\n  R²（基准=训练均值；0 表示不优于直接用 PO）")
    print(f"  {'特征集':<20}{'口径':<8}{'随机':>9}{'空间':>9}{'镜像':>9}")
    for tag, Xs in (("局部特征", Xfull), ("局部+绝对坐标", Xctrl)):
        for mtag, mk in (("全部", None), ("仅受照面", lit_flat)):
            row = {nm: eval_splits(Xs, Y, fd, mk) for nm, fd in folds.items()}
            (out["r2"] if mk is None else out["r2_lit"])[tag] = row
            label = "局部特征" if tag == "局部特征" else "  +绝对坐标"
            print(f"  {label:<20}{mtag:<8}{row['random']:>9.3f}{row['spatial']:>9.3f}"
                  f"{row['mirror']:>9.3f}")

    # 非线性检验（小 MLP，空间划分，仅受照面）
    try:
        sel = np.where(lit_flat)[0]
        sub = rng.choice(sel, size=min(150000, len(sel)), replace=False)
        Xt = torch.tensor(Xfull[sub], dtype=torch.float32)
        Yt = torch.tensor(Y[sub], dtype=torch.float32)
        ft = torch.tensor(np.tile(f_spat, Nang)[sub], dtype=torch.long)
        mu = Xt.mean(0); sd = Xt.std(0); sd[sd < 1e-12] = 1.0
        Xn = (Xt - mu) / sd
        pred = np.zeros((len(sub), 4), dtype=np.float32)
        for f in np.unique(ft.numpy()):
            tr = ft != f; te = ~tr
            if te.sum() < 30:
                continue
            net = torch.nn.Sequential(torch.nn.Linear(Xfull.shape[1], 128), torch.nn.GELU(),
                                      torch.nn.Linear(128, 128), torch.nn.GELU(),
                                      torch.nn.Linear(128, 4))
            opt = torch.optim.Adam(net.parameters(), lr=2e-3)
            for ep in range(60):
                perm = torch.randperm(int(tr.sum()))
                for b in range(0, len(perm), 8192):
                    s2 = perm[b:b + 8192]
                    loss = ((net(Xn[tr][s2]) - Yt[tr][s2]) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                pred[te.numpy()] = net(Xn[te]).numpy()
        den = np.sum((Y[sub] - Y[sub].mean(0)) ** 2)
        out["r2_mlp_spatial_lit"] = float(1.0 - np.sum((pred - Y[sub]) ** 2) / den)
        print(f"  {'小 MLP（非线性）':<20}{'仅受照面':<8}{'':>9}{out['r2_mlp_spatial_lit']:>9.3f}")
    except Exception as ex:
        print(f"  [MLP 跳过] {ex}")
        out["r2_mlp_spatial_lit"] = None

    # ---------- 远场 ----------
    print("\n[远场] NFFFT 结果（对照 FEKO 真值，468 角度）")
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    P = np.exp(1j * k * (rsurf @ rhat.T))
    sm = smooth3(metal.astype(np.float64))
    grad = np.stack(np.gradient(sm, gx, gy, gz), axis=-1)
    nsm = -grad / np.maximum(np.linalg.norm(grad, axis=-1, keepdims=True), 1e-12)
    nsm_f = nsm[owners[:, 0], owners[:, 1], owners[:, 2]]
    nsm_f = nsm_f / np.maximum(np.linalg.norm(nsm_f, axis=1, keepdims=True), 1e-12)

    def farstats(Js, tag):
        """Js:(Nang,Ns,3) 已经过 e0 归一化；P 只含远场相位核（与入射角无关），逐角度做 matmul。"""
        rcs = np.zeros((Nang, *shape), dtype=np.float64)
        for a in range(Nang):
            N = P.T @ (Js[a] * dA[:, None])
            Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
            E_ff = (1j * k / (4 * np.pi)) * (-ETA0 * Nperp)
            rcs[a] = (4 * np.pi * (np.abs((E_ff * th).sum(1)) ** 2
                                   + np.abs((E_ff * ph).sum(1)) ** 2)
                      / e0mag[a] ** 2).reshape(shape)
        st = _rcs_stats(rcs, rcs_true_all)
        print(f"  {tag:<26} med {st['rcs_dB_err_median']:6.2f} dB   "
              f"P90 {st['rcs_dB_err_p90']:6.2f} dB   corr {st['rcs_corr_median']:.3f}")
        return st

    po1 = farstats(Jpo_surf, "PO 阶梯面法向")
    Jsm = np.zeros_like(Jpo_surf)
    for i in range(Nang):
        ki = khat[i].astype(np.float64)
        psi_s = np.exp(-1j * beta[i] * (rsurf @ ki))
        Hv = np.cross(ki, e0[i].astype(np.complex128)) / ETA0
        lit_s = (nsm_f @ ki < 0) & (~occl_all[i])
        Jsm[i] = 2.0 * np.cross(nsm_f, np.broadcast_to(Hv, (Ns, 3))) \
            * psi_s[:, None] * lit_s[:, None] / e0mag[i]
    po2 = farstats(Jsm, "PO 光滑法向")
    ref = farstats(Jnum_phys / e0mag[:, None, None], "truth J_num（复现 floor）")
    out["farfield"] = {"truth_floor": ref, "po_staircase": po1, "po_smooth": po2}

    r_rand = out["r2_lit"]["局部特征"]["random"]
    r_spat = out["r2_lit"]["局部特征"]["spatial"]
    r_mirr = out["r2_lit"]["局部特征"]["mirror"]
    ok = (r_spat >= 0.5 * max(r_rand, 0)) and (r_spat > 0.3)
    out["verdict"] = {"rule": "仅受照面：R2_spatial >= 0.5*R2_random 且 R2_spatial > 0.3",
                      "r2_random": r_rand, "r2_spatial": r_spat, "r2_mirror": r_mirr,
                      "hypothesis_holds": bool(ok)}
    print("\n" + "=" * 78)
    print(f"判定（仅受照面）: R²_random={r_rand:.3f}  R²_spatial={r_spat:.3f}  "
          f"R²_mirror={r_mirr:.3f}")
    print(f"假设 H（残差只依赖局部信息）→ {'成立' if ok else '不成立'}")
    print("=" * 78)
    jp = os.path.join(RESULT_DIR, "exp_po_locality.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False, default=float)
    print(f"已存: {jp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
