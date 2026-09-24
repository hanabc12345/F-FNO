# -*- coding: utf-8 -*-
"""
po_patch_data.py — 局部 patch 数据集构建（"PO 粗解 + 局部算子修正"路线第 2 步）
================================================================================
把"整机 → 近场"的算子学习问题，改写成"局部 patch → 电流残差"的**局部算子**问题：

  对每个受照面元 s 与入射角 i：
    Ĵ_num(s,i) = (n̂×H_tot)|voxel · conj(ψ) · η₀/|e0|      （去快相位 + 无量纲化）
    J_PO(s,i)  = 2 n̂×û                                     （PO 粗解，û = k̂×ê0）
    ΔĴ(s,i)    = Ĵ_num − J_PO   ∈ span(e1,e2)               （4 个实分量：e1/e2 的 Re/Im）
  输入 = 以面元为原点、局部规范帧 (e1,e2,n̂) 上的 **占用立方体** patch O[i,j,k]
         + 7 个全局标量（k̂、û 在局部基下的分量 + kh）。

两个关键约定
------------
1. **局部规范帧**（决定 patch 可复用的前提）：
     n̂ = 面元外法向（阶梯面 → 严格轴向）
     e1 = k̂ 在切平面的投影归一化（掠射退化时改用 û 的切向分量，保证法向入射也可定义）
     e2 = n̂ × e1
   该帧由"面元的法向 + 入射平面"唯一确定，与面元在机体上的绝对位置无关。
2. **采样对齐**（避免半体素歧义）：
     采样点索引 = round( p + i·e1 + j·e2 + (k+0.5)·n̂ )，p = 金属体素索引
   → k=−1 层落在金属体素 p，k=0 层落在空气体素 q，**表面平面恰在 k=−1/0 之间**，
     与 surface_parts 的面元中心（p+0.5h·n̂）严格一致。取整用 floor(x+0.5)（tie 向上），
     否则 round-half-even 会把 k=0 与 k=−1 折到同一层。

多尺度（Maxwell 尺度律，物理严格、零额外求解成本）
--------------------------------------------------
  目标缩放 s 倍 ⟺ 同一几何、频率 f/s。绕质心 c 缩放：
    metal_s(r′) = metal(c + (r′−c)/s)，  H_scat_s(r′) = H_scat(c + (r′−c)/s)，
    ψ_s = exp(−j(β/s) k̂·(r′−c))，  kh_s = β·h/s
  （入射波解严格满足：E_inc′(r′) = E_inc(c+(r′−c)/s)，见 docs 说明。全局常数相位
    exp(−jβk̂·c) 在 Ĵ 的归一化中相消，故可略。）
  注意：本项目 voxel 网格 h=λ/3.2 是**粗网格**——s 变化会同时改变 h/λ′（离散分辨率等级），
  因此跨尺度 ≠ 严格"同离散不同电尺寸"，这一条会在报告里如实标注。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" po_patch_data.py --scales 0.7,0.925,1.0,1.15 --step 4 --P 11
产出：results/po_patches/s<scale>.npz + results/po_patches/index.json
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
from fno_f16_3d_p4_nffft import surface_parts
from exp_po_locality import facet_geometry, ray_occlusion

ETA0 = 119.9169832 * np.pi
H = 0.03125
OUT_DIR = os.path.join(BASE, "results", "po_patches")
CHUNK = 800                     # patch 采样分块（控制中间张量内存）


# ============================================================
# 一、几何重采样（绕质心，避免 s≠1 时目标漂移出盒/被截断）
# ============================================================

def _tri(field, U, V, W):
    """三线性插值：U/V/W 为源网格浮点索引（(Nx,Ny,Nz)），越界按 0 计。"""
    Nx, Ny, Nz = field.shape[:3]
    u0 = np.floor(U).astype(np.int64); v0 = np.floor(V).astype(np.int64)
    w0 = np.floor(W).astype(np.int64)
    fu = (U - u0)[..., None]; fv = (V - v0)[..., None]; fw = (W - w0)[..., None]
    out = np.zeros_like(field)
    for du in (0, 1):
        uu = u0 + du; ok_u = (uu >= 0) & (uu < Nx); uu = uu.clip(0, Nx - 1)
        wu = fu if du else (1 - fu)
        for dv in (0, 1):
            vv = v0 + dv; ok_v = (vv >= 0) & (vv < Ny); vv = vv.clip(0, Ny - 1)
            wv = fv if dv else (1 - fv)
            for dw in (0, 1):
                ww = w0 + dw; ok_w = (ww >= 0) & (ww < Nz); ww = ww.clip(0, Nz - 1)
                ok = (ok_u & ok_v & ok_w)[..., None]
                out += np.where(ok, field[uu, vv, ww] * (wu * wv * (fw if dw else (1 - fw))), 0)
    return out


def resample_about(field, scale, center, gx, gy, gz, h):
    """new(r′) = field(c + (r′−c)/s)。field:(Nx,Ny,Nz[,C])。"""
    u = (center[0] + (gx - center[0]) / scale - gx[0]) / h
    v = (center[1] + (gy - center[1]) / scale - gy[0]) / h
    w = (center[2] + (gz - center[2]) / scale - gz[0]) / h
    U, V, W = np.meshgrid(u, v, w, indexing="ij")
    return _tri(field, U, V, W)


# ============================================================
# 二、局部规范帧 与 patch 采样
# ============================================================

def local_frame(nvec, khat, uhat):
    """返回 (e1,e2)：(e1,e2,n̂) 右手正交基。
    e1 = k̂ 的切向投影；掠射/法向入射退化时退化为 û 的切向投影（法向入射时 û⊥n̂，必有定义）。"""
    t = khat[None, :] - (khat[None, :] * nvec).sum(1, keepdims=True) * nvec
    nt = np.linalg.norm(t, axis=1)
    bad = nt < 1e-9
    if bad.any():
        t2 = uhat - (uhat * nvec).sum(1, keepdims=True) * nvec
        t = np.where(bad[:, None], t2, t)
        nt = np.linalg.norm(t, axis=1)
    nt = np.maximum(nt, 1e-12)
    e1 = t / nt[:, None]
    e2 = np.cross(nvec, e1)
    return e1, e2


def sample_patches(metal, p_idx, e1, e2, nvec, P):
    """O[i,j,k] = metal[ floor(p + i·e1 + j·e2 + (k+0.5)·n̂ + 0.5) ]，(N,P,P,P) uint8。
    采样点索引与体素索引差 0（p 已是体素索引），故无需网格原点。"""
    N = len(p_idx)
    m = P // 2
    rng = np.arange(-m, m + 1, dtype=np.float64)
    ii, jj, kk = np.meshgrid(rng, rng, rng, indexing="ij")
    ii = ii.ravel(); jj = jj.ravel(); kk = (kk.ravel() + 0.5)
    Nx, Ny, Nz = metal.shape
    lim = np.array([Nx, Ny, Nz])
    out = np.zeros((N, P ** 3), dtype=np.uint8)
    for a in range(0, N, CHUNK):
        sl = slice(a, min(a + CHUNK, N))
        base = p_idx[sl].astype(np.float64)
        idx = (base[:, None, :]
               + ii[None, :, None] * e1[sl][:, None, :]
               + jj[None, :, None] * e2[sl][:, None, :]
               + kk[None, :, None] * nvec[sl][:, None, :])
        q = np.floor(idx + 0.5).astype(np.int64)
        ok = ((q >= 0) & (q < lim)).all(-1)
        qc = np.clip(q, 0, lim - 1)
        out[sl] = np.where(ok, metal[qc[..., 0], qc[..., 1], qc[..., 2]], 0)
    return out.reshape(N, P, P, P)


# ============================================================
# 三、单尺度构建
# ============================================================

def canon_phase(ei):
    """把入射场 e0 的**全局相位**归一：取最大分量的相位为 0。
    线极化下归一后 e0 为实向量，于是 û = k̂×ê0 是实单位矢（局部基、PO 粗解都只应
    依赖这个几何量，不该带 e0 的任意全局相位）。入射场与散射场必须**同乘**该相位，
    才能保持二者相对相位不变（J 与 RCS 均不变）。"""
    jm = int(np.argmax(np.abs(ei)))
    return np.exp(-1j * np.angle(ei[jm]))


def build_scale(scale, fh, angles, e0, khat, beta, gx, gy, gz, P, aidx, verbose=True):
    t0 = time.time()
    eps0 = fh["eps_field"][:]
    metal0 = (eps0 > 1.5)
    center = np.array(np.argwhere(metal0).mean(0), dtype=np.float64)
    center = np.array([gx[int(round(center[0]))], gy[int(round(center[1]))],
                       gz[int(round(center[2]))]])
    # 绕质心缩放掩膜（对**二值掩膜**做插值，再按 0.5 判决——不要对 1e6 的 eps 值插值）
    mask_f = resample_about(metal0.astype(np.float32)[:, :, :, None], scale, center,
                            gx, gy, gz, H)[..., 0]
    metal = mask_f > 0.5
    eps = np.where(metal, np.float32(1e6), np.float32(1.0))
    beta_s = beta / scale
    print(f"[s={scale}] 金属体素 {int(metal.sum())}（原始 {int(metal0.sum())}）  "
          f"质心 {center.round(3)}  kh={float(beta_s[0])*H:.3f}", flush=True)

    owners, rsurf, dS = surface_parts(eps, gx, gy, gz)
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / dA[:, None]
    p_idx = owners - np.rint(dS / (H * H)).astype(np.int64)
    r_air = np.stack([gx[owners[:, 0]], gy[owners[:, 1]], gz[owners[:, 2]]], axis=1)
    geo = facet_geometry(owners, np.rint(dS / (H * H)).astype(np.int64), metal)
    X_geo = np.stack([geo["planar_deg"], geo["edge_dist"], geo["n_concave"],
                      geo["n_convex"], geo["nb_metal26"]], axis=1).astype(np.float32)
    Ns = len(dA)

    P_list, G_list, D_list, FI, AI, R_list = [], [], [], [], [], []
    n_lit_tot = 0
    for i in aidx:
        ki = khat[i].astype(np.float64)
        ei = e0[i].astype(np.complex128)
        e0m = float(np.linalg.norm(ei))
        ph0 = canon_phase(ei)
        ei = ei * ph0                                        # 归一后为实向量
        Hs = resample_about(fh["H_scat"][i], scale, center,
                            gx, gy, gz, H).astype(np.complex128) * ph0
        u_inc = (np.cross(ki, ei) / e0m).real                # 实单位入射 H 方向
        assert abs(np.linalg.norm(u_inc) - 1.0) < 1e-3, "û 非实单位矢（入射极化非线极化？）"
        # 全局常数相位 Φ = exp(j(β−β/s)k̂·c)：使变换后解的**入射**部分严格写成
        # (k̂×e0)/η₀·exp(−j(β/s)k̂·r′)（绝对参考点），与 s=1 时的数据约定完全一致。
        # 缺了它，ΔĴ 会整体旋转一个与角度有关的常数相位，而 J_PO 不带 → 残差被污染。
        Phi = np.exp(1j * (beta[i] - beta_s[i]) * float(center @ ki))
        psi = np.exp(-1j * beta_s[i] * (r_air @ ki))          # 绝对参考点，s=1 退化为原约定
        h_inc = np.cross(ki[None, :], np.broadcast_to(ei, (Ns, 3))) / ETA0
        H_tot = h_inc * psi[:, None] + Phi * Hs[owners[:, 0], owners[:, 1], owners[:, 2]]
        J_num = np.cross(nvec, H_tot)
        Jn = J_num * (np.conj(psi) * (ETA0 / e0m))[:, None]

        cosi = -(nvec @ ki)
        occl = ray_occlusion(owners, -ki, metal)
        lit = (cosi > 0) & (~occl)
        n_lit_tot += int(lit.sum())
        if not lit.any():
            continue
        Jpo = 2.0 * np.cross(nvec, np.broadcast_to(u_inc, (Ns, 3)))
        dJ = Jn - Jpo

        e1, e2 = local_frame(nvec, ki, u_inc)
        sl = np.where(lit)[0]
        P_list.append(sample_patches(metal, p_idx[sl], e1[sl], e2[sl], nvec[sl], P))
        G_list.append(np.stack([(ki[None, :] * e1[sl]).sum(1), (ki[None, :] * e2[sl]).sum(1),
                                (ki[None, :] * nvec[sl]).sum(1),
                                (u_inc[None, :] * e1[sl]).sum(1), (u_inc[None, :] * e2[sl]).sum(1),
                                (u_inc[None, :] * nvec[sl]).sum(1),
                                np.full(len(sl), float(beta_s[i]) * H)], axis=1).astype(np.float32))
        dd = dJ[sl]
        D_list.append(np.stack([(dd * e1[sl]).sum(1).real, (dd * e1[sl]).sum(1).imag,
                                (dd * e2[sl]).sum(1).real, (dd * e2[sl]).sum(1).imag],
                               axis=1).astype(np.float32))
        FI.append(sl.astype(np.int32)); AI.append(np.full(len(sl), i, dtype=np.int16))
        R_list.append((np.linalg.norm(dJ[sl], axis=1)
                       / np.maximum(np.linalg.norm(Jn[sl], axis=1), 1e-30)).astype(np.float32))
        if verbose and (len(P_list) % 30 == 0):
            print(f"    s={scale} 角 {len(P_list)}/{len(aidx)}  "
                  f"patch {sum(len(x) for x in FI):,}  ({time.time()-t0:.0f}s)", flush=True)

    if not P_list:
        print(f"[s={scale}] 无受照面元，跳过")
        return None
    rel = np.concatenate(R_list)
    out = {"patch": np.concatenate(P_list), "glob": np.concatenate(G_list),
           "dJ": np.concatenate(D_list), "geo": X_geo[np.concatenate(FI)],
           "fidx": np.concatenate(FI), "aidx": np.concatenate(AI).astype(np.int32),
           "fpos": r_air[np.concatenate(FI)].astype(np.float32),
           "meta": {"scale": scale, "P": P, "h": H, "n_facet": Ns,
                    "n_angle": len(aidx), "n_patch": int(sum(len(x) for x in FI)),
                    "lit_frac": float(n_lit_tot / (Ns * len(aidx))),
                    "dJ_over_J_median": float(np.median(rel)),
                    "dJ_abs_median": float(np.median(np.linalg.norm(np.concatenate(D_list), axis=1))),
                    "kh": float(beta_s[0]) * H, "center": center.tolist()}}
    print(f"[s={scale}] 完成: {out['meta']['n_patch']:,} patch  "
          f"受照占比 {out['meta']['lit_frac']*100:.1f}%  |ΔĴ|/|Ĵ| 中位 "
          f"{out['meta']['dJ_over_J_median']*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.7,0.925,1.0,1.15")
    ap.add_argument("--step", type=int, default=4, help="角度子采样步长（468/step）")
    ap.add_argument("--P", type=int, default=11, help="patch 边长（奇数个体素）")
    ap.add_argument("--only-scale", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    scales = [float(s) for s in args.scales.split(",")]
    if args.only_scale is not None:
        scales = [args.only_scale]
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        assert np.allclose(angles, f["angles"][:])
        aidx = np.arange(0, len(angles), args.step)
        print(f"角度子集 {len(aidx)}（step={args.step}）  尺度 {scales}  P={args.P}")
        idx = {}
        for s in scales:
            d = build_scale(s, f, angles, e0, khat, beta, gx, gy, gz, args.P, aidx)
            if d is None:
                continue
            path = os.path.join(OUT_DIR, f"s{s:g}.npz")
            np.savez_compressed(path, patch=d["patch"], glob=d["glob"], dJ=d["dJ"],
                                geo=d["geo"], fidx=d["fidx"], aidx=d["aidx"],
                                fpos=d["fpos"],
                                meta=np.array([json.dumps(d["meta"])], dtype=object))
            idx[f"{s:g}"] = {"path": os.path.basename(path), **d["meta"]}
            print(f"  已存 {path}  ({os.path.getsize(path)/1e6:.0f} MB)", flush=True)
    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as fo:
        json.dump({"step": args.step, "P": args.P, "h": H, "eta0": ETA0,
                   "beta0": float(beta[0]), "scales": idx}, fo, indent=2, ensure_ascii=False)
    print(f"索引已存: {os.path.join(OUT_DIR, 'index.json')}")


if __name__ == "__main__":
    main()
