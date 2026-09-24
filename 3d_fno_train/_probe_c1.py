# -*- coding: utf-8 -*-
"""C1 前置探针 3：自写 z-parity 体素化 + 与 FEKO eps 网格对拍（用后即删）"""
import os, sys, time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np, h5py
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from _diag_nffft_audit2 import read_stl
from exp_po_mesh import STL


def voxelize_axis(tri, ax_u, ax_v, ax_w, gu, gv, gw, h, wmin):
    """沿 axis w 做奇偶填充：把闭合曲面体素化成 (nu,nv,nw) bool。
    ax_* = 0/1/2 轴号。返回 metal 数组。"""
    nu, nv, nw = len(gu), len(gv), len(gw)
    cnt = np.zeros((nu, nv, nw), dtype=np.int16)
    P = tri[:, :, [ax_u, ax_v]]
    W = tri[:, :, ax_w]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    area2 = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    keep = np.abs(area2) > 1e-12
    gu0, gv0 = gu[0], gv[0]
    for t in np.flatnonzero(keep):
        p0, p1, p2 = P[t]; w0, w1, w2 = W[t]
        lo = [min(p0[0], p1[0], p2[0]), min(p0[1], p1[1], p2[1])]
        hi = [max(p0[0], p1[0], p2[0]), max(p0[1], p1[1], p2[1])]
        i0 = max(int(np.ceil((lo[0] - gu0) / h)), 0)
        i1 = min(int(np.floor((hi[0] - gu0) / h)), nu - 1)
        j0 = max(int(np.ceil((lo[1] - gv0) / h)), 0)
        j1 = min(int(np.floor((hi[1] - gv0) / h)), nv - 1)
        if i1 < i0 or j1 < j0:
            continue
        xs = gu[i0:i1 + 1][:, None]
        ys = gv[j0:j1 + 1][None, :]
        # 重心（2D 边函数，符号与 area2 同向）
        dx = xs - p2[0]; dy = ys - p2[1]
        f = 1.0 / area2[t]
        a = ((p1[1] - p2[1]) * dx + (p2[0] - p1[0]) * dy) * f
        b = ((p2[1] - p0[1]) * dx + (p0[0] - p2[0]) * dy) * f
        c = 1.0 - a - b
        m = (a >= 0) & (b >= 0) & (c >= 0)
        if not m.any():
            continue
        zc = a * w0 + b * w1 + c * w2
        kk = np.floor((zc[m] - gw[0]) / h).astype(np.int64) + 1
        ii, jj = np.nonzero(m)
        ok = (kk >= 0) & (kk < nw)
        np.add.at(cnt, (ii[ok] + i0, jj[ok] + j0, kk[ok]), 1)
    return (np.cumsum(cnt, axis=2) % 2).astype(bool)


def build_grid(h, bbmin, bbmax, g0):
    ax = [g0[a] + np.arange(int(np.ceil((bbmax[a] - g0[a]) / h)) + 2) * h for a in range(3)]
    return ax


def voxelize(tri, h, g0, bbmin, bbmax):
    ax = build_grid(h, bbmin, bbmax, g0)
    m = voxelize_axis(tri, 0, 1, 2, ax[0], ax[1], ax[2], h, ax[2][0])
    return ax, m


with h5py.File(M.H5, "r") as f:
    eps = f["eps_field"][:]
    gx = f["grid_x"][:].astype(np.float64); gy = f["grid_y"][:].astype(np.float64)
    gz = f["grid_z"][:].astype(np.float64)
metal_feko = eps > 1.5
print("FEKO metal", int(metal_feko.sum()), metal_feko.shape, flush=True)

tri, _ = read_stl(STL)
bbmin = tri.reshape(-1, 3).min(0); bbmax = tri.reshape(-1, 3).max(0)
g0 = np.array([gx[0], gy[0], gz[0]])
print("bbox", bbmin, bbmax, flush=True)

for h in (0.03125, 0.015625, 0.0078125):
    t0 = time.time()
    ax, m = voxelize(tri, h, g0, bbmin, bbmax)
    print(f"h={h:.7f} shape={m.shape} metal={int(m.sum())} t={time.time()-t0:.1f}s", flush=True)

ax, m = voxelize(tri, 0.03125, g0, bbmin, bbmax)
n = [min(m.shape[a], metal_feko.shape[a]) for a in range(3)]
A = m[:n[0], :n[1], :n[2]]; B = metal_feko[:n[0], :n[1], :n[2]]
print(f"h0 对拍 IoU={int((A&B).sum())/int((A|B).sum()):.3f} mine={int(A.sum())} feko={int(B.sum())}")
for sh in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
    Bs = np.roll(B, sh, [0,1,2])
    print(f"   shift{sh} IoU={int((A&Bs).sum())/int((A|Bs).sum()):.3f}")
