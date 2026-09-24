# -*- coding: utf-8 -*-
"""
nffft_lite.py — 部署包专用：RCS 服务端所需的 NFFFT 表面提取与远场积分（纯 numpy）
============================================================================
从 fno_f16_3d_p4_nffft.py 抽取服务端运行时必需的函数，去除训练/绘图/数据集依赖，
使目标机器只需 numpy/h5py/onnxruntime。

函数：
  surface_parts(eps, gx, gy, gz, h) → (idxs, rsurf, dS)
  nffft(J, M, rsurf, dS, rhat, k)   → E_ff
常量：
  ETA0 = 119.9169832 * pi
"""
import numpy as np

ETA0 = 119.9169832 * np.pi   # 376.7303 Ω


# 表面抽取（face-based：金属体素→空气邻居的面，法向严格轴向）
def surface_parts(eps, gx, gy, gz, h=0.03125):
    """对每个"金属体素-空气体素"相邻面对生成面元。
    体素轴序为 (x,y,z)（meshgrid ij：axis0=x, axis1=y, axis2=z）。
    返回 (idxs_owner (Nf,3) 金属体素索引[场值采样用], rsurf (Nf,3) 面元中心, dS (Nf,3) 外法向·h²)。"""
    metal = eps > 1.5
    Nx, Ny, Nz = metal.shape
    grid = [gx, gy, gz]
    neigh = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
             np.array([0, 1, 0]), np.array([0, -1, 0]),
             np.array([0, 0, 1]), np.array([0, 0, -1])]
    owners, poss, ds = [], [], []
    for p in np.argwhere(metal):
        for d in neigh:
            q = p + d
            if (q < 0).any() or (q[0] >= Nx) or (q[1] >= Ny) or (q[2] >= Nz):
                continue                          # 域外跳过（金属体素不在盒边界）
            if metal[tuple(q)]:
                continue                          # 金属邻居（内部面）跳过
            ax = int(np.argmax(np.abs(d)))        # 面法向轴
            pos = [float(grid[a][p[a]]) for a in range(3)]
            pos[ax] += float(d[ax]) * (h / 2.0)   # 面元中心在体素间中点
            dS_v = np.zeros(3); dS_v[ax] = float(d[ax]) * h * h
            owners.append(q)                      # 场值取空气侧体素（表面外推近似）
            poss.append(pos); ds.append(dS_v)
    print(f"  表面面元: {len(owners)}（金属体素 {int(metal.sum())}）", flush=True)
    return np.asarray(owners), np.asarray(poss), np.asarray(ds)


def nffft(J, M, rsurf, dS, rhat, k):
    """J,M:(Ns,3) complex 等效电流/磁流（已含法向）；rsurf:(Ns,3) 面元中心；
    dS:(Ns,3) 面元矢量（积分用其标量面积 |dS|）；rhat:(Ndir,3) 观测单位矢量；k: 波数。
    返回 E_ff:(Ndir,3) complex（不含 e^{-jkr}/r 因子）。"""
    dA = np.linalg.norm(dS, axis=1)                      # (Ns,) 标量面元面积
    phase = np.exp(1j * k * (rsurf @ rhat.T))            # (Ns,Ndir)
    N = phase.T @ (J * dA[:, None])                      # (Ndir,3)
    L = phase.T @ (M * dA[:, None])
    Nrhat = (N * rhat).sum(axis=1, keepdims=True) * rhat
    cross = np.cross(rhat, L)
    E = (1j * k / (4.0 * np.pi)) * (ETA0 * (N - Nrhat) + cross)
    return E
