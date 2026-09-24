# -*- coding: utf-8 -*-
"""角度离散化误差分析：量化 10° 网格下最近角查表的真实场差异（加密收益上限）。
思路：查表时把查询角量化到最近训练角，其误差 = |E(θ_q,φ_q) − E(θ_nn,φ_nn)| / |E(θ_nn,φ_nn)|。
用现有 468 组 10° 数据统计相邻角真实场相对差异，作为 10° 离散化误差的度量；
若该值 >> 模型记忆误差(48.5%)，则角度加密有意义（可降低查表误差），否则无意义。
"""
import h5py
import numpy as np

H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"

with h5py.File(H5, "r") as f:
    Es = f["E_scat"][:]           # (468,64,48,32,3) complex
    ang = f["angles"][:]          # (468,2) [theta, phi]

thetas = sorted(set(ang[:, 0]))
phis = sorted(set(ang[:, 1]))
ti = {t: i for i, t in enumerate(thetas)}
pi = {p: i for i, p in enumerate(phis)}
idx_map = {(int(a[0]), int(a[1])): k for k, a in enumerate(ang)}

def rel_diff(i, j):
    a, b = Es[i], Es[j]
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))

# 1) 最近邻（θ 或 φ 单轴 10°）差异统计
d_phi, d_theta = [], []
for (t, p), i in idx_map.items():
    if p + 10 in pi and (t, p + 10) in idx_map:
        d_phi.append(rel_diff(i, idx_map[(t, p + 10)]))
    if t + 10 in ti and (t + 10, p) in idx_map:
        d_theta.append(rel_diff(i, idx_map[(t + 10, p)]))
d_phi, d_theta = np.array(d_phi), np.array(d_theta)

# 2) 每角最小邻（8 邻域内最近）差异——查表真实误差下界
d_min = []
for (t, p), i in idx_map.items():
    best = 1e9
    for dt, dp in [(0, 10), (0, -10), (10, 0), (-10, 0), (10, 10), (10, -10), (-10, 10), (-10, -10)]:
        nt, np_ = t + dt, p + dp
        if nt in ti and np_ in pi and (nt, np_) in idx_map:
            best = min(best, rel_diff(i, idx_map[(nt, np_)]))
    if best < 1e8:
        d_min.append(best)
d_min = np.array(d_min)

def stat(name, a):
    print(f"{name:24s}: med={np.median(a)*100:5.1f}%  p25={np.percentile(a,25)*100:5.1f}% "
          f"p75={np.percentile(a,75)*100:5.1f}%  mean={a.mean()*100:5.1f}%")

print(f"角度网格: θ={thetas[0]}..{thetas[-1]}步长{thetas[1]-thetas[0]}°, "
      f"φ={phis[0]}..{phis[-1]}步长{phis[1]-phis[0]}°  共{len(ang)}组")
stat("10° φ 邻域差异", d_phi)
stat("10° θ 邻域差异", d_theta)
stat("最近邻(8方向)差异", d_min)
print(f"\n对照：full-w128 模型记忆误差 48.5%（查表误差下限）")
print(f"结论：若 10° 邻域差异中位数 >> 48.5%，角度加密显著可降查表误差；否则加密收益有限")
