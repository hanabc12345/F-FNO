# -*- coding: utf-8 -*-
"""F-16 模型尺寸/结构/网格质量检查"""
import sys
import numpy as np
import trimesh

STL = r"f:\MyWorkSpace\UAVGame\3d_feko_run\f16_refined.stl"
FREQ_HZ = 3e9
LAMBDA = 2.99792458e8 / FREQ_HZ

mesh = trimesh.load(STL, process=False)
print("== 基础 ==")
print(f"faces: {len(mesh.faces)}   vertices: {len(mesh.vertices)}")
print(f"watertight: {mesh.is_watertight}")

# 修复检查
mesh2 = trimesh.load(STL, process=True)
print(f"process=True 后 faces: {len(mesh2.faces)}  watertight: {mesh2.is_watertight}")

v = mesh.vertices
lo, hi = v.min(axis=0), v.max(axis=0)
ext = hi - lo
print(f"\n== 包围盒 ==")
print(f"min: {np.round(lo,4)}  max: {np.round(hi,4)}")
print(f"尺寸: L(x)={ext[0]:.3f} m  W(y)={ext[1]:.3f} m  H(z)={ext[2]:.3f} m")
print(f"电尺寸: L={ext[0]/LAMBDA:.1f}λ  W={ext[1]/LAMBDA:.1f}λ  H={ext[2]/LAMBDA:.1f}λ")
print(f"面积: {mesh.area:.3f} m²")

# 真实 F-16 对比
print(f"\n== 真实 F-16 (15.06m×9.45m×5.09m) 对比 ==")
real = np.array([15.06, 9.45, 5.09])
print(f"缩放比(模型/真实): {ext/real}")
print(f"比例 L:W:H 模型={ext[0]/ext[1]:.2f}:1:{ext[2]/ext[1]:.2f}  真实={real[0]/real[1]:.2f}:1:{real[2]/real[1]:.2f}")

# 对称性（F-16 关于 y=0 平面近似对称）—— 量化匹配
print(f"\n== y 对称性（关于 y=0 平面）==")
mm = mesh.copy()
mm.merge_vertices()
mv = mm.vertices
print(f"合并后顶点数: {len(mv)}")
for tol_mm in (0.5, 2.0):
    q = 1e-3 * tol_mm
    k1 = np.round(mv / q, 0).astype(np.int64)
    k2 = np.round(mv * np.array([1, -1, 1]) / q, 0).astype(np.int64)
    u1 = set(map(tuple, k1))
    u2 = set(map(tuple, k2))
    n_common = len(u1 & u2)
    print(f"容差{tol_mm}mm: 唯一顶点 {len(u1)} 与镜像重合 {n_common} ({100*n_common/max(len(u1),1):.1f}%)")
print(f"顶点 |y|: max={np.abs(mv[:,1]).max():.4f}  mean={np.abs(mv[:,1]).mean():.4f} m")

# 面积统计（三角形网格质量）
faces = mesh.faces
p0, p1, p2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
areas = 0.5 * np.linalg.norm(np.cross(p1-p0, p2-p0), axis=1)
print(f"\n== 网格质量 ==")
print(f"三角形面积: min={areas.min():.3e}  median={np.median(areas):.3e}  max={areas.max():.3e} m²")
print(f"退化面(面积<1e-10): {(areas<1e-10).sum()} 个")
# 边长大致估计
edge = np.sqrt(areas)
print(f"等效边长: median={np.median(edge)*1000:.2f} mm = λ/{LAMBDA/np.median(edge):.1f}")
# 网格密度 vs λ
print(f"目标典型尺寸 λ/10 = {LAMBDA/10*1000:.1f} mm → 网格需 <{LAMBDA/10*1000:.1f} mm")

# 近场盒覆盖检查
X0,Y0,Z0,X1,Y1,Z1 = -0.75,-0.75,-0.60,1.21875,0.71875,0.36875
print(f"\n== 近场盒覆盖 ==")
print(f"近场盒: x[{X0},{X1}] y[{Y0},{Y1}] z[{Z0},{Z1}]")
print(f"模型  : x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}] z[{lo[2]:.3f},{hi[2]:.3f}]")
print(f"余量(x): 前 {(X1-hi[0])/LAMBDA:.1f}λ 后 {(lo[0]-X0)/LAMBDA:.1f}λ")

# 检查是否有孤立碎片（连通性）
cc = trimesh.graph.connected_components(mesh.face_adjacency)
print(f"\n== 连通性 ==")
print(f"连通分量数: {len(cc)}")
sizes = sorted(len(c) for c in cc)
print(f"最大分量面数: {sizes[-1]}  其余: {sizes[:-1][-5:] if len(sizes)>1 else []}")
