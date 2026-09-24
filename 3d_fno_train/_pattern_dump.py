# -*- coding: utf-8 -*-
"""离线输出方向图37个实际值，验证算法是否产出变化的数据。"""
import ue_rcs_service as svc

eng = svc.RcsEngine()
phis, rcs_db = eng.pattern_cut()
print(f"\n=== 方向图 θ=90° 水平面 37 点实测值 ===")
for p, r in zip(phis, rcs_db):
    print(f"  φ={p:6.1f}°   RCS={r:8.2f} dBsm")
print(f"\n范围: {rcs_db.min():.2f} ~ {rcs_db.max():.2f} dBsm, 差异 {rcs_db.max()-rcs_db.min():.1f} dB")
print(f"是否全部相同: {np.ptp(rcs_db) < 1e-6 if 'np' in dir() else ''}")
