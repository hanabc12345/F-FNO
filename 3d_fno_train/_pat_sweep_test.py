# -*- coding: utf-8 -*-
"""验证 pattern_sweep：1) 后向散射点与 rcs_single 一致；2) 不同入射角下方向图随θ变化；3) 耗时。"""
import time
import numpy as np
import ue_rcs_service as svc

eng = svc.RcsEngine()

# 1) 一致性：pattern_sweep(θ,φ) 在 φ=φ_los 处 == rcs_single(θ,φ)
th, ph = 90.0, 0.0
x = eng.make_input(th, ph)
pred = eng.predict(x)
rcs1 = eng.rcs_single(pred, th, ph)
phis, rcs_db = eng.pattern_sweep(th, ph, pred)
i_los = int(np.argmin(np.abs(phis - ph)))
db_single = 10 * np.log10(rcs1 + 1e-9)
print(f"[一致性] rcs_single({th:.0f}°,{ph:.0f}°)={db_single:.3f} dBsm  vs  "
      f"pattern_sweep 在 φ={ph:.0f}°={rcs_db[i_los]:.3f} dBsm  "
      f"→ 差 {abs(db_single-rcs_db[i_los]):.2e} dB")

# 2) 不同入射角的方向图应不同（θ 切面变化）
for th, ph in [(90.0, 0.0), (75.0, 120.0), (60.0, 200.0)]:
    x = eng.make_input(th, ph)
    pred = eng.predict(x)
    t0 = time.perf_counter()
    phis, rcs_db = eng.pattern_sweep(th, ph, pred)
    dt = (time.perf_counter() - t0) * 1000
    print(f"[θ={th:5.1f}° φ={ph:5.1f}°] n={len(phis)} 耗时{dt:6.1f}ms  "
          f"RCS 范围 {rcs_db.min():7.2f}~{rcs_db.max():7.2f} dBsm  "
          f"后向散射点(φ={ph:.0f}°)={rcs_db[i_los]:7.2f} dBsm")

# 3) 72 点（5°步长）耗时
x = eng.make_input(90.0, 30.0)
pred = eng.predict(x)
t0 = time.perf_counter()
phis72, rcs72 = eng.pattern_sweep(90.0, 30.0, pred, phi_step=5.0)
dt72 = (time.perf_counter() - t0) * 1000
print(f"[5°步长72点] 耗时{dt72:6.1f}ms  RCS 范围 {rcs72.min():7.2f}~{rcs72.max():7.2f} dBsm")
