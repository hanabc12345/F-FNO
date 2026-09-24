# -*- coding: utf-8 -*-
"""
verify_dataset_v2.py — f16_3d_rcs_dataset_v2.h5 完整性 & 质量核查
检查：结构/形状、角度网格覆盖、NaN·Inf、金属体素、与 pilot 交叉核对、互易性
用法: D:\\shared\\Python39_64\\python.exe verify_dataset_v2.py
"""
import os
import numpy as np
import h5py

H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"
ok = True


def chk(name, cond, detail=""):
    global ok
    mark = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{mark}] {name}" + (f"  {detail}" if detail else ""))


print(f"文件: {H5}  ({os.path.getsize(H5)/1e9:.2f} GB)")
with h5py.File(H5, "r") as f:
    print(f"attrs: {dict(f.attrs)}")
    expect = {"eps_field", "grid_x", "grid_y", "grid_z", "angles",
              "E_scat", "H_scat", "E_scat_ortho", "H_scat_ortho",
              "rcs", "rcs_ortho", "E_ff", "E_ff_ortho", "ff_theta", "ff_phi"}
    chk("全部 15 个数据集存在", expect <= set(f.keys()),
        f"缺失: {sorted(expect - set(f.keys())) or '无'}")

    n = f.attrs["n_angles"]
    chk("n_angles=468", n == 468, str(n))

    ang = f["angles"][:]
    thetas = np.unique(ang[:, 0]); phis = np.unique(ang[:, 1])
    chk("angles 形状 [468,2]", ang.shape == (468, 2))
    chk("θ 覆盖 30..150/10 (13 值)", thetas.tolist() == list(range(30, 151, 10)), str(thetas))
    chk("φ 覆盖 0..350/10 (36 值)", phis.tolist() == list(range(0, 360, 10)), str(phis))
    chk("角度行唯一", len(np.unique(ang, axis=0)) == 468)

    # 形状与 dtype
    s = {"E_scat": (468, 64, 48, 32, 3), "H_scat": (468, 64, 48, 32, 3),
         "E_scat_ortho": (468, 64, 48, 32, 3), "H_scat_ortho": (468, 64, 48, 32, 3),
         "rcs": (468, 37, 73), "rcs_ortho": (468, 37, 73),
         "E_ff": (468, 37, 73, 2), "E_ff_ortho": (468, 37, 73, 2),
         "eps_field": (64, 48, 32)}
    for k, sh in s.items():
        chk(f"{k} shape={sh}", f[k].shape == sh, f"实际 {f[k].shape}")

    # 数值有限性
    for k in ["E_scat", "H_scat", "E_scat_ortho", "H_scat_ortho", "rcs", "rcs_ortho"]:
        d = f[k][:]
        chk(f"{k} 全有限(无 NaN/Inf)", np.all(np.isfinite(d)))
    # rcs 物理范围（F-16 在 3GHz：0 ~ 数百 m²）
    r = f["rcs"][:]; ro = f["rcs_ortho"][:]
    chk(f"rcs 范围合理 (min={r.min():.4f}, max={r.max():.1f} m²)", r.min() >= 0 and 0 < r.max() < 1e4)

    eps = f["eps_field"][:]
    metal = np.sum(eps > 1.5)
    chk("金属体素计数合理", metal > 0, f"{metal} 个 (v1 为 699)")

    # 与 pilot case_216 交叉核对（θ=90,φ=0）
    i216 = int(np.where((ang == [90, 0]).all(axis=1))[0][0])
    e216 = f["E_scat"][i216]; r216 = f["rcs"][i216]; ro216 = f["rcs_ortho"][i216]
    chk("case_216 |E_scat|max≈4.425 (pilot)",
        abs(np.abs(e216).max() - 4.425) < 1e-3, f"{np.abs(e216).max():.4f}")
    chk("case_216 RCSmax≈4.304 m² (pilot)", abs(r216.max() - 4.304) < 1e-3, f"{r216.max():.4f}")
    chk("case_216 正交 RCSmax≈13.593 m² (pilot)", abs(ro216.max() - 13.593) < 1e-3, f"{ro216.max():.4f}")

    # 互易性（双站）：σ(入射A→观测O) = σ(入射O→观测A)（v1 已验证 ~1e-3）。
    # 即 rcs[case_i][θ_j,φ_j] == rcs[case_j][θ_i,φ_i]（(θ_i,φ_i) 为 case_i 的入射方向）。
    def mono(theta, phi):  # 远场网格索引: θ 0-180/5, φ 0-360/5
        return theta // 5, phi // 5

    rec_pairs = [((60, 0), (120, 180)), ((90, 0), (90, 180)),
                 ((60, 0), (60, 180)), ((90, 180), (60, 0))]
    for (t1, p1), (t2, p2) in rec_pairs:
        i1 = int(np.where((ang == [t1, p1]).all(axis=1))[0][0])
        i2 = int(np.where((ang == [t2, p2]).all(axis=1))[0][0])
        k2, l2 = mono(t2, p2); k1, l1 = mono(t1, p1)
        v1c, v1o = f["rcs"][i1][k2, l2], f["rcs_ortho"][i1][k2, l2]
        v2c, v2o = f["rcs"][i2][k1, l1], f["rcs_ortho"][i2][k1, l1]
        chk(f"互易 σ({t1},{p1}→{t2},{p2})≈σ({t2},{p2}→{t1},{p1}) 同极化",
            abs(v1c - v2c) < 1e-2, f"{v1c:.4f} vs {v2c:.4f} diff={abs(v1c-v2c):.2e}")
        chk(f"互易 σ({t1},{p1}→{t2},{p2})≈σ({t2},{p2}→{t1},{p1}) 正交极化",
            abs(v1o - v2o) < 1e-2, f"{v1o:.4f} vs {v2o:.4f} diff={abs(v1o-v2o):.2e}")

    # E/H 物理关系 |H|/|E| ~ 1/η0≈0.00265（近场数量级）
    h = f["H_scat"][i216]; ratio = np.abs(h).max() / np.abs(e216).max()
    chk("|H|/|E| 近场数量级合理(~0.001-0.005)", 0.0005 < ratio < 0.02, f"{ratio:.4f} (1/η0=0.00265)")

print("\n" + ("=" * 50))
print("  结论: 数据集完整且通过全部核查" if ok else "  结论: 存在 FAIL 项，需处理")
