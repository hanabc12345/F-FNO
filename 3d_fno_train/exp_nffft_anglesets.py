# -*- coding: utf-8 -*-
"""
exp_nffft_anglesets.py — 统一角度集合下的 NFFFT 分解（floor vs pred）
====================================================================
目的（回应审稿意见 #1 与 #4）：

  ① 参考链路（truth→NFFFT vs FEKO）与代理链路（pred→NFFFT vs FEKO）必须在
     **同一组角度**上评估，才能唯一分解 4.05 / 5.28 dB 中"来自网络"与"来自管线"
     的份额；原稿 Table II 的 floor 行来自 40 角度子集，in-sample 行来自 468 角度，
     两个口径不一致，无法做分解。

  ② 原稿的 "40-angle subset" 是 np.linspace(0,467,40) 的均匀抽稀，实测只覆盖
     4 个方位面（φ=0°, 110°, 230°, 350°）且其中 39/40 个是 interp 测试角，
     既不是随机子集、也不能代表"未见角度"（与训练角混在一起）；
     本脚本在 **全部 234 个 interp 测试角度**上评估（φ=10,30,...,350）。

同时输出：
  · 近场 rel_E / rel_H（与 metrics_*.json 同口径，可交叉校验）；
  · 保留 legacy40 子集以复核历史数字（floor 3.55 / pred 5.28）；
  · 相位核 P = exp(jk r̂·r′) 只算一次。

远场公式（Balanis 12-10 相对符号，e^{jωt} 约定）：
  E_ff = (jk/4π)[ −η₀ N⊥ + r̂×L ],  N=∮J e^{jk r̂·r′}dS′,  L=∮M e^{jk r̂·r′}dS′
  PEC 目标 M=0；对 |E_ff|² 而言全局符号无影响，但 J/M 相对符号必须一致。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_nffft_anglesets.py
产出：results/exp_nffft_anglesets.json
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
import fno_f16_3d_p3 as P3
from fno_f16_3d_p4_nffft import surface_parts, direction_grid, pred_to_surface
from _exp_common import _rcs_stats, build_x_single

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi


def load_model(name, device):
    """兼容两种 ckpt 键名（P3 训练：y_mean/y_std/x_inc_*；exp_improve：ym/ys/xm/xs）"""
    ck = torch.load(os.path.join(RESULT_DIR, name), map_location="cpu", weights_only=False)
    cfg = ck["config"]; st = ck["stats"]
    ym = st.get("ym", st.get("y_mean")); ys = st.get("ys", st.get("y_std"))
    xm = st.get("xm", st.get("x_inc_mean")); xs = st.get("xs", st.get("x_inc_std"))
    model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=12).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"  [ckpt] {name}: width={cfg['width']} modes={tuple(cfg['modes'])} "
          f"params={sum(p.numel() for p in model.parameters()):,}")
    return (model, np.asarray(ym, np.float32).reshape(12, 1, 1, 1),
            np.asarray(ys, np.float32).reshape(12, 1, 1, 1),
            np.asarray(xm, np.float32).reshape(1, 7, 1, 1, 1),
            np.asarray(xs, np.float32).reshape(1, 7, 1, 1, 1))


def nffft_cached(J, P, dA, rhat, k):
    """P:(Ns,Ndir) 预先算好的相位核；dA:(Ns,) 标量面元面积。PEC: M=0。"""
    N = P.T @ (J * dA[:, None])
    Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
    return (1j * k / (4.0 * np.pi)) * (-ETA0 * Nperp)


def rcs_of(H_s, P, dA, nvec, rhat, th, ph, k, e0mag):
    J = np.cross(nvec, H_s)                      # PEC: J = n̂×H_tot, M = 0
    E_ff = nffft_cached(J, P, dA, rhat, k)
    return 4.0 * np.pi * (np.abs((E_ff * th).sum(axis=1)) ** 2
                          + np.abs((E_ff * ph).sum(axis=1)) ** 2) / e0mag ** 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        rcs_true_all = f["rcs"][:]
        assert np.allclose(angles, f["angles"][:])
    k = float(beta[0])

    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / dA[:, None]
    P = np.exp(1j * k * (rsurf @ rhat.T))          # (Ns,Ndir) 只算一次
    print(f"  表面面元 {len(dA)}（金属体素 {int((eps > 1.5).sum())}）  观测方向 {rhat.shape[0]}"
          f" ({shape[0]}x{shape[1]})  k={k:.4f} rad/m")

    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)

    # ---------------- 角度集合 ----------------
    all_idx = np.arange(len(angles))
    te = np.where(angles[:, 1] % 20 == 10)[0]           # interp 测试：φ=10,30,...,350
    legacy = np.linspace(0, len(angles) - 1, 40).astype(int)
    sets = {"full468": all_idx, "interp234_test": te, "legacy40": legacy}
    plan = {"full468": ["ckpt_full_p3.pt"],
            "interp234_test": ["ckpt_imp_interp_plain_s0.pt", "ckpt_imp_interp_augms_s0.pt"],
            "legacy40": ["ckpt_imp_interp_plain_s0.pt"]}
    for nm, ix in sets.items():
        phis = sorted(set(angles[ix, 1].astype(int).tolist()))
        print(f"  [{nm}] n={len(ix)} 方位面 {len(phis)} 个 "
              f"{phis if len(phis) <= 6 else str(phis[:6]) + '...'}"
              f" 其中 interp-test {int(np.isin(ix, te).sum())}")

    models = {nm: load_model(nm, device) for nm in sorted({m for v in plan.values() for m in v})}

    out = {"k": k, "h_m": 0.03125, "n_surf": int(len(dA)),
           "n_dir": int(rhat.shape[0]), "device": device, "sets": {}}
    arrays = {nm: {} for nm in sets}               # 落盘 (n,37,73) 供复算/bootstrap
    fh = h5py.File(M.H5, "r")

    for nm, ix in sets.items():
        print(f"\n===== {nm} (n={len(ix)}) =====")
        t0 = time.time()
        rcs_true = rcs_true_all[ix]
        rcs_floor = np.zeros((len(ix), *shape), dtype=np.float64)
        for a, i in enumerate(ix):
            Hs = fh["H_scat"][i]
            phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
            e_inc = e0[i][None, None, None, :] * phase[..., None]
            h_inc = np.cross(khat[i][None, None, None, :],
                             np.broadcast_to(e_inc, e_inc.shape)) / ETA0
            H_tot = h_inc + Hs
            rcs_floor[a] = rcs_of(H_tot[idxs[:, 0], idxs[:, 1], idxs[:, 2]],
                                  P, dA, nvec, rhat, th, ph, k,
                                  float(np.linalg.norm(e0[i]))).reshape(shape)
        rec = {"n_angles": int(len(ix)), "n_test_angles": int(np.isin(ix, te).sum()),
               "phi_layers": sorted(set(angles[ix, 1].astype(int).tolist())),
               "angle_idx": [int(x) for x in ix],
               "floor": _rcs_stats(rcs_floor, rcs_true), "pred": {}, "nearfield": {},
               "sur": {}}
        print(f"  [floor] med {rec['floor']['rcs_dB_err_median']:.2f} dB  "
              f"P90 {rec['floor']['rcs_dB_err_p90']:.2f} dB  "
              f"corr {rec['floor']['rcs_corr_median']:.3f}   ({time.time()-t0:.0f}s)")

        for mname in plan[nm]:
            model, ym, ys, xm, xs = models[mname]
            rcs_p = np.zeros((len(ix), *shape), dtype=np.float64)
            acc = {"E": [0.0, 0.0], "H": [0.0, 0.0]}
            for a, i in enumerate(ix):
                Es = fh["E_scat"][i]; Hs = fh["H_scat"][i]
                y_raw = P3.clip12(np.concatenate(
                    [Es.real.transpose(3, 0, 1, 2), Es.imag.transpose(3, 0, 1, 2),
                     Hs.real.transpose(3, 0, 1, 2), Hs.imag.transpose(3, 0, 1, 2)])[None])
                x = build_x_single(i, eps, e0, khat, beta, gx, gy, gz)
                with torch.no_grad():
                    pred = model(torch.from_numpy((x - xm) / xs).to(device))[0].cpu().numpy()
                pred = pred * ys + ym                                     # (12,64,48,32)
                E_s, H_s = pred_to_surface(pred, idxs)
                rcs_p[a] = rcs_of(H_s, P, dA, nvec, rhat, th, ph, k,
                                  float(np.linalg.norm(e0[i]))).reshape(shape)
                for tag, slc in (("E", slice(0, 6)), ("H", slice(6, 12))):
                    p = pred[slc]; t = y_raw[0][slc]
                    acc[tag][0] += float(((p - t) ** 2).sum())
                    acc[tag][1] += float((t ** 2).sum())
            st = _rcs_stats(rcs_p, rcs_true)
            sur = _rcs_stats(rcs_p, rcs_floor)     # E_sur：可归因于代理的直接误差
            nf = {f"rel_{tag}": float((acc[tag][0] / acc[tag][1]) ** 0.5) for tag in ("E", "H")}
            rec["pred"][mname] = st
            rec["sur"][mname] = sur
            rec["nearfield"][mname] = nf
            arrays[nm][mname] = rcs_p.astype(np.float32)
            print(f"  [{mname}] pred med {st['rcs_dB_err_median']:.2f} dB  "
                  f"P90 {st['rcs_dB_err_p90']:.2f} dB  corr {st['rcs_corr_median']:.3f}  "
                  f"| E_sur med {sur['rcs_dB_err_median']:.2f} dB  "
                  f"| relE {nf['rel_E']*100:.2f}% relH {nf['rel_H']*100:.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        arrays[nm]["truth"] = rcs_true.astype(np.float32)
        arrays[nm]["floor"] = rcs_floor.astype(np.float32)
        out["sets"][nm] = rec

    fh.close()
    jp = os.path.join(RESULT_DIR, "exp_nffft_anglesets.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(out, fo, indent=2, ensure_ascii=False, default=float)
    npz = os.path.join(RESULT_DIR, "exp_nffft_anglesets_rcs.npz")
    np.savez_compressed(npz, **{f"{nm}__{key}": arr
                                for nm, d in arrays.items() for key, arr in d.items()})
    print(f"\n已存: {jp}\n已存: {npz}")


if __name__ == "__main__":
    main()
