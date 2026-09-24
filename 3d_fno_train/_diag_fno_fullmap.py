# -*- coding: utf-8 -*-
"""
_diag_fno_fullmap.py — FNO（部署链路）口径的全向图：468 角 × 2701 方向
================================================================================
动机（P1 理论问题）
--------------------------------------------------------------------------------
项目里所有标定/残差系数（`calib_affine.json`、`resid_mlp.onnx`）都是在**解析 PO
链路**的误差图（`_diag_po_fullmap.npz`）上拟合的，而部署链路是 **FNO→NFFFT**。
本脚本用**部署包原件**（models/fno_f16_3d_p3.onnx + data/stats.npz +
data/geometry.h5 + nffft_lite.py）算出同 schema 的 FNO 口径误差图，使"跨链路借用"
可以被直接检验（对照 PO 口径的三条口径数字）。

同时量化一个实测到的**输入约定差异**
--------------------------------------------------------------------------------
· 训练输入：`e0` 取自 FEKO `.out` 解析表（`incidence_table.npz`）；
· 部署 `make_input`：解析式 `e0 = [cosθcosφ, cosθsinφ, −sinθ]`；
· 实测两者**恰为相反数**（468/468 角，比值 −1，虚部 1e-16）⇒ 部署喂给模型的是
  **整体取负**的入射场。
物理上线性的（E_scat 关于 E_inc 奇），RCS 只取模方 ⇒ 理论上无影响；但模型含 GELU
非线性，奇对称只是近似。故本脚本把两套约定都跑全量，用 ΔRCS 直接量出该约定代价：
  · conv "tab"（与训练一致，P4 pred 用的就是它）→ 可与 P4 封板数字对拍自检；
  · conv "dep"（部署实际所喂）→ 作为交付口径写入 npz 主字段。

口径定义（与 `_diag_decision_err.py` 逐位同源）
--------------------------------------------------------------------------------
· 单站   ：观测方向 = 入射方向 (θ_i,φ_i) → 索引 (it,ip)；
· 方向图：θ_s=θ_i 的锥面，全部 φ（73 点）；
· 池化   ：全部 2701 方向。

用法
--------------------------------------------------------------------------------
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fno_fullmap.py
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fno_fullmap.py --ncase 24   # 快检
产出
--------------------------------------------------------------------------------
  results/_diag_fno_fullmap.npz   （与 _diag_po_fullmap.npz 同 schema；d_full=部署约定）
  results/_diag_fno_fullmap.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(os.path.dirname(BASE), "deploy_rcs_server")
sys.path.insert(0, BASE)
sys.path.insert(0, DEPLOY)

import fno_f16_3d as M                                   # noqa: E402
from nffft_lite import surface_parts, nffft              # noqa: E402
from _diag_decision_err import _sep_and_psi, DELTA_EDGES  # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
ONNX = os.path.join(DEPLOY, "models", "fno_f16_3d_p3.onnx")
STATS = os.path.join(DEPLOY, "data", "stats.npz")
GEOM = os.path.join(DEPLOY, "data", "geometry.h5")
PO_NPZ = os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
BETA0 = 62.8754
GRID = (64, 48, 32)


def dir_grid(ft, fp):
    """(θ,φ) 网格 → (rhat, theta_hat, phi_hat)，与部署 nffft_lite 调用一致。"""
    T, P = np.meshgrid(np.deg2rad(ft), np.deg2rad(fp), indexing="ij")
    T, P = T.ravel(), P.ravel()
    st, ct, sp, cp = np.sin(T), np.cos(T), np.sin(P), np.cos(P)
    rhat = np.stack([st * cp, st * sp, ct], axis=1)
    thv = np.stack([ct * cp, ct * sp, -st], axis=1)
    phv = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
    return rhat, thv, phv


def rcs_dirs(pred, idxs, nrm, rsurf, dS, rhat, thv, phv, e0m, chunk=700):
    """pred (12,64,48,32) → 全部观测方向的线性 σ（部署 rcs_directions 的向量化版）。"""
    i0, i1, i2 = idxs[:, 0], idxs[:, 1], idxs[:, 2]
    H_s = (pred[6:9] + 1j * pred[9:12])[:, i0, i1, i2].T        # (Ns,3)
    J = np.cross(nrm, H_s)                                       # n̂×H（PEC: M=0）
    Zero = np.zeros_like(J)
    out = np.empty(len(rhat), np.float64)
    for a in range(0, len(rhat), chunk):
        b = min(a + chunk, len(rhat))
        E_ff = nffft(J, Zero, rsurf, dS, rhat[a:b], BETA0)
        Eth = (E_ff * thv[a:b]).sum(axis=1)
        Eph = (E_ff * phv[a:b]).sum(axis=1)
        out[a:b] = 4.0 * np.pi * (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / e0m ** 2
    return out


def caliber_stats(a_full, d_full, it, ip):
    """三口径：单站 / 方向图锥面（θ_s=θ_i 全 φ）/ 池化。"""
    na = a_full.shape[0]
    ar = np.arange(na)
    mono = a_full[ar, it, ip]
    cone = a_full[ar, it, :].reshape(-1)
    return {"单站": {"n": int(mono.size), "med": float(np.median(mono)),
                     "p90": float(np.percentile(mono, 90)),
                     "bias": float(np.median(d_full[ar, it, ip]))},
            "锥面": {"n": int(cone.size), "med": float(np.median(cone)),
                     "p90": float(np.percentile(cone, 90))},
            "池化": {"n": int(a_full.size), "med": float(np.median(a_full)),
                     "p90": float(np.percentile(a_full, 90)),
                     "bias": float(np.median(d_full))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncase", type=int, default=0, help="只跑前 n 个角（0=全部 468）")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    import onnxruntime as ort
    angles, e0t, khat, beta = M.build_incidence_table()
    na_all = len(angles)
    aidx = np.arange(na_all)
    if args.ncase:
        aidx = aidx[:args.ncase]
    na = len(aidx)

    # ---- 部署包原件 ----
    with h5py.File(GEOM, "r") as f:
        eps = f["eps_field"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    h = float(np.median(np.diff(gx)))
    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz, h)
    nrm = dS / (np.linalg.norm(dS, axis=1, keepdims=True) + 1e-12)
    st = np.load(STATS)
    xm, xs = st["x_inc_mean"].astype(np.float32), st["x_inc_std"].astype(np.float32)
    ym, ys = st["y_mean"].astype(np.float32), st["y_std"].astype(np.float32)

    providers = (["CPUExecutionProvider"] if args.cpu else
                 ["CUDAExecutionProvider", "CPUExecutionProvider"])
    sess = ort.InferenceSession(ONNX, providers=providers)
    print(f"ONNX provider={sess.get_providers()[0]}  表面面元={len(idxs)}  h={h:.5f} m",
          flush=True)

    # ---- 真值/网格（训练数据集） + 与 PO npz 一致性核对 ----
    with h5py.File(M.H5, "r") as f:
        rcs_true = f["rcs"][:].astype(np.float64)
        ft, fp = f["ff_theta"][:].astype(np.float64), f["ff_phi"][:].astype(np.float64)
    nT, nP = len(ft), len(fp)
    sto = rcs_true.reshape(na_all, nT, nP)
    rhat, thv, phv = dir_grid(ft, fp)

    zp = np.load(PO_NPZ)
    chk = {"po_sel_equal": bool(np.allclose(zp["sel"], angles, atol=1e-9)),
           "po_ff_grid_equal": bool(np.allclose(zp["ff_theta"], ft) and np.allclose(zp["ff_phi"], fp)),
           "po_itip_equal": None}
    it = np.rint((angles[:, 0] - ft[0]) / (ft[1] - ft[0])).astype(int)
    ip = np.rint((np.mod(angles[:, 1], 360.0) - fp[0]) / (fp[1] - fp[0])).astype(int)
    chk["po_itip_equal"] = bool(np.array_equal(it[zp["aidx"]], zp["it"])
                               and np.array_equal(ip[zp["aidx"]], zp["ip"]))
    sto_po = zp["sp_full"].astype(np.float64) / 10.0 ** (zp["d_full"].astype(np.float64) / 10.0)
    chk["truth_recon_maxdiff_dB"] = float(np.max(np.abs(
        10 * np.log10(np.maximum(sto[zp["aidx"]], 1e-30))
        - 10 * np.log10(np.maximum(sto_po, 1e-30)))))
    print(f"一致性： {chk}", flush=True)

    # ---- 逐角推理（两套 e0 约定） ----
    Xg, Yg, Zg = np.meshgrid(gx, gy, gz, indexing="ij")
    Xg, Yg, Zg = Xg.astype(np.float64), Yg.astype(np.float64), Zg.astype(np.float64)
    metal = (eps > 1.5).astype(np.float32)
    sp_tab = np.zeros((na, nT * nP), np.float64)
    sp_dep = np.zeros((na, nT * nP), np.float64)
    t0 = time.time()
    for c, i in enumerate(aidx):
        th, ph = np.deg2rad(angles[i, 0]), np.deg2rad(angles[i, 1])
        st_, ct, sp_, cp = np.sin(th), np.cos(th), np.sin(ph), np.cos(ph)
        kh = np.array([-st_ * cp, -st_ * sp_, -ct], np.float64)
        phs = np.exp(-1j * BETA0 * (kh[0] * Xg + kh[1] * Yg + kh[2] * Zg))
        e0_ana = np.array([ct * cp, ct * sp_, -st_], np.float64)
        for tag, e0 in (("tab", e0t[i].astype(np.complex128)), ("dep", e0_ana)):
            ei = e0[None, None, None, :] * phs[..., None]
            x = np.zeros((1, 7, *GRID), np.float32)
            x[0, 0] = metal
            x[0, 1:4] = ei.real.transpose(3, 0, 1, 2)
            x[0, 4:7] = ei.imag.transpose(3, 0, 1, 2)
            x = (x - xm) / xs
            pred = sess.run(None, {"input": x})[0][0]
            pred = pred * ys.reshape(12, 1, 1, 1) + ym.reshape(12, 1, 1, 1)
            sig = rcs_dirs(pred, idxs, nrm, rsurf, dS, rhat, thv, phv, float(np.linalg.norm(e0)))
            (sp_tab if tag == "tab" else sp_dep)[c] = sig
        if c == 0 or (c + 1) % 20 == 0 or c + 1 == na:
            el = time.time() - t0
            print(f"  {c+1}/{na}  {el:.0f}s  ETA {(el/(c+1))*(na-c-1)/60:.1f} min", flush=True)

    sp_tab3 = sp_tab.reshape(na, nT, nP)
    sp_dep3 = sp_dep.reshape(na, nT, nP)
    st3 = sto[aidx]
    d_tab = 10 * np.log10(np.maximum(sp_tab3, 1e-30) / np.maximum(st3, 1e-30))
    d_dep = 10 * np.log10(np.maximum(sp_dep3, 1e-30) / np.maximum(st3, 1e-30))
    a_tab, a_dep = np.abs(d_tab), np.abs(d_dep)
    it_a, ip_a = it[aidx], ip[aidx]

    res = {"n_angle": int(na), "n_dir": int(nT * nP), "wall_s": float(time.time() - t0),
           "surface_facets": int(len(idxs)), "provider": sess.get_providers()[0],
           "consistency": chk,
           "caliber": {"conv_tab（与训练一致）": caliber_stats(a_tab, d_tab, it_a, ip_a),
                       "conv_dep（部署实际所喂）": caliber_stats(a_dep, d_dep, it_a, ip_a)},
           "e0_convention": {
               "tab_vs_dep_ratio": "−1（468/468 角实测，见 AGENTS 台账）",
               "delta_rcs_med_dB": float(np.median(np.abs(d_dep - d_tab))),
               "delta_rcs_p90_dB": float(np.percentile(np.abs(d_dep - d_tab), 90))}}
    print("\n[口径统计]  med|Δ| / P90 (dB)")
    for k, v in res["caliber"].items():
        print(f"  {k}")
        for kk, vv in v.items():
            print(f"    {kk:<4} n={vv['n']:<8} med {vv['med']:6.3f}  P90 {vv['p90']:6.3f}"
                  + (f"  bias {vv['bias']:+.3f}" if "bias" in vv else ""))
    print(f"  两套 e0 约定的 ΔRCS：中位 {res['e0_convention']['delta_rcs_med_dB']:.4f} dB  "
          f"P90 {res['e0_convention']['delta_rcs_p90_dB']:.4f} dB")

    # 自检：conv_tab 池化中位应与 P4 pred 封板值对拍
    p4 = os.path.join(RESULT_DIR, "p4_pred.json")
    if os.path.exists(p4) and na == na_all:
        ref = json.load(open(p4, encoding="utf-8"))
        res["crosscheck_p4_pred"] = {"p4_pooled_med_dB": ref["rcs_dB_err_median"],
                                     "here_conv_tab_pooled_med_dB":
                                         res["caliber"]["conv_tab（与训练一致）"]["池化"]["med"],
                                     "diff_dB": abs(ref["rcs_dB_err_median"]
                                                    - res["caliber"]["conv_tab（与训练一致）"]["池化"]["med"])}
        print(f"  自检 P4 pred 池化 {ref['rcs_dB_err_median']:.3f} vs 本脚本 conv_tab "
              f"{res['crosscheck_p4_pred']['here_conv_tab_pooled_med_dB']:.3f} "
              f"（差 {res['crosscheck_p4_pred']['diff_dB']:.3f} dB）")

    if na == na_all:
        np.savez_compressed(os.path.join(RESULT_DIR, "_diag_fno_fullmap.npz"),
                            aidx=aidx, sel=angles[aidx], d_full=d_dep.astype(np.float32),
                            sp_full=sp_dep3.astype(np.float32),
                            d_full_conv_tab=d_tab.astype(np.float32),
                            sp_full_conv_tab=sp_tab3.astype(np.float32),
                            it=it_a, ip=ip_a, ff_theta=ft, ff_phi=fp)
        json.dump(res, open(os.path.join(RESULT_DIR, "_diag_fno_fullmap.json"), "w",
                            encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
        print("\n已存 results/_diag_fno_fullmap.npz / .json")
    else:
        print(f"\n（--ncase={args.ncase} 快检，未写盘）")


if __name__ == "__main__":
    main()
