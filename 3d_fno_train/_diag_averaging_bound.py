# -*- coding: utf-8 -*-
"""
_diag_averaging_bound.py — P3：误差平均化定律的适用边界（i.i.d. 前提的刻画）
================================================================================
要回答的理论问题（P3）
--------------------------------------------------------------------------------
现行定律（exp_errprop.py / _diag_errprop_exact.py）：

    |ΔRCS|_med ≈ 4.144 · ε / √G          （G = 相干增益）

它的推导**只用了一个假设**：每个面元的相对复误差 δ_s 满足
    (a) 零均值；(b) 等方差 σ²；(c) **互不相关**（i.i.d.）。
真实代理模型的误差场显然不满足 (c)：`interp_augms` 的实测远场误差是理想平均化
预言的 **3.2×**（exp_errprop 记录），而 `full_p3` 是 1.0×。

本脚本把 (c) 参数化并推过其失效点，给出可引用的边界。

一、合成扫描（把相关长度当自变量）
  对 truth 表面场注入**空间相关**复高斯噪声（3D 网格高斯平滑，σ_vox 可控），
  扫描 σ_vox = 0, 0.25, 0.5, 1, 2, 3, 5 个体素（h=0.03125 m，λ=0.0999 m）：
    · K_meas(σ) = median|ΔRCS|_dB / ε                       （完整链路实测，dB/ε）
    · A(σ)      = Σ_d|Σ_s a_s δ_s|² / Σ_d Σ_s |a_s|²
                  其中 a_s = e^{jk r̂·r_s}(p̂·ΔJ_s)dA_s，ΔJ = n̂×(H δ)。
                  **分母用 ΔJ 自己**算非相干和 ⇒ A 就是"相关相对 i.i.d. 的方差放大"。
                  σ=0 时应有 A≈1（脚本自检）。
    · K_pred(σ) = K_meas(0)·√A(σ)                           （一阶预言）

二、真实模型闭环（用量到的相关长度解释 1.0× / 1.2× / 3.2×）
  对 ckpt_full_p3 / ckpt_imp_interp_plain / ckpt_imp_interp_augms 三臂，
  在 interp 留出角上取近场**散射场误差** δ = H_scat_pred − H_scat_true：
    · ε_H（口径同 metrics_* 的 rel_H）、误差场自相关 ρ(d)、ℓ50 / ℓ1e；
    · 该 δ 的一阶放大 A（同一公式，无噪声模型假设）；
    · K_pred = K_meas_synth(σ=0)·√A·ε_H，与实测对照。
  另加一条**口径对照**：同一预测场，J = n̂×H_scat（历史口径，exp_nffft_anglesets 用）
  vs J = n̂×(h_inc+H_scat)（物理口径）谁更接近 rcs_true。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_averaging_bound.py
产出
  results/_diag_averaging_bound.json
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
sys.path.insert(0, BASE)
import fno_f16_3d as M                                              # noqa: E402
from fno_f16_3d_p4_nffft import (surface_parts, direction_grid, sample_surface_field,
                                 pred_to_surface, rcs_from_surface)  # noqa: E402
from _exp_common import build_x_single                              # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi
K_MED_FACTOR = 4.144      # 8.69 · 0.6745 / √2，一阶定律常数


def dBs(x):
    return 10.0 * np.log10(np.maximum(x, 1e-12))


def med_err(a, ref, mask):
    """median |ΔRCS|_dB（只在中 RCS 处统计，掩码同 rcs_stats）。"""
    return float(np.median(np.abs(dBs(a) - dBs(ref))[mask]))


def coherent_gain(Ph, J, dA, th, ph):
    """G_d = (|Σaθ|²+|Σaφ|²)/(Σ|aθ|²+Σ|aφ|²)，两极化求和口径。"""
    N = Ph.T @ (J * dA[:, None])
    E0t = (N * th).sum(axis=1)
    E0p = (N * ph).sum(axis=1)
    St = (np.abs((J @ th.T) * dA[:, None]) ** 2).sum(axis=0)
    Sp = (np.abs((J @ ph.T) * dA[:, None]) ** 2).sum(axis=0)
    return (np.abs(E0t) ** 2 + np.abs(E0p) ** 2) / (St + Sp)


def first_order_A(Ph, Jerr, dA, th, ph):
    """A = Σ_d |Σ_s a_s|² / Σ_d Σ_s |a_s|² ，a 由**误差流** Jerr 构造。
    i.i.d. 误差 ⇒ A ≈ 1；正相关 ⇒ A > 1。分母用 Jerr 自身保证量纲一致。"""
    Wt = (Jerr @ th.T) * dA[:, None]              # (Ns,Ndir)
    Wp = (Jerr @ ph.T) * dA[:, None]
    ut = (Ph * Wt).sum(axis=0)
    up = (Ph * Wp).sum(axis=0)
    num = float((np.abs(ut) ** 2 + np.abs(up) ** 2).sum())
    den = float((np.abs(Wt) ** 2 + np.abs(Wp) ** 2).sum())
    return num / den


def corr_noise_field(shape, sigma_vox, rng):
    """3D 网格白噪声经高斯平滑 ⇒ 相关长度可控的复噪声（未归一化）。"""
    w = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)
    if sigma_vox <= 0:
        return w
    from scipy.ndimage import gaussian_filter
    return (gaussian_filter(w.real, sigma_vox, mode="constant")
            + 1j * gaussian_filter(w.imag, sigma_vox, mode="constant"))


def empirical_rho(rsurf, delta, h=0.03125, npair=200000, seed=1):
    """表面点对距离分箱平均相关（复场 Frobenius 内积）。返回 (d/h, rho)。
    必须剔除 i==j 自对：自对距离恒为 0、相关恒为 1，会把第一个距离箱整体污染。"""
    rng = np.random.default_rng(seed)
    n = rsurf.shape[0]
    i = rng.integers(0, n, npair)
    j = rng.integers(0, n, npair)
    keep = i != j
    i, j = i[keep], j[keep]
    d = np.linalg.norm(rsurf[i] - rsurf[j], axis=1) / h
    ip = (delta[i].conj() * delta[j]).sum(axis=1)
    den = float((np.abs(delta) ** 2).sum() / n)
    rho = np.real(ip) / den
    edges = np.arange(0.5, 25.5, 1.0)
    ds, rs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (d >= a) & (d < b)
        if m.sum() >= 20:
            ds.append(0.5 * (a + b)); rs.append(float(rho[m].mean()))
    return ds, rs


def rho_half_life(ds, rs):
    out = {}
    for tag, thr in (("l50_h", 0.5), ("l1e_h", 1.0 / np.e)):
        v = np.nan
        for d, r in zip(ds, rs):
            if r < thr:
                v = float(d); break
        out[tag] = v
    return out


def load_model(name, device):
    import torch
    ck = torch.load(os.path.join(RESULT_DIR, name), map_location="cpu", weights_only=False)
    cfg = ck["config"]; st = ck["stats"]
    ym = st.get("ym", st.get("y_mean")); ys = st.get("ys", st.get("y_std"))
    xm = st.get("xm", st.get("x_inc_mean")); xs = st.get("xs", st.get("x_inc_std"))
    model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=12).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    return (model,
            np.asarray(ym, np.float32).reshape(12, 1, 1, 1),
            np.asarray(ys, np.float32).reshape(12, 1, 1, 1),
            np.asarray(xm, np.float32).reshape(1, 7, 1, 1, 1),
            np.asarray(xs, np.float32).reshape(1, 7, 1, 1, 1))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--nangles", type=int, default=6)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.5)
    ap.add_argument("--sigmas", default="0,0.25,0.5,1,2,3,5")
    ap.add_argument("--n_holdout", type=int, default=6)
    ap.add_argument("--skip_models", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        rcs_true_all = f["rcs"][:]
    k = float(beta[0]); lam = 2.0 * np.pi / k; h = 0.03125
    print(f"k={k:.4f} rad/m  λ={lam:.5f} m  h={h}  λ/h={lam/h:.4f}  "
          f"λ/(2π)/h={lam/(2*np.pi)/h:.4f}  kh={k*h:.4f} rad")

    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / dA[:, None]
    Ph = np.exp(1j * k * (rsurf @ rhat.T))
    print(f"面元 {len(dA)}  观测方向 {rhat.shape[0]}")
    X3, Y3, Z3 = np.meshgrid(gx, gy, gz, indexing="ij")
    fh = h5py.File(M.H5, "r")

    def inc_H_surface(i):
        phase = np.exp(-1j * beta[i] * (khat[i, 0] * X3 + khat[i, 1] * Y3 + khat[i, 2] * Z3))
        e_inc = e0[i][None, None, None, :] * phase[..., None]
        return sample_surface_field((1.0 / ETA0) * np.cross(
            np.broadcast_to(khat[i], e_inc.shape), e_inc), idxs)

    def scat_H_surface(i):
        return sample_surface_field(fh["H_scat"][i], idxs)

    # ---------- 各角度：真值总场 / J / G / 干净估计 / 掩码 ----------
    idx_list = list(np.linspace(0, len(angles) - 1, args.nangles).astype(int))
    base = []
    for i in idx_list:
        Hs = inc_H_surface(i) + scat_H_surface(i)
        J = np.cross(nvec, Hs)
        base.append({"i": int(i), "Hs": Hs, "J": J, "G_d": coherent_gain(Ph, J, dA, th, ph),
                     "e0mag": float(np.linalg.norm(e0[i]))})
    G_all = np.concatenate([b["G_d"] for b in base])
    G_med = float(np.median(G_all))
    K_iid_theory = K_MED_FACTOR / np.sqrt(G_med)
    print(f"G_med={G_med:.4f} P25={np.percentile(G_all,25):.4f} P90={np.percentile(G_all,90):.4f}"
          f"  ⇒ 一阶理论 K=4.144/√G_med={K_iid_theory:.3f} dB/ε")

    clean, masks = [], []
    for b in base:
        clean.append(rcs_from_surface(np.zeros_like(b["Hs"]), b["Hs"], dS, rsurf, rhat, th,
                                      ph, k, b["e0mag"]).reshape(shape))
        masks.append(rcs_true_all[b["i"]].reshape(shape) > 1e-6)

    res = {"k": k, "h_m": h, "lambda_over_h": float(lam / h),
           "lambda_over_2pi_over_h": float(lam / (2 * np.pi) / h), "kh_rad": float(k * h),
           "G_med": G_med, "G_p25": float(np.percentile(G_all, 25)),
           "G_p90": float(np.percentile(G_all, 90)), "K_iid_theory": K_iid_theory,
           "n_surf": int(len(dA)), "n_dir": int(rhat.shape[0]),
           "angles": idx_list, "eps": args.eps}

    # ================= 一、合成相关长度扫描 =================
    print(f"\n=== 一、合成相关噪声扫描（ε={args.eps}，reps={args.reps}）===")
    NS = eps.shape
    sweep = []
    for sg in [float(s) for s in args.sigmas.split(",")]:
        t0 = time.time()
        med_vals, A_vals, rho_curve = [], [], None
        for r in range(args.reps):
            nz = corr_noise_field(NS, sg, rng)
            zs_all = nz[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
            zs_all = zs_all / np.sqrt(np.mean(np.abs(zs_all) ** 2))
            per_angle = []
            for a, b in enumerate(base):
                Hs_n = b["Hs"] * (1.0 + args.eps * zs_all[:, None])
                rcs_n = rcs_from_surface(np.zeros_like(Hs_n), Hs_n, dS, rsurf, rhat, th, ph,
                                         k, b["e0mag"]).reshape(shape)
                per_angle.append(med_err(rcs_n, clean[a], masks[a]))
                if r == 0:
                    A_vals.append(first_order_A(Ph, np.cross(nvec, b["Hs"]) * zs_all[:, None],
                                                dA, th, ph))
                    if a == 0:
                        ds, rr = empirical_rho(rsurf, np.repeat(zs_all[:, None], 3, axis=1))
                        rho_curve = {"d_over_h": ds, "rho": rr, **rho_half_life(ds, rr)}
            med_vals.append(float(np.median(per_angle)))
        K_meas = float(np.median(med_vals)) / args.eps
        A_agg = float(np.mean(A_vals))
        row = {"sigma_vox": sg, "sigma_m": sg * h, "sigma_over_lambda": sg * h / lam,
               "A_agg": A_agg, "A_sqrt": float(np.sqrt(A_agg)),
               "K_meas_dB_per_eps": K_meas,
               "K_meas_rep_std": float(np.std(med_vals) / args.eps),
               "K_pred_1st": K_meas * float(np.sqrt(A_agg)),
               "corr_length": rho_curve}
        sweep.append(row)
        print(f"  σ={sg:>4} vox（{sg*h*1000:6.2f} mm, {sg*h/lam:5.3f}λ）  A={A_agg:6.3f} "
              f"(√A={np.sqrt(A_agg):5.3f})  K_meas={K_meas:6.3f}  "
              f"K_pred={row['K_pred_1st']:6.3f}  ℓ50={rho_curve['l50_h']} "
              f"ℓ1e={rho_curve['l1e_h']}  ({time.time()-t0:.0f}s)", flush=True)
    res["synthetic_sweep"] = sweep
    K_synth0 = sweep[0]["K_meas_dB_per_eps"]
    res["K_synth_iid_measured"] = K_synth0
    # 自检：σ=0 应 A≈1 且 K_meas 与 _diag_errprop_exact 的 4.501 dB/ε 同量级
    print(f"  [自检] σ=0: A={sweep[0]['A_agg']:.3f}（应≈1）；K_meas={K_synth0:.3f} dB/ε "
          f"（_diag_errprop_exact 4.501）")

    # ================= 二、真实模型闭环 =================
    res["model_closed_loop"] = {}
    if not args.skip_models:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        te = np.where(angles[:, 1] % 20 == 10)[0]
        hold = list(te[np.linspace(0, len(te) - 1, args.n_holdout).astype(int)])
        print(f"\n=== 二、真实模型闭环（interp 留出角 {len(hold)} 个，device={device}）===")
        arms = [("ckpt_full_p3.pt", "P3 full"),
                ("ckpt_imp_interp_plain_s0.pt", "interp_plain"),
                ("ckpt_imp_interp_augms_s0.pt", "interp_augms")]
        tdat = {}
        for i in hold:
            hinc = inc_H_surface(i); hsc = scat_H_surface(i)
            tdat[i] = {"hinc": hinc, "hsc": hsc}
        for fname, tag in arms:
            model, ym, ys, xm, xs = load_model(fname, device)
            eps_list, A_list, rhos, ka, kb = [], [], [], [], []
            for i in hold:
                x = build_x_single(i, eps, e0, khat, beta, gx, gy, gz)
                with torch.no_grad():
                    pred = model(torch.from_numpy((x - xm) / xs).to(device))[0].cpu().numpy()
                pred = pred * ys + ym
                _, H_s = pred_to_surface(pred, idxs)            # 预测散射场
                dlt = H_s - tdat[i]["hsc"]                      # 散射场误差
                eps_list.append(float(np.sqrt((np.abs(dlt) ** 2).sum()
                                              / (np.abs(tdat[i]["hsc"]) ** 2).sum())))
                A_list.append(first_order_A(Ph, np.cross(nvec, dlt), dA, th, ph))
                ds, rr = empirical_rho(rsurf, dlt)
                rhos.append({"d_over_h": ds, "rho": rr, **rho_half_life(ds, rr)})
                e0m = float(np.linalg.norm(e0[i]))
                m = rcs_true_all[i].reshape(shape) > 1e-6
                rt = rcs_true_all[i].reshape(shape)
                ka.append(med_err(rcs_from_surface(np.zeros_like(H_s), H_s, dS, rsurf, rhat,
                                                   th, ph, k, e0m).reshape(shape), rt, m))
                kb.append(med_err(rcs_from_surface(np.zeros_like(H_s), H_s + tdat[i]["hinc"],
                                                   dS, rsurf, rhat, th, ph, k, e0m).reshape(shape),
                                  rt, m))
            eps_H = float(np.mean(eps_list)); A_agg = float(np.mean(A_list))
            l50 = float(np.nanmedian([r["l50_h"] for r in rhos]))
            l1e = float(np.nanmedian([r["l1e_h"] for r in rhos]))
            K_pred = K_synth0 * float(np.sqrt(A_agg)) * eps_H
            res["model_closed_loop"][tag] = {
                "eps_H": eps_H, "A_agg": A_agg, "A_sqrt": float(np.sqrt(A_agg)),
                "l50_h": l50, "l1e_h": l1e, "K_pred_1st_dB": K_pred,
                "K_meas_scat_only_dB": float(np.median(ka)),
                "K_meas_total_field_dB": float(np.median(kb)),
                "ratio_scat_only_over_pred": float(np.median(ka) / K_pred),
                "ratio_total_over_pred": float(np.median(kb) / K_pred),
                "corr_examples": rhos[:2]}
            r = res["model_closed_loop"][tag]
            print(f"  [{tag:<13}] ε_H={eps_H*100:6.2f}%  A={A_agg:6.3f}(√A={r['A_sqrt']:5.3f})  "
                  f"ℓ50={l50:4.1f}h ℓ1e={l1e:4.1f}h  "
                  f"K_meas(仅散射)={r['K_meas_scat_only_dB']:6.3f} "
                  f"K_meas(总场)={r['K_meas_total_field_dB']:6.3f}  "
                  f"K_pred={K_pred:6.3f}  比值(总场)={r['ratio_total_over_pred']:5.3f}"
                  f"  比值(散射)={r['ratio_scat_only_over_pred']:5.3f}")
    fh.close()

    jp = os.path.join(RESULT_DIR, "_diag_averaging_bound.json")
    with open(jp, "w", encoding="utf-8") as fo:
        json.dump(res, fo, indent=2, ensure_ascii=False, default=float)
    print(f"\n已存: {jp}")


if __name__ == "__main__":
    main()
