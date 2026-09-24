# -*- coding: utf-8 -*-
"""
_diag_angle_gen.py — P6：角度泛化的归因（结构 vs 覆盖）与「内插 / 外推」代价
================================================================================
回答的问题（理论问题 P6）
--------------------------------------------------------------------------------
FNO 的角向泛化到底靠什么？
  · 「结构」假说：解析相位嵌入 e^{-jβ k̂·r} + 因子化谱卷积把入射角变成一个
    光滑的连续参数，网络自然学会角向内插；
  · 「覆盖」假说：训练集 468 角（θ 10°×13、φ 10°×36）足够密，任何查询角都离
    某个训练角 ≤5°，所以只是"就近查表"。

本脚本给出**可判读的量化对照**（全部复用既有落盘数据，不重训模型）：

  A. 覆盖基线（纯查表，零模型）
     用「相邻训练角在 dB 域线性插值」预测目标角的**整幅方向图**，与真值比。
     这就是 10° 间距的查找表在最优插值下能达到的水平 —— 覆盖假说的**上界参照**。
     分三种：φ 内插(±10°)、θ 内插(±10°)、θ 外推(30°/150° 边缘外推 10°)。

  B. 模型实测（匹配留出角，同真值）
     · 全量训练模型（`exp_seeds_ffno_full_rcs.npz`）在 **interp 留出 234 角**上的误差
       （样本内，乐观）；
     · interp 半量训练模型（`exp_seeds_ffno_interp_rcs.npz`）在**同一 234 角**上的误差
       （真留出）。
     两者之差 = 「没见到这些角」的代价（同时叠加了一半点量的代价 ⇒ 是**上界**）。

  判读：
   · 若 B(留出) ≈ A(覆盖基线) ⇒ 泛化由**覆盖**解释，模型没多做任何事；
   · 若 B(留出) ≪ A(覆盖基线) ⇒ 模型在训练角之间做出了比线性插值更好的预测，
     即**结构**在起作用；
   · 若 A 本身就很大（方向图在 10° 尺度上已去相关）⇒ 10° 网格根本不可能靠覆盖
     内插，结构是**必要**条件。

  C. 角向相干尺度
     从 FEKO 真值 (468,37,73)（观测步长 5°）量方向图的角向自相关，给出相关长度，
     与 10° 训练间距对照。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_angle_gen.py
产出
  results/_diag_angle_gen.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M                                              # noqa: E402
from _diag_decision_err import _sep_and_psi                          # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
FULL_RCS = os.path.join(RESULT_DIR, "exp_seeds_ffno_full_rcs.npz")
INTERP_RCS = os.path.join(RESULT_DIR, "exp_seeds_ffno_interp_rcs.npz")
PO_NPZ = os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")


def to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def stats(a_full, sel_ang, it, ip, tag=""):
    """把 (na, nT, nP) 的 |ΔdB| 图按 单站 / 锥面 / 池化 汇总。"""
    a = np.abs(a_full)
    na, nT, nP = a.shape
    ar = np.arange(na)
    mono = a[ar, it, ip]
    cone = a[ar, it, :].reshape(-1)
    return {f"{tag}池化": {"n": int(a.size), "med": float(np.median(a)),
                           "p90": float(np.percentile(a, 90))},
            f"{tag}单站": {"n": int(mono.size), "med": float(np.median(mono)),
                           "p90": float(np.percentile(mono, 90))},
            f"{tag}锥面": {"n": int(cone.size), "med": float(np.median(cone)),
                           "p90": float(np.percentile(cone, 90))}}


def coverage_baseline(TdB, key, targets, mode, step):
    """相邻训练角 dB 域线性（外推则线性外推）预测目标角整幅方向图。

    TdB (468,nT,nP) 真值 dB；key{(θ,φ)->角索引}；targets[(θ,φ)] 目标角列表。
    返回 (pred, truth) 两个 (n_target, nT, nP) 数组。"""
    P = np.empty((len(targets),) + TdB.shape[1:], np.float64)
    Y = np.empty_like(P)
    for i, (th, ph) in enumerate(targets):
        if mode == "phi":
            n1 = key[(th, (ph - step) % 360)]
            n2 = key[(th, (ph + step) % 360)]
            P[i] = 0.5 * (TdB[n1] + TdB[n2])
        elif mode == "theta":
            n1 = key[(th - step, ph)]
            n2 = key[(th + step, ph)]
            P[i] = 0.5 * (TdB[n1] + TdB[n2])
        elif mode == "theta_extrap":
            if th == 30:
                n1, n2 = key[(40, ph)], key[(50, ph)]
            else:
                n1, n2 = key[(140, ph)], key[(130, ph)]
            P[i] = 2.0 * TdB[n1] - TdB[n2]
        else:
            raise ValueError(mode)
        Y[i] = TdB[key[(th, ph)]]
    return P, Y


def autocorr_lag(P, axis, step_deg):
    """沿指定轴的一阶自相关（去均值），返回 (lag_deg, corr) 曲线（lag 到 12 步）。"""
    Q = P - P.mean(axis=axis, keepdims=True)
    denom = (Q * Q).sum(axis=axis)
    out = []
    n = P.shape[axis]
    for L in range(1, min(12, n // 3) + 1):
        a1 = np.take(Q, range(0, n - L), axis=axis)
        a2 = np.take(Q, range(L, n), axis=axis)
        num = (a1 * a2).sum(axis=axis)
        out.append(float(np.median(num / np.maximum(denom, 1e-30)) * 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_diag_angle_gen.json")
    args = ap.parse_args()

    zp = np.load(PO_NPZ)
    sel = zp["sel"].astype(np.float64)          # (468,2) 入射角
    it, ip = zp["it"], zp["ip"]
    ft = zp["ff_theta"].astype(np.float64)
    fp = zp["ff_phi"].astype(np.float64)
    with h5py.File(M.H5, "r") as f:
        rcs_true = f["rcs"][:].astype(np.float64)
    nT, nP = len(ft), len(fp)
    nA = len(sel)
    assert rcs_true.shape == (nA, nT, nP)
    TdB = to_db(rcs_true)
    key = {(int(round(t)), int(round(p)) % 360): i for i, (t, p) in enumerate(sel)}
    print(f"入射角表：{nA} 角，θ 取值 {sorted({int(t) for t, _ in sel})[:3]}..."
          f"，φ 取值步长 {int(round(sel[1,1]-sel[0,1]))}°")

    res = {"n_angle": nA, "n_dir": nT * nP, "theta_step": 10.0, "phi_step": 10.0}

    # ---------- A. 覆盖基线 ----------
    th_all = sorted({int(round(t)) for t, _ in sel})
    ph_all = sorted({int(round(p)) % 360 for _, p in sel})
    phi_targets = [(t, p) for t in th_all for p in ph_all]
    theta_targets = [(t, p) for t in th_all[1:-1] for p in ph_all]
    edge_targets = [(t, p) for t in (th_all[0], th_all[-1]) for p in ph_all]

    cov = {}
    for tag, targets, mode, step in (
            ("φ 内插 ±10°", phi_targets, "phi", 10),
            ("φ 内插 ±20°", phi_targets, "phi", 20),
            ("θ 内插 ±10°", theta_targets, "theta", 10),
            ("θ 外推 10°（边缘层）", edge_targets, "theta_extrap", 10)):
        P, Y = coverage_baseline(TdB, key, targets, mode, step)
        itt = np.array([key[(t, p)] for t, p in targets])
        cov[tag] = {"n_angle": len(targets)}
        cov[tag].update(stats(P - Y, None, it[itt], ip[itt], tag=""))
    res["coverage_baseline"] = cov
    print("\n[A] 覆盖基线（纯查表：相邻训练角 dB 域插值/外推）  med|Δ| dB")
    for k, v in cov.items():
        print(f"  {k:<22} 池化 {v['池化']['med']:6.3f} | 单站 {v['单站']['med']:6.3f}"
              f" | 锥面 {v['锥面']['med']:6.3f}   (n_angle={v['n_angle']}, n_ang_sec={v['单站']['n']})")

    # ---------- B. 模型实测（匹配留出角） ----------
    zf = np.load(FULL_RCS)
    zi = np.load(INTERP_RCS)
    interp_idx = np.where(sel[:, 1] % 20 == 10)[0]         # 奇数 φ 层 = 234 角
    assert np.array_equal(interp_idx, np.arange(1, nA, 2))
    assert np.allclose(zi["rcs_true"], zf["rcs_true"][interp_idx]), "两条 npz 真值不一致"
    arms = {"floor（NFFFT 管线基线）": to_db(zi["floor"]) - to_db(zi["rcs_true"])}
    for s in (0, 1, 2):
        arms[f"全量训练 s{s}（样本内）"] = to_db(zf[f"pred_s{s}"][interp_idx]) - to_db(zf["rcs_true"][interp_idx])
        arms[f"interp 半量训练 s{s}（真留出）"] = to_db(zi[f"pred_s{s}"]) - to_db(zi["rcs_true"])
    res["matched_holdout"] = {}
    print("\n[B] 匹配留出角（234 角，φ≡10 mod 20）  med|Δ| dB")
    for k, v in arms.items():
        res["matched_holdout"][k] = stats(v, None, it[interp_idx], ip[interp_idx])
        st = res["matched_holdout"][k]
        print(f"  {k:<30} 池化 {st['池化']['med']:6.3f} | 单站 {st['单站']['med']:6.3f}"
              f" | 锥面 {st['锥面']['med']:6.3f}")

    def mean3(prefix):
        ks = [k for k in arms if k.startswith(prefix)]
        return {kk: float(np.mean([res["matched_holdout"][k][kk]["med"] for k in ks]))
                for kk in ("池化", "单站", "锥面")}
    res["matched_holdout_summary"] = {"全量训练（样本内）_mean": mean3("全量训练"),
                                     "interp 半量训练（真留出）_mean": mean3("interp 半量"),
                                     "floor": mean3("floor")}
    print("\n  —— 3 seed 均值 ——")
    for k, v in res["matched_holdout_summary"].items():
        print(f"  {k:<34} 池化 {v['池化']:6.3f} | 单站 {v['单站']:6.3f} | 锥面 {v['锥面']:6.3f}")

    # 全量 468 池化自检（应 ≈ p4_pred 4.045 / exp_seeds_ffno_full 4.02）
    full_all = float(np.mean([np.median(np.abs(to_db(zf[f"pred_s{s}"]) - TdB)) for s in (0, 1, 2)]))
    res["selfcheck_full468_pooled_med"] = full_all
    print(f"\n  [自检] 全量训练模型在全部 468 角的池化中位 = {full_all:.3f} dB"
          f"（封板 4.045 / exp_seeds 4.025）")

    # ---------- C. 角向相干尺度 ----------
    # 沿观测 θ_s 与 φ_s 的一阶自相关（去均值后求中位），lag 以真值网格步长为单位
    ac_t = autocorr_lag(rcs_true, axis=1, step_deg=float(ft[1] - ft[0]))
    ac_p = autocorr_lag(rcs_true, axis=2, step_deg=float(fp[1] - fp[0]))
    # 相邻 5° 的单步变化幅度（方向图粗糙度）
    step_t = np.abs(np.diff(to_db(rcs_true), axis=1))
    step_p = np.abs(np.diff(to_db(rcs_true), axis=2))
    corr_len = {}
    for name, ac, ds in (("theta_s", ac_t, float(ft[1] - ft[0])),
                         ("phi_s", ac_p, float(fp[1] - fp[0]))):
        arr = np.asarray(ac)
        k = int(np.argmax(arr < 0.5)) if (arr < 0.5).any() else len(arr) - 1
        corr_len[name] = {"step_deg": ds, "corr_per_lag": [float(x) for x in arr],
                          "lag_half_deg": float((k + 1) * ds)}
    res["angular_coherence"] = {
        "corr": corr_len,
        "roughness_1step_deg": {
            "theta_s": {"step_deg": float(ft[1] - ft[0]),
                        "med_dB": float(np.median(step_t)),
                        "p90_dB": float(np.percentile(step_t, 90))},
            "phi_s": {"step_deg": float(fp[1] - fp[0]),
                      "med_dB": float(np.median(step_p)),
                      "p90_dB": float(np.percentile(step_p, 90))}}}
    print("\n[C] 角向相干尺度（FEKO 真值方向图）")
    for name, v in corr_len.items():
        print(f"  {name}: 步长 {v['step_deg']:.0f}°，自相关降到 0.5 的滞后 ≈ {v['lag_half_deg']:.0f}°")
    for name, v in res["angular_coherence"]["roughness_1step_deg"].items():
        print(f"  {name} 单步（{v['step_deg']:.0f}°）变化中位 {v['med_dB']:.3f} dB / P90 {v['p90_dB']:.3f} dB")

    json.dump(res, open(os.path.join(RESULT_DIR, args.out), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 results/{args.out}")


if __name__ == "__main__":
    main()
