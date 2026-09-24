# -*- coding: utf-8 -*-
"""
_diag_fusion_gate.py — B1：PO×FNO 融合后处理的可学性闸门（HGB，CPU）
================================================================================
问题（用户批准的 A 系列清单 B1）
  两条链各自的定位（来自 item 6）：
      PO 链   池化 3.161 / 锥面 2.296 / 极优在中远距 —— 但单站判决路径 5.889
      FNO 链  池化 4.131 / 锥面 2.793 —— 但单站判决路径 **4.306（比 PO 好 1.58）**
  ⇒ 两条链**在不同子集上各占优**，这是融合的真实动机（不是"取长补短的修辞"）。
  本闸门回答：**把两条链的 σ 一起喂给一个后处理模型，能不能稳定超过两条单链 +
  现行 1D Δ 表？** 只有答案是"能、且超过无学习基线"，才值得做 T1（MLP 训练 + ONNX）。

数据源（唯一口径，不再重跑模型）
  results/_diag_po_fullmap.npz   PO 链：sp_full（σ 估计）/ d_full（= dB估计 − dB真值）
  results/_diag_fno_fullmap.npz  FNO 链：同结构
  ⇒ 真值 Y = X_po − d_po 应逐位等于 X_fno − d_fno（**先做口径对拍再谈收益**，
    本项目已有两次口径致伤史：item 9 的 12.3 dB、item 6 的 e0 符号 2.4 dB）。

无学习基线（判据的锚）
  raw_PO / raw_FNO           不标定
  1D Δ 表（PO）/（FNO）      现行部署形态，留折 CV
  oracle min(|d_po|,|d_fno|) **乐观上界**（逐方向选对，部署不可得）
  最优固定混合               α·X_po+(1−α)·X_fno，α 在训练折上选（留折 CV，可部署）

HGB 臂（同一套折 / 子集 / 目标，全部留折 CV + 3 seed 聚合）
  G0 [σ_PO]                              现行单标量上界（item 5 的 F0）
  G1 [σ_PO,Δ,θ_i]                        现行 F1（PO 链，item 5 主口径）
  G2 [σ_FNO,Δ,θ_i]                       F1 换成 FNO 链
  G3 [σ_PO,σ_FNO,Δσ,Δ,θ_i]               ★ **融合主问**
  G4 G3+ψ                                融合 + 绕入射轴方位
  G5 G3+ψ+θ_s,φ_s,φ_i                    融合 + 完整几何（上界）

判据（预注册）
  · G3 相对 **G1（PO 链 F1）** 与 **最优固定混合** 都取得 ≥0.3 dB 改善，且四条子集
    无一退化 ⇒ **融合成立** ⇒ 进 T1（MLP + ONNX）。
  · G3 与 G1/固定混合 打平 ⇒ **融合不成立**：FNO 链在单站上的优势已被 1D Δ 表/残差
    吸收，维持「PO 链 + 残差模型」单一链路，停止 T1（省掉一次训练与一次部署改动）。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fusion_gate.py
产出
  results/_diag_fusion_gate.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import time
import argparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from exp_resid_model import load_flat, fold_ids, SUBSET_ORDER          # noqa: E402
from _diag_decision_err import _cv_errors, DELTA_EDGES                 # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")

FEATURE_SETS = {
    "G0 [σ_PO]": ["Xp"],
    "G1 [σ_PO,Δ,θ_i]": ["Xp", "sep", "thi"],
    "G2 [σ_FNO,Δ,θ_i]": ["Xf", "sep", "thi"],
    "G3 [σ_PO,σ_FNO,Δσ,Δ,θ_i]": ["Xp", "Xf", "dX", "sep", "thi"],
    "G4 G3+ψ": ["Xp", "Xf", "dX", "sep", "thi", "psi"],
    "G5 G3+完整几何": ["Xp", "Xf", "dX", "sep", "thi", "psi",
                       "thj", "sinpj", "cospj", "sinpi", "cospi"],
}


def _stats(e, subs):
    return {k: {"n": int(np.isfinite(e[v]).sum()),
                "med": float(np.median(e[v][np.isfinite(e[v])])) if np.isfinite(e[v]).any()
                else float("nan")} for k, v in subs.items()}


def hgb_cv(F, Y, foldf, subs, nfold, nsub, max_iter, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    err = np.full(Y.size, np.nan)
    for f in range(nfold):
        tr = np.flatnonzero(foldf != f)
        te = np.flatnonzero(foldf == f)
        if nsub and tr.size > nsub:
            rng = np.random.default_rng(seed + f)
            tr = rng.choice(tr, nsub, replace=False)
        m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1e-3, random_state=seed + f)
        m.fit(F[tr], Y[tr])
        err[te] = np.abs(Y[te] - m.predict(F[te]))
        print(f"    fold {f} 完成", flush=True)
    return _stats(err, subs), err


def mix_cv(Xp, Xf, Y, foldf, subs, nfold, nalpha=101):
    """最优固定混合 α·X_po+(1−α)·X_fno：α 在**训练折**上按 |Δ| 中位最小化选，
    作用于测试折（真·可部署口径）。返回逐样本误差。"""
    alphas = np.linspace(0.0, 1.0, nalpha)
    e = np.full(Y.size, np.nan)
    for f in range(nfold):
        tr, te = foldf != f, foldf == f
        best, bestv = 0.5, np.inf
        for a in alphas:
            v = np.median(np.abs(a * Xp[tr] + (1 - a) * Xf[tr] - Y[tr]))
            if v < bestv:
                best, bestv = float(a), float(v)
        e[te] = np.abs(best * Xp[te] + (1 - best) * Xf[te] - Y[te])
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfold", type=int, default=6)
    ap.add_argument("--nsub", type=int, default=150000)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    p_po = os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    p_fno = os.path.join(RESULT_DIR, "_diag_fno_fullmap.npz")

    # ---------- 口径对拍（先做，再谈收益）----------
    zpo, zfno = np.load(p_po), np.load(p_fno)
    for k in ("aidx", "it", "ip", "ff_theta", "ff_phi"):
        assert np.array_equal(zpo[k], zfno[k]), f"两链 {k} 不一致 ⇒ 口径不可比，停止"
    assert np.array_equal(zpo["sel"], zfno["sel"]), "两链 sel 不一致 ⇒ 口径不可比，停止"

    Xp_raw, d_po, cols, ph, subs, shape = load_flat(p_po)
    na, nT, nP = shape
    Xpo = 10.0 * np.log10(np.maximum(zpo["sp_full"].astype(np.float64), 1e-30))
    Xfno = 10.0 * np.log10(np.maximum(zfno["sp_full"].astype(np.float64), 1e-30))
    Y_po = (Xpo - zpo["d_full"].astype(np.float64)).reshape(-1)
    Y_fno = (Xfno - zfno["d_full"].astype(np.float64)).reshape(-1)
    dv = float(np.median(np.abs(Y_po - Y_fno)))
    dmax = float(np.abs(Y_po - Y_fno).max())
    print(f"[口径对拍] 两链反推的真值 Y 差：中位 {dv:.3e} dB，最大 {dmax:.3e} dB")
    assert dmax < 1e-3, f"两链真值对不上（最大 {dmax:.3e} dB）⇒ 口径不一致，停止"
    print("           ⇒ 通过：两链指向同一 FEKO 真值，可逐位比较")
    assert np.allclose(Xp_raw, Xpo), "load_flat 的 X 与 sp_full 不同源"

    Xp = Xpo.reshape(-1)
    Xf = Xfno.reshape(-1)
    Y = Y_po
    assert np.allclose(cols["X"], Xp), "load_flat 的特征列 X 与展平后的 σ_PO 不同源"
    d_po_f = zpo["d_full"].astype(np.float64).reshape(-1)
    d_fno_f = zfno["d_full"].astype(np.float64).reshape(-1)
    cols = dict(cols)
    cols["Xp"], cols["Xf"] = Xp, Xf
    cols["dX"] = Xp - Xf
    foldf = fold_ids(ph, nT, nP, args.nfold)

    out = {"nfold": args.nfold, "nsub": args.nsub, "max_iter": args.maxiter,
           "seeds": seeds, "n_samples": int(Y.size),
           "caliber_check": {"med": dv, "max": dmax},
           "baselines": {}, "hgb": {}}

    print(f"\n===== B1 融合闸门：{na} 角 × {nT}×{nP} = {Y.size} 样本，"
          f"{args.nfold} 折（按入射 φ 分块，两链同折）=====")
    print(f"  {'口径':<30}" + "".join(f"{k[:12]:>14}" for k in SUBSET_ORDER))

    # ---------- 无学习基线 ----------
    base_rows = {
        "raw_PO（不标定）": np.abs(d_po_f),
        "raw_FNO（不标定）": np.abs(d_fno_f),
        "1D Δ 表（PO，现行部署）": _cv_errors(Xp, Y, cols["sep"], None, None,
                                              DELTA_EDGES, foldf, args.nfold),
        "1D Δ 表（FNO）": _cv_errors(Xf, Y, cols["sep"], None, None,
                                     DELTA_EDGES, foldf, args.nfold),
        "最优固定混合（α 留折选）": mix_cv(Xp, Xf, Y, foldf, subs, args.nfold),
        "oracle min(|d_PO|,|d_FNO|)": np.minimum(np.abs(d_po_f), np.abs(d_fno_f)),
    }
    for k, e in base_rows.items():
        st = _stats(e, subs)
        out["baselines"][k] = st
        tag = k + ("  [乐观上界]" if "oracle" in k else "")
        print(f"  {tag:<30}" + "".join(f"{st[s]['med']:>14.3f}" for s in SUBSET_ORDER))

    # ---------- HGB 臂 ----------
    for name, use in FEATURE_SETS.items():
        F = np.column_stack([cols[c] for c in use]).astype(np.float32)
        t0 = time.time()
        meds = {k: [] for k in SUBSET_ORDER}
        ns = {}
        for sd in seeds:
            print(f"  -- HGB {name}  seed={sd}  ({F.shape[1]} 特征) --", flush=True)
            st, _ = hgb_cv(F, Y, foldf, subs, args.nfold, args.nsub, args.maxiter, sd)
            for k in SUBSET_ORDER:
                meds[k].append(st[k]["med"])
                ns[k] = st[k]["n"]
        out["hgb"][name] = {k: {"n": ns[k], "med": float(np.mean(meds[k])),
                                "med_std": float(np.std(meds[k])),
                                "med_per_seed": [float(x) for x in meds[k]]}
                            for k in SUBSET_ORDER}
        print(f"  {name+'  [HGB]':<30}"
              + "".join(f"{out['hgb'][name][k]['med']:>14.3f}" for k in SUBSET_ORDER)
              + "   ±" + "".join(f"{out['hgb'][name][k]['med_std']:>13.3f}"
                                 for k in SUBSET_ORDER)
              + f"   [{time.time()-t0:.0f}s]", flush=True)

    # ---------- 判据 ----------
    g1 = out["hgb"]["G1 [σ_PO,Δ,θ_i]"]
    g3 = out["hgb"]["G3 [σ_PO,σ_FNO,Δσ,Δ,θ_i]"]
    mix = out["baselines"]["最优固定混合（α 留折选）"]
    print(f"\n  ── 判据（G3 融合 vs G1 现行 / vs 最优固定混合）──")
    verdict = []
    for k in SUBSET_ORDER:
        d1 = g1[k]["med"] - g3[k]["med"]
        dm = mix[k]["med"] - g3[k]["med"]
        verdict.append({"subset": k, "gain_vs_G1": float(d1), "gain_vs_mix": float(dm)})
        print(f"  {k:<24} vs G1 {d1:+7.3f} dB   vs 固定混合 {dm:+7.3f} dB")
    out["verdict_rows"] = verdict
    worst = min(min(r["gain_vs_G1"], r["gain_vs_mix"]) for r in verdict)
    pool = [r for r in verdict if "池化" in r["subset"]][0]
    ok = (pool["gain_vs_G1"] >= 0.3) and (pool["gain_vs_mix"] >= 0.3) and (worst > -0.05)
    out["verdict"] = ("融合成立 ⇒ 进 T1（MLP + ONNX）" if ok
                      else "融合不成立 ⇒ 维持「PO 链 + 残差模型」单链，停止 T1")
    print(f"\n  判决 → {out['verdict']}")
    print(f"    （池化：vs G1 {pool['gain_vs_G1']:+.3f} dB，vs 固定混合 "
          f"{pool['gain_vs_mix']:+.3f} dB；四条子集最差 {worst:+.3f} dB）")

    jp = os.path.join(RESULT_DIR, "_diag_fusion_gate.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")


if __name__ == "__main__":
    main()
