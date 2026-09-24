# -*- coding: utf-8 -*-
"""
_diag_fno_gate.py — P2：在**部署链路（FNO）**上重跑 Δ-标量化的可行性闸门
================================================================================
回答的问题（理论问题 P2）
--------------------------------------------------------------------------------
Δ = ∠(u_i, u_j) 是**绕入射轴旋转不变**的量；把它当唯一的几何自变量，隐含
"观测方向绕入射轴的方位 ψ 无关"这一假设。单站（Δ=0）与锥面（θ_s=θ_i）下该假设
被几何约束，但一般双站下物理上不成立。
PO 链路上已用 `--mode gate2d` 测过（ψ 分档无增益、θ_i 分档有增益）；本脚本把**同一
套闸门**（同折划分、同特征、同子集；直接复用 exp_resid_model 的口径实现，保证逐位
同源）搬到 FNO 链路上，回答：**部署链路上 Δ-标量化是否仍然成立**。

三种输入集：
  F1 [σ,Δ,θ_i]            —— 现行可用集
  F2 F1+ψ                 —— 检验"绕入射轴方位"是否含信息
  F3 F2+θ_s,φ_s,φ_i       —— 完整几何（上界，检验 Δ 压缩丢失了多少）

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fno_gate.py
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fno_gate.py --chain po   # 复算 PO 对照
产出
  results/_diag_fno_gate.json
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

from exp_resid_model import load_flat, fold_ids, FEATURE_SETS, SUBSET_ORDER   # noqa: E402
from _diag_decision_err import _cv_errors, DELTA_EDGES                        # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
SETS = ["F1 [σ_PO,Δ,θ_i]", "F2 F1+ψ",
        "F3 F2+θ_s,φ_s,φ_i"]


def hgb_cv(F, d, foldf, subs, nfold, nsub, max_iter, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    err = np.full(d.size, np.nan)
    for f in range(nfold):
        tr = np.flatnonzero(foldf != f)
        te = np.flatnonzero(foldf == f)
        if nsub and tr.size > nsub:
            rng = np.random.default_rng(seed + f)
            tr = rng.choice(tr, nsub, replace=False)
        m = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=0.06,
                                          max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1e-3, random_state=seed + f)
        m.fit(F[tr], d[tr])
        err[te] = np.abs(d[te] - m.predict(F[te]))
        print(f"    fold {f} 完成", flush=True)
    return {k: {"n": int(np.isfinite(err[v]).sum()),
                "med": float(np.median(err[v][np.isfinite(err[v])])),
                "p90": float(np.percentile(err[v][np.isfinite(err[v])], 90))}
            for k, v in subs.items()}, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain", choices=["fno", "po", "both"], default="fno")
    ap.add_argument("--nfold", type=int, default=6)
    ap.add_argument("--nsub", type=int, default=150000)
    ap.add_argument("--maxiter", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0,
                    help="单 seed（给出 --seeds 时被忽略，仅为兼容保留）")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="逗号分隔的 seed 列表；跨 seed 取均值 ± 标准差（项目约定）")
    args = ap.parse_args()
    args.seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    paths = {"fno": os.path.join(RESULT_DIR, "_diag_fno_fullmap.npz"),
             "po": os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")}
    todo = ["fno", "po"] if args.chain == "both" else [args.chain]
    out = {"nfold": args.nfold, "nsub": args.nsub, "max_iter": args.maxiter,
           "feature_sets": SETS, "chains": {}}
    for tag in todo:
        X, d, cols, ph, subs, shape = load_flat(paths[tag])
        na, nT, nP = shape
        foldf = fold_ids(ph, nT, nP, args.nfold)
        Y = cols["X"] - d
        print(f"\n===== 链路 {tag.upper()}：{na} 角 × {nT}×{nP}，{args.nfold} 折 ===", flush=True)
        from _diag_decision_err import _th_nodes_from_edges, _cv_errors_thetanodes
        rows = {"1D Δ 表（现行部署）": _cv_errors(cols["X"], Y, cols["sep"], None, None,
                                                  DELTA_EDGES, foldf, args.nfold)}
        rows["2D 表（θ4×Δ6）"] = _cv_errors_thetanodes(
            cols["X"], Y, cols["sep"], cols["thi"],
            _th_nodes_from_edges(np.linspace(25.0, 155.0, 5)), DELTA_EDGES,
            foldf, args.nfold)
        rows["raw（不标定）"] = np.abs(d)
        base = {}
        for k, e in rows.items():
            base[k] = {kk: {"n": int(np.isfinite(e[v]).sum()),
                            "med": float(np.median(e[v][np.isfinite(e[v])])),
                            "p90": float(np.percentile(e[v][np.isfinite(e[v])], 90))}
                       for kk, v in subs.items()}
        hgb = {}
        for name in SETS:
            cols_use = FEATURE_SETS[name]
            F = np.column_stack([cols[c] for c in cols_use]).astype(np.float32)
            t0 = time.time()
            meds = {k: [] for k in SUBSET_ORDER}
            for sd in args.seeds:
                print(f"  -- HGB {name}  seed={sd} --", flush=True)
                r, e = hgb_cv(F, d, foldf, subs, args.nfold, args.nsub, args.maxiter, sd)
                for k in SUBSET_ORDER:
                    meds[k].append(r[k]["med"])
            # 跨 seed 聚合（项目约定：单 seed 数字不写结论）
            hgb[name] = {k: {"n": int(r[k]["n"]), "med": float(np.mean(meds[k])),
                             "med_std": float(np.std(meds[k])),
                             "med_per_seed": [float(x) for x in meds[k]]}
                         for k in SUBSET_ORDER}
            print(f"  {name:<28}" + "".join(f"{hgb[name][k]['med']:>13.3f}" for k in SUBSET_ORDER)
                  + "   ±" + "".join(f"{hgb[name][k]['med_std']:>12.3f}" for k in SUBSET_ORDER)
                  + f"   [{time.time()-t0:.0f}s]", flush=True)
        out["chains"][tag] = {"baselines": base, "hgb": hgb, "seeds": list(args.seeds)}
        print(f"  ── 汇总（中位 |Δ| dB）──")
        for k, r in base.items():
            print(f"  {k:<28}" + "".join(f"{r[kk]['med']:>13.3f}" for kk in SUBSET_ORDER))
        for k, r in hgb.items():
            print(f"  {k+'  [HGB]':<28}" + "".join(f"{r[kk]['med']:>13.3f}" for kk in SUBSET_ORDER))

    json.dump(out, open(os.path.join(RESULT_DIR, "_diag_fno_gate.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print("\n已存 results/_diag_fno_gate.json")


if __name__ == "__main__":
    main()
