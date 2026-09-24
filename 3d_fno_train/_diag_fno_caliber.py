# -*- coding: utf-8 -*-
"""
_diag_fno_caliber.py — P1：FNO 口径对齐 + 标定/残差层的跨链路迁移检验
================================================================================
回答的问题（理论问题 P1）
--------------------------------------------------------------------------------
`calib_affine.json`（1D Δ 表）、`calib_affine_2d.json`（2D 表）、`resid_mlp.onnx`
的系数全部在**解析 PO 链路**的误差图上拟合，而部署链路是 **FNO→NFFFT**。
本脚本把同一条后处理流水线分别施加到 PO 与 FNO 的完整误差图上，按**三条部署口径**
（单站 / 方向图锥面 / 池化）报告：

  · raw（不标定）
  · PO 拟合 1D Δ 表（= 现行部署）
  · PO 拟合 2D 表（θ4×Δ6）
  · PO 训练的残差 MLP（特征 X 取 链自身 σ，即部署真实用法）
  · 残差 MLP（特征 X 取 σ_PO，即其训练分布内用法 → 隔离"链路"与"分布"两个因素）

并给出**FNO 自拟合上界**（同折留角 CV 重拟 1D/2D 表），用于区分：
  「跨链路借用不成立」 vs 「表本身形式不够」。

用法
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_fno_caliber.py
产出
  results/_diag_fno_caliber.json
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import json
import argparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(os.path.dirname(BASE), "deploy_rcs_server")
sys.path.insert(0, BASE)
sys.path.insert(0, DEPLOY)

from _diag_decision_err import (_sep_and_psi, DELTA_EDGES, _cv_errors,      # noqa: E402
                                _th_nodes_from_edges, _cv_errors_thetanodes,
                                _dec_metrics)

RESULT_DIR = os.path.join(BASE, "results")
PO_NPZ = os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
FN_NPZ = os.path.join(RESULT_DIR, "_diag_fno_fullmap.npz")
SUBS = ["单站(Δ=0, 判决路径)", "锥面环扫(方向图包)", "其余双站 σ_ij", "全方向池化"]


def load_chain(zpath):
    """→ X(dB)、Y(dB)、sep、thi、子集掩码、线性 σ 对（供判决级指标）。"""
    z = np.load(zpath)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft = z["ff_theta"].astype(np.float64)
    fp = z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape
    X = 10.0 * np.log10(np.maximum(sp, 1e-30))
    Y = X - dfull
    sep, _psi = _sep_and_psi(sel, ft, fp)

    def flat(v):
        return np.ascontiguousarray(np.broadcast_to(v, X.shape)).reshape(-1)

    tj = np.broadcast_to(ft[None, :, None], (na, nT, nP))
    ti = np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))
    m_mono = flat(sep < 1e-6).astype(bool)
    m_cone = flat((np.abs(tj - ti) < 1e-9) & (sep >= 1e-6)).astype(bool)
    subs = {SUBS[0]: m_mono, SUBS[1]: m_cone,
            SUBS[2]: ~m_mono & ~m_cone, SUBS[3]: np.ones(X.size, bool)}
    return {"X": X.reshape(-1), "Y": Y.reshape(-1), "sep": flat(sep),
            "thi": flat(ti), "subs": subs, "sp_lin": sp.reshape(-1),
            "st_lin": (10 ** (Y / 10.0)).reshape(-1), "shape": (na, nT, nP)}


def tab1d(path):
    d = json.load(open(path, encoding="utf-8"))
    return np.asarray(d["delta_deg"], np.float64), np.asarray(d["a"], np.float64), \
        np.asarray(d["b"], np.float64)


def apply1d(X, sep, tab):
    dn, a, b = tab
    return np.interp(sep, dn, a) + np.interp(sep, dn, b) * X


def apply2d(X, sep, thi, path):
    d = json.load(open(path, encoding="utf-8"))
    tn = np.asarray(d["theta_nodes_deg"], np.float64)
    dn = np.asarray(d["delta_nodes_deg"], np.float64)
    A = np.asarray(d["a"], np.float64)
    B = np.asarray(d["b"], np.float64)

    def bilinear(M, t, s):
        t = np.clip(t, tn[0], tn[-1])
        s = np.clip(s, dn[0], dn[-1])
        it = np.clip(np.searchsorted(tn, t) - 1, 0, len(tn) - 2)
        isd = np.clip(np.searchsorted(dn, s) - 1, 0, len(dn) - 2)
        wt = (t - tn[it]) / (tn[it + 1] - tn[it])
        ws = (s - dn[isd]) / (dn[isd + 1] - dn[isd])
        return ((1 - wt) * (1 - ws) * M[it, isd] + wt * (1 - ws) * M[it + 1, isd]
                + (1 - wt) * ws * M[it, isd + 1] + wt * ws * M[it + 1, isd + 1])

    return bilinear(A, thi, sep) + bilinear(B, thi, sep) * X


def summarize(Y, est, subs):
    out = {}
    for k, m in subs.items():
        e = np.abs(est[m] - Y[m])
        e = e[np.isfinite(e)]
        b = (est[m] - Y[m])
        b = b[np.isfinite(b)]
        out[k] = {"n": int(e.size), "med": float(np.median(e)),
                  "p90": float(np.percentile(e, 90)), "bias": float(np.median(b))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nfold", type=int, default=6)
    args = ap.parse_args()
    import onnxruntime as ort

    tabpo = tab1d(os.path.join(RESULT_DIR, "calib_affine.json"))
    tab2d = os.path.join(RESULT_DIR, "calib_affine_2d.json")
    resid_onnx = os.path.join(RESULT_DIR, "resid_mlp.onnx")
    resid_stats = np.load(os.path.join(RESULT_DIR, "resid_mlp_stats.npz"))
    feat_names = [str(s) for s in resid_stats["feat_names"]]
    sess = ort.InferenceSession(resid_onnx, providers=["CPUExecutionProvider"])
    print(f"[残差模型] 特征 {feat_names}，入参 {sess.get_inputs()[0].name}")

    Y_po = None
    chains = {}
    for tag, zp in (("PO", PO_NPZ), ("FNO", FN_NPZ)):
        C = load_chain(zp)
        C["tab_FNO_fit"] = None
        chains[tag] = C
        if tag == "PO":
            Y_po = C["Y"]
    # 两条链的真值由 (sp_full, d_full) 各自重建（均为 float32 落盘）⇒ 允许 ~1e-6 dB 量级
    # 的舍入差；实测 max |ΔY| = 2.15e-06 dB（正是 float32 单精度极限的必然结果）。
    y_fno = chains["FNO"]["Y"]
    y_gap = float(np.max(np.abs(Y_po - y_fno)))
    assert y_gap < 1e-4, f"两条链的真值不一致：max|ΔY|={y_gap:.3e} dB"
    print(f"[真值一致性] max|ΔY(PO−FNO)| = {y_gap:.3e} dB（float32 落盘舍入）")

    # 同折留角 CV：FNO 自拟合上界
    c = chains["FNO"]
    na, nT, nP = c["shape"]
    ph = np.load(FN_NPZ)["sel"][:, 1]
    fold = np.minimum((np.mod(ph, 360.0) // (360.0 / args.nfold)).astype(int), args.nfold - 1)
    foldf = np.repeat(fold, nT * nP)
    e1d = _cv_errors(c["X"], c["Y"], c["sep"], None, None, DELTA_EDGES, foldf, args.nfold)
    nodes = _th_nodes_from_edges(np.linspace(25.0, 155.0, 5))
    e2d = _cv_errors_thetanodes(c["X"], c["Y"], c["sep"], c["thi"], nodes,
                                DELTA_EDGES, foldf, args.nfold)
    cv = {}
    for k, e in (("1D Δ 表（FNO 自拟合 CV）", e1d), ("2D 表（FNO 自拟合 CV）", e2d)):
        cv[k] = {kk: {"n": int(np.isfinite(e[m]).sum()),
                      "med": float(np.median(e[m][np.isfinite(e[m])])),
                      "p90": float(np.percentile(e[m][np.isfinite(e[m])], 90))}
                 for kk, m in c["subs"].items()}

    # 残差 MLP 特征矩阵（按 stats 里的名字顺序构造）
    def resid_est(X, sep, thi):
        cols = {"X": X, "sep": sep, "thi": thi}
        F = np.column_stack([cols[n] for n in feat_names]).astype(np.float32)
        d = sess.run(None, {"feat": F})[0].reshape(-1).astype(np.float64)
        return X - d

    out = {"feature_names": feat_names, "nfold": args.nfold, "chains": {}}
    for tag, C in chains.items():
        X, Y, sep, thi = C["X"], C["Y"], C["sep"], C["thi"]
        est = {
            "raw（不标定）": X,
            "1D Δ 表（PO 拟合，现行部署）": apply1d(X, sep, tabpo),
            "2D 表（PO 拟合）": apply2d(X, sep, thi, tab2d),
            "残差 MLP（X=本链 σ）": resid_est(X, sep, thi),
        }
        if tag != "PO":
            est["残差 MLP（X=σ_PO，训练分布内）"] = resid_est(
                chains["PO"]["X"], chains["PO"]["sep"], chains["PO"]["thi"])
        rows = {k: summarize(Y, v, C["subs"]) for k, v in est.items()}
        mono = C["subs"][SUBS[0]]
        dec = _dec_metrics(C["sp_lin"][mono], C["st_lin"][mono])
        out["chains"][tag] = {"rows": rows, "decision_mono": dec}
        print(f"\n===== 链路 {tag} =====")
        print(f"  {'方案':<34}" + "".join(f"{k[:4]:>19}" for k in SUBS))
        for k, r in rows.items():
            print(f"  {k:<34}" + "".join(
                f"{r[kk]['med']:>13.3f}/{r[kk]['p90']:<5.1f}" for kk in SUBS))
        print(f"  判决级（单站 468 角）：R_det 相对误差中位 "
              f"{dec['r_det_rel_abs_median']*100:.1f}%  四态不一致 "
              f"{dec['mismatch_frac']*100:.1f}%")
    out["cv_FNO_self_fit"] = cv
    print("\n===== FNO 自拟合上界（同折留角 CV，按入射 φ 分块）=====")
    for k, r in cv.items():
        print(f"  {k:<34}" + "".join(f"{r[kk]['med']:>13.3f}/{r[kk]['p90']:<5.1f}" for kk in SUBS))

    json.dump(out, open(os.path.join(RESULT_DIR, "_diag_fno_caliber.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print("\n已存 results/_diag_fno_caliber.json")


if __name__ == "__main__":
    main()
