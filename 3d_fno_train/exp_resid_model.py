# -*- coding: utf-8 -*-
"""
exp_resid_model.py — RCS 残差模型训练与离线评估
================================================================================
动机
--------------------------------------------------------------------------------
`_diag_decision_err.py --mode residgate`（item 4）用灵活函数类（HGB）在
**可部署输入空间**上量出残差收益上界：判决路径 2D 表 3.248/3.101 → F3 2.445/2.406
（nf6/nf12），判据 ≥0.3 dB ⇒ 值得训。本脚本把那个**上界**落成**模型**：

  · 训练目标 = 残差 `d = σ_PO_dB − σ_true_dB`（即 `_diag_po_fullmap.npz::d_full`），
    输出 `σ̂_true_dB = σ_PO_dB − d̂`（与 `DeltaCalib` 的 `a + b·σ_PO` 同一个用法位置）；
  · 输入 = 与闸门**逐位同源**的特征集 F0–F3（全部是部署时已算出的量）；
  · 验证 = 与闸门**同一折**的留角 CV（按入射 φ 分块），量出真实收益相对上界的折扣。

为什么要真训一遍而不是直接信闸门：闸门是 **HGB 上界**，其收益来自树模型的分段常数
逼近能力；小 MLP 未必拿得到。本脚本必须如实报告折扣，并给出**模型规模—收益曲线**
（F0 只需一个标量 σ_PO，最易部署；F3 最强但要姿态角全量）。

口径（与全项目一致，勿凭直觉改）
--------------------------------------------------------------------------------
· 数据源：`results/_diag_po_fullmap.npz`（468 角 × 37×73 方向，PO vs FEKO 真值）。
· `X = 10·log10(sp_full)`（PO σ dB）、`Y = X − d_full`（真值 σ dB）、目标 `d_full`。
· 折划分：`folds = min(⌊φ/ (360/nfold)⌋, nfold−1)`（与 `residgate`/`gate2d` 完全相同）。
· 子集（互斥）：单站 Δ=0（判决路径）/ 锥面环扫（θ_j=θ_i）/ 其余双站 σ_ij / 全方向池化。
· 评估量：中位 |Δ| dB 与 P90；误差 = |d − d̂| = |Y − σ̂|（两者恒等）。

用法
--------------------------------------------------------------------------------
  & "F:/miniconda3/envs/isaac311/python.exe" exp_resid_model.py                  # 全量（F0–F3, nf6）
  & "F:/miniconda3/envs/isaac311/python.exe" exp_resid_model.py --nfold 12
  & "F:/miniconda3/envs/isaac311/python.exe" exp_resid_model.py --sets F0,F1 --epochs 60
  & "F:/miniconda3/envs/isaac311/python.exe" exp_resid_model.py --no-onnx        # 只做 CV

产出
--------------------------------------------------------------------------------
  results/exp_resid_model.json          全部数字（含逐折明细）
  results/exp_resid_model_nf{nfold}.log 本脚本 stdout（调用方重定向）
  results/resid_mlp.onnx                全量训练的最终模型（--no-onnx 时跳过）
  results/resid_mlp_stats.npz           特征名 + 标准化参数 + 目标标准化参数
"""
import os
import sys
import json
import time
import argparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# 复用闸门/标定表的**同源**口径实现（折划分、Δ/ψ、1D/2D 表 CV），保证逐位可比
from _diag_decision_err import (_sep_and_psi, DELTA_EDGES, RESULT_DIR,   # noqa: E402
                                _cv_errors, _th_nodes_from_edges, _cv_errors_thetanodes)

import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402

# 特征集：键 = 报告名，值 = 特征列名（顺序即网络输入顺序）
FEATURE_SETS = {
    "F0 [σ_PO]": ["X"],
    "F1 [σ_PO,Δ,θ_i]": ["X", "sep", "thi"],
    "F2 F1+ψ": ["X", "sep", "thi", "psi"],
    "F3 F2+θ_s,φ_s,φ_i": ["X", "sep", "thi", "psi",
                          "thj", "sinpj", "cospj", "sinpi", "cospi"],
}
SUBSET_ORDER = ["单站(Δ=0, 判决路径)", "锥面环扫(方向图包)", "其余双站 σ_ij", "全方向池化"]


# ============================================================
# 一、数据与特征
# ============================================================

def load_flat(zpath=None):
    """读 fullmap → 展平后的 (X, d, 特征列字典, 折号, 子集掩码)。"""
    zp = zpath or os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    z = np.load(zp)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft, fp = z["ff_theta"].astype(np.float64), z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape

    X = 10.0 * np.log10(np.maximum(sp, 1e-30))
    sep, psi = _sep_and_psi(sel, ft, fp)

    def flat(v):
        return np.ascontiguousarray(np.broadcast_to(v, X.shape)).reshape(-1)

    d2r = np.deg2rad
    cols = {
        "X": X.reshape(-1),
        "sep": flat(sep),
        "psi": flat(psi),
        "thi": flat(np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))),
        "thj": flat(np.broadcast_to(ft[None, :, None], (na, nT, nP))),
        "sinpj": flat(np.broadcast_to(np.sin(d2r(fp))[None, None, :], (na, nT, nP))),
        "cospj": flat(np.broadcast_to(np.cos(d2r(fp))[None, None, :], (na, nT, nP))),
        "sinpi": flat(np.broadcast_to(np.sin(d2r(sel[:, 1]))[:, None, None], (na, nT, nP))),
        "cospi": flat(np.broadcast_to(np.cos(d2r(sel[:, 1]))[:, None, None], (na, nT, nP))),
    }
    tj = np.broadcast_to(ft[None, :, None], (na, nT, nP))
    ti = np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))
    m_mono = sep < 1e-6
    m_cone = (np.abs(tj - ti) < 1e-9) & ~m_mono
    subs = {
        SUBSET_ORDER[0]: flat(m_mono).astype(bool),
        SUBSET_ORDER[1]: flat(m_cone).astype(bool),
        SUBSET_ORDER[2]: flat(~m_mono & ~m_cone).astype(bool),
        SUBSET_ORDER[3]: np.ones(X.size, bool),
    }
    ph = np.mod(sel[:, 1], 360.0)
    return X, dfull.reshape(-1), cols, ph, subs, (na, nT, nP)


def fold_ids(ph, nT, nP, nfold):
    """按入射 φ 分块（与 residgate/gate2d 完全相同的折定义）。"""
    f = np.minimum((ph // (360.0 / nfold)).astype(int), nfold - 1)
    return np.repeat(f, nT * nP)


# ============================================================
# 二、模型与训练
# ============================================================

class MLP(nn.Module):
    """小 MLP（SiLU），输出 1 维标准化残差。默认 2×128 隐层 ≈ 18k 参数。"""

    def __init__(self, nin, hidden=128, nlayer=2):
        super().__init__()
        layers, d = [], nin
        for _ in range(nlayer):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Ftr, dtr, Fte, cfg, seed):
    """在训练折上训 MLP，返回 (测试折预测 d̂, 标准化参数, 参数量)。"""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fm, fs = Ftr.mean(0), Ftr.std(0) + 1e-8
    tm, ts = dtr.mean(), dtr.std() + 1e-8
    xtr = torch.tensor((Ftr - fm) / fs, dtype=torch.float32, device=dev)
    ytr = torch.tensor((dtr - tm) / ts, dtype=torch.float32, device=dev)
    net = MLP(Ftr.shape[1], cfg["hidden"], cfg["nlayer"]).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"],
                                                       eta_min=cfg["lr"] * 0.1)
    lossf = nn.L1Loss()
    n = xtr.shape[0]
    bs = min(cfg["batch"], n)
    steps = int(np.ceil(n / bs))
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, device=dev)
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(net(xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        xte = torch.tensor((Fte - fm) / fs, dtype=torch.float32, device=dev)
        pred = (net(xte).cpu().numpy() * ts + tm).astype(np.float64)
    return pred, {"feat_mean": fm, "feat_std": fs, "tgt_mean": float(tm),
                  "tgt_std": float(ts)}, npar


# ============================================================
# 三、评估
# ============================================================

def stats(mask, err):
    v = err[mask]
    v = v[np.isfinite(v)]
    return {"n": int(v.size), "med_dB": float(np.median(v)),
            "p90_dB": float(np.percentile(v, 90))}


def run_cv(Xf, df, cols, foldf, subs, cfg, feat_sets, nfold):
    """对每个特征集做留角 CV，返回 {名字: {子集: stats} + 逐折明细}。"""
    out = {}
    for name in feat_sets:
        cols_use = FEATURE_SETS[name]
        F = np.column_stack([cols[c] for c in cols_use]).astype(np.float32)
        err = np.full(Xf.size, np.nan)
        pred_all = np.full(Xf.size, np.nan)
        per_fold, t0, npar = [], time.time(), 0
        for f in range(nfold):
            tr = np.flatnonzero(foldf != f)
            te = np.flatnonzero(foldf == f)
            if cfg["nsub"] and tr.size > cfg["nsub"]:
                rng = np.random.default_rng(cfg["seed"] + f)
                tr = rng.choice(tr, cfg["nsub"], replace=False)
            pred, _, npar = train_mlp(F[tr], df[tr], F[te], cfg, seed=cfg["seed"] + f)
            pred_all[te] = pred
            e = np.abs(df[te] - pred)
            err[te] = e
            fm = subs[SUBSET_ORDER[0]][te]
            per_fold.append({"fold": f, "n_test": int(te.size),
                             "mono_n": int(fm.sum()),
                             "mono_med_dB": float(np.median(e[fm])) if fm.any() else None,
                             "pool_med_dB": float(np.median(e))})
            print(f"    fold {f:>2}  单站 {per_fold[-1]['mono_med_dB']:.3f}  "
                  f"池化 {per_fold[-1]['pool_med_dB']:.3f}", flush=True)
        rows = {k: stats(v, err) for k, v in subs.items()}
        mono_meds = [r["mono_med_dB"] for r in per_fold if r["mono_med_dB"] is not None]
        out[name] = {"rows": rows, "per_fold": per_fold,
                     "mono_fold_min": float(np.min(mono_meds)),
                     "mono_fold_max": float(np.max(mono_meds)),
                     "n_params": int(npar), "train_s": float(time.time() - t0)}
        print(f"  {name:<20} " + "  ".join(
            f"{k[:4]} {rows[k]['med_dB']:.3f}" for k in SUBSET_ORDER)
            + f"   [{time.time()-t0:.0f}s, {npar} 参]", flush=True)
    return out


def baselines(Xf, df, cols, foldf, nfold):
    """同口径重算 raw / 1D 表 / 2D 表（与闸门逐位核对用的基线行）。"""
    Yf = cols["X"] - df                       # 真值 σ dB
    e = {}
    e["raw（不标定）"] = np.abs(Yf - cols["X"])
    e["1D Δ 表（现行部署）"] = _cv_errors(cols["X"], Yf, cols["sep"], None, None,
                                          DELTA_EDGES, foldf, nfold)
    nodes = _th_nodes_from_edges(np.linspace(25.0, 155.0, 5))
    e["2D 表（θ4×Δ6，现行最优）"] = _cv_errors_thetanodes(
        cols["X"], Yf, cols["sep"], cols["thi"], nodes, DELTA_EDGES, foldf, nfold)
    return e


# ============================================================
# 四、ONNX 导出与核对
# ============================================================

def export_onnx(F, df, cols_use, cfg, path, stats_path):
    """全量训练 → 导出 ONNX → torch/onnxruntime 数值核对 + 时延。"""
    import onnxruntime as ort
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fm, fs = F.mean(0), F.std(0) + 1e-8
    tm, ts = df.mean(), df.std() + 1e-8
    torch.manual_seed(cfg["seed"])
    xtr = torch.tensor((F - fm) / fs, dtype=torch.float32, device=dev)
    ytr = torch.tensor((df - tm) / ts, dtype=torch.float32, device=dev)
    net = MLP(F.shape[1], cfg["hidden"], cfg["nlayer"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"],
                                                       eta_min=cfg["lr"] * 0.1)
    lossf = nn.L1Loss()
    n, bs = xtr.shape[0], cfg["batch"]
    for ep in range(cfg["epochs"]):
        perm = torch.randperm(n, device=dev)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            lossf(net(xtr[idx]), ytr[idx]).backward()
            opt.step()
        sched.step()
    net.eval()

    # 把标准化搬进图（部署端只喂原始特征）：w = (x−fm)/fs, out = y*ts+tm
    class Deploy(nn.Module):
        def __init__(self, net, fm, fs, tm, ts):
            super().__init__()
            self.net = net
            self.register_buffer("fm", torch.tensor(fm, dtype=torch.float32))
            self.register_buffer("fs", torch.tensor(fs, dtype=torch.float32))
            self.tm, self.ts = float(tm), float(ts)

        def forward(self, x):
            return self.net((x - self.fm) / self.fs) * self.ts + self.tm

    dep = Deploy(net, fm, fs, tm, ts).to(dev).eval()
    dummy = torch.zeros(1, F.shape[1], dtype=torch.float32, device=dev)
    torch.onnx.export(dep, dummy, path, input_names=["feat"], output_names=["resid_dB"],
                      dynamic_axes={"feat": {0: "n"}, "resid_dB": {0: "n"}},
                      opset_version=17, dynamo=False)
    np.savez(stats_path, feat_names=np.array(cols_use), feat_mean=fm, feat_std=fs,
             tgt_mean=np.float64(tm), tgt_std=np.float64(ts),
             epochs=cfg["epochs"], nlayer=cfg["nlayer"], hidden=cfg["hidden"])

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    # 核对口径：torch 侧手动归一化（与原训练口径一致），ONNX 侧喂**原始特征**（图内已含归一化）
    xt = np.ascontiguousarray((F - fm) / fs, dtype=np.float32)
    Fraw = np.ascontiguousarray(F, dtype=np.float32)
    with torch.no_grad():
        y_t = (net(torch.tensor(xt, device=dev)).cpu().numpy() * ts + tm)
    y_o = sess.run(None, {"feat": Fraw})[0].reshape(-1)
    maxdiff = float(np.max(np.abs(y_t - y_o)))

    x1 = np.ascontiguousarray(Fraw[:1])
    for _ in range(50):
        sess.run(None, {"feat": x1})
    t0 = time.time()
    for _ in range(1000):
        sess.run(None, {"feat": x1})
    lat1 = (time.time() - t0) / 1000 * 1e6
    t0 = time.time()
    sess.run(None, {"feat": Fraw})
    lat_all = time.time() - t0
    return {"onnx_path": path, "stats_path": stats_path,
            "torch_vs_ort_max_abs_dB": maxdiff,
            "latency_us_per_call_b1": lat1,
            "latency_s_fullmap": lat_all,
            "n_samples": int(Fraw.shape[0]),
            "onnx_input": "原始特征（图内含标准化）"}


# ============================================================
# 五、主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zpath", default=None, help="默认 results/_diag_po_fullmap.npz")
    ap.add_argument("--nfold", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--nlayer", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--nsub", type=int, default=300000, help="每折训练子采样上限（与闸门同）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sets", default="F0,F1,F2,F3", help="只跑部分特征集，逗号分隔前缀")
    ap.add_argument("--tag", default="", help="产物文件名后缀（多 seed 复跑用，避免互相覆盖）")
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in
           ("epochs", "hidden", "nlayer", "lr", "batch", "nsub", "seed")}

    X, df, cols, ph, subs, shape = load_flat(args.zpath)
    na, nT, nP = shape
    Xf = cols["X"]
    foldf = fold_ids(ph, nT, nP, args.nfold)

    want = [k for k in FEATURE_SETS
            if any(k.startswith(p.strip()) for p in args.sets.split(","))]
    print(f"[残差模型] {na} 角 × {nT}×{nP} 方向 = {Xf.size} 样本，{args.nfold} 折留角 CV"
          f"（按入射 φ 分块），训练子采样 {args.nsub}", flush=True)
    print(f"  训练配置：MLP {args.nlayer}×{args.hidden}  SiLU  L1  Adam lr={args.lr}  "
          f"epochs={args.epochs}  batch={args.batch}  "
          f"device={'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)
    print(f"  特征集：{want}\n", flush=True)

    print("── 基线（同口径重算，应与 residgate 逐位一致）──", flush=True)
    bl = baselines(Xf, df, cols, foldf, args.nfold)
    for k, e in bl.items():
        r = {kk: stats(v, e) for kk, v in subs.items()}
        print(f"  {k:<20} " + "  ".join(f"{kk[:4]} {r[kk]['med_dB']:.3f}"
                                        for kk in SUBSET_ORDER), flush=True)
    bl_rows = {k: {kk: stats(v, e) for kk, v in subs.items()} for k, e in bl.items()}

    print("\n── MLP 留角 CV ──", flush=True)
    models = run_cv(Xf, df, cols, foldf, subs, cfg, want, args.nfold)

    # HGB 上界（闸门产出，直接读，不重算）
    hp = os.path.join(RESULT_DIR, f"_diag_residgate_nf{args.nfold}.json")
    hgb = json.load(open(hp, encoding="utf-8")) if os.path.exists(hp) else None

    print(f"\n  ── 汇总（中位 |Δ| dB，{args.nfold} 折留角 CV）──", flush=True)
    print(f"  {'方案':<34}" + "".join(f"{k:>16}" for k in SUBSET_ORDER), flush=True)
    for k in ["raw（不标定）", "1D Δ 表（现行部署）", "2D 表（θ4×Δ6，现行最优）"]:
        r = bl_rows[k]
        print(f"  {k:<34}" + "".join(f"{r[kk]['med_dB']:>12.3f}/{r[kk]['p90_dB']:<3.0f}"
                                     for kk in SUBSET_ORDER), flush=True)
    if hgb:
        for k in want:
            if k in hgb["rows"]:
                r = hgb["rows"][k]
                print(f"  {k+'  [HGB 上界]':<34}"
                      + "".join(f"{r[kk]['med_dB']:>12.3f}/{r[kk]['p90_dB']:<3.0f}"
                                for kk in SUBSET_ORDER), flush=True)
    for k in want:
        r = models[k]["rows"]
        print(f"  {k+'  [MLP]':<34}"
              + "".join(f"{r[kk]['med_dB']:>12.3f}/{r[kk]['p90_dB']:<3.0f}"
                        for kk in SUBSET_ORDER)
              + f"   {models[k]['n_params']}参 {models[k]['train_s']:.0f}s", flush=True)

    mono = SUBSET_ORDER[0]
    base2d = bl_rows["2D 表（θ4×Δ6，现行最优）"][mono]["med_dB"]
    base1d = bl_rows["1D Δ 表（现行部署）"][mono]["med_dB"]
    summary = {}
    for k in want:
        m = models[k]["rows"][mono]["med_dB"]
        h = hgb["rows"][k][mono]["med_dB"] if (hgb and k in hgb["rows"]) else None
        summary[k] = {"mono_med_dB": m, "mono_p90_dB": models[k]["rows"][mono]["p90_dB"],
                      "gain_vs_2D_dB": base2d - m, "gain_vs_1D_dB": base1d - m,
                      "mono_fold_min": models[k]["mono_fold_min"],
                      "mono_fold_max": models[k]["mono_fold_max"],
                      "n_params": models[k]["n_params"], "train_s": models[k]["train_s"],
                      "hgb_upper_med_dB": h, "hgb_minus_mlp_dB": (m - h) if h else None}
    print(f"\n  判决路径（单站）1D 表 {base1d:.3f} / 2D 表 {base2d:.3f} ⇒", flush=True)
    for k, s in summary.items():
        up = f"，HGB 上界 {s['hgb_upper_med_dB']:.3f}（差 {s['hgb_minus_mlp_dB']:+.3f}）" \
            if s["hgb_upper_med_dB"] else ""
        print(f"    {k:<20} MLP {s['mono_med_dB']:.3f}（vs 2D 表 {s['gain_vs_2D_dB']:+.3f} dB，"
              f"逐折 {s['mono_fold_min']:.3f}~{s['mono_fold_max']:.3f}）{up}", flush=True)

    onnx_info = None
    if not args.no_onnx:
        best = max(want, key=lambda k: summary[k]["gain_vs_2D_dB"])
        cols_use = FEATURE_SETS[best]
        Ffull = np.column_stack([cols[c] for c in cols_use]).astype(np.float32)
        op = os.path.join(RESULT_DIR, "resid_mlp.onnx")
        sp = os.path.join(RESULT_DIR, "resid_mlp_stats.npz")
        print(f"\n── 全量训练 + ONNX 导出（特征集 {best}）──", flush=True)
        onnx_info = export_onnx(Ffull, df, cols_use, cfg, op, sp)
        onnx_info["feature_set"] = best
        print(f"  {op}", flush=True)
        print(f"  torch vs onnxruntime 最大差 {onnx_info['torch_vs_ort_max_abs_dB']:.2e} dB；"
              f"单样本 {onnx_info['latency_us_per_call_b1']:.1f} μs/次，"
              f"全图 {onnx_info['n_samples']} 样本 {onnx_info['latency_s_fullmap']:.3f} s",
              flush=True)

    out = {"n_angle": int(na), "n_dir": int(nT * nP), "nfold": args.nfold,
           "config": cfg, "feature_sets": want,
           "subsets": SUBSET_ORDER,
           "baselines": bl_rows, "models": models, "summary": summary,
           "hgb_upper_source": os.path.basename(hp) if hgb else None,
           "onnx": onnx_info}
    jp = os.path.join(RESULT_DIR, f"exp_resid_model{args.tag}.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}", flush=True)


if __name__ == "__main__":
    main()
