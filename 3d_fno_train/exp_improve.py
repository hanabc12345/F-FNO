# -*- coding: utf-8 -*-
"""
exp_improve.py — 建议3：精度提升实验（角度/极化增广 + 多尺度损失 + ensemble）
==============================================================================
三条改进（可单独/组合消融，验证对 E_rel/H_rel 的提升并传导到远场）：

A. 增广（--aug）：利用电磁散射的线性性，对入射场与散射场同时施加
   ① 全局相位旋转 ψ∈U[0,2π)（E 场乘 e^{jψ}）② 极化基绕入射方向 k̂ 旋转 α∈U[0,2π)
   ——这两类变换在"线性各向同性介质 + 远场（平面波入射）"下都是严格成立的
   （散射场与入射场同变换），无需重跑求解器即可扩充训练流形。

B. 多尺度损失（--ms λ）：loss = rel_mse(E,H) + λ·rel_mse(avgpool4(E,H))
   ——低频分量加权，压"整体结构"误差（低频为能量主载，远场积分对低频敏感）。

C. ensemble（--seeds N）：N 个随机 seed 独立训练，原始量纲预测取平均
   ——降低随机误差分量（而远场 NFFFT 积分对随机误差天然平均，此为其近场对应）。

评估：E/H rel（全局相对L2）+ corr；--nffft 时对角度子集跑 NFFFT 远场 RCS，
对比改进前后 |ΔRCS| 中位/P90（回答"近场改进是否传导到远场"）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_improve.py --aug --epochs 300
  & "F:/miniconda3/envs/isaac311/python.exe" exp_improve.py --ms 0.3 --epochs 300
  & "F:/miniconda3/envs/isaac311/python.exe" exp_improve.py --aug --ms 0.3 --seeds 3 --nffft
产出（results/）：exp_improve_<tag>.json / exp_improve_<tag>.png / ckpt_imp_<tag>_s<k>.pt
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_p3 as P3
import exp_baselines as EB          # 复用 load_p3_cached / make_split
import _exp_common as EC            # 复用 nffft_eval

RESULT_DIR = os.path.join(BASE, "results")


# ============================================================
# 一、增广（严格物理：线性系统 + 相位/极化旋转）
# ============================================================

def rodrigues(k, alpha):
    """绕 k（单位矢量）旋转 alpha 的旋转矩阵 (B,3,3)"""
    k = F.normalize(k, dim=1)
    B = k.shape[0]
    ca = torch.cos(alpha); sa = torch.sin(alpha)
    K = torch.zeros(B, 3, 3, device=k.device)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=k.device)
    return (I[None] * ca[:, None, None] + sa[:, None, None] * K
            + (1 - ca[:, None, None]) * torch.einsum("bi,bj->bij", k, k))


def aug_fields(x_raw, y_raw, khat, alpha, psi):
    """对入射场(E_inc)与散射场(E_scat,H_scat)施加同一旋转+相位。
    x_raw/y_raw 为原始物理量纲 (B,...)；khat (B,3) 入射方向；alpha/psi (B,) 随机角度。"""
    B = x_raw.shape[0]
    R = rodrigues(khat, alpha).to(x_raw.dtype)   # (B,3,3) 实数（保持 float）
    Rc = R.to(torch.complex64)                   # 复数版本用于复场旋转
    ej = torch.exp(1j * psi).view(B, 1, 1, 1, 1)
    xr = x_raw.clone()
    E = torch.complex(xr[:, 1:4], xr[:, 4:7])  # (B,3,...)
    E = torch.einsum("bij,bj...->bi...", Rc, E) * ej
    xr[:, 1:4] = E.real; xr[:, 4:7] = E.imag
    yr = y_raw.clone()
    for rs, ims in (((0, 3), (3, 6)), ((6, 9), (9, 12))):
        F = torch.complex(yr[:, rs[0]:rs[1]], yr[:, ims[0]:ims[1]])
        F = torch.einsum("bij,bj...->bi...", Rc, F) * ej
        yr[:, rs[0]:rs[1]] = F.real; yr[:, ims[0]:ims[1]] = F.imag
    return xr, yr


# ============================================================
# 二、多尺度损失
# ============================================================

def rel_mse_ms(pred, tgt, lam, ksize=4):
    """rel_mse + λ·rel_mse(4×4×4 均值池化)。池化核 4 整除 64/48/32。"""
    l = M.rel_mse_loss(pred, tgt)
    if lam <= 0:
        return l
    p = F.avg_pool3d(pred, ksize, ksize)
    t = F.avg_pool3d(tgt, ksize, ksize)
    return l + lam * M.rel_mse_loss(p, t)


# ============================================================
# 三、训练（单 seed）
# ============================================================

def train_one(tag, seed, x_tr, y_all_c, idx_tr, ym, ys, khat_tr, device,
              epochs, batch, lr, width, modes, aug, ms_lam, verbose=25):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr = torch.from_numpy(np.ascontiguousarray(x_tr[idx_tr])).to(device)
    ytr_raw = torch.from_numpy(np.ascontiguousarray(y_all_c[idx_tr])).to(device)
    khat_t = torch.tensor(khat_tr, dtype=torch.float32).to(device)   # (n,3)
    # 标准化统计（main 中赋值到模块全局；x 的 E_inc 通道增广须在原始域进行）
    xmean_t = torch.tensor(_XMEAN, device=device, dtype=torch.float32)
    xstd_t = torch.tensor(_XSTD, device=device, dtype=torch.float32)
    ystd_t = torch.tensor(ys, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ym, device=device, dtype=torch.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    model = M.FFNO3D(modes=modes, width=width, in_ch=7, out_ch=12).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xtr)
    print(f"\n=== {tag} seed={seed}: aug={aug} ms={ms_lam} width={width} "
          f"params={model.count_params():,} ===")
    t0 = time.time()
    hist = []
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            xb = xtr[b]; yb = ytr_raw[b]
            if aug:
                x_raw = xb * xstd_t + xmean_t                       # 反标准化回原始量纲
                alpha = torch.rand(len(b), device=device) * 2 * np.pi
                psi = torch.rand(len(b), device=device) * 2 * np.pi
                x_raw, yb = aug_fields(x_raw, yb, khat_t[b], alpha, psi)
                xb = (x_raw - xmean_t) / xstd_t                     # 再标准化
            pred_raw = model(xb) * ystd_t + ymean_t                 # 原始量纲
            loss = (rel_mse_ms(pred_raw[:, 0:6], yb[:, 0:6], ms_lam)
                    + rel_mse_ms(pred_raw[:, 6:12], yb[:, 6:12], ms_lam))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % verbose == 0 or ep == epochs:
            relE, relH, _, _, cE, cH = P3.evaluate12(
                model, x_tr[idx_tr], (y_all_c[idx_tr] - ym) / ys, ym, ys,
                device, batch=2)
            hist.append({"ep": ep + 1, "rel_E": relE, "rel_H": relH})
            print(f"  ep{ep+1:4d}/{epochs} loss={ep_loss:.4f} E_rel={relE*100:.2f}% "
                  f"H_rel={relH*100:.2f}% ({time.time()-t0:.0f}s)", flush=True)
    return model, hist


class Ensemble(nn.Module):
    """多个 seed 模型的平均（原始量纲域平均由调用方处理）"""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        return torch.stack([m(x) for m in self.models]).mean(0)


# ============================================================
# 四、主流程
# ============================================================

def main():
    global _XMEAN, _XSTD
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="输出标识（默认由开关自动组合）")
    ap.add_argument("--aug", action="store_true", help="启用相位+极化旋转增广")
    ap.add_argument("--ms", type=float, default=0.0, help="多尺度损失权重 λ（0=关）")
    ap.add_argument("--seeds", type=int, default=1, help="ensemble seed 数")
    ap.add_argument("--split", default="full", choices=["full", "interp"])
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--modes", default="20,16,10")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed0", type=int, default=0, help="ensemble 起始 seed")
    ap.add_argument("--nffft", action="store_true", help="跑 NFFFT 远场 RCS 评估")
    ap.add_argument("--n-angles", type=int, default=40, help="NFFFT 角度子集数")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    x_all, y_all, _, angles, eps = EB.load_p3_cached(use_cache=not args.no_cache)
    y_all_c = P3.clip12(y_all)
    idx_tr, idx_te = EB.make_split(args.split, angles)
    _, xm, xs = M.standardize(x_all, idx_tr)
    y_tr, ym, ys = M.standardize(y_all_c, idx_tr)
    _XMEAN, _XSTD = xm, xs
    _, _, khat, _ = M.build_incidence_table()
    khat_tr = khat[idx_tr]
    modes = tuple(int(m) for m in args.modes.split(","))
    tag = args.tag or f"aug{int(args.aug)}_ms{args.ms}_s{args.seeds}"
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 训练 N 个 seed
    models, all_hist = [], []
    x_tr, _, _ = M.standardize(x_all, idx_tr)
    for k in range(args.seeds):
        seed = args.seed0 + k
        m, hist = train_one(f"{tag}#{k}", seed, x_tr, y_all_c, idx_tr, ym, ys,
                            khat_tr, device, args.epochs, args.batch, args.lr,
                            args.width, modes, args.aug, args.ms)
        models.append(m); all_hist.append(hist)
        torch.save({"model_state": m.state_dict(), "config": {"modes": list(modes),
                    "width": args.width}, "stats": {"ym": ym, "ys": ys, "xm": xm, "xs": xs},
                    "tag": tag, "seed": seed, "aug": args.aug, "ms": args.ms},
                   os.path.join(RESULT_DIR, f"ckpt_imp_{tag}_s{seed}.pt"))

    # 评估：单模型 vs ensemble
    def ev(m):
        x_trn, _, _ = M.standardize(x_all, idx_tr)
        y_trn, _, _ = M.standardize(y_all_c, idx_tr)
        out = {}
        for t_, idx in (("train", idx_tr), ("test", idx_te)):
            if len(idx) == 0:
                continue
            rE, rH, mE, mH, cE, cH = P3.evaluate12(m, x_trn[idx], y_trn[idx],
                                                   ym, ys, device, batch=2)
            out[t_] = {"rel_E": rE, "rel_H": rH, "med_E": mE, "med_H": mH,
                       "corr_E": cE, "corr_H": cH}
        return out

    res = {"tag": tag, "aug": args.aug, "ms": args.ms, "seeds": args.seeds,
           "width": args.width, "epochs": args.epochs, "split": args.split}
    res["single_mean"] = ev(Ensemble(models)) if args.seeds > 1 else None
    res["ensemble"] = ev(Ensemble(models))
    for k, m in enumerate(models):
        res[f"seed{k}"] = ev(m)
    print("\n[改进结果]")
    for k, m in enumerate(models):
        print(f"  seed{k}: E={res[f'seed{k}']['train']['rel_E']*100:.2f}% "
              f"H={res[f'seed{k}']['train']['rel_H']*100:.2f}%")
    print(f"  ensemb: E={res['ensemble']['train']['rel_E']*100:.2f}% "
          f"H={res['ensemble']['train']['rel_H']*100:.2f}%")

    # NFFFT 远场评估（对比 truth 基线）
    if args.nffft:
        n = min(args.n_angles, len(angles))
        a_idx = np.linspace(0, len(angles) - 1, n).astype(int)
        st, _ = EC.nffft_eval(Ensemble(models) if args.seeds > 1 else models[0],
                              a_idx, xm, xs, ym, ys, device, tag=tag,
                              save_png=os.path.join(RESULT_DIR, f"exp_improve_{tag}_nffft.png"))
        res["nffft"] = st
        print(f"  [nffft] truth med={st['truth']['rcs_dB_err_median']:.2f} dB | "
              f"pred med={st['pred']['rcs_dB_err_median']:.2f} dB")

    with open(os.path.join(RESULT_DIR, f"exp_improve_{tag}.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print(f"  结果已存: results/exp_improve_{tag}.json")

    # 训练曲线（可选）
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, h in enumerate(all_hist):
        ax.plot([e["ep"] for e in h], [e["rel_E"] * 100 for e in h],
                label=f"seed{k} E")
    ax.set_xlabel("epoch"); ax.set_ylabel("E rel%")
    ax.set_title(tag); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, f"exp_improve_{tag}.png"), dpi=110)


if __name__ == "__main__":
    main()
