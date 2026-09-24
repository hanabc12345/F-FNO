# -*- coding: utf-8 -*-
"""
exp_baselines.py — 建议1：代理模型公平对比（F-FNO vs MLP vs 3D U-Net vs DeepONet）
==============================================================================
目的：回答审稿人第一疑问——F-FNO 相对主流回归结构（MLP / 卷积 / DeepONet）
在"3D 散射近场 E+H 代理"任务上是否有优势。同数据、同划分、同指标。

任务：X(468,7,64,48,32) → Y(468,12,64,48,32)，E/H 十二实通道（同 P3）。
指标：E_rel / H_rel（全局相对 L2）、逐样本中位 rel、|E|/|H| 结构相关 corr。

模型：
  ffno     : 3D F-FNO（M.FFNO3D，因子化谱卷积）——本文方法
  mlp      : 全局池化编码 + 逐体素坐标条件 MLP（无卷积/谱混合）
  unet     : 3D U-Net（卷积编解码）
  deeponet : branch=入射场全局特征, trunk=坐标, 逐通道内积（标准 DeepONet 范式）

显存/速度控制：mlp/deeponet 训练时每样本随机采 --pts 点（默认 4096，每样本独立
采样，逐样本相对损失仍严格成立）；评估全网格。ffno/unet 训练全网格。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_baselines.py --model all --split full
  & "F:/miniconda3/envs/isaac311/python.exe" exp_baselines.py --model deeponet --epochs 100
  & "F:/miniconda3/envs/isaac311/python.exe" exp_baselines.py --ablate      # ffno width/modes 消融
产出（results/）：exp_baselines_<split>.json + exp_baselines_<split>.png / exp_ablate.json
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

RESULT_DIR = os.path.join(BASE, "results")
CACHE = {k: os.path.join(RESULT_DIR, k) for k in
         ("_p3_X.npy", "_p3_Y.npy", "_p3_angles.npy", "_p3_eps.npy")}


# ============================================================
# 一、数据（复用 P3 的 E+H 十二通道组装，带磁盘缓存）
# ============================================================

def load_p3_cached(use_cache=True):
    if use_cache and all(os.path.exists(p) for p in CACHE.values()):
        print("  [cache] 载入 P3 数据集缓存")
        x_all = np.load(CACHE["_p3_X.npy"])
        y_all = np.load(CACHE["_p3_Y.npy"])
        angles = np.load(CACHE["_p3_angles.npy"])
        eps = np.load(CACHE["_p3_eps.npy"])
    else:
        print("  [build] 组装 P3 数据集（首次约 1-2 分钟）")
        x_all, y_all, _, angles, eps = P3.load_p3()
        if use_cache:
            os.makedirs(RESULT_DIR, exist_ok=True)
            np.save(CACHE["_p3_X.npy"], x_all)
            np.save(CACHE["_p3_Y.npy"], y_all)
            np.save(CACHE["_p3_angles.npy"], angles)
            np.save(CACHE["_p3_eps.npy"], eps)
    return x_all, y_all, np.arange(len(angles)), angles, eps


def make_split(split, angles):
    if split == "full":
        return np.arange(len(angles)), np.array([], dtype=int)
    if split == "interp":
        te = np.where(angles[:, 1] % 20 == 10)[0]     # φ=10,30,...,350 奇数层
        tr = np.setdiff1d(np.arange(len(angles)), te)
        return tr, te
    raise ValueError(split)


# ============================================================
# 二、Baseline 模型（统一接口 forward(x, pts=None) → (B,12,Nx,Ny,Nz)）
# ============================================================

class PointMLP(nn.Module):
    """全局入射场编码 + 逐体素坐标条件回归。
    有全局信息（池化编码）但无局部卷积/谱混合 → 检验结构先验的价值。"""
    def __init__(self, in_ch=7, out_ch=12, grid=(64, 48, 32), gfeat=(8, 6, 4),
                 hidden=128):
        super().__init__()
        self.in_ch, self.out_ch, self.grid = in_ch, out_ch, grid
        self.gfeat = gfeat
        self.g_enc = nn.Sequential(
            nn.Linear(in_ch * int(np.prod(gfeat)), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU())
        self.reg = nn.Sequential(
            nn.Linear(hidden + 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_ch))
        coord = PointMLP._mesh_coord(grid)
        self.register_buffer("coord", torch.tensor(coord, dtype=torch.float32))
        self.sample_pts = True

    @staticmethod
    def _mesh_coord(grid):
        ax = [np.linspace(-1, 1, n) for n in grid]
        G = np.meshgrid(*ax, indexing="ij")
        return np.stack([g.ravel() for g in G], axis=1)   # (Np,3)

    def forward(self, x, idx=None):
        """idx: (B,pts) 展平体素索引（训练采样）；None=全网格（评估）。"""
        B = x.shape[0]
        g = self.g_enc(F.adaptive_avg_pool3d(x, self.gfeat).flatten(1))   # (B,hidden)
        if idx is None:
            Np = int(np.prod(self.grid))
            idx = torch.arange(Np, device=x.device)
            c = self.coord[idx][None].expand(B, -1, -1)   # (B,Np,3)
            inp = torch.cat([g[:, None, :].expand(B, c.shape[1], -1), c], dim=-1)
            out = self.reg(inp).permute(0, 2, 1)          # (B,out_ch,Np)
            return out.reshape(B, self.out_ch, *self.grid)
        c = self.coord[idx]                               # (B,pts,3)
        inp = torch.cat([g[:, None, :].expand(B, c.shape[1], -1), c], dim=-1)
        return self.reg(inp).permute(0, 2, 1)             # (B,out_ch,pts)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class DeepONet3D(nn.Module):
    """DeepONet 范式：branch 编码入射场全局特征，trunk 编码坐标，
    输出 u(x)=Σ_k branch_k(x)·trunk_k(x)（逐输出通道）。"""
    def __init__(self, in_ch=7, out_ch=12, grid=(64, 48, 32), gfeat=(8, 6, 4),
                 p=64, hidden=256):
        super().__init__()
        self.in_ch, self.out_ch, self.grid = in_ch, out_ch, grid
        self.p = p
        self.gfeat = gfeat
        self.branch = nn.Sequential(
            nn.Linear(in_ch * int(np.prod(gfeat)), hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p * out_ch))
        self.trunk = nn.Sequential(
            nn.Linear(3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, p))
        coord = PointMLP._mesh_coord(grid)
        self.register_buffer("coord", torch.tensor(coord, dtype=torch.float32))
        self.sample_pts = True

    def forward(self, x, idx=None):
        """idx: (B,pts) 展平体素索引（训练采样）；None=全网格（评估）。"""
        B = x.shape[0]
        br = self.branch(F.adaptive_avg_pool3d(x, self.gfeat).flatten(1)).reshape(B, self.p, self.out_ch)
        if idx is None:
            tk = self.trunk(self.coord)                   # (Np,p)
            return torch.einsum("bpo,np->bon", br, tk).reshape(
                B, self.out_ch, *self.grid)
        c = self.coord[idx]                               # (B,pts,3)
        tk = self.trunk(c.reshape(-1, 3)).reshape(B, -1, self.p)
        return torch.einsum("bpo,bnp->bon", br, tk)       # (B,out_ch,pts)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class DoubleConv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.GELU(),
            nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.GELU())

    def forward(self, x):
        return self.net(x)


class UNet3D(nn.Module):
    """轻量 3D U-Net（levels=3，64→32→16→8）。局部卷积先验。"""
    def __init__(self, in_ch=7, out_ch=12, base=32, levels=3):
        super().__init__()
        encs, chs, c = [], [], in_ch
        for l in range(levels):
            ch = base * (2 ** l)
            encs.append(DoubleConv(c, ch)); chs.append(ch); c = ch
        self.enc = nn.ModuleList(encs)
        self.pools = nn.ModuleList([nn.MaxPool3d(2) for _ in range(levels)])
        self.bottleneck = DoubleConv(c, c * 2)
        bot = c * 2
        dec = []
        for l in reversed(range(levels)):
            ch = chs[l]
            dec.append(nn.ConvTranspose3d(bot, ch, 2, stride=2))
            dec.append(DoubleConv(ch * 2, ch))
            bot = ch
        self.dec = nn.ModuleList(dec)
        self.head = nn.Conv3d(chs[0], out_ch, 1)
        self.sample_pts = False

    def forward(self, x, pts=None):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x); skips.append(x)
            x = self.pools[i](x)
        x = self.bottleneck(x)
        for l in range(len(self.dec) // 2):
            x = self.dec[2 * l](x)
            x = torch.cat([x, skips[len(skips) - 1 - l]], dim=1)
            x = self.dec[2 * l + 1](x)
        return self.head(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def build_model(name, width, modes, in_ch, out_ch):
    if name == "ffno":
        return M.FFNO3D(modes=modes, width=width, in_ch=in_ch, out_ch=out_ch)
    if name == "mlp":
        return PointMLP(in_ch, out_ch)
    if name == "unet":
        return UNet3D(in_ch, out_ch, base=width)
    if name == "deeponet":
        return DeepONet3D(in_ch, out_ch)
    raise ValueError(name)


# ============================================================
# 三、训练 + 评估
# ============================================================

def train_model(name, model, x_tr, y_all_c, idx_tr, ym, ys, device,
                epochs, batch, lr, pts, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    xtr = torch.from_numpy(np.ascontiguousarray(x_tr[idx_tr])).to(device)
    ytr_raw = torch.from_numpy(np.ascontiguousarray(y_all_c[idx_tr])).to(device)
    ystd_t = torch.tensor(ys, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ym, device=device, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xtr)
    use_pts = getattr(model, "sample_pts", False)
    Np = 64 * 48 * 32
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            yb = ytr_raw[b]
            if use_pts:
                # 共享同一组采样点：模型与目标必须在相同位置比较
                idx = torch.argsort(torch.rand(len(b), Np, device=device),
                                    dim=1)[:, :pts]
                pred = model(xtr[b], idx=idx)             # (B,12,pts)
                yb = torch.gather(yb.reshape(len(b), 12, Np), 2,
                                  idx.unsqueeze(1).expand(-1, 12, -1))
            else:
                pred = model(xtr[b])                     # (B,12,Nx,Ny,Nz)
            if use_pts:                                  # pred (B,12,pts) 3D
                pred_raw = pred * ystd_t.reshape(1, 12, 1) + ymean_t.reshape(1, 12, 1)
            else:                                        # pred (B,12,Nx,Ny,Nz) 5D
                pred_raw = pred * ystd_t + ymean_t
            loss = (M.rel_mse_loss(pred_raw[:, 0:6], yb[:, 0:6])
                    + M.rel_mse_loss(pred_raw[:, 6:12], yb[:, 6:12]))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == epochs:
            print(f"  {name} ep{ep+1:4d}/{epochs} loss={ep_loss:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model


def final_eval(model, x_all, y_all_c, idx_tr, idx_te, ym, ys, device):
    """标准评估：标准化用训练集统计；预测反标准化后比较（P3 口径）。"""
    model.eval()
    res = {}
    for tag, idx in (("train", idx_tr), ("test", idx_te)):
        if len(idx) == 0:
            continue
        x_tr, _, _ = M.standardize(x_all, idx_tr)
        y_tr, _, _ = M.standardize(y_all_c, idx_tr)
        rE, rH, mE, mH, cE, cH = P3.evaluate12(model, x_tr[idx], y_tr[idx],
                                              ym, ys, device, batch=1)
        res[tag] = {"rel_E": rE, "rel_H": rH, "med_E": mE, "med_H": mH,
                    "corr_E": cE, "corr_H": cH}
        print(f"  [{tag:5s}] E_rel={rE*100:.2f}% H_rel={rH*100:.2f}% "
              f"E_corr={cE:.3f} H_corr={cH:.3f}")
    return res


def plot_baselines(rows, split, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"] for r in rows]
    e_rel = [r["metrics"].get("train", {}).get("rel_E", 0) * 100 for r in rows]
    h_rel = [r["metrics"].get("train", {}).get("rel_H", 0) * 100 for r in rows]
    x = np.arange(len(names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - w / 2, e_rel, w, label="E rel%")
    ax.bar(x + w / 2, h_rel, w, label="H rel%")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("relative L2 error (%)")
    ax.set_title(f"Baseline comparison (split={split})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110)
    print(f"  图已存: {out_path}")


def run_ablate(x_all, y_all_c, angles, ym, ys, device, epochs=150):
    """F-FNO 消融：width ∈ {32,64,128} × modes × n_layers 三轴（full-split 训练拟合口径）。
    配置：w32/w64/w128（m=(20,16,10), L=4）+ m=(10,8,5)（w64, L=4）+ L=2（w64, m=(20,16,10)）。"""
    idx_tr, _ = make_split("full", angles)
    print("\n=== 消融: width / modes / n_layers（F-FNO, full-split）===")
    out = []
    base_modes, base_width, base_layers = (20, 16, 10), 64, 4

    def _one(tag, width, modes, n_layers):
        torch.manual_seed(0); np.random.seed(0)
        model = M.FFNO3D(modes=modes, width=width, in_ch=7, out_ch=12,
                         n_layers=n_layers).to(device)
        train_model(tag, model, M.standardize(x_all, idx_tr)[0], y_all_c, idx_tr,
                    ym, ys, device, epochs, 8, 3e-3, 0, 0)
        res = final_eval(model, x_all, y_all_c, idx_tr, idx_tr, ym, ys, device)
        out.append({"width": width, "modes": list(modes), "n_layers": n_layers,
                    "params": model.count_params(), **res["train"]})
        print(f"  {tag} params={model.count_params():,}", flush=True)

    for w in (32, 64, 128):                       # 轴1：width
        _one(f"ffno-w{w}", w, base_modes, base_layers)
    _one("ffno-m10", base_width, (10, 8, 5), base_layers)   # 轴2：modes 下限
    _one("ffno-L2", base_width, base_modes, 2)              # 轴3：n_layers 下限

    with open(os.path.join(RESULT_DIR, "exp_ablate.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("  已存 results/exp_ablate.json")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "ffno", "mlp", "unet", "deeponet"])
    ap.add_argument("--split", default="full", choices=["full", "interp"])
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", default="20,16,10")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--pts", type=int, default=4096, help="mlp/deeponet 训练采样点数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--ablate", action="store_true", help="只跑 F-FNO 消融")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    x_all, y_all, _, angles, eps = load_p3_cached(use_cache=not args.no_cache)
    y_all_c = P3.clip12(y_all)
    idx_tr, idx_te = make_split(args.split, angles)
    # 标准化统计（P3 口径：clip 后全量统计）
    _, xm, xs = M.standardize(x_all, idx_tr)
    y_tr, ym, ys = M.standardize(y_all_c, idx_tr)

    if args.ablate:
        run_ablate(x_all, y_all_c, angles, ym, ys, device, args.epochs)
        return

    models = ["ffno", "mlp", "unet", "deeponet"] if args.model == "all" else [args.model]
    modes = tuple(int(m) for m in args.modes.split(","))
    rows = []
    for name in models:
        print(f"\n=== Baseline: {name} (split={args.split}, w={args.width}) ===")
        model = build_model(name, args.width, modes, 7, 12).to(device)
        print(f"  参数量: {model.count_params():,}")
        x_tr, _, _ = M.standardize(x_all, idx_tr)
        train_model(name, model, x_tr, y_all_c, idx_tr, ym, ys, device,
                    args.epochs, 1 if name in ("mlp", "deeponet") else args.batch,
                    args.lr, args.pts, args.seed)
        res = final_eval(model, x_all, y_all_c, idx_tr, idx_te, ym, ys, device)
        rows.append({"name": name, "width": args.width, "params": model.count_params(),
                     "metrics": res})
        # 保存单模型 ckpt 供后续（改进对比/NFFFT）复用
        torch.save({"model_state": model.state_dict(), "name": name,
                    "metrics": res}, os.path.join(RESULT_DIR, f"bl_{name}.pt"))

    if args.model == "all":
        jpath = os.path.join(RESULT_DIR, f"exp_baselines_{args.split}.json")
        with open(jpath, "w") as f:
            json.dump({"split": args.split, "epochs": args.epochs, "rows": rows}, f, indent=2)
        print(f"\n  结果已存: {jpath}")
        plot_baselines(rows, args.split,
                       os.path.join(RESULT_DIR, f"exp_baselines_{args.split}.png"))


if __name__ == "__main__":
    main()
