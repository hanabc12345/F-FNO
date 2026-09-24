# -*- coding: utf-8 -*-
"""
exp_po_patchnet.py — 局部 patch 残差网络 + 装配验证（路线第 2 步）
================================================================================
输入：po_patch_data.py 产出的 results/po_patches/s*.npz
任务：学 ΔĴ = Ĵ_num − 2n̂×û（局部 patch → 4 个残差实分量），再装配回整机：

    Ĵ(facet) = 2n̂×û·[受照] + ΔĴ_net(patch)      →  J = Ĵ·ψ·|e0|/η₀
    → NFFFT → RCS 角谱 → 对比 FEKO 真值

回答三个问题：
  ① patch 级：残差网络能否在**未见过的尺度/空间块**上预测残差？（相对 L2、R²）
  ② 装配级：PO 基线（med 3.75 / P90 10.11 dB）能被推低多少？离"完美电流"floor（3.48/8.90）多远？
  ③ 半体素相位：远场相位核取 r_air（与"J 采样于空气体素中心"自洽）还是 r_surf（现基线）？
     两者差 βh/2 = 0.98 rad = 56°，量化它对 floor 的贡献。

四个训练协议（F-FNO patch 主干 + MLP 基线对照）
  ins : 全部尺度训练（s=1.0 完全在训练集内）→ 装配为**上界**
  loo : train {0.7,0.925,1.15} → test {1.0}   → 装配的**诚实数字**（未见尺度）
  ext : train {0.7,0.925,1.0}  → test {1.15}  （尺度外推探针）
  spa : 空间 8 簇留一（全尺度）→ 未见空间块

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_po_patchnet.py [--epochs 15] [--assemble-angles 117]
产出：results/exp_po_patchnet.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py
import torch
import torch.nn as nn

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import surface_parts, direction_grid
from exp_po_locality import ray_occlusion
from _exp_common import _rcs_stats
from po_patch_data import local_frame, sample_patches, canon_phase, OUT_DIR, H

ETA0 = 119.9169832 * np.pi
P = 11
SCALES = [0.7, 0.925, 1.0, 1.15]
RESULT_DIR = M.RESULT_DIR


# ============================================================
# 一、数据装载
# ============================================================

def load_patches(scales=SCALES):
    data = {}
    for s in scales:
        d = np.load(os.path.join(OUT_DIR, f"s{s:g}.npz"), allow_pickle=True)
        data[s] = {k: d[k] for k in ("patch", "glob", "dJ", "geo", "fidx", "aidx", "fpos")}
        data[s]["meta"] = json.loads(d["meta"][0])
    return data


def stack_scales(data, scales):
    keys = ("patch", "glob", "dJ", "geo", "fidx", "aidx", "fpos")
    out = {k: np.concatenate([data[s][k] for s in scales]) for k in keys}
    out["scale"] = np.concatenate([np.full(len(data[s]["dJ"]), s, dtype=np.float32) for s in scales])
    # 全局面元键 = 尺度序号*1e5 + 面元序号：按面元分组划分，避免同面元跨角度泄漏
    out["facet_key"] = np.concatenate(
        [np.full(len(data[s]["dJ"]), i) * 100000 + data[s]["fidx"] for i, s in enumerate(scales)]
    ).astype(np.int64)
    return out


def make_x(patch_u8, gn, device):
    """patch_u8:(n,P,P,P) uint8; gn:(n,7) 已标准化 → (n,8,P,P,P)"""
    n = len(patch_u8)
    x = torch.zeros(n, 8, P ** 3, device=device)
    x[:, 0] = torch.from_numpy(patch_u8.reshape(n, -1).astype(np.float32)).to(device)
    x[:, 1:8] = torch.from_numpy(gn.astype(np.float32)).to(device)[:, :, None]
    return x.view(n, 8, P, P, P)


class PatchNet(nn.Module):
    """沿用现有 F-FNO 主干；只读回 k=0 层（= 空气体素中心，即残差定义处）的 4 个分量。"""
    def __init__(self, width=64, modes=(5, 5, 5), n_layers=4):
        super().__init__()
        self.fno = M.FFNO3D(modes=modes, width=width, in_ch=8, out_ch=4,
                            n_layers=n_layers, grid=(P, P, P))

    def forward(self, x):
        return self.fno(x)[:, :, P // 2, P // 2, P // 2]


class PatchMLP(nn.Module):
    """基线：占用均值 + 7 维入射量 + 5 维局部几何（与第 1 步特征集同源）。"""
    def __init__(self, n_in=13, width=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.GELU(),
                                 nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))

    def forward(self, ex):
        return self.net(ex)


# ============================================================
# 二、训练 / 评估
# ============================================================

def train_model(kind, big, tr, te, device, epochs, batch, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    mu = big["glob"][tr].mean(0); sd = big["glob"][tr].std(0); sd[sd < 1e-8] = 1.0
    ym = big["dJ"][tr].mean(0); ys = big["dJ"][tr].std(0); ys[ys < 1e-8] = 1.0
    net = (PatchNet() if kind == "fno" else PatchMLP()).to(device)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    nstep = max(1, len(tr) // batch)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(tr))
        tot, cnt = 0.0, 0
        for i in range(0, len(perm) - batch + 1, batch):
            b = tr[perm[i:i + batch]]
            gn = (big["glob"][b] - mu) / sd
            if kind == "fno":
                p = net(make_x(big["patch"][b], gn, device))
            else:
                ex = np.hstack([gn, big["geo"][b], big["patch"][b].mean((1, 2, 3))[:, None]])
                p = net(torch.from_numpy(ex.astype(np.float32)).to(device))
            tgt = torch.from_numpy(((big["dJ"][b] - ym) / ys).astype(np.float32)).to(device)
            loss = ((p - tgt) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); cnt += 1
        sched.step()
        if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f"    [{kind}] ep{ep+1:3d}/{epochs} loss={tot/max(cnt,1):.4f} "
                  f"params={npar:,} ({time.time()-t0:.0f}s)", flush=True)
    return {"net": net, "mu": mu, "sd": sd, "ym": ym, "ys": ys, "kind": kind}


def predict(model, big, idx, device, batch=512):
    net = model["net"]; out = np.zeros((len(idx), 4), dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            gn = (big["glob"][b] - model["mu"]) / model["sd"]
            if model["kind"] == "fno":
                p = net(make_x(big["patch"][b], gn, device))
            else:
                ex = np.hstack([gn, big["geo"][b], big["patch"][b].mean((1, 2, 3))[:, None]])
                p = net(torch.from_numpy(ex.astype(np.float32)).to(device))
            out[i:i + len(b)] = p.cpu().numpy() * model["ys"] + model["ym"]
    return out


def patch_metrics(pred, y):
    """以"零残差(= 直接用 PO)"为基准的 R²：R²=0 表示不比 PO 好，=1 表示完全复现残差。"""
    ss = float(((pred - y) ** 2).sum()); tot = float((y ** 2).sum())
    return {"rel_l2": float(np.sqrt(ss / tot)), "r2_vs_po0": float(1.0 - ss / tot),
            "y_rms": float(np.sqrt(tot / len(y))),
            "pred_rms": float(np.sqrt(float((pred ** 2).sum()) / len(y)))}


# ============================================================
# 三、装配：Ĵ → J → NFFFT → RCS 角谱
# ============================================================

def _extra_stats(est, true):
    """补充指标：RCS 的 dB 误差中位数会被**零点（深衰落）**主导——那里绝对误差极小却
    产生几十 dB 偏差，导致"3.5 dB floor"其实主要在描述零点结构。这里补三个不受零点
    主导的量：① 线性域相对 L2；② 逐角度按峰值归一后的相对 L2；③ 只在真值 ≥ 峰值−20 dB
    的波瓣内统计 dB 误差。"""
    n = len(true)
    e2 = est.reshape(n, -1); t2 = true.reshape(n, -1)
    pk = t2.max(1, keepdims=True)
    msk = t2 >= pk * 1e-2
    dberr = np.abs(10 * np.log10(np.maximum(e2, 1e-30) / np.maximum(t2, 1e-30)))
    v = dberr[msk]
    return {"rel_lin": float(np.linalg.norm(e2 - t2) / np.linalg.norm(t2)),
            "rel_lin_peaknorm": float(np.linalg.norm(e2 / pk - t2 / pk) / np.linalg.norm(t2 / pk)),
            "corr_global": float(np.corrcoef(e2.ravel(), t2.ravel())[0, 1]),
            "db_err_peak20_med": float(np.median(v)),
            "db_err_peak20_p90": float(np.percentile(v, 90))}


def assembly(models, geo_s1, device, aidx=None, verbose=True):
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        rcs_true = f["rcs"][:]
        k = float(beta[0])
        if aidx is None:
            aidx = np.arange(len(angles))
        owners, rsurf, dS = surface_parts(eps, gx, gy, gz)
        dA = np.linalg.norm(dS, axis=1)
        nvec = dS / dA[:, None]
        p_idx = owners - np.rint(dS / (H * H)).astype(np.int64)
        r_air = np.stack([gx[owners[:, 0]], gy[owners[:, 1]], gz[owners[:, 2]]], axis=1)
        metal = eps > 1.5
        Ns = len(dA)
        assert len(geo_s1) == Ns, f"geo 面元数 {len(geo_s1)} != {Ns}"
        rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
        P_air = np.exp(1j * k * (r_air @ rhat.T))
        P_surf = np.exp(1j * k * (rsurf @ rhat.T))
        print(f"[装配] 面元 {Ns}  角度 {len(aidx)}  观测方向 {rhat.shape[0]}", flush=True)

        variants = ["truth", "truth_lit", "po"] + [f"po+{m}" for m in models]
        acc = {v: {o: np.zeros((len(aidx), *shape)) for o in ("air", "surf")} for v in variants}
        tim = {"dda": 0.0, "patch": 0.0, "infer": 0.0, "nffft": 0.0}
        n_lit_tot = 0
        fh = h5py.File(M.H5, "r")
        for a, i in enumerate(aidx):
            ki = khat[i].astype(np.float64)
            ei = e0[i].astype(np.complex128)
            e0m = float(np.linalg.norm(ei))
            ph0 = canon_phase(ei); ei = ei * ph0
            u_inc = (np.cross(ki, ei) / e0m).real
            Hs = fh["H_scat"][i].astype(np.complex128) * ph0
            psi = np.exp(-1j * beta[i] * (r_air @ ki))
            h_inc = np.cross(ki, ei) / ETA0
            H_tot = h_inc[None, :] * psi[:, None] + Hs[owners[:, 0], owners[:, 1], owners[:, 2]]
            J_truth = np.cross(nvec, H_tot)
            cosi = -(nvec @ ki)
            t1 = time.time()
            occl = ray_occlusion(owners, -ki, metal)
            lit = (cosi > 0) & (~occl)
            tim["dda"] += time.time() - t1
            J_po = 2.0 * np.cross(nvec, np.broadcast_to(u_inc, (Ns, 3))) \
                * psi[:, None] * lit[:, None] * (e0m / ETA0)      # 2n̂×H_inc，H_inc=(k̂×e0)/η₀·ψ
            J_lit = np.zeros_like(J_truth); J_lit[lit] = J_truth[lit]

            Jset = {"truth": J_truth, "truth_lit": J_lit, "po": J_po}
            sl = np.where(lit)[0]
            n_lit_tot += len(sl)
            if len(sl):
                e1, e2 = local_frame(nvec, ki, u_inc)
                t1 = time.time()
                pat = sample_patches(metal, p_idx[sl], e1[sl], e2[sl], nvec[sl], P)
                gl = np.stack([(ki[None, :] * e1[sl]).sum(1), (ki[None, :] * e2[sl]).sum(1),
                               (ki[None, :] * nvec[sl]).sum(1),
                               (u_inc[None, :] * e1[sl]).sum(1), (u_inc[None, :] * e2[sl]).sum(1),
                               (u_inc[None, :] * nvec[sl]).sum(1),
                               np.full(len(sl), float(beta[i]) * H)], axis=1).astype(np.float32)
                tim["patch"] += time.time() - t1
                Jpo_n = 2.0 * np.cross(nvec[sl], np.broadcast_to(u_inc, (len(sl), 3)))
                t1 = time.time()
                for mtag, mdl in models.items():
                    gn = (gl - mdl["mu"]) / mdl["sd"]
                    mdl["net"].eval()
                    with torch.no_grad():
                        if mdl["kind"] == "fno":
                            p = mdl["net"](make_x(pat, gn, device))
                        else:
                            ex = np.hstack([gn, geo_s1[sl], pat.mean((1, 2, 3))[:, None]])
                            p = mdl["net"](torch.from_numpy(ex.astype(np.float32)).to(device))
                    p = (p.cpu().numpy() * mdl["ys"] + mdl["ym"]).astype(np.float64)
                    dJ = (p[:, 0][:, None] + 1j * p[:, 1][:, None]) * e1[sl] \
                        + (p[:, 2][:, None] + 1j * p[:, 3][:, None]) * e2[sl]
                    J = np.zeros((Ns, 3), dtype=np.complex128)
                    J[sl] = (Jpo_n + dJ) * psi[sl, None] * (e0m / ETA0)
                    Jset[f"po+{mtag}"] = J
                tim["infer"] += time.time() - t1

            t1 = time.time()
            for v, J in Jset.items():
                for o, Pf in (("air", P_air), ("surf", P_surf)):
                    N = Pf.T @ (J * dA[:, None])
                    Nperp = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
                    E_ff = (1j * k / (4 * np.pi)) * (-ETA0 * Nperp)
                    acc[v][o][a] = (4 * np.pi * (np.abs((E_ff * th).sum(1)) ** 2
                                                 + np.abs((E_ff * ph).sum(1)) ** 2)
                                    / e0m ** 2).reshape(shape)
            tim["nffft"] += time.time() - t1
            if verbose and (a + 1) % 50 == 0:
                print(f"    角 {a+1}/{len(aidx)}  dda{tim['dda']:.0f}s patch{tim['patch']:.0f}s "
                      f"infer{tim['infer']:.0f}s nffft{tim['nffft']:.0f}s", flush=True)
        fh.close()

    res = {"n_angle": len(aidx), "timing_s": {k2: round(v, 1) for k2, v in tim.items()},
           "per_angle_ms": {k2: round(v / len(aidx) * 1000, 1) for k2, v in tim.items()},
           "per_facet_ms": {k2: round(v / max(1, n_lit_tot) * 1000, 3) for k2, v in tim.items()},
           "n_lit_total": int(n_lit_tot),
           "variants": {}}
    print("\n[远场] vs FEKO 真值    med / P90 / corr   (air = 相位核用 r_air 自洽, surf = 现基线)")
    print("       另: relL2 = 线性域相对 L2; pk20 = 峰值−20dB 波瓣内 dB 误差 med/P90")
    for v in variants:
        res["variants"][v] = {}
        for o in ("air", "surf"):
            st = _rcs_stats(acc[v][o], rcs_true[aidx])
            ex = _extra_stats(acc[v][o], rcs_true[aidx])
            res["variants"][v][o] = {**st, "extra": ex}
            print(f"  {v:<14} [{o:>4}] med {st['rcs_dB_err_median']:6.2f}  "
                  f"P90 {st['rcs_dB_err_p90']:6.2f}  corr {st['rcs_corr_median']:.3f}  "
                  f"relL2 {ex['rel_lin']*100:5.1f}%  pk20 {ex['db_err_peak20_med']:5.2f}/"
                  f"{ex['db_err_peak20_p90']:5.2f}")
    return res


# ============================================================
# 四、主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--assemble-angles", type=int, default=0, help=">0 只用 N 个角度装配（快测）")
    ap.add_argument("--train-cap", type=int, default=120000)
    ap.add_argument("--skip-assembly", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备 {device}  P={P}  尺度 {SCALES}  epochs={args.epochs} batch={args.batch}")

    data = load_patches()
    big = stack_scales(data, SCALES)
    print(f"patch 总数 {len(big['dJ']):,}  每尺度 "
          f"{ {s: int(len(data[s]['dJ'])) for s in SCALES} }")
    geo_s1 = np.zeros((int(data[1.0]["fidx"].max()) + 1, 5), dtype=np.float32)
    geo_s1[data[1.0]["fidx"]] = data[1.0]["geo"]      # 逐面元几何描述子（角度无关）

    rng = np.random.default_rng(0)
    # 随机留出面元（留出尺度协议也会用同一套，但 loo 的测试集是 s=1.0 全量，故 ins 才有 20% 留出）
    fac = np.unique(big["facet_key"])
    hold = np.isin(big["facet_key"], rng.choice(fac, size=int(0.2 * len(fac)), replace=False))
    # 空间 8 簇（在 (x,y,z) 上，跨尺度共用同一形状族）
    C = big["fpos"][rng.choice(len(big["fpos"]), 8, replace=False)].copy()
    for _ in range(25):
        lab = ((big["fpos"][:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for kk in range(8):
            m = lab == kk
            if m.any():
                C[kk] = big["fpos"][m].mean(0)
    big["spa"] = lab
    in_s = {s: np.where(big["scale"] == s)[0] for s in SCALES}
    print(f"空间簇 {np.bincount(lab, minlength=8).tolist()}  随机留出面元样本 {int(hold.sum()):,}")

    proto = {
        "ins": (np.where(~hold)[0], np.where(hold)[0]),
        "loo": (np.concatenate([in_s[s] for s in (0.7, 0.925, 1.15)]), in_s[1.0]),
        "ext": (np.concatenate([in_s[s] for s in (0.7, 0.925, 1.0)]), in_s[1.15]),
        "spa": (np.where(lab != 0)[0], np.where(lab == 0)[0]),
    }
    out = {"n_patch": int(len(big["dJ"])), "epochs": args.epochs, "batch": args.batch, "P": P,
           "scales": SCALES, "protocols": {}, "assembly": None}
    trained = {}
    for tag, (tr, te) in proto.items():
        if len(tr) > args.train_cap:
            tr = rng.choice(tr, args.train_cap, replace=False)
        print(f"\n=== 协议 {tag}:  train {len(tr):,}  test {len(te):,} ===", flush=True)
        for kind in ("fno", "mlp"):
            mdl = train_model(kind, big, tr, te, device, args.epochs, args.batch)
            pred = predict(mdl, big, te, device)
            m = patch_metrics(pred, big["dJ"][te])
            out["protocols"][f"{tag}_{kind}"] = {"n_train": int(len(tr)), "n_test": int(len(te)), **m}
            print(f"    -> 测试 rel_L2 {m['rel_l2']*100:6.2f}%   R²(以零残差=PO 为基准) "
                  f"{m['r2_vs_po0']:6.3f}   |ΔĴ|rms {m['y_rms']:.3f}  pred_rms {m['pred_rms']:.3f}",
                  flush=True)
            trained[f"{tag}_{kind}"] = mdl

    if args.skip_assembly:
        jp = os.path.join(RESULT_DIR, "exp_po_patchnet.json")
        json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
        print(f"已存 {jp}（跳过装配）")
        return

    aidx = None
    if args.assemble_angles:
        aidx = np.linspace(0, 467, args.assemble_angles).astype(int)
    asm = {"ins": trained["ins_fno"], "loo": trained["loo_fno"], "loo_mlp": trained["loo_mlp"]}
    print("\n=== 装配（s=1.0，FEKO 真值可用） ===", flush=True)
    t0 = time.time()
    out["assembly"] = assembly(asm, geo_s1, device, aidx=aidx)
    out["assembly"]["wall_s"] = round(time.time() - t0, 1)
    jp = os.path.join(RESULT_DIR, "exp_po_patchnet.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 {jp}   总耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
