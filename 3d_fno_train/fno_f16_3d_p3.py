# -*- coding: utf-8 -*-
"""
fno_f16_3d_p3.py — P3 输出扩展：E+H 主极化（12 实通道）训练
================================================================
背景：P4 NFFFT 需 PEC 表面电流 J=n̂×H_tot，仅 E 场不足以合成远场。
输入：eps 二值掩膜 + E_inc Re/Im×3（7 通道，同 P1；H_inc 由平面波解析，不进网络）
输出：E_scat Re/Im×3 + H_scat Re/Im×3（12 通道，主极化 θ-pol）
归一化：standardize 按通道（E/H 量级差 ~400 倍自动隔离）
损失：rel_mse(E 分量) + rel_mse(H 分量)（分别逐样本相对，等权，H 不被 E 淹没）
评估：E 部分与 H 部分分别报 rel/corr（原始量纲反标准化后比较）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" fno_f16_3d_p3.py --width 128 --epochs 300
产出（results/ 下）：ckpt_full_p3.pt / metrics_full_p3.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import json
import argparse

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M   # 复用 build_incidence_table / FFNO3D / standardize / rel_mse_loss

H5 = M.H5
RESULT_DIR = os.path.join(BASE, "results")
CLIP_E, CLIP_H = 15.0, 0.01


def load_p3():
    """X(468,7,64,48,32) / Y(468,12,64,48,32)：E Re/Im×3 + H Re/Im×3（主极化）"""
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(H5, "r") as f:
        E = f["E_scat"][:]
        H = f["H_scat"][:]
        eps = f["eps_field"][:]
        h_ang = f["angles"][:]
        gx = f["grid_x"][:].astype(np.float32)
        gy = f["grid_y"][:].astype(np.float32)
        gz = f["grid_z"][:].astype(np.float32)
    assert np.allclose(angles, h_ang), "h5 angles 与入射表不一致"
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    n = len(angles)
    x_all = np.zeros((n, 7, 64, 48, 32), dtype=np.float32)
    y_all = np.zeros((n, 12, 64, 48, 32), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
        ei = e0[i][None, None, None, :] * phase[..., None]
        x_all[i, 0] = (eps > 1.5).astype(np.float32)
        x_all[i, 1:4] = ei.real.transpose(3, 0, 1, 2)
        x_all[i, 4:7] = ei.imag.transpose(3, 0, 1, 2)
        y_all[i, 0:3] = E[i].real.transpose(3, 0, 1, 2)
        y_all[i, 3:6] = E[i].imag.transpose(3, 0, 1, 2)
        y_all[i, 6:9] = H[i].real.transpose(3, 0, 1, 2)
        y_all[i, 9:12] = H[i].imag.transpose(3, 0, 1, 2)
        if (i + 1) % 100 == 0:
            print(f"  预处理 {i+1}/{n}, {(time.time()-t0):.0f}s", flush=True)
    print(f"  P3 数据集: X{x_all.shape} Y{y_all.shape}")
    return x_all, y_all, np.arange(n), angles, (eps > 1.5).astype(np.float32)


def clip12(data, ce=CLIP_E, ch=CLIP_H):
    """按样本钳制：E 分量峰值 ≤ ce，H 分量峰值 ≤ ch"""
    d = data.copy()
    em = np.sqrt(np.sum(d[:, 0:3] ** 2 + d[:, 3:6] ** 2, axis=1, keepdims=True))
    sE = np.minimum(1.0, ce / np.maximum(em.max(axis=(2, 3, 4), keepdims=True), 1e-12))
    d[:, 0:6] *= sE
    hm = np.sqrt(np.sum(d[:, 6:9] ** 2 + d[:, 9:12] ** 2, axis=1, keepdims=True))
    sH = np.minimum(1.0, ch / np.maximum(hm.max(axis=(2, 3, 4), keepdims=True), 1e-12))
    d[:, 6:12] *= sH
    return d


def evaluate12(model, X, Y, ymean, ystd, device, batch=8):
    """E/H 分别报 (relE, relH, medE, medH, corrE, corrH)，反标准化回原始量纲"""
    model.eval()
    ystd_t = torch.tensor(ystd, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ymean, device=device, dtype=torch.float32)

    def part(slc):
        sd = sn = 0.0
        rels, corrs = [], []
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.from_numpy(X[i:i + batch]).to(device)
                yb = Y[i:i + batch, slc] * ystd[0, slc] + ymean[0, slc]
                pb = (model(xb)[:, slc] * ystd_t[0, slc] + ymean_t[0, slc]).cpu().numpy()
                sd += float(np.sum((pb - yb) ** 2))
                sn += float(np.sum(yb ** 2))
                for j in range(pb.shape[0]):
                    rels.append(float(np.linalg.norm(pb[j] - yb[j]) / (np.linalg.norm(yb[j]) + 1e-12)))
                    pm = np.sqrt(np.sum(pb[j, :3] ** 2 + pb[j, 3:] ** 2, axis=0))
                    tm = np.sqrt(np.sum(yb[j, :3] ** 2 + yb[j, 3:] ** 2, axis=0))
                    corrs.append(float(np.corrcoef(pm.ravel(), tm.ravel())[0, 1]))
        return (sd / sn) ** 0.5, float(np.median(rels)), float(np.median(corrs))

    relE, medE, corrE = part(slice(0, 6))
    relH, medH, corrH = part(slice(6, 12))
    return relE, relH, medE, medH, corrE, corrH


def train_p3(width, modes, epochs, batch, lr, gain, seed, device):
    x_all, y_all, idx_all, angles, eps_mask = load_p3()
    y_all_c = clip12(y_all)
    x_tr, xm, xs = M.standardize(x_all, idx_all)
    y_tr, ym, ys = M.standardize(y_all_c, idx_all)

    xtr = torch.from_numpy(x_tr[idx_all]).to(device)
    ytr_raw = torch.from_numpy(y_all_c[idx_all]).to(device)
    ystd_t = torch.tensor(ys, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ym, device=device, dtype=torch.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    model = M.FFNO3D(modes=modes, width=width, in_ch=7, out_ch=12, gain=gain).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(xtr)
    print(f"\n=== P3 full E+H: train={n} width={width} modes={modes} "
          f"params={model.count_params():,} lr={lr} ===")
    t0 = time.time()
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            pred_raw = model(xtr[b]) * ystd_t + ymean_t
            loss = (M.rel_mse_loss(pred_raw[:, 0:6], ytr_raw[b][:, 0:6])
                    + M.rel_mse_loss(pred_raw[:, 6:12], ytr_raw[b][:, 6:12]))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == epochs - 1:
            relE, relH, medE, medH, cE, cH = evaluate12(model, x_tr[idx_all], y_tr[idx_all], ym, ys, device)
            print(f"  ep{ep+1:4d}/{epochs} loss={ep_loss:.4f} "
                  f"E_rel={relE*100:.2f}% corr={cE:.3f} | H_rel={relH*100:.2f}% corr={cH:.3f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    relE, relH, medE, medH, cE, cH = evaluate12(model, x_tr[idx_all], y_tr[idx_all], ym, ys, device)
    res = {"train": {"rel_E": relE, "rel_H": relH, "med_E": medE, "med_H": medH,
                     "corr_E": cE, "corr_H": cH}}
    print(f"  [P3] E_rel={relE*100:.2f}% H_rel={relH*100:.2f}% E_corr={cE:.3f} H_corr={cH:.3f}")
    os.makedirs(RESULT_DIR, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "config": {"modes": list(modes), "width": width, "out_ch": 12},
                "stats": {"y_mean": ym, "y_std": ys, "x_inc_mean": xm, "x_inc_std": xs,
                          "clip_e": CLIP_E, "clip_h": CLIP_H},
                "metrics": res}, os.path.join(RESULT_DIR, "ckpt_full_p3.pt"))
    with open(os.path.join(RESULT_DIR, "metrics_full_p3.json"), "w") as f:
        json.dump({"name": "full_p3", "width": width, "modes": list(modes), "epochs": epochs,
                   "out_ch": 12, "metrics": res}, f, indent=2)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--modes", default="20,16,10")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--gain", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    modes = tuple(int(m) for m in args.modes.split(","))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")
    train_p3(args.width, modes, args.epochs, args.batch, args.lr, args.gain, args.seed, device)


if __name__ == "__main__":
    main()
