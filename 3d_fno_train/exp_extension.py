# -*- coding: utf-8 -*-
"""
exp_extension.py — 建议4：扩维度实验（尺寸-频率对偶 · 几何参数化 FNO）
=========================================================================
核心思想（Maxwell 尺度不变性，物理严格，无需重跑求解器）：
  目标几何整体缩放 s 倍（λ 不变）⟺ 目标不变、频率变 f/s（λ 变 s 倍）。
  因此：对 F-16 掩膜与散射场做"坐标缩放 + 三线性重采样"，同时把入射场相位
  改为 β/s，即可低成本合成"同一形状族内不同尺寸目标/不同频率"的训练数据。
  这使 FNO 首次面对"几何/频率连续维度"——回答审稿人"只含单一目标"的质疑。

变换规则（严格）：设新网格点 r'，源点 r = r'/s，则
  eps'(r')      = eps(r'/s)                     （掩膜重采样，>0.5 二值化）
  E_scat'(r')   = E_scat(r'/s)                  （散射场重采样，幅度不变）
  H_scat'(r')   = H_scat(r'/s)
  E_inc'(r')    = E0·exp(−j(β/s)k̂·r')           （解析重建）
验证：E_inc'(s·r)=E0·exp(−jβk̂·r)=E_inc(r) ✓；Maxwell 方程在 (r', λ') 下自洽 ✓。

数据规模控制：默认角度子采样 step=8（468→59 角）× 4 个尺度 = 236 样本（~3.5GB 内存）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" exp_extension.py --step 8 --epochs 150
  & "F:/miniconda3/envs/isaac311/python.exe" exp_extension.py --nffft        # 仅 s=1.0 子集远场验证
产出（results/）：exp_extension.json + exp_extension.png（per-scale rel/corr 曲线）
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
import fno_f16_3d as M
import fno_f16_3d_p3 as P3

RESULT_DIR = os.path.join(BASE, "results")


# ============================================================
# 一、三线性重采样（numpy 向量化，无 scipy 依赖）
# ============================================================

def trilinear_resample(field, scale, gx, gy, gz, h):
    """new_field(r') = field(r'/s)。field:(Nx,Ny,Nz,C) complex64/float32。
    源坐标超出源网格 → 置 0（目标缩小后盒外无散射场；放大后截断由尺度范围保证不触发）。"""
    Nx, Ny, Nz = field.shape[:3]
    u = (gx / scale - gx[0]) / h        # 体素中心 → 源网格索引
    v = (gy / scale - gy[0]) / h
    w = (gz / scale - gz[0]) / h
    U, V, W = np.meshgrid(u, v, w, indexing="ij")     # (Nx,Ny,Nz)
    u0 = np.floor(U).astype(np.int64)
    v0 = np.floor(V).astype(np.int64)
    w0 = np.floor(W).astype(np.int64)
    fu = U - u0; fv = V - v0; fw = W - w0
    out = np.zeros_like(field)
    for du in (0, 1):
        uu = u0 + du; wu = (1 - fu) if du == 0 else fu
        uv = (uu >= 0) & (uu < Nx)
        uu = uu.clip(0, Nx - 1)
        for dv in (0, 1):
            vv = v0 + dv; wv = (1 - fv) if dv == 0 else fv
            vv_ok = (vv >= 0) & (vv < Ny)
            vv = vv.clip(0, Ny - 1)
            for dw in (0, 1):
                ww = w0 + dw; ww_ = (1 - fw) if dw == 0 else fw
                ww_ok = (ww >= 0) & (ww < Nz)
                ww = ww.clip(0, Nz - 1)
                ok = uv & vv_ok & ww_ok                       # (Nx,Ny,Nz)
                wgt = (wu * wv * ww_)[..., None]              # (Nx,Ny,Nz,1)
                out += np.where(ok[..., None],
                                field[uu, vv, ww] * wgt, 0)
    return out


# ============================================================
# 二、尺度数据集组装（P3 通道布局）
# ============================================================

def build_scale_dataset(scale, angle_idx, E_scat, H_scat, eps, angles, e0, khat, beta,
                        gx, gy, gz, h):
    """s 缩放下的数据集 X(n,7,...)/Y(n,12,...)。"""
    eps_s = (trilinear_resample(eps[:, :, :, None], scale, gx, gy, gz, h)[..., 0] > 0.5)
    eps_s = eps_s.astype(np.float32)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)
    n = len(angle_idx)
    x_all = np.zeros((n, 7, *eps.shape), dtype=np.float32)
    y_all = np.zeros((n, 12, *eps.shape), dtype=np.float32)
    beta_s = beta / scale
    for a, i in enumerate(angle_idx):
        phase = np.exp(-1j * beta_s[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
        ei = e0[i][None, None, None, :] * phase[..., None]
        x_all[a, 0] = eps_s
        x_all[a, 1:4] = ei.real.transpose(3, 0, 1, 2)
        x_all[a, 4:7] = ei.imag.transpose(3, 0, 1, 2)
        Es = trilinear_resample(E_scat[i], scale, gx, gy, gz, h)
        Hs = trilinear_resample(H_scat[i], scale, gx, gy, gz, h)
        y_all[a, 0:3] = Es.real.transpose(3, 0, 1, 2)
        y_all[a, 3:6] = Es.imag.transpose(3, 0, 1, 2)
        y_all[a, 6:9] = Hs.real.transpose(3, 0, 1, 2)
        y_all[a, 9:12] = Hs.imag.transpose(3, 0, 1, 2)
        if (a + 1) % 30 == 0:
            print(f"    scale={scale} 组装 {a+1}/{n}", flush=True)
    return x_all, y_all, eps_s


# ============================================================
# 三、主流程
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=8, help="角度子采样步长（468/step 个角度）")
    ap.add_argument("--scales", default="0.7,0.85,1.0,1.15", help="全部尺度")
    ap.add_argument("--train-scales", default="0.7,0.85,1.0")
    ap.add_argument("--test-scales", default="0.925,1.15")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--modes", default="20,16,10")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nffft", action="store_true", help="对 s=1.0 子集跑 NFFFT 远场验证")
    ap.add_argument("--n-angles", type=int, default=20, help="NFFFT 角度子集数")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}  {torch.cuda.get_device_name(0) if device=='cuda' else ''}")

    scales = [float(s) for s in args.scales.split(",")]
    tr_scales = {float(s) for s in args.train_scales.split(",")}
    te_scales = {float(s) for s in args.test_scales.split(",")}
    assert tr_scales.isdisjoint(te_scales), "train/test 尺度不能重叠"

    # 读 h5（子集角度）
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]
        H_scat = f["H_scat"][:]
        eps = f["eps_field"][:]
        gx = f["grid_x"][:].astype(np.float32)
        gy = f["grid_y"][:].astype(np.float32)
        gz = f["grid_z"][:].astype(np.float32)
    h = float(gx[1] - gx[0])
    a_idx = np.arange(0, len(angles), args.step)
    print(f"角度子集: {len(a_idx)}（step={args.step}），尺度: {scales}，h={h:.5f}")

    # 组装各尺度数据
    X_all, Y_all, tags = [], [], []
    for s in scales:
        x_s, y_s, _ = build_scale_dataset(s, a_idx, E_scat, H_scat, eps, angles,
                                          e0, khat, beta, gx, gy, gz, h)
        X_all.append(x_s); Y_all.append(y_s)
        tags.append(np.full(len(a_idx), s, dtype=np.float32))
    X_all = np.concatenate(X_all); Y_all = np.concatenate(Y_all)
    tags = np.concatenate(tags)
    y_all_c = P3.clip12(Y_all)

    idx_tr = np.where(np.isin(tags, list(tr_scales)))[0]
    idx_te = np.where(np.isin(tags, list(te_scales)))[0]
    print(f"样本: 总 {len(tags)}（train {len(idx_tr)} / test {len(idx_te)}）")

    x_tr, xm, xs = M.standardize(X_all, idx_tr)
    y_tr, ym, ys = M.standardize(y_all_c, idx_tr)

    # 训练
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    modes = tuple(int(m) for m in args.modes.split(","))
    model = M.FFNO3D(modes=modes, width=args.width, in_ch=7, out_ch=12).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    n = len(idx_tr)
    xtr = torch.from_numpy(np.ascontiguousarray(x_tr[idx_tr])).to(device)
    ytr_raw = torch.from_numpy(np.ascontiguousarray(y_all_c[idx_tr])).to(device)
    ystd_t = torch.tensor(ys, device=device, dtype=torch.float32)
    ymean_t = torch.tensor(ym, device=device, dtype=torch.float32)
    print(f"\n=== 尺寸泛化: train scales {sorted(tr_scales)} / test scales {sorted(te_scales)} "
          f"params={model.count_params():,} ===")
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n)
        ep_loss = 0.0
        for i in range(0, n, args.batch):
            b = perm[i:i + args.batch]
            pred_raw = model(xtr[b]) * ystd_t + ymean_t
            loss = (M.rel_mse_loss(pred_raw[:, 0:6], ytr_raw[b][:, 0:6])
                    + M.rel_mse_loss(pred_raw[:, 6:12], ytr_raw[b][:, 6:12]))
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        sched.step()
        if (ep + 1) % 25 == 0 or ep == args.epochs:
            print(f"  ep{ep+1:4d}/{args.epochs} loss={ep_loss:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # 分尺度评估
    def ev(idx):
        rE, rH, _, _, cE, cH = P3.evaluate12(model, x_tr[idx], y_tr[idx],
                                             ym, ys, device, batch=2)
        return {"rel_E": rE, "rel_H": rH, "corr_E": cE, "corr_H": cH}

    res = {"scales": scales, "train_scales": sorted(tr_scales),
           "test_scales": sorted(te_scales), "n_train": len(idx_tr), "n_test": len(idx_te),
           "per_scale": {}, "aggregate": {"train": ev(idx_tr), "test": ev(idx_te)}}
    print("\n[分尺度指标]")
    for s in scales:
        idx = np.where(tags == s)[0]
        r = ev(idx)
        res["per_scale"][str(s)] = r
        mark = "train" if s in tr_scales else "TEST "
        print(f"  s={s:.3f} [{mark}] E_rel={r['rel_E']*100:.2f}% H_rel={r['rel_H']*100:.2f}% "
              f"corrE={r['corr_E']:.3f} corrH={r['corr_H']:.3f}")
    print(f"  [汇总] train E={res['aggregate']['train']['rel_E']*100:.2f}% | "
          f"test E={res['aggregate']['test']['rel_E']*100:.2f}%")

    # 保存
    os.makedirs(RESULT_DIR, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": {"modes": list(modes),
                "width": args.width}, "stats": {"ym": ym, "ys": ys, "xm": xm, "xs": xs},
                "scales": scales}, os.path.join(RESULT_DIR, "ckpt_ext_scale.pt"))

    # 图：rel vs scale
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ss = scales
    ax.semilogx(ss, [res["per_scale"][str(s)]["rel_E"] * 100 for s in ss], "o-", label="E rel%")
    ax.semilogx(ss, [res["per_scale"][str(s)]["rel_H"] * 100 for s in ss], "s--", label="H rel%")
    for s in tr_scales:
        ax.axvspan(s * 0.98, s * 1.02, color="green", alpha=0.12)
    ax.set_xlabel("scale s (target size × s)"); ax.set_ylabel("relative L2 error (%)")
    ax.set_title("Geometry-scale generalization (shaded = training scales)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULT_DIR, "exp_extension.png"), dpi=110)

    # NFFFT 远场验证（仅 s=1.0，与 h5 真值可比；非 1.0 尺度缺 FEKO 真值，见文档）
    if args.nffft:
        import _exp_common as EC
        n = min(args.n_angles, len(a_idx))
        ff_idx = np.linspace(0, len(a_idx) - 1, n).astype(int)
        # 注意：s=1.0 时输入与 h5 完全一致，可用 h5 真值比对
        st, _ = EC.nffft_eval(model, a_idx[ff_idx], xm, xs, ym, ys, device,
                              tag="scale-model",
                              save_png=os.path.join(RESULT_DIR, "exp_extension_nffft.png"))
        res["nffft_s1"] = st
        print(f"  [nffft s=1.0] truth med={st['truth']['rcs_dB_err_median']:.2f} dB | "
              f"pred med={st['pred']['rcs_dB_err_median']:.2f} dB")

    with open(os.path.join(RESULT_DIR, "exp_extension.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    print("  结果已存: results/exp_extension.json + exp_extension.png")


if __name__ == "__main__":
    main()
