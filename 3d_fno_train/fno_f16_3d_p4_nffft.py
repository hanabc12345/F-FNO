# -*- coding: utf-8 -*-
"""
fno_f16_3d_p4_nffft.py — P4 近→远场 NFFFT 校验
================================================
问题裁决：P3 模型近场误差（E 部分 ~48%）经 Huygens 等效流积分后，
在远场是否被平均抵消（远场为积分量，存在平均效应）？

流程（mode=truth，实现正确性校验）：
  真实 E_scat/H_scat(+解析 E_inc/H_inc) → 表面等效流 J=n̂×H_tot, M=−n̂×E_tot
  → 远场积分 → RCS=4π|E_ff|²/|E0|² → 对比 h5 真实 rcs（应接近，验证 NFFFT 实现）

流程（mode=pred，核心裁决）：
  P3 ckpt 预测 E_scat/H_scat → 同上 NFFFT → 对比 h5 rcs
  → 回答：远场是否比近场更准/更稳？

实现要点：
- 表面提取：eps_field 二值场 → np.gradient 表面层 → 面元矢量 dS_vec = −∇eps·h²
  （体素化阶梯表面面积修正 |∇eps|·h²，法向含于矢量中，指向金属外部）
- 相位约定与数据集一致：入射场 exp(−jβ k̂·r)；远场因子 e^{jkr̂·r'}
- E_ff = (jk/4π)[−η₀(N − (N·r̂)r̂) + r̂×L]，N=∬J e^{jk r̂·r'}dS，L=∬M e^{jk r̂·r'}dS
  （Balanis 12-10；J/M 相对符号不可交换）
- 观测网格 ff_theta 0..180/5° × ff_phi 0..360/5°（37×73）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" fno_f16_3d_p4_nffft.py --mode truth
  & "F:/miniconda3/envs/isaac311/python.exe" fno_f16_3d_p4_nffft.py --mode pred
产出（results/）：p4_truth.json + p4_truth.png / p4_pred.json + p4_pred.png
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
import matplotlib
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi   # 376.7303 Ω


# ============================================================
# 一、表面提取（eps 二值场 → 表面体素 + 面元矢量）
# ============================================================

# 表面抽取（face-based：金属体素→空气邻居的面，法向严格轴向）
def surface_parts(eps, gx, gy, gz, h=0.03125):
    """对每个"金属体素-空气体素"相邻面对生成面元。
    注意：体素轴序为 (x,y,z)（meshgrid ij：axis0=x, axis1=y, axis2=z）。
    返回 (idxs_owner (Nf,3) 金属体素索引[场值采样用], rsurf (Nf,3) 面元中心, dS (Nf,3) 外法向·h²)。
    为何不用 np.gradient：中心差分对单层体素表面梯度为 0，会漏表面。"""
    metal = eps > 1.5
    Nx, Ny, Nz = metal.shape
    grid = [gx, gy, gz]
    neigh = [np.array([1, 0, 0]), np.array([-1, 0, 0]),
             np.array([0, 1, 0]), np.array([0, -1, 0]),
             np.array([0, 0, 1]), np.array([0, 0, -1])]
    owners, poss, ds = [], [], []
    for p in np.argwhere(metal):
        for d in neigh:
            q = p + d
            if (q < 0).any() or (q[0] >= Nx) or (q[1] >= Ny) or (q[2] >= Nz):
                continue                          # 域外跳过（金属体素不在盒边界）
            if metal[tuple(q)]:
                continue                          # 金属邻居（内部面）跳过
            ax = int(np.argmax(np.abs(d)))        # 面法向轴
            pos = [float(grid[a][p[a]]) for a in range(3)]
            pos[ax] += float(d[ax]) * (h / 2.0)   # 面元中心在体素间中点
            dS_v = np.zeros(3); dS_v[ax] = float(d[ax]) * h * h
            owners.append(q)                      # 场值取空气侧体素（表面外推近似）
            poss.append(pos); ds.append(dS_v)
    print(f"  表面面元: {len(owners)}（金属体素 {int(metal.sum())}）")
    return np.asarray(owners), np.asarray(poss), np.asarray(ds)


def sample_surface_field(field, idxs):
    """体素场 field:(64,48,32,3) complex → 表面体素处值 (Ns,3)"""
    return field[idxs[:, 0], idxs[:, 1], idxs[:, 2]]


def pred_to_surface(pred, idxs):
    """P3 预测场 pred:(12,Nx,Ny,Nz)（前 6=E Re/Im, 后 6=H Re/Im）
    → (E_s (Ns,3), H_s (Ns,3)) complex"""
    i0, i1, i2 = idxs[:, 0], idxs[:, 1], idxs[:, 2]
    E3 = pred[0:3] + 1j * pred[3:6]      # (3,Nx,Ny,Nz)
    H3 = pred[6:9] + 1j * pred[9:12]
    E_s = E3[:, i0, i1, i2].T            # (Ns,3)
    H_s = H3[:, i0, i1, i2].T
    return E_s, H_s


# ============================================================
# 二、NFFFT：等效流 → 远场
# ============================================================

def nffft(J, M, rsurf, dS, rhat, k):
    """J,M:(Ns,3) complex 等效电流/磁流（已含法向）；rsurf:(Ns,3) 面元中心；
    dS:(Ns,3) 面元矢量（积分用其标量面积 |dS|）；rhat:(Ndir,3) 观测单位矢量；k: 波数。
    返回 E_ff:(Ndir,3) complex（不含 e^{-jkr}/r 因子）。
    注意：J ⊥ n̂，而 dS ∥ n̂，逐元素乘恒为 0 → 必须用标量面积 |dS|。"""
    dA = np.linalg.norm(dS, axis=1)                      # (Ns,) 标量面元面积
    phase = np.exp(1j * k * (rsurf @ rhat.T))            # (Ns,Ndir)
    N = phase.T @ (J * dA[:, None])                      # (Ndir,3)
    L = phase.T @ (M * dA[:, None])
    Nrhat = (N * rhat).sum(axis=1, keepdims=True) * rhat
    cross = np.cross(rhat, L)
    # Balanis 12-10（e^{jωt}）：E_ff = (jk/4π)[−η₀N⊥ + r̂×L]。
    # J 项与 M 项的**相对符号**必须如此；PEC 路径 M=0，全局符号对 |E_ff|² 无影响，
    # 但含磁流（介质/M≠0 或闭合盒面等效原理）时旧写法会得到错误的干涉。
    # 数据侧独立佐证：h5 的 E_ff 对 (η₀N⊥, r̂×L) 做双系数最小二乘得 (−1.00∠−172°, +1.00∠−2°)。
    E = (1j * k / (4.0 * np.pi)) * (-ETA0 * (N - Nrhat) + cross)
    return E


def direction_grid(ff_theta, ff_phi):
    """(θ,φ) 网格 → rhat(Ndir,3), theta_hat, phi_hat (Ndir,3)"""
    T, P = np.meshgrid(np.deg2rad(ff_theta), np.deg2rad(ff_phi), indexing="ij")
    T = T.ravel(); P = P.ravel()
    st, ct = np.sin(T), np.cos(T)
    sp, cp = np.sin(P), np.cos(P)
    rhat = np.stack([st * cp, st * sp, ct], axis=1)
    th = np.stack([ct * cp, ct * sp, -st], axis=1)
    ph = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
    return rhat, th, ph, (len(ff_theta), len(ff_phi))


def build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i):
    """第 i 个角度：解析入射场 + 散射场 → 总场 E_tot/H_tot。
    返回 (Etot (64,48,32,3), Htot (64,48,32,3)) complex。"""
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
    e_inc = e0[i][None, None, None, :] * phase[..., None]      # (64,48,32,3)
    h_inc = (1.0 / ETA0) * np.cross(khat[i][None, None, None, :],
                                    np.broadcast_to(e_inc, (64, 48, 32, 3)))  # k̂×E/η₀
    Etot = e_inc + E_scat[i]
    Htot = h_inc + H_scat[i]
    return Etot, Htot


# 表面等效流 → RCS（Ndir,）公共函数
def rcs_from_surface(E_s, H_s, dS, rsurf, rhat, th, ph, k, e0mag=1.0, use_mag=False):
    n = dS / (np.linalg.norm(dS, axis=1, keepdims=True) + 1e-12)
    J = np.cross(n, H_s)                        # n̂×H_tot
    Mm = -np.cross(n, E_s) if use_mag else np.zeros_like(J)  # PEC: M=0（物理正确，实测更优）
    E_ff = nffft(J, Mm, rsurf, dS, rhat, k)
    E_theta = (E_ff * th).sum(axis=1)
    E_phi = (E_ff * ph).sum(axis=1)
    return 4.0 * np.pi * (np.abs(E_theta) ** 2 + np.abs(E_phi) ** 2) / e0mag ** 2


# ============================================================
# 三、主流程
# ============================================================


def main():
    matplotlib.use("Agg")   # P4 仅保存图片，无头后端
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["truth", "pred"], default="truth")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--nangles", type=int, default=-1, help="评估角度数（-1=全部 468）")
    args = ap.parse_args()

    # 载入几何与入射表
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        E_scat = f["E_scat"][:]              # (468,64,48,32,3) complex
        H_scat = f["H_scat"][:]
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]               # (468,37,73)
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    k = float(beta[0])                       # 波数（各角度一致）

    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    idx_list = list(range(args.nangles)) if args.nangles > 0 else list(range(len(angles)))

    print(f"\n=== P4 NFFFT mode={args.mode} angles={len(idx_list)} k={k:.3f} rad/m "
          f"eta0={ETA0:.4f} 表面体素 {len(rsurf)} ===")

    rcs_est = []
    t0 = time.time()
    if args.mode == "truth":
        # 真实场 → 校验 NFFFT 实现正确性
        for a, i in enumerate(idx_list):
            Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i)
            E_s = sample_surface_field(Etot, idxs)
            H_s = sample_surface_field(Htot, idxs)
            rcs = rcs_from_surface(E_s, H_s, dS, rsurf, rhat, th, ph, k, np.linalg.norm(e0[i]))
            rcs_est.append(rcs.reshape(shape))
            if (a + 1) % 50 == 0:
                print(f"  [{a+1}/{len(idx_list)}] {(time.time()-t0):.0f}s", flush=True)
    else:
        # P3 预测场 → 核心裁决
        ckpt = torch.load(os.path.join(RESULT_DIR, "ckpt_full_p3.pt"),
                          map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        ym, ys = ckpt["stats"]["y_mean"], ckpt["stats"]["y_std"]
        xm, xs = ckpt["stats"]["x_inc_mean"], ckpt["stats"]["x_inc_std"]
        model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                         in_ch=7, out_ch=12).to("cuda")
        model.load_state_dict(ckpt["model_state"]); model.eval()

        # 重建 X（同 P3 load_p3 的通道布局）
        X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
        X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)
        with torch.no_grad():
            for a, i in enumerate(idx_list):
                phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
                ei = e0[i][None, None, None, :] * phase[..., None]
                x = np.zeros((1, 7, 64, 48, 32), dtype=np.float32)
                x[0, 0] = (eps > 1.5).astype(np.float32)
                x[0, 1:4] = ei.real.transpose(3, 0, 1, 2)
                x[0, 4:7] = ei.imag.transpose(3, 0, 1, 2)
                x = (x - xm) / xs
                pred = model(torch.from_numpy(x).to("cuda"))[0].cpu().numpy()
                pred = pred * ys.reshape(12, 1, 1, 1) + ym.reshape(12, 1, 1, 1)  # 反标准化，保持 4D（5D 广播会生成空 0 维）
                E_s, H_s = pred_to_surface(pred, idxs)
                rcs = rcs_from_surface(E_s, H_s, dS, rsurf, rhat, th, ph, k, np.linalg.norm(e0[i]))
                rcs_est.append(rcs.reshape(shape))
                if (a + 1) % 50 == 0:
                    print(f"  [{a+1}/{len(idx_list)}] {(time.time()-t0):.0f}s", flush=True)
    rcs_est = np.asarray(rcs_est)            # (n,37,73)

    # 统计与可视化
    tgt = rcs_true[idx_list]                                  # (n,37,73)
    mask = tgt > 1e-6                                         # 忽略 ~0 RCS 方向
    dB_est = 10 * np.log10(np.maximum(rcs_est, 1e-9))
    dB_tgt = 10 * np.log10(np.maximum(tgt, 1e-9))
    diff = dB_est - dB_tgt
    med = float(np.median(np.abs(diff[mask])))
    p90 = float(np.percentile(np.abs(diff[mask]), 90))
    # 结构相关（每角度 RCS 图与目标的相关中位数）
    corrs = []
    for a in range(len(idx_list)):
        r1 = dB_est[a][mask[a]]; r2 = dB_tgt[a][mask[a]]
        if r1.std() > 1e-9 and r2.std() > 1e-9:
            corrs.append(float(np.corrcoef(r1, r2)[0, 1]))
    corr = float(np.median(corrs))
    # 近场 E/H 对照（pred 模式）
    out = {"mode": args.mode, "n_angles": len(idx_list),
           "rcs_dB_err_median": med, "rcs_dB_err_p90": p90,
           "rcs_corr_median": corr}
    print(f"\n  [P4-{args.mode}] |ΔRCS| 中位 {med:.2f} dB, P90 {p90:.2f} dB, 结构相关 {corr:.3f}")
    if args.mode == "truth":
        # 顺带核对 h5 E_ff 约定：case 0 的 E_θ/E_φ 与 h5 E_ff 幅度比
        with h5py.File(M.H5, "r") as f:
            eff_h5 = f["E_ff"][idx_list[0]]                   # (37,73,2)
        Etot, Htot = build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, idx_list[0])
        E_s = sample_surface_field(Etot, idxs)
        H_s = sample_surface_field(Htot, idxs)
        n = dS / (np.linalg.norm(dS, axis=1, keepdims=True) + 1e-12)
        E_ff = nffft(np.cross(n, H_s), np.zeros_like(np.cross(n, H_s)), rsurf, dS, rhat, k)
        E_theta = (E_ff * th).sum(axis=1).reshape(shape)
        E_phi = (E_ff * ph).sum(axis=1).reshape(shape)
        # h5 E_ff 列序（0/1 = θ/φ）实测比对
        mag0 = np.abs(E_theta) / (np.abs(eff_h5[..., 0]) + 1e-12)
        mag1 = np.abs(E_phi) / (np.abs(eff_h5[..., 1]) + 1e-12)
        r0 = np.median(mag0[mag0 > 0]); r1 = np.median(mag1[mag1 > 0])
        out["E_ff_ratio_med"] = {"ch0/theta": float(r0), "ch1/phi": float(r1)}
        print(f"  h5 E_ff 幅度比（应≈1）：ch0/θ={r0:.4f} ch1/φ={r1:.4f}")

    # 可视化：前 3 个角度（θ=90° 俯仰切 + φ 方位切）
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for a in range(min(3, len(idx_list))):
        i = idx_list[a]
        # φ cut at θ=90（若存在）
        t90 = np.argmin(np.abs(ff_theta - 90))
        axes[0, a].semilogy(ff_phi, rcs_est[a][t90], "-", lw=1.2, label=f"NFFFT {args.mode}")
        axes[0, a].semilogy(ff_phi, rcs_true[i][t90], "--", lw=1.2, label="FEKO")
        axes[0, a].set_title(f"case_{i:03d} θ=90° φ-cut")
        axes[0, a].set_xlabel("φ"); axes[0, a].legend(fontsize=8)
        # θ cut at φ=0（若存在）
        p0 = np.argmin(np.abs(ff_phi - 0))
        axes[1, a].semilogy(ff_theta, rcs_est[a][:, p0], "-", lw=1.2, label=f"NFFFT {args.mode}")
        axes[1, a].semilogy(ff_theta, rcs_true[i][:, p0], "--", lw=1.2, label="FEKO")
        axes[1, a].set_title(f"case_{i:03d} φ=0° θ-cut")
        axes[1, a].set_xlabel("θ"); axes[1, a].legend(fontsize=8)
    fig.suptitle(f"P4 NFFFT vs FEKO RCS (mode={args.mode}, med|ΔdB|={med:.2f})", fontsize=13)
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, f"p4_{args.mode}.png")
    fig.savefig(png, dpi=110)
    print(f"  图已存: {png}")

    jpath = os.path.join(RESULT_DIR, f"p4_{args.mode}.json")
    with open(jpath, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  结果已存: {jpath}")


if __name__ == "__main__":
    main()
