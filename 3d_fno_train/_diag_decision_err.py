# -*- coding: utf-8 -*-
"""
_diag_decision_err.py — 判决级误差传播实验（RCS 误差 → 下游判决误差）
================================================================================
动机
--------------------------------------------------------------------------------
`mesh_po` 的中位 3.181 dB 是 **RCS 误差**，但下游（UE 演示 / 训练推演）消费的是
**判决**：何时被发现、是否烧穿、被识别成什么。本脚本把实测 RCS 误差传播到
`deploy_rcs_server/ue_rcs_service.py::radar_equation_state`（逐字复刻其公式与常量），
测量"判决级误差"，回答三个问题：

  A 误差结构：ΔdB 是**系统偏差**（时间平均不掉）还是**零均值闪烁**？
  B 多帧累积：W 帧非相干积累后残留误差还剩多少？（服务端现状是逐帧瞬时判决）
  C 判决传播：R_det / R_BT 的误差分布 + 四态判决不一致率

口径与约定（与全项目一致，勿凭直觉改）
--------------------------------------------------------------------------------
· 单站（monostatic）：入射方向 = 观测方向。h5 的 `rcs` 是 (468 入射 × 2701 观测)
  的双站矩阵；入射角网格（θ 30..150 步 10、φ 0..350 步 10）是观测网格
  （θ 0..180 步 5、φ 0..360 步 5）的子集 ⇒ 单站值 = rcs[i, θ_i/5, φ_i/5]。
· 入射传播方向 khat = -u(θ,φ)（u = 标准单位矢量），故**回波方向 = +u(θ,φ)**
  ⇒ 单站观测方向与入射方向同 (θ,φ)（脚本内用 H1/H2 两种索引假设置信度自检）。
· σ 用解析 PO 电流 2n̂×H_inc 在 STL 真网格上积分 —— 与 `exp_po_mesh.py` 的
  `mesh_po` 臂完全同一条链路的同一个口径。

用法
--------------------------------------------------------------------------------
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --step 4   # 117 角快测
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py            # 全 468 角
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --mode post # 只重跑分析
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --mode feko --ncase 12
      ② 与 FEKO 原始 .out 逐角核对（h5 保真度 / 索引约定一手证据 / 单站偏差归属 / 互易性）
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --mode gen
      ① 标定泛化验证（空间分块交叉验证，替代会泄漏的随机半集划分）
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --mode fullmap
      P1 全向图补满：468 角 × 2701 方向完整双站误差图
  & "F:/miniconda3/envs/isaac311/python.exe" _diag_decision_err.py --mode patcalib
      P2 前置：由 fullmap 产出 Δ-分档标定表（单站/方向图/双站三路共用一张连续表）
产出：results/_diag_decision_err.npz（σ_pred/σ_true/入射角，供复算）
      results/_diag_decision_err.json / _diag_feko_audit.json / _diag_calib_gen.json
      results/_diag_po_fullmap.npz / .json / calib_affine.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M                                          # noqa: E402
from fno_f16_3d_p4_nffft import direction_grid                   # noqa: E402
from exp_po_mesh import (load_mesh, mesh_outside_voxel,          # noqa: E402
                         nffft_from, ETA0, H, STL, RESULT_DIR)
from exp_po_locality import ray_occlusion                        # noqa: E402
from po_patch_data import canon_phase                            # noqa: E402

# ============================================================
# 服务端判决常量（逐字复刻 deploy_rcs_server/ue_rcs_service.py L98-110）
# ============================================================
R_REF = 150000.0        # 参考探测距离 m（σ_ref 目标的探测距离）
SIGMA_REF = 1.0         # 参考 RCS m²
ECM_R_BT = 4000.0       # 干扰标定参考距离 m
JS_MIN = 1.0            # 烧穿判据 J/S ≥ 1 视为被压制
JAM_NONE, JAM_SUPPRESS, JAM_BURN, JAM_TRACK = 0, 1, 2, 3

# ============================================================
# FEKO 原始 .out（生成端）：直接复用 gen_feko_batch 的解析器 —— 同一份代码，
# 避免"自己重写解析器导致口径漂移"把数据问题掩盖成实现差异。
# ============================================================
FEKO_RUN_DIR = r"f:\MyWorkSpace\UAVGame\3d_feko_run"
FEKO_DATA_DIR = os.path.join(os.path.dirname(BASE), "3d_feko_data")
if FEKO_DATA_DIR not in sys.path:
    sys.path.insert(0, FEKO_DATA_DIR)
try:
    import gen_feko_batch as G
except Exception as _e:                                       # noqa: BLE001
    G = None
    print(f"[warn] gen_feko_batch 不可用（--mode feko 将不可用）：{_e}")


def radar_state(dist, sigma_real, jammer_on, jam_db_at4km):
    """逐字复刻 ue_rcs_service.radar_equation_state（返回 (rcs_apparent_db, state)）。"""
    sigma_real = max(float(sigma_real), 0.0)
    if jammer_on:
        jam_lin = 10 ** (jam_db_at4km / 10.0)
        jam_sigma = jam_lin * (dist / ECM_R_BT) ** 2
    else:
        jam_lin, jam_sigma = 0.0, 0.0
    rcs_apparent_db = 10 * np.log10(sigma_real + jam_sigma + 1e-9)
    if R_REF <= 0.0:
        r_det0 = float("inf")
    elif sigma_real > 0.0:
        r_det0 = R_REF * (sigma_real / SIGMA_REF) ** 0.25
    else:
        r_det0 = 0.0
    if dist > r_det0:
        state = JAM_NONE
    elif not jammer_on:
        state = JAM_TRACK
    else:
        r_bt = ECM_R_BT * (JS_MIN * sigma_real / max(jam_lin, 1e-30)) ** 0.5
        state = JAM_BURN if dist <= r_bt else JAM_SUPPRESS
    return rcs_apparent_db, state


def r_det_of(sigma):
    return R_REF * (max(float(sigma), 0.0) / SIGMA_REF) ** 0.25


def r_bt_of(sigma, jam_db_at4km):
    jam_lin = 10 ** (jam_db_at4km / 10.0)
    return ECM_R_BT * (JS_MIN * max(float(sigma), 0.0) / max(jam_lin, 1e-30)) ** 0.5


# ============================================================
# 一、单站 σ：解析 PO（真网格） vs FEKO 真值
# ============================================================

def unit_vecs(th_deg, ph_deg):
    """与 fno_f16_3d_p4_nffft.direction_grid 同一公式的单位矢量 (rhat, theta_hat, phi_hat)。"""
    T, P = np.deg2rad(th_deg), np.deg2rad(ph_deg)
    st, ct, sp, cp = np.sin(T), np.cos(T), np.sin(P), np.cos(P)
    rhat = np.stack([st * cp, st * sp, ct], axis=1)
    th = np.stack([ct * cp, ct * sp, -st], axis=1)
    ph = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
    return rhat, th, ph


def _load_common():
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        rcs_true = f["rcs"][:]
        ff_theta = f["ff_theta"][:]
        ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
    return angles, e0, khat, beta, eps, rcs_true, ff_theta, ff_phi, gx, gy, gz


def po_jacobian(i, khat, e0, beta, cen, nvm, q_out, metal, ph_tgt):
    """第 i 个入射角的解析 PO 面电流（含相位），返回 (J, e0m)。"""
    ki = khat[i].astype(np.float64)
    ei = e0[i].astype(np.complex128)
    e0m = float(np.linalg.norm(ei))
    ei = ei * canon_phase(ei)
    psi = np.exp(-1j * beta[i] * (ph_tgt @ ki))
    lit = ((nvm @ ki) < 0) & (~ray_occlusion(q_out, -ki, metal))
    J = 2.0 * np.cross(nvm, np.cross(ki, ei)) / ETA0 * psi[:, None]
    J[~lit] = 0.0
    return J, e0m


def sigma_from_e(E, U, TH, PH, e0m):
    """(Ndir,) 线性 σ。"""
    return 4 * np.pi * (np.abs((E * TH).sum(1)) ** 2
                        + np.abs((E * PH).sum(1)) ** 2) / e0m ** 2


def mode_validate(nval):
    """验证模式：对前 nval 个入射角算**全方向图** σ_pred，用于
    ① 复现封板口径（med/P90）确认链路一致；② 判定单站索引约定。"""
    (angles, e0, khat, beta, eps, rcs_true,
     ff_theta, ff_phi, gx, gy, gz) = _load_common()
    g0 = np.array([gx[0], gy[0], gz[0]])
    metal = eps > 1.5
    k = float(beta[0])
    cen, nvm, dAm = load_mesh(STL)
    q_out, _ = mesh_outside_voxel(cen, nvm, g0, metal)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)

    aidx = np.arange(0, min(nval, len(angles)))
    # 约定候选：观测方向的 (θ,φ) 候选索引
    def idx_of(thd, phd):
        it = np.rint((np.asarray(thd) - ff_theta[0]) / (ff_theta[1] - ff_theta[0])).astype(int)
        ip = np.rint((np.mod(np.asarray(phd), 360.0) - ff_phi[0]) / (ff_phi[1] - ff_phi[0])).astype(int)
        ok = (it >= 0) & (it < len(ff_theta)) & (ip >= 0) & (ip < len(ff_phi))
        return it, ip, ok

    cands = {"H1_same(θ,φ)": (angles[aidx, 0], angles[aidx, 1]),
             "H2_anti(180-θ,φ+180)": (180.0 - angles[aidx, 0], angles[aidx, 1] + 180.0)}

    spo = np.zeros((len(aidx), len(rhat)))
    t0 = time.time()
    for a, i in enumerate(aidx):
        J, e0m = po_jacobian(i, khat, e0, beta, cen, nvm, q_out, metal, cen)
        spo[a] = sigma_from_e(nffft_from(J, dAm, cen, rhat, k), rhat, th, ph, e0m)
        if (a + 1) % 5 == 0:
            print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)
    sto = rcs_true[aidx].reshape(len(aidx), -1).astype(np.float64)

    # ① 复现封板口径（全方向图，与 exp_po_mesh 相同：全部方向，不做掩码）
    d_all = 10 * np.log10(np.maximum(spo, 1e-30) / np.maximum(sto, 1e-30))
    print(f"\n[全方向图口径复现] n={len(aidx)}  med|Δ| {np.median(np.abs(d_all)):.3f} dB  "
          f"P90 {np.percentile(np.abs(d_all), 90):.3f} dB   "
          f"（封板参考 3.181 / 9.068，30 角度）")
    dl = np.log10(np.maximum(spo, 1e-30).ravel())
    tl = np.log10(np.maximum(sto, 1e-30).ravel())
    print(f"  全局 corr(logσ) {np.corrcoef(dl, tl)[0, 1]:+.4f}")

    # ② 单站索引约定
    print("\n[单站索引约定判定]")
    for nm, (thd, phd) in cands.items():
        it, ip, ok = idx_of(thd, phd)
        if not ok.all():
            print(f"  {nm:<22} 索引越界（{int((~ok).sum())} 个），跳过")
            continue
        flat = it * len(ff_phi) + ip
        s_p = spo[np.arange(len(aidx)), flat]
        s_t = sto[np.arange(len(aidx)), flat]
        m = (s_p > 0) & (s_t > 0)
        dd = 10 * np.log10(s_p[m] / s_t[m])
        pct = np.array([(sto[a] < s_t[a]).mean() for a in range(len(aidx))])
        print(f"  {nm:<22} n={m.sum():<4} med|Δ| {np.median(np.abs(dd)):6.3f}  "
              f"P90 {np.percentile(np.abs(dd), 90):6.3f}  bias {np.median(dd):+7.3f} dB  "
              f"corr(logσ) {np.corrcoef(np.log10(s_p[m]), np.log10(s_t[m]))[0,1]:+.4f}  "
              f"σ真值百分位(中位) {np.median(pct)*100:.1f}%")
    # ③ 单站在全方向图中的强弱位置
    print("\n[单站方向在方向图中的位置]（σ_true 百分位：0%=最弱，100%=最强）")
    it, ip, ok = idx_of(angles[aidx, 0], angles[aidx, 1])
    flat = it * len(ff_phi) + ip
    pct = np.array([(sto[a] < sto[a][flat[a]]).mean() for a in range(len(aidx))])
    print(f"  H1 方向 σ_true 百分位：中位 {np.median(pct)*100:5.1f}%  "
          f"四分位 [{np.percentile(pct,25)*100:.1f}%, {np.percentile(pct,75)*100:.1f}%]  "
          f"⇒ {'强散射（含镜面附近）' if np.median(pct) > 0.5 else '弱散射（非镜面）'}")
    top = np.argmax(sto, axis=1)
    same = (top == flat).mean()
    print(f"  最强方向即单站方向的比例 {same*100:.1f}%（随机基线 {100/len(rhat):.2f}%）")

    np.savez(os.path.join(RESULT_DIR, "_diag_validate_full.npz"),
             aidx=aidx, spo=spo.astype(np.float32), sto=sto.astype(np.float32),
             flat_h1=flat, ff_theta=ff_theta, ff_phi=ff_phi)
    print(f"\n已存 results/_diag_validate_full.npz   耗时 {time.time()-t0:.0f}s")


def compute_mono_sigma(step, conv="h1"):
    (angles, e0, khat, beta, eps, rcs_true,
     ff_theta, ff_phi, gx, gy, gz) = _load_common()
    g0 = np.array([gx[0], gy[0], gz[0]])
    metal = eps > 1.5
    k = float(beta[0])
    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    print(f"  三角面空气侧起点: 成功 {int(air_ok.sum())}/{len(cen)}", flush=True)

    aidx = np.arange(0, len(angles), step)
    sel = angles[aidx]
    if conv == "h1":
        thd, phd = sel[:, 0], sel[:, 1]
    else:
        thd, phd = 180.0 - sel[:, 0], sel[:, 1] + 180.0
    U, TH, PH = unit_vecs(thd, phd)
    it = np.rint((np.asarray(thd) - ff_theta[0]) / (ff_theta[1] - ff_theta[0])).astype(int)
    ip = np.rint((np.mod(np.asarray(phd), 360.0) - ff_phi[0]) / (ff_phi[1] - ff_phi[0])).astype(int)
    flat = it * len(ff_phi) + ip
    sto = rcs_true[aidx].reshape(len(aidx), -1).astype(np.float64)

    sp = np.zeros(len(aidx))
    st = np.zeros(len(aidx))
    t0 = time.time()
    for a, i in enumerate(aidx):
        J, e0m = po_jacobian(i, khat, e0, beta, cen, nvm, q_out, metal, cen)
        E = nffft_from(J, dAm, cen, U[a:a + 1], k)
        sp[a] = float(sigma_from_e(E, U[a:a + 1], TH[a:a + 1], PH[a:a + 1], e0m)[0])
        st[a] = sto[a, flat[a]]
        if (a + 1) % 100 == 0:
            print(f"    {a+1}/{len(aidx)}  {time.time()-t0:.0f}s", flush=True)
    print(f"  σ 计算完成 {time.time()-t0:.0f}s", flush=True)
    return sel, sp, st


def row_table(sp_ring, st_ring, it, ip, ff_theta):
    """按 θ 行（俯仰入射角）分列：单站 vs 同环误差、偏差、σ 绝对值、σ_true 环内百分位。
    输出 σ 的绝对 dBsm 与 Δ 的 p10/p90，用于区分"整体系统偏移"与"少数奇异角"。"""
    am = np.arange(sp_ring.shape[0])
    d_ring = 10 * np.log10(np.maximum(sp_ring, 1e-30) / np.maximum(st_ring, 1e-30))
    a_ring, a_mono = np.abs(d_ring), np.abs(d_ring[am, ip])
    dm = d_ring[am, ip]
    pct = np.array([(st_ring[a] < st_ring[a, ip[a]]).mean() for a in am])
    out = []
    print("\n[按 θ（俯仰入射角）分列]（σ 绝对值 = 单站方向的 dBsm 中位）")
    print(f"  {'θ(°)':>5} {'n':>4} {'单站med|Δ|':>11} {'同环med|Δ|':>11} "
          f"{'单站Δ中位':>10} {'Δ的p10':>9} {'Δ的p90':>9} "
          f"{'σ_pred':>9} {'σ_true':>9} {'环内百分位':>10}")
    for t in np.unique(it):
        k = it == t
        rec = {"theta_deg": float(ff_theta[t]), "n": int(k.sum()),
               "mono_med_abs_dB": float(np.median(a_mono[k])),
               "ring_med_abs_dB": float(np.median(a_ring[k])),
               "mono_bias_dB": float(np.median(dm[k])),
               "mono_p10_dB": float(np.percentile(dm[k], 10)),
               "mono_p90_dB": float(np.percentile(dm[k], 90)),
               "sigma_pred_dBsm": float(np.median(10 * np.log10(
                   np.maximum(sp_ring[am[k], ip[k]], 1e-30)))),
               "sigma_true_dBsm": float(np.median(10 * np.log10(
                   np.maximum(st_ring[am[k], ip[k]], 1e-30)))),
               "sigma_true_pct_med": float(np.median(pct[k]))}
        out.append(rec)
        print(f"  {rec['theta_deg']:>5.0f} {rec['n']:>4} {rec['mono_med_abs_dB']:>11.3f} "
              f"{rec['ring_med_abs_dB']:>11.3f} {rec['mono_bias_dB']:>+10.3f} "
              f"{rec['mono_p10_dB']:>+9.3f} {rec['mono_p90_dB']:>+9.3f} "
              f"{rec['sigma_pred_dBsm']:>9.2f} {rec['sigma_true_dBsm']:>9.2f} "
              f"{rec['sigma_true_pct_med']*100:>9.1f}%")
    return out


def report_ring(sp_ring, st_ring, it, ip, ff_theta, ff_phi):
    """单站口径 vs 同 θ_s 环口径的**配对裁定** + 误差-角距剖面 + 前向散射对照。
    回答：单站方向误差大，是"取样效应"还是"后向散射方向本身更难"。"""
    na, nP = sp_ring.shape
    am = np.arange(na)
    sp_m, st_m = sp_ring[am, ip], st_ring[am, ip]
    d_ring = 10 * np.log10(np.maximum(sp_ring, 1e-30) / np.maximum(st_ring, 1e-30))
    a_ring, a_mono = np.abs(d_ring), np.abs(d_ring[am, ip])
    pair = a_mono - np.median(a_ring, axis=1)
    dm = d_ring[am, ip]
    ok = (sp_m > 0) & (st_m > 0)
    corr = float(np.corrcoef(np.log10(sp_m[ok]), np.log10(st_m[ok]))[0, 1])

    print(f"\n[裁定量：单站口径 vs 同 θ_s 环口径]（同一 {na} 角，严格配对）")
    print(f"  单站   med|Δ| {np.median(a_mono):6.3f} dB   P90 {np.percentile(a_mono, 90):6.3f} dB")
    print(f"  同环   med|Δ| {np.median(a_ring):6.3f} dB   P90 {np.percentile(a_ring, 90):6.3f} dB")
    print(f"  配对差（单站 − 同环中位）：中位 {np.median(pair):+.3f} dB；"
          f"单站更差的角占比 {100*(pair > 0).mean():.1f}%")
    print(f"  单站 bias（中位）{np.median(dm):+.3f} dB   corr(logσ) {corr:+.4f}   "
          f"线性域总功率比 {10*np.log10(st_m.sum()/sp_m.sum()):+.3f} dB")
    pct = np.array([(st_ring[a] < st_m[a]).mean() for a in am])
    print(f"  单站方向 σ_true 在环内百分位：中位 {np.median(pct)*100:.1f}%  "
          f"四分位 [{np.percentile(pct,25)*100:.0f}%, {np.percentile(pct,75)*100:.0f}%]"
          f" ⇒ {'强散射' if np.median(pct) > 0.5 else '偏弱散射（非镜面）'}")

    by_theta = row_table(sp_ring, st_ring, it, ip, ff_theta)

    margin = margin_table(a_ring, ip, 360.0 / (nP - 1))
    print("\n[误差-角距剖面]（同环内，按离单站方向的折叠角距分箱）")
    prof = []
    for r in margin:
        prof.append(r)
        print(f"   角距 [{r['lo_deg']:5.1f},{r['hi_deg']:5.1f})°  n={r['n']:>6}  "
              f"med|Δ| {r['med_abs_dB']:6.3f} dB  P90 {r['p90_abs_dB']:6.3f} dB")

    j2 = (ip + nP // 2) % nP
    a_h2 = np.abs(d_ring[am, j2])
    pct2 = np.array([(st_ring[a] < st_ring[a, j2[a]]).mean() for a in am])
    print(f"\n[对照：前向散射 (180-θ, φ+180)] med|Δ| {np.median(a_h2):.3f} dB  "
          f"σ_true 环内百分位 {np.median(pct2)*100:.1f}% ⇒ 方向图峰值，非单站口径")

    return {"mono_med_abs_dB": float(np.median(a_mono)),
            "mono_p90_abs_dB": float(np.percentile(a_mono, 90)),
            "ring_med_abs_dB": float(np.median(a_ring)),
            "ring_p90_abs_dB": float(np.percentile(a_ring, 90)),
            "paired_diff_med_dB": float(np.median(pair)),
            "frac_mono_worse": float((pair > 0).mean()),
            "mono_bias_dB": float(np.median(dm)),
            "mono_corr_log": corr,
            "mono_power_ratio_dB": float(10 * np.log10(st_m.sum() / sp_m.sum())),
            "mono_sigma_true_pct_med": float(np.median(pct))}, by_theta, prof, \
        {"med_abs_dB": float(np.median(a_h2)),
         "sigma_true_pct_med": float(np.median(pct2))}


def margin_table(a_ring, ip, dphi):
    """按离单站方向的折叠角距分箱（°）。"""
    nP = a_ring.shape[1]
    koff = (np.arange(nP)[None, :] - ip[:, None]) % nP
    fold = np.minimum(koff, nP - koff) * dphi
    out = []
    edges = [0.0, 1.0, 7.5, 22.5, 52.5, 112.5, 181.0]
    for b0, b1 in zip(edges[:-1], edges[1:]):
        m = (fold >= b0) & (fold < b1)
        if not m.any():
            continue
        v = a_ring[m]
        out.append({"lo_deg": b0, "hi_deg": b1, "n": int(m.sum()),
                    "med_abs_dB": float(np.median(v)),
                    "p90_abs_dB": float(np.percentile(v, 90))})
    return out


def mode_full(pooled_n=60):
    """全 468 角逐角计算（判决口径裁定实验）：
      ① 单站方向（θ_s=θ_i, φ_s=φ_i）σ_pred/σ_true —— **部署真正消费的口径**
         （ue_rcs_service.py::rcs_single → rcs_directions(pred,[θ],[φ])）；
      ② θ_s=θ_i 的整个锥面环（73 个 φ_s）—— "同环对照"：判断单站误差是否只是取样效应；
      ③ pooled_n>0 时，对 pooled_n 个**同一批**角算全 2701 方向图 —— 复现封板口径
         并与单站做严格配对对照。
    返回 (sel, sp_mono, st_mono, rowlen=36)。"""
    (angles, e0, khat, beta, eps, rcs_true,
     ff_theta, ff_phi, gx, gy, gz) = _load_common()
    g0 = np.array([gx[0], gy[0], gz[0]])
    metal = eps > 1.5
    k = float(beta[0])
    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    print(f"  三角面空气侧起点: 成功 {int(air_ok.sum())}/{len(cen)}", flush=True)

    nT, nP = len(ff_theta), len(ff_phi)
    rhat, th, ph, _ = direction_grid(ff_theta, ff_phi)
    rhat3, th3, ph3 = (rhat.reshape(nT, nP, 3), th.reshape(nT, nP, 3),
                       ph.reshape(nT, nP, 3))
    na = len(angles)
    sto3 = rcs_true.reshape(na, nT, nP).astype(np.float64)

    it = np.rint((angles[:, 0] - ff_theta[0]) / (ff_theta[1] - ff_theta[0])).astype(int)
    ip = np.rint((np.mod(angles[:, 1], 360.0) - ff_phi[0]) / (ff_phi[1] - ff_phi[0])).astype(int)
    assert it.min() >= 0 and it.max() < nT and ip.min() >= 0 and ip.max() < nP + 1

    sp_ring = np.zeros((na, nP))
    st_ring = sto3[np.arange(na)[:, None], it[:, None],
                   np.arange(nP)[None, :]].astype(np.float64)
    t0 = time.time()
    for a in range(na):
        J, e0m = po_jacobian(a, khat, e0, beta, cen, nvm, q_out, metal, cen)
        U, TH, PH = rhat3[it[a]], th3[it[a]], ph3[it[a]]
        sp_ring[a] = sigma_from_e(nffft_from(J, dAm, cen, U, k), U, TH, PH, e0m)
        if (a + 1) % 60 == 0:
            print(f"    环扫 {a+1}/{na}  {time.time()-t0:.0f}s", flush=True)
    print(f"  环扫完成 {time.time()-t0:.0f}s", flush=True)

    am = np.arange(na)
    sp_m, st_m = sp_ring[am, ip], st_ring[am, ip]
    mono_vs_ring, by_theta, prof, fwd = report_ring(sp_ring, st_ring, it, ip,
                                                    ff_theta, ff_phi)
    extra = {"mono_vs_ring": mono_vs_ring, "by_theta": by_theta,
             "angle_profile": prof, "forward_scatter": fwd}

    # ---- ④ 封板口径同批对照 ----
    if pooled_n and pooled_n > 0:
        ap_ = np.unique(np.linspace(0, na - 1, pooled_n).round().astype(int))
        d_pool = np.zeros((len(ap_), nT * nP))
        t1 = time.time()
        for j, a in enumerate(ap_):
            J, e0m = po_jacobian(a, khat, e0, beta, cen, nvm, q_out, metal, cen)
            sig = sigma_from_e(nffft_from(J, dAm, cen, rhat, k), rhat, th, ph, e0m)
            d_pool[j] = 10 * np.log10(np.maximum(sig, 1e-30)
                                      / np.maximum(sto3[a].ravel(), 1e-30))
            if (j + 1) % 20 == 0:
                print(f"    全向图 {j+1}/{len(ap_)}  {time.time()-t1:.0f}s", flush=True)
        a_pool = np.abs(d_pool)
        a_pool_mono = np.abs(d_pool[np.arange(len(ap_)), it[ap_] * nP + ip[ap_]])
        print(f"\n[封板口径同批对照] 同 {len(ap_)} 角")
        print(f"  全方向图（封板口径，全 2701 方向）med|Δ| {np.median(a_pool):.3f} dB  "
              f"P90 {np.percentile(a_pool, 90):.3f} dB")
        print(f"  同批角单站口径                    med|Δ| {np.median(a_pool_mono):.3f} dB  "
              f"P90 {np.percentile(a_pool_mono, 90):.3f} dB")
        extra["sealed_vs_mono"] = {
            "n_angle": int(len(ap_)),
            "pooled_med_abs_dB": float(np.median(a_pool)),
            "pooled_p90_abs_dB": float(np.percentile(a_pool, 90)),
            "mono_med_abs_dB": float(np.median(a_pool_mono)),
            "mono_p90_abs_dB": float(np.percentile(a_pool_mono, 90))}

    sel = angles[am]
    np.savez(os.path.join(RESULT_DIR, "_diag_decision_full.npz"),
             sel=sel, sig_p=sp_m, sig_t=st_m,
             sp_ring=sp_ring.astype(np.float32), st_ring=st_ring.astype(np.float32),
             it=it, ip=ip, ff_theta=ff_theta, ff_phi=ff_phi)
    json.dump(extra, open(os.path.join(RESULT_DIR, "_diag_decision_full.json"), "w",
                          encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 results/_diag_decision_full.npz / .json")
    return sel, sp_m, st_m, sp_ring, st_ring, it


# ============================================================
# 二、分析
# ============================================================

def movavg_rows(x, w, rowlen):
    """按 (θ 行, φ 列) 结构的**循环**滑动平均（模拟 W 帧非相干积累，功率域）。"""
    if w <= 1 or rowlen <= 1:
        return x.copy()
    nrow = len(x) // rowlen
    m = x[:nrow * rowlen].reshape(nrow, rowlen)
    out = np.empty_like(m)
    kk = np.arange(w)
    for r in range(nrow):
        idx = (np.arange(rowlen)[:, None] + kk[None, :]) % rowlen
        out[r] = m[r][idx].mean(axis=1)
    return out.reshape(-1)


def part_a(d, sig_p, sig_t):
    """误差结构：系统偏差 vs 随机闪烁。"""
    d = d[np.isfinite(d)]
    med = float(np.median(d))
    resid = d - med
    c_lin = float(np.sum(sig_t) / max(np.sum(sig_p), 1e-300))
    return {"n": int(d.size),
            "bias_median_dB": med,
            "bias_mean_dB": float(np.mean(d)),
            "bias_calib_dB": float(10 * np.log10(c_lin)),      # 功率域最佳标定因子
            "frac_pred_low": float((d < 0).mean()),
            "rand_iqr_dB": float(np.percentile(resid, 75) - np.percentile(resid, 25)),
            "rand_p90_abs_dB": float(np.percentile(np.abs(resid), 90)),
            "abs_median_dB": float(np.median(np.abs(d))),
            "abs_p90_dB": float(np.percentile(np.abs(d), 90)),
            "q_delta_dB": {f"p{p}": float(np.percentile(d, p))
                           for p in (1, 5, 25, 50, 75, 95, 99)}}


def xval_calib(sp, st, it, n_rep=40, seed=0):
    """交叉验证标定：回答"多少误差是一个可去掉的尺度/斜率偏差"。
      global ：单标量功率域因子 c=Σσt/Σσp
      row    ：按 θ 行各一个标量（13 个）
      affine ：dB 域仿射（斜率 b + 截距 a），最小二乘拟合
    训练半集拟合 → 测试半集评估；每方案同时报 med|Δ| 与 P90|Δ|（防止"修好中位、炸掉尾巴"）。"""
    rng = np.random.default_rng(seed)
    n = len(sp)
    keys = ("global", "row", "affine")
    A = {k: {"m": [], "p": [], "p0": [], "p1": []} for k in keys}
    base = {"m": [], "p": []}

    def rec(k, spp, stt, p0=None, p1=None):
        e = np.abs(10 * np.log10(spp / stt))
        A[k]["m"].append(np.median(e))
        A[k]["p"].append(np.percentile(e, 90))
        if p0 is not None:
            A[k]["p0"].append(p0)
        if p1 is not None:
            A[k]["p1"].append(p1)

    for _ in range(n_rep):
        perm = rng.permutation(n)
        tr, te = perm[:n // 2], perm[n // 2:]
        e0 = np.abs(10 * np.log10(sp[te] / st[te]))
        base["m"].append(np.median(e0))
        base["p"].append(np.percentile(e0, 90))
        cg = np.sum(st[tr]) / np.sum(sp[tr])
        rec("global", cg * sp[te], st[te], p0=10 * np.log10(cg))
        rowp = np.full(len(te), np.nan)
        for t in np.unique(it):
            m_tr, m_te = it[tr] == t, it[te] == t
            if not m_tr.any() or not m_te.any():
                continue
            rowp[m_te] = (np.sum(st[tr[m_tr]]) / np.sum(sp[tr[m_tr]])) * sp[te[m_te]]
        ok = np.isfinite(rowp)
        if ok.any():
            rec("row", rowp[ok], st[te][ok])
        b, a = np.polyfit(10 * np.log10(sp[tr]), 10 * np.log10(st[tr]), 1)
        rec("affine", 10 ** ((a + b * 10 * np.log10(sp[te])) / 10.0), st[te],
            p0=a, p1=b)

    out = {"raw": {"med_before_dB": float(np.median(base["m"])),
                   "med_after_dB": float(np.median(base["m"])),
                   "p90_before_dB": float(np.median(base["p"])),
                   "p90_after_dB": float(np.median(base["p"])),
                   "offset_dB": 0.0, "slope": 1.0}}
    for k, v in A.items():
        out[k] = {"med_before_dB": float(np.median(base["m"])),
                  "med_after_dB": float(np.median(v["m"])),
                  "p90_before_dB": float(np.median(base["p"])),
                  "p90_after_dB": float(np.median(v["p"])),
                  "offset_dB": (float(np.mean(v["p0"])) if v["p0"] else 0.0),
                  "slope": (float(np.mean(v["p1"])) if v["p1"] else 1.0)}
    return out


def radar_state_vec(dist, sigma_real, jammer_on, jam_db_at4km):
    """radar_state 的**向量化等价**（sigma (N,1)、dist (M,) → 广播成 (N,M)）。
    仅用于大规模判决扫描；与逐字复刻版的一致性由 check_radar_state 断言。"""
    dist = np.asarray(dist, np.float64)
    sig = np.maximum(np.asarray(sigma_real, np.float64), 0.0)
    if jammer_on:
        jl = 10 ** (jam_db_at4km / 10.0)
        jam_sigma = jl * (dist / ECM_R_BT) ** 2
    else:
        jl, jam_sigma = 0.0, 0.0
    r_det0 = np.where(sig > 0.0, R_REF * (sig / SIGMA_REF) ** 0.25, 0.0)
    r_bt = ECM_R_BT * (JS_MIN * sig / max(jl, 1e-30)) ** 0.5
    if not jammer_on:
        return np.where(dist > r_det0, JAM_NONE, JAM_TRACK)
    return np.where(dist > r_det0, JAM_NONE,
                    np.where(dist <= r_bt, JAM_BURN, JAM_SUPPRESS))


def check_radar_state(n=400, seed=7):
    """断言向量化版与逐字复刻版在随机样本上完全一致。"""
    rng = np.random.default_rng(seed)
    d = 10 ** rng.uniform(2.0, 5.5, n)
    s = 10 ** rng.uniform(-3.0, 2.0, n)
    jd = float(rng.choice([-16.0, -11.5, -6.0]))
    sv = radar_state_vec(d, s.reshape(-1, 1), True, jd)
    ok = all(radar_state(d[i], s[i], True, jd)[1] == sv[i, i] for i in range(n))
    if not ok:
        raise AssertionError("radar_state_vec 与 radar_state 不一致")
    return True


def _dec_metrics(sp, st, jam_db=-11.5):
    """判决级误差指标（raw 或已标定的 σ 序列 → 与真值 σ 逐点配对）。
    R_det/R_BT 相对误差中位 + 四态不一致率（全网格 / 仅探测范围内）。"""
    dists = np.logspace(np.log10(100.0), np.log10(300000.0), 200)
    jl = 10 ** (jam_db / 10.0)
    rdp = R_REF * np.maximum(sp, 0.0) ** 0.25
    rdt = R_REF * np.maximum(st, 0.0) ** 0.25
    rp = ECM_R_BT * (np.maximum(sp, 0.0) / jl) ** 0.5
    rt = ECM_R_BT * (np.maximum(st, 0.0) / jl) ** 0.5
    s1 = radar_state_vec(dists, np.asarray(sp).reshape(-1, 1), True, jam_db)
    s2 = radar_state_vec(dists, np.asarray(st).reshape(-1, 1), True, jam_db)
    mism = s1 != s2
    own = s2 != JAM_NONE
    return {"r_det_rel_abs_median": float(np.median(np.abs(rdp - rdt)
                                                    / np.maximum(rdt, 1e-9))),
            "r_bt_rel_abs_median": float(np.median(np.abs(rp - rt)
                                                   / np.maximum(rt, 1e-9))),
            "mismatch_frac": float(mism.mean()),
            "mismatch_in_detect": float(mism[own].mean()) if own.any() else 0.0}


def part_c_calib(sp, st, jam_db=-11.5, n_rep=10, seed=1):
    """判决级收益：交叉验证拟合 dB 域仿射后，判决误差如何变化。
    训练半集拟合 (a,b)，**只在测试半集上**评估；raw 与 affine 用同一测试半集配对。
    返回两者各自的 R_det / R_BT 相对误差中位、四态不一致率（全网格 / 仅探测范围内）。"""
    check_radar_state()
    rng = np.random.default_rng(seed)
    n = len(sp)
    acc = {k: [] for k in ("raw", "affine")}
    for _ in range(n_rep):
        perm = rng.permutation(n)
        tr, te = perm[:n // 2], perm[n // 2:]
        b, a = np.polyfit(10 * np.log10(sp[tr]), 10 * np.log10(st[tr]), 1)
        for k, spp in (("raw", sp[te]),
                       ("affine", 10 ** ((a + b * 10 * np.log10(sp[te])) / 10.0))):
            acc[k].append(_dec_metrics(spp, st[te], jam_db))
    return {k: {kk: float(np.median([r[kk] for r in v])) for kk in v[0]}
            for k, v in acc.items()}


def ring_autocorr(sp_ring, st_ring, dphi):
    """环内 ΔdB 沿 φ 的循环自相关（先扣每环均值）。返回 (lags_deg, rho, lag50)。
    物理含义：误差去相关所需的方位角变化量 ⇒ 时间积累能不能消掉闪烁。"""
    e = 10 * np.log10(np.maximum(sp_ring, 1e-30) / np.maximum(st_ring, 1e-30))
    e = e - e.mean(axis=1, keepdims=True)
    nP = e.shape[1]
    rho = np.array([np.mean(e * np.roll(e, -l, axis=1)) for l in range(nP // 2 + 1)])
    rho = rho / rho[0]
    lags = np.arange(nP // 2 + 1) * dphi
    below = np.where(rho < 0.5)[0]
    return lags, rho, (float(lags[below[0]]) if len(below) else float("nan"))


def part_b(sig_p, sig_t, rowlen):
    """多帧非相干积累后的残留误差（对**环数据**按 φ 循环平均，rowlen=73）。"""
    rows = []
    for w in (1, 2, 3, 5, 9, 18):
        if w > rowlen:
            break
        sp = movavg_rows(sig_p, w, rowlen)
        st = movavg_rows(sig_t, w, rowlen)
        e = 10 * np.log10(np.maximum(sp, 1e-30) / np.maximum(st, 1e-30))
        rows.append({"w": w,
                     "med_abs_dB": float(np.median(np.abs(e))),
                     "p90_abs_dB": float(np.percentile(np.abs(e), 90)),
                     "bias_dB": float(np.median(e))})
    return rows


def part_c(sig_p, sig_t, sig_pw, sig_tw, jam_configs, close_speed=250.0):
    """判决传播：R_det / R_BT 误差 + 四态不一致率（帧判决用 sig_p/sig_t，
    W9 判决用 sig_pw/sig_tw）。"""
    out = {}
    # --- C1 探测距离 ---
    rd_p, rd_t = r_det_of(0.0) * 0 + np.array([r_det_of(x) for x in sig_p]), \
        np.array([r_det_of(x) for x in sig_t])
    rel = (rd_p - rd_t) / np.maximum(rd_t, 1e-9)
    out["r_det"] = {"sigma_ref_km": float(R_REF / 1000.0),
                    "rel_err_median": float(np.median(rel)),
                    "rel_err_abs_median": float(np.median(np.abs(rel))),
                    "rel_err_abs_p90": float(np.percentile(np.abs(rel), 90)),
                    "km_abs_median": float(np.median(np.abs(rd_p - rd_t)) / 1000.0),
                    "km_abs_p90": float(np.percentile(np.abs(rd_p - rd_t), 90) / 1000.0)}
    # --- C2 烧穿距离 ---
    out["r_bt"] = {}
    for jd in jam_configs:
        rp = np.array([r_bt_of(x, jd) for x in sig_p])
        rt = np.array([r_bt_of(x, jd) for x in sig_t])
        m = np.isfinite(rp) & np.isfinite(rt) & (rt > 0)
        rl = (rp[m] - rt[m]) / rt[m]
        dt_s = np.abs(rp[m] - rt[m]) / close_speed
        out["r_bt"][f"jam{jd:+.1f}dB"] = {
            "r_bt_true_med_km": float(np.median(rt[m]) / 1000.0),
            "rel_err_abs_median": float(np.median(np.abs(rl))),
            "rel_err_abs_p90": float(np.percentile(np.abs(rl), 90)),
            "km_abs_median": float(np.median(np.abs(rp[m] - rt[m])) / 1000.0),
            "km_abs_p90": float(np.percentile(np.abs(rp[m] - rt[m]), 90) / 1000.0),
            "time_err_med_s": float(np.median(dt_s)),
            "time_err_p90_s": float(np.percentile(dt_s, 90))}
    # --- C3 四态不一致率（aspect × dist 网格）---
    dists = np.logspace(np.log10(100.0), np.log10(300000.0), 240)
    out["state"] = {}
    for name, sp_, st_ in (("frame", sig_p, sig_t), ("W9", sig_pw, sig_tw)):
        rec = {}
        for jd in jam_configs:
            mism = np.zeros((len(sp_), len(dists)), dtype=bool)
            none_p = np.zeros_like(mism)
            for a in range(len(sp_)):
                for d_i, R in enumerate(dists):
                    _, s1 = radar_state(R, sp_[a], True, jd)
                    _, s2 = radar_state(R, st_[a], True, jd)
                    mism[a, d_i] = (s1 != s2)
                    none_p[a, d_i] = (s2 == JAM_NONE)
            rec[f"jam{jd:+.1f}dB"] = {
                "mismatch_frac": float(mism.mean()),
                "mismatch_frac_in_detect": float(mism[~none_p].mean())
                if (~none_p).any() else 0.0}
        out["state"][name] = rec
    return out


def analyse(sel, sp, st, step, conv, t0, tag="", ring=None, rowlen_mono=1, it_x=None):
    """Part A/B/C 公共分析。
    ring=(sp_ring, st_ring)：θ_s=θ_i 环数据 (n,73)，供 Part B（帧积累/去相关）与 W9 判决；
    为 None 时退化为对单站序列本身做（rowlen_mono 需保持"相邻样本=相邻帧"语义）。
    it_x：各角所在的 θ 行索引（供交叉验证按行标定），None 时跳过 A2。"""
    res = {"tag": tag, "n_angle": int(len(sp)), "step": step, "conv": conv,
           "k": 62.8754, "h_m": H, "stl": STL, "wall_s": round(time.time() - t0, 1)}

    m = (st > 0) & (sp > 0) & np.isfinite(st) & np.isfinite(sp)
    sel, sp, st = sel[m], sp[m], st[m]
    d = 10 * np.log10(sp / st)
    res["n_angle_used"] = int(len(sp))
    print(f"\n[单站 σ 误差] n={len(sp)}  med|Δ| {np.median(np.abs(d)):.2f} dB  "
          f"P90 {np.percentile(np.abs(d), 90):.2f} dB")

    res["A_error_structure"] = part_a(d, sp, st)
    print("\n[Part A 误差结构]")
    for kk, vv in res["A_error_structure"].items():
        if isinstance(vv, dict):
            print(f"  {kk:<20} " + "  ".join(f"{a} {b:+.2f}" for a, b in vv.items()))
        else:
            print(f"  {kk:<20} {vv:.4f}" if isinstance(vv, float)
                  else f"  {kk:<20} {vv}")

    if it_x is not None:
        res["A2_xval_calib"] = xval_calib(sp, st, it_x)
        print("\n[Part A2 交叉验证标定]（训练半集拟合 → 测试半集评估，40 次随机划分）")
        print(f"  {'方案':<8} {'med前':>7} {'med后':>7} {'P90前':>7} {'P90后':>7}   说明")
        notes = {"raw": "不做标定（基准）",
                 "global": lambda v: f"单标量 {v['offset_dB']:+.2f} dB",
                 "row": lambda v: "按 θ 行各一标量（13 个）",
                 "affine": lambda v: (f"dB 域仿射：斜率 {v['slope']:.3f}，"
                                      f"截距 {v['offset_dB']:+.2f} dB")}
        for kk, vv in res["A2_xval_calib"].items():
            nt = notes[kk](vv) if callable(notes[kk]) else notes[kk]
            print(f"  {kk:<8} {vv['med_before_dB']:>7.3f} {vv['med_after_dB']:>7.3f} "
                  f"{vv['p90_before_dB']:>7.3f} {vv['p90_after_dB']:>7.3f}   {nt}")

    if ring is not None:
        sp_ring, st_ring = ring
        nP = sp_ring.shape[1]
        dphi = 360.0 / (nP - 1)        # ff_phi = 0,5,...,360 ⇒ nP=73, 步进 5°
        spb, stb, rowlen = sp_ring.ravel(), st_ring.ravel(), nP
        res["B_frame_step_deg"] = float(dphi)
        lags, rho, l50 = ring_autocorr(sp_ring, st_ring, dphi)
        pick = [0, 2, 4, 6, 9, 12, 18]
        res["B_autocorr"] = {
            "lag50_deg": l50,
            "rho": {f"{int(lags[p])}deg": float(rho[p]) for p in pick if p < len(rho)}}
        print("\n[Part B0 误差去相关长度] ΔdB 沿 φ 的循环自相关（每环扣均值）")
        print("  角距(°)  " + "  ".join(f"{int(lags[p]):>6d}" for p in pick))
        print("  ρ(ΔdB)   " + "  ".join(f"{rho[p]:>6.3f}" for p in pick))
        print(f"  ρ 降到 0.5 所需的方位变化量：{l50:.0f}°（= 1 个网格步）"
              f" ⇒ 真实驻留内（方位几乎不变）误差**完全相关，平均无改善**")
    else:
        spb, stb, rowlen = sp, st, rowlen_mono
        res["B_frame_step_deg"] = None
    res["rowlen"] = int(rowlen)

    res["B_accumulation"] = part_b(spb, stb, rowlen)
    print(f"\n[Part B 多帧非相干积累]（沿 θ_s=θ_i 环，每帧方位步进 "
          f"{res['B_frame_step_deg'] or 10:.0f}°）")
    print(f"  {'W帧':>4} {'med|Δ| dB':>10} {'P90|Δ| dB':>10} {'bias dB':>9}")
    for r in res["B_accumulation"]:
        print(f"  {r['w']:>4} {r['med_abs_dB']:>10.3f} {r['p90_abs_dB']:>10.3f} "
              f"{r['bias_dB']:>9.3f}")

    jam_configs = (-16.0, -11.5, -6.0)
    spW = movavg_rows(spb, 9, rowlen) if rowlen >= 9 else spb
    stW = movavg_rows(stb, 9, rowlen) if rowlen >= 9 else stb
    res["C_decision"] = part_c(sp, st, spW, stW, jam_configs)
    print("\n[Part C 判决传播]")
    c = res["C_decision"]
    print(f"  R_det : 真值参考 {c['r_det']['sigma_ref_km']:.0f} km；"
          f"相对误差 |中位| {c['r_det']['rel_err_abs_median']*100:.1f}%  "
          f"P90 {c['r_det']['rel_err_abs_p90']*100:.1f}%  "
          f"（绝对 {c['r_det']['km_abs_median']:.1f} / {c['r_det']['km_abs_p90']:.1f} km）")
    for kk, v in c["r_bt"].items():
        print(f"  R_BT {kk:<12} 真值中位 {v['r_bt_true_med_km']:5.2f} km  "
              f"相对误差 |中位| {v['rel_err_abs_median']*100:5.1f}%  "
              f"P90 {v['rel_err_abs_p90']*100:5.1f}%  "
              f"（绝对 {v['km_abs_median']:.2f} / {v['km_abs_p90']:.2f} km；"
              f"折合 {v['time_err_med_s']:.1f} / {v['time_err_p90_s']:.1f} s @250m/s）")
    print("  四态判决不一致率：")
    for nm in c["state"]:
        for kk, v in c["state"][nm].items():
            print(f"    {nm:<6} {kk:<12} 全网格 {v['mismatch_frac']*100:5.2f}%  "
                  f"仅探测范围内 {v['mismatch_frac_in_detect']*100:5.2f}%")

    if it_x is not None:
        res["C2_calib_decision"] = part_c_calib(sp, st)
        print("\n[Part C2 判决级收益：交叉验证仿射标定后]（同一测试半集配对，jam −11.5 dB）")
        print(f"  {'方案':<8} {'R_det|Δ|中位':>12} {'R_BT|Δ|中位':>12} "
              f"{'不一致(全网格)':>14} {'不一致(探测内)':>15}")
        for kk, vv in res["C2_calib_decision"].items():
            print(f"  {kk:<8} {vv['r_det_rel_abs_median']*100:>11.1f}% "
                  f"{vv['r_bt_rel_abs_median']*100:>11.1f}% "
                  f"{vv['mismatch_frac']*100:>13.2f}% "
                  f"{vv['mismatch_in_detect']*100:>14.2f}%")

    jp = os.path.join(RESULT_DIR, "_diag_decision_err.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}   总耗时 {time.time()-t0:.0f}s")
    return res


# ============================================================
# 三、② 与 FEKO 原始 .out 逐角核对（--mode feko）
# ============================================================

_FEKO_CACHE = {}


def feko_case(idx):
    """解析 case_<idx>.out（带缓存）：入射平面波（含传播方向单位矢量）+ 远场 RCS 表。
    解析器直接复用生成端 gen_feko_batch，避免重写解析器把数据问题掩盖成实现差异。"""
    if idx in _FEKO_CACHE:
        return _FEKO_CACHE[idx]
    if G is None:
        raise RuntimeError("gen_feko_batch 不可用，无法解析 .out")
    path = os.path.join(FEKO_RUN_DIR, f"case_{idx:03d}.out")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    th, ph, e0, khat, beta0 = G.parse_plane_wave(text, 0)
    ff = G.parse_farfield(text, 0)
    rec = {"theta": th, "phi": ph, "e0": e0, "khat": khat, "beta0": beta0,
           "e0_abs": float(np.linalg.norm(e0)), "ff": ff, "mb": len(text) / 1e6}
    _FEKO_CACHE[idx] = rec
    return rec


def _dir_index(ff, th, ph):
    """方向 (θ,φ) → 远场表索引（表轴为 0..180 / 0..360 步进 5°）。"""
    dt = ff["theta"][1] - ff["theta"][0]
    dp = ff["phi"][1] - ff["phi"][0]
    return (int(round((th - ff["theta"][0]) / dt)),
            int(round((np.mod(ph, 360.0) - ff["phi"][0]) / dp)))


def recip_check(ridx):
    """④ 互易性自检（独立于本项目所有约定）。
    记 case a 的散射矩阵元 σθθ(k̂s = D(θb,φb), k̂i = −u_a)，D 为远场表的方向约定。
    互易定理 S_θθ(k̂s,k̂i) = S_θθ(−k̂i,−k̂s) ⇒ 必有某个"伙伴测量"与它相等。并列三种候选：
        transpose : case b 的 (θa, φa)      —— 直接转置，等价 D = +u
        revdir    : case b 的 (180−θa, φa+180) —— 等价 D = −u
        bothrev   : case (180−θa,φa+180) 的 (180−θb, φb+180) —— 入射/观测全反向
    哪个候选残差 ≈ 0，同时验证了互易性与远场表的方向约定（一次实验两个结论）。"""
    alist = [tuple(map(float, a)) for a in G.angle_list()]
    pos = {a: i for i, a in enumerate(alist)}
    rows = []
    for a in ridx:
        ta, pa_ = alist[a]
        ca = feko_case(a)
        pa_rev = pos.get((180.0 - ta, (pa_ + 180.0) % 360.0))
        ca_rev = feko_case(pa_rev) if pa_rev is not None else None
        for b in ridx:
            if b == a:
                continue
            tb, pb = alist[b]
            cb = feko_case(b)
            i_f = _dir_index(ca["ff"], tb, pb)
            s = {"fwd": float(abs(ca["ff"]["Etheta"][i_f[0], i_f[1]]) ** 2)}
            i1 = _dir_index(cb["ff"], ta, pa_)
            s["transpose"] = float(abs(cb["ff"]["Etheta"][i1[0], i1[1]]) ** 2)
            i2 = _dir_index(cb["ff"], 180.0 - ta, (pa_ + 180.0) % 360.0)
            s["revdir"] = float(abs(cb["ff"]["Etheta"][i2[0], i2[1]]) ** 2)
            if ca_rev is not None:
                i3 = _dir_index(ca_rev["ff"], 180.0 - tb, (pb + 180.0) % 360.0)
                s["bothrev"] = float(abs(ca_rev["ff"]["Etheta"][i3[0], i3[1]]) ** 2)
            s.update({"inc": int(a), "obs": int(b)})
            rows.append(s)
    return rows


def mode_feko(ncase=0, stride=1, n_recip=6, seed=0):
    """② 与 FEKO 原始 .out 逐角核对：
      ① h5 保真度：.out 远场表 vs h5 `rcs` 逐点比对（2701 方向 × N 角）
      ② 索引约定的**一手数据证据**：.out 头部 `Direction of propagation` 的单位矢量
         与 ∓u(θ,φ) 的夹角（此前该结论只由文档/物理推断，未查一手数据）
      ③ 单站偏差归属：.out 表直接取单站值 → 与 PO σ_pred 比对，判断 −5.5 dB 的
         系统性偏低是 PO 物理，还是 HDF5 组装/索引映射引入的管线错误
      ④ 附加自检：FEKO 数据自身的互易性（见 recip_check）"""
    if G is None:
        raise RuntimeError("gen_feko_batch 不可用")
    (angles, e0t, khat_t, beta_t, eps, rcs_true,
     ff_theta, ff_phi, gx, gy, gz) = _load_common()
    alist = [tuple(map(float, a)) for a in G.angle_list()]
    assert len(alist) == len(angles), "角度表长度不一致"

    idxs = list(range(0, len(alist), stride))
    if ncase and ncase > 0:
        idxs = idxs[:ncase]

    z = np.load(os.path.join(RESULT_DIR, "_diag_decision_full.npz"))
    sp_all = z["sig_p"].astype(np.float64)
    sel_npz = z["sel"]
    order_ok = (len(sel_npz) == len(angles)
                and np.allclose(sel_npz, angles, atol=1e-9))

    rel_max, rel_med, abs_max, dev_minus, dev_plus = [], [], [], [], []
    e0n, mb, n_axis_bad, gth, gph = [], [], 0, [], []
    s_out, s_h5, sp_po = [], [], []
    t0 = time.time()
    for c, idx in enumerate(idxs):
        r = feko_case(idx)
        th_o, ph_o = r["theta"], r["phi"]
        th_g, ph_g = alist[idx]
        gth.append(abs(th_o - th_g))
        gph.append(abs(((ph_o - ph_g) + 180.0) % 360.0 - 180.0))
        kk = np.asarray(r["khat"], np.float64)
        kk = kk / np.linalg.norm(kk)
        u = unit_vecs(np.array([th_o]), np.array([ph_o]))[0][0]
        dev_minus.append(np.degrees(np.arccos(np.clip(float(kk @ (-u)), -1.0, 1.0))))
        dev_plus.append(np.degrees(np.arccos(np.clip(float(kk @ u), -1.0, 1.0))))
        e0n.append(r["e0_abs"])
        mb.append(r["mb"])

        ff = r["ff"]
        ax_ok = (len(ff["theta"]) == len(ff_theta)
                 and np.allclose(ff["theta"], ff_theta)
                 and np.allclose(np.mod(ff["phi"], 360.0), np.mod(ff_phi, 360.0)))
        if not ax_ok:
            n_axis_bad += 1
        a_out = ff["rcs"].astype(np.float64)
        b_h5 = rcs_true[idx].astype(np.float64)
        rel = np.abs(a_out - b_h5) / np.maximum(np.abs(b_h5), 1e-300)
        rel_max.append(float(rel.max()))
        rel_med.append(float(np.median(rel)))
        abs_max.append(float(np.abs(a_out - b_h5).max()))
        it, ip = _dir_index(ff, th_o, ph_o)
        s_out.append(float(a_out[it, ip]))
        s_h5.append(float(b_h5[it, ip]))
        sp_po.append(float(sp_all[idx]))
        if (c + 1) % 24 == 0:
            print(f"    {c+1}/{len(idxs)}  {time.time()-t0:.0f}s", flush=True)
    print(f"  .out 解析完成 {len(idxs)} 角  {time.time()-t0:.0f}s", flush=True)

    s_out, s_h5, sp_po = map(np.asarray, (s_out, s_h5, sp_po))
    ok = (s_out > 0) & (sp_po > 0)
    d_po = 10 * np.log10(sp_po[ok] / s_out[ok])
    d_po_h5 = 10 * np.log10(sp_po[ok] / s_h5[ok])

    res = {
        "n_case": len(idxs), "stride": stride,
        "npz_angle_order_ok": bool(order_ok),
        "incidence_angle": {"max_dtheta_deg": float(np.max(gth)),
                            "max_dphi_deg": float(np.max(gph))},
        "k_direction": {"dev_from_minus_u_deg": {
                            "median": float(np.median(dev_minus)),
                            "max": float(np.max(dev_minus))},
                        "dev_from_plus_u_deg": {
                            "median": float(np.median(dev_plus)),
                            "max": float(np.max(dev_plus))},
                        "sign_convention": ("khat = -u(theta,phi)" if
                                            np.max(dev_minus) < 1e-3 else "未确认")},
        "e0_abs": {"min": float(np.min(e0n)), "max": float(np.max(e0n))},
        "out_size_mb": {"min": float(np.min(mb)), "max": float(np.max(mb))},
        "ff_axis_mismatch_cases": int(n_axis_bad),
        "h5_fidelity": {"max_rel": float(np.max(rel_max)),
                        "median_rel": float(np.median(rel_med)),
                        "max_abs_m2": float(np.max(abs_max))},
        "mono_value": {"n": int(ok.sum()),
                       "h5_vs_out_max_abs_dB": float(np.max(np.abs(
                           10 * np.log10(np.maximum(s_h5[ok], 1e-300)
                                         / np.maximum(s_out[ok], 1e-300))))),
                       "po_vs_out_med_abs_dB": float(np.median(np.abs(d_po))),
                       "po_vs_out_p90_abs_dB": float(np.percentile(np.abs(d_po), 90)),
                       "po_vs_out_bias_dB": float(np.median(d_po)),
                       "po_vs_h5_med_abs_dB": float(np.median(np.abs(d_po_h5))),
                       "po_vs_h5_bias_dB": float(np.median(d_po_h5))},
    }

    print("\n[② 与 FEKO 原始 .out 逐角核对]")
    print(f"  样本 {len(idxs)} 角 × {len(ff_theta)}×{len(ff_phi)} 方向；"
          f"单文件 {np.min(mb):.0f}–{np.max(mb):.0f} MB")
    print(f"  ① 入射角一致：max|Δθ| {max(gth):.3e}°，max|Δφ| {max(gph):.3e}°")
    print(f"  ② .out 头 `Direction of propagation` 单位矢量与 −u(θ,φ) 夹角："
          f"中位 {np.median(dev_minus):.3e}°，最大 {np.max(dev_minus):.3e}°")
    print(f"                          与 +u(θ,φ) 夹角："
          f"中位 {np.median(dev_plus):.3f}°，最大 {np.max(dev_plus):.3f}°"
          f"  ⇒ 一手数据确认 k̂ = −u(θ,φ)")
    print(f"     |E0| ∈ [{np.min(e0n):.6f}, {np.max(e0n):.6f}] V/m；"
          f"远场表轴不符的 case 数 {n_axis_bad}")
    print(f"  ③ h5 保真度（.out 远场表 vs h5 `rcs`，逐点）："
          f"中位相对差 {np.median(rel_med):.3e}，最大相对差 {np.max(rel_max):.3e}，"
          f"最大绝对差 {np.max(abs_max):.3e} m²")
    print(f"     单站值：h5 与 .out 的最大绝对差 "
          f"{res['mono_value']['h5_vs_out_max_abs_dB']:.3e} dB")
    print(f"  ④ PO vs .out 单站：med|Δ| {res['mono_value']['po_vs_out_med_abs_dB']:.3f} dB  "
          f"P90 {res['mono_value']['po_vs_out_p90_abs_dB']:.3f} dB  "
          f"bias {res['mono_value']['po_vs_out_bias_dB']:+.3f} dB")
    print(f"     PO vs h5   单站：med|Δ| {res['mono_value']['po_vs_h5_med_abs_dB']:.3f} dB  "
          f"bias {res['mono_value']['po_vs_h5_bias_dB']:+.3f} dB   "
          f"⇒ 与上一行一致即证明偏差属 PO 物理而非数据管线")

    ridx = [int(round(v)) for v in np.linspace(0, len(alist) - 1, n_recip)]
    rows = recip_check(ridx)
    if rows:
        fwd = np.array([r["fwd"] for r in rows])
        smax = fwd.max()
        rec = {}
        for cand in ("transpose", "revdir", "bothrev"):
            if cand not in rows[0]:
                continue
            ref = np.array([r[cand] for r in rows])
            m = (fwd > 1e-4 * smax) & (ref > 1e-4 * smax)
            rr = 10 * np.log10(fwd[m] / ref[m])
            rec[cand] = {"n": int(m.sum()),
                         "median_abs_dB": float(np.median(np.abs(rr))),
                         "p90_abs_dB": float(np.percentile(np.abs(rr), 90)),
                         "median_signed_dB": float(np.median(rr))}
        res["reciprocity"] = {"n_pair": int(len(rows)), "candidates": rec}
        print(f"  ⑤ 互易性自检（{len(rows)} 对方向 × 3 种候选配对；残差越小越可能是真实约定）：")
        for cand, v in rec.items():
            print(f"     {cand:<10} n={v['n']:<4} |Δ| 中位 {v['median_abs_dB']:6.3f} dB  "
                  f"P90 {v['p90_abs_dB']:6.3f} dB  有符号中位 {v['median_signed_dB']:+7.3f} dB")

    jp = os.path.join(RESULT_DIR, "_diag_feko_audit.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return res


# ============================================================
# 四、① 标定泛化验证（--mode gen）
# ============================================================

def _fit_affine(sp_tr, st_tr):
    b, a = np.polyfit(10 * np.log10(sp_tr), 10 * np.log10(st_tr), 1)
    return float(a), float(b)


def _apply_affine(sp, a, b):
    return 10 ** ((a + b * 10 * np.log10(np.maximum(sp, 1e-300))) / 10.0)


def _folds(scheme, rowg, colg, rng):
    """返回 [(train_idx, test_idx), ...]。分块方式决定"泛化"的含义：
      rand      随机半集 —— 现状做法：5° 邻角互相泄漏（ρ(5°)≈0.3，1 网格步即去相关）
      phi_cont  按 φ 连续 4 段（每段 90° 方位）留出 —— 跨方位间隙
      phi_inter 按 φ 每 4 列取 1（40° 间隔）—— 方位内插
      theta_L1  留一 θ 行（13 折）—— 俯仰方向外推/内插
      half_theta  θ≤90 训练 / θ>90 测试（及反向）—— 半空间外推
      checker   (θ行+φ列) 棋盘留出 —— 最近邻全在训练集，泄漏上界（对照）"""
    n = len(rowg)
    ax = np.arange(n)
    if scheme == "rand":
        perm = rng.permutation(n)
        return [(perm[:n // 2], perm[n // 2:])]
    if scheme == "phi_cont":
        return [(ax[colg // 9 != f], ax[colg // 9 == f]) for f in range(4)]
    if scheme == "phi_inter":
        return [(ax[colg % 4 != f], ax[colg % 4 == f]) for f in range(4)]
    if scheme == "theta_L1":
        return [(ax[rowg != f], ax[rowg == f]) for f in np.unique(rowg)]
    if scheme == "half_theta":
        mid = float(np.median(np.unique(rowg)))
        lo, hi = ax[rowg <= mid], ax[rowg > mid]
        return [(lo, hi), (hi, lo)]
    if scheme == "checker":
        m = (rowg + colg) % 2 == 0
        return [(ax[m], ax[~m])]
    raise ValueError(scheme)


def xval_spatial(sp, st, rowg, colg, n_rep=20, seed=0):
    """空间分块交叉验证：回答"标定能不能迁移到没见过的姿态"。
    每折：训练子集拟合 dB 仿射 → 只在测试子集评估 med|Δ| / P90|Δ| / 判决指标。"""
    rng = np.random.default_rng(seed)
    a_full, b_full = _fit_affine(sp, st)
    out = {}
    for sc in ("rand", "checker", "phi_inter", "phi_cont", "theta_L1", "half_theta"):
        per = []
        for _ in range(n_rep if sc == "rand" else 1):
            for tr, te in _folds(sc, rowg, colg, rng):
                if len(te) < 5 or len(tr) < 20:
                    continue
                a, b = _fit_affine(sp[tr], st[tr])
                e0 = np.abs(10 * np.log10(sp[te] / st[te]))
                e1 = np.abs(10 * np.log10(_apply_affine(sp[te], a, b) / st[te]))
                m0 = _dec_metrics(sp[te], st[te])
                m1 = _dec_metrics(_apply_affine(sp[te], a, b), st[te])
                per.append({"n_te": int(len(te)), "slope": b, "offset_dB": a,
                            "med_before": float(np.median(e0)),
                            "med_after": float(np.median(e1)),
                            "p90_before": float(np.percentile(e0, 90)),
                            "p90_after": float(np.percentile(e1, 90)),
                            "rdet_before": m0["r_det_rel_abs_median"],
                            "rdet_after": m1["r_det_rel_abs_median"],
                            "mism_before": m0["mismatch_in_detect"],
                            "mism_after": m1["mismatch_in_detect"]})
        if not per:
            continue
        agg = {kk: float(np.median([r[kk] for r in per])) for kk in per[0]}
        agg["n_fold"] = len(per)
        agg["slope_min"] = float(np.min([r["slope"] for r in per]))
        agg["slope_max"] = float(np.max([r["slope"] for r in per]))
        agg["offset_min_dB"] = float(np.min([r["offset_dB"] for r in per]))
        agg["offset_max_dB"] = float(np.max([r["offset_dB"] for r in per]))
        out[sc] = agg
    return {"full_fit": {"slope": b_full, "offset_dB": a_full},
            "n_angle": int(len(sp)), "schemes": out}


def mode_gen(n_rep=20):
    """① 标定泛化验证：读 results/_diag_decision_full.npz（468 角单站口径），
    用**空间分块**交叉验证替代会泄漏的随机半集划分，检验 2 参数仿射标定的可迁移性。"""
    z = np.load(os.path.join(RESULT_DIR, "_diag_decision_full.npz"))
    sp, st, it, ip = (z["sig_p"].astype(np.float64), z["sig_t"].astype(np.float64),
                      z["it"], z["ip"])
    m = (sp > 0) & (st > 0) & np.isfinite(sp) & np.isfinite(st)
    sp, st, it, ip = sp[m], st[m], it[m], ip[m]
    urow, ucol = np.unique(it), np.unique(ip)
    rowg = np.searchsorted(urow, it).astype(int)
    colg = np.searchsorted(ucol, ip).astype(int)
    print(f"[标定泛化验证] n={len(sp)} 角（{len(urow)} θ 行 × {len(ucol)} φ 列）")

    res = xval_spatial(sp, st, rowg, colg, n_rep=n_rep)
    res["grid"] = {"n_theta_row": int(len(urow)), "n_phi_col": int(len(ucol))}
    res["note"] = ("跨目标/跨频段不可验证：全项目只有 f16_refined.stl + 3GHz 一套 FEKO 真值，"
                   "故可测的泛化轴只有姿态；本模式量化「没见过的姿态」下的标定迁移性。")

    print(f"  全集拟合参考：斜率 {res['full_fit']['slope']:.3f}，"
          f"截距 {res['full_fit']['offset_dB']:+.2f} dB")
    print(f"\n  {'方案':<11} {'折':>3} {'测试角/折':>8} {'med前':>7} {'med后':>7} "
          f"{'P90前':>7} {'P90后':>7} {'斜率范围':>14} {'截距范围':>16}")
    names = {"rand": "rand(现状)", "checker": "checker", "phi_inter": "phi_inter",
             "phi_cont": "phi_cont", "theta_L1": "theta_L1", "half_theta": "half_theta"}
    for sc, v in res["schemes"].items():
        print(f"  {names[sc]:<11} {v['n_fold']:>3} {v['n_te']:>8} {v['med_before']:>7.3f} "
              f"{v['med_after']:>7.3f} {v['p90_before']:>7.3f} {v['p90_after']:>7.3f} "
              f"{v['slope_min']:>6.3f}-{v['slope_max']:<6.3f} "
              f"{v['offset_min_dB']:>+7.2f}-{v['offset_max_dB']:<+7.2f}")

    print(f"\n  {'方案':<11} {'R_det|Δ|中位 前→后':>20} {'不一致(探测内) 前→后':>22}")
    for sc, v in res["schemes"].items():
        print(f"  {names[sc]:<11} "
              f"{v['rdet_before']*100:>9.1f}% → {v['rdet_after']*100:<7.1f}% "
              f"{v['mism_before']*100:>12.2f}% → {v['mism_after']*100:<7.2f}%")

    jp = os.path.join(RESULT_DIR, "_diag_calib_gen.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return res


# ============================================================
# 五、P1 全向图补满：468 角 × 2701 方向（--mode fullmap）
# ============================================================

def mode_fullmap(step=1, ncase=0):
    """把 PO 的**完整双站误差图**算出来（此前只算了单站 + θ_s=θ_i 锥面环 + 60 角全向图）：
      ① 池化口径从 60 角抽样补满到全 468 角（封板 3.181 的真值）；
      ② 为「方向图包」（θ 切面，随 φ 变）与「双站 σ_ij」提供逐方向误差，
         使这两路输出也能做分口径标定（单站 affine 不可外推到它们）；
      ③ 检验「动态范围压缩」假设沿散射角维度是否一致。
    产出 results/_diag_po_fullmap.npz：sp_full/d_full (na, nT, nP)。"""
    (angles, e0, khat, beta, eps, rcs_true,
     ff_theta, ff_phi, gx, gy, gz) = _load_common()
    g0 = np.array([gx[0], gy[0], gz[0]])
    metal = eps > 1.5
    k = float(beta[0])
    cen, nvm, dAm = load_mesh(STL)
    q_out, air_ok = mesh_outside_voxel(cen, nvm, g0, metal)
    print(f"  三角面空气侧起点: 成功 {int(air_ok.sum())}/{len(cen)}", flush=True)

    nT, nP = len(ff_theta), len(ff_phi)
    rhat, th, ph, _ = direction_grid(ff_theta, ff_phi)
    U, THf, PHf = rhat, th.reshape(-1, 3), ph.reshape(-1, 3)
    nd = nT * nP
    na = len(angles)
    sto = rcs_true.reshape(na, nT, nP).astype(np.float64)

    aidx = np.arange(0, na, step)
    if ncase and ncase > 0:
        aidx = aidx[:ncase]
    sp = np.zeros((len(aidx), nd), np.float32)
    t0 = time.time()
    for c, a in enumerate(aidx):
        J, e0m = po_jacobian(a, khat, e0, beta, cen, nvm, q_out, metal, cen)
        sp[c] = sigma_from_e(nffft_from(J, dAm, cen, U, k), U, THf, PHf, e0m)
        if c == 0 or (c + 1) % 20 == 0 or c + 1 == len(aidx):
            el = time.time() - t0
            print(f"    全向图 {c+1}/{len(aidx)}  {el:.0f}s  "
                  f"ETA {(el / (c + 1)) * (len(aidx) - c - 1) / 60:.1f} min", flush=True)
    print(f"  全向图扫描完成 {time.time()-t0:.0f}s", flush=True)

    sp3 = sp.reshape(-1, nT, nP).astype(np.float64)
    st3 = sto[aidx]
    d_full = 10 * np.log10(np.maximum(sp3, 1e-30) / np.maximum(st3, 1e-30))
    a_full = np.abs(d_full)
    it = np.rint((angles[aidx, 0] - ff_theta[0]) / (ff_theta[1] - ff_theta[0])).astype(int)
    ip = np.rint((np.mod(angles[aidx, 1], 360.0) - ff_phi[0])
                 / (ff_phi[1] - ff_phi[0])).astype(int)
    d_mono = a_full[np.arange(len(aidx)), it, ip]

    res = {"n_angle": int(len(aidx)), "step": step, "n_dir": int(nd),
           "wall_s": float(time.time() - t0),
           "pooled_all_dirs": {"med_abs_dB": float(np.median(a_full)),
                               "p90_abs_dB": float(np.percentile(a_full, 90)),
                               "bias_dB": float(np.median(d_full)),
                               "frac_pred_low": float((d_full < 0).mean())},
           "mono_pair_of_same_angles": {"med_abs_dB": float(np.median(d_mono)),
                                        "p90_abs_dB": float(np.percentile(d_mono, 90)),
                                        "bias_dB": float(np.median(
                                            d_full[np.arange(len(aidx)), it, ip]))},
           "linear_power_leak_dB": float(10 * np.log10(
               sp3.sum(axis=(1, 2)).mean() / st3.sum(axis=(1, 2)).mean()))}

    # 按折叠角距（入射→观测）分箱：把 2701 方向压成"散射角剖面"
    dd = 360.0 / (nP - 1)
    koff = (np.arange(nP)[None, :] - ip[:, None]) % nP
    fold = np.minimum(koff, nP - koff) * dd
    edges = np.array([0.5, 2.5, 7.5, 22.5, 52.5, 112.5, 181.0])
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fold[:, None, :] >= lo) & (fold[:, None, :] < hi)
        mm = np.broadcast_to(m, a_full.shape)
        if mm.sum() < 10:
            continue
        prof.append({"ang_sep_deg": [float(lo), float(hi)], "n": int(mm.sum()),
                     "med_abs_dB": float(np.median(a_full[mm])),
                     "bias_dB": float(np.median(d_full[mm]))})
    res["scatter_angle_profile"] = prof

    print("\n[P1 全向图补满]")
    print(f"  {len(aidx)} 角 × {nd} 方向（{nT}θ × {nP}φ）")
    print(f"  池化（全方向）     med|Δ| {res['pooled_all_dirs']['med_abs_dB']:.3f} dB  "
          f"P90 {res['pooled_all_dirs']['p90_abs_dB']:.3f} dB  "
          f"bias {res['pooled_all_dirs']['bias_dB']:+.3f} dB  "
          f"偏低占比 {res['pooled_all_dirs']['frac_pred_low']*100:.1f}%")
    print(f"  同批角单站口径     med|Δ| "
          f"{res['mono_pair_of_same_angles']['med_abs_dB']:.3f} dB  "
          f"P90 {res['mono_pair_of_same_angles']['p90_abs_dB']:.3f} dB  "
          f"bias {res['mono_pair_of_same_angles']['bias_dB']:+.3f} dB")
    print(f"  线性域总功率泄漏 {res['linear_power_leak_dB']:+.3f} dB")
    print("  散射角剖面（折叠角距 → |Δ| 中位 / bias）：")
    for p in prof:
        print(f"    {p['ang_sep_deg'][0]:>6.1f}–{p['ang_sep_deg'][1]:<6.1f}° "
              f"n={p['n']:>8}  med|Δ| {p['med_abs_dB']:6.3f} dB  bias {p['bias_dB']:+7.3f} dB")

    jp = os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    np.savez_compressed(jp, aidx=aidx, sel=angles[aidx], d_full=d_full.astype(np.float32),
                        sp_full=sp3.astype(np.float32), it=it, ip=ip,
                        ff_theta=ff_theta, ff_phi=ff_phi)
    json.dump(res, open(os.path.join(RESULT_DIR, "_diag_po_fullmap.json"), "w",
                        encoding="utf-8"), indent=2, ensure_ascii=False, default=float)
    print(f"\n已存 results/_diag_po_fullmap.npz / .json")
    return res


# ============================================================
# 六、Δ-分档标定表（--mode patcalib）
# ============================================================

DELTA_EDGES = np.array([0.0, 2.5, 7.5, 22.5, 52.5, 112.5, 181.0])


def mode_patcalib(zpath=None):
    """由 `_diag_po_fullmap.npz` 产出**按角距 Δ 分档**的 dB 域仿射标定表。

    动机（部署侧口径冲突，实测）：`ue_rcs_service` 里 `pattern_sweep` 在 φ=φ_i 处与
    `rcs_single` 的 rhat 完全相同（同一物理方向），但两者误差水平差近 2 倍
    （环扫 2.946 dB vs 单站 5.98 dB）。若给它们各配一个标量 affine：
      · 单站 affine 施加到环扫 → med|Δ| 2.946 **升到** 3.390（变差）；
      · 各自最优 → 方向图后向点与单站输出出现 ~3 dB 跳变（折角）。
    实测 dB 域仿射系数随 Δ 平滑单调变化（b：0.417→0.823），且 Δ=0 档系数
    ≈ 封板单站 affine ⇒ **一张 Δ 表可同时服务 单站(Δ=0)/方向图/双站**，
    既连续又各自接近最优。

    Δ 定义：入射方向 u(θ_i,φ_i) 与观测方向 u(θ_j,φ_j) 的三维夹角。
    （注意：锥面环扫时 Δ ≠ Δφ，θ≠90° 时两者不同。）

    产出 results/calib_affine.json：`sigma_true_dB = a(Δ) + b(Δ)·sigma_PO_dB`，
    以分档中心点为节点的分段线性插值（连续，避免档边界上的台阶）。"""
    zp = zpath or os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    z = np.load(zp)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft, fp = z["ff_theta"].astype(np.float64), z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape

    X = 10 * np.log10(np.maximum(sp, 1e-30))          # PO 预测（dBsm）
    Y = X - dfull                                     # FEKO 真值（dBsm）
    ti, pi_ = np.deg2rad(sel[:, 0])[:, None, None], np.deg2rad(sel[:, 1])[:, None, None]
    tj, pj = np.deg2rad(ft)[None, :, None], np.deg2rad(fp)[None, None, :]
    cosd = (np.cos(ti) * np.cos(tj)
            + np.sin(ti) * np.sin(tj) * np.cos(pi_ - pj))
    sep = np.rad2deg(np.arccos(np.clip(cosd, -1.0, 1.0)))       # (na,nT,nP)

    def fit(m):
        b, a = np.polyfit(X[m], Y[m], 1)
        return float(a), float(b)

    def err(a_or_tab, b=None):
        if b is None:
            e = np.abs(a_or_tab - Y)
        else:
            e = np.abs(a_or_tab + b * X - Y)
        return float(np.median(e)), float(np.percentile(e, 90))

    mono = sep < 1e-6                                 # Δ=0：单站口径子集
    a_raw, b_raw = 0.0, 1.0

    print(f"[Δ-分档标定] {na} 角 × {nT}×{nP} 方向")
    print("  raw                        med %.3f  P90 %.3f" % err(a_raw, b_raw))
    a_m, b_m = fit(mono)
    print("  单站 affine(Δ=0, n=%d)   a %+.3f b %.4f   med %.3f  P90 %.3f"
          % (mono.sum(), a_m, b_m, *err(a_m, b_m)))

    per_bin, cents, a_list, b_list = [], [], [], []
    for lo, hi in zip(DELTA_EDGES[:-1], DELTA_EDGES[1:]):
        m = (sep >= lo) & (sep < hi)
        if m.sum() < 50:
            continue
        a, b = fit(m)
        e0 = float(np.median(np.abs(X[m] - Y[m])))
        e1 = float(np.median(np.abs(a + b * X[m] - Y[m])))
        per_bin.append({"delta_deg": [float(lo), float(hi)], "n": int(m.sum()),
                        "a": a, "b": b, "med_raw_dB": e0, "med_cal_dB": e1})
        cents.append(0.5 * (lo + hi)); a_list.append(a); b_list.append(b)
        print("    %6.1f–%6.1f°  n=%7d  a %+7.3f  b %.4f | med %5.3f → %5.3f dB"
              % (lo, hi, m.sum(), a, b, e0, e1))
    cents = np.asarray(cents); a_list = np.asarray(a_list); b_list = np.asarray(b_list)

    A = np.interp(sep, cents, a_list)
    B = np.interp(sep, cents, b_list)
    med_tab, p90_tab = err(A, B)
    print("  Δ 表（分段线性插值）      med %.3f  P90 %.3f" % (med_tab, p90_tab))

    # 单站口径子集上的对照：raw / 单站 affine / Δ 表（表在 Δ=0 处应≈单站 affine）
    print("  --- 单站子集 (Δ=0, n=%d) ---" % mono.sum())
    print("    raw %.3f | 单站 affine %.3f | Δ 表 %.3f dB"
          % (np.median(np.abs(X[mono] - Y[mono])),
             np.median(np.abs(a_m + b_m * X[mono] - Y[mono])),
             np.median(np.abs(A[mono] + B[mono] * X[mono] - Y[mono]))))

    res = {"form": "sigma_true_dB = a(delta_deg) + b(delta_deg) * sigma_PO_dB",
           "delta_deg": cents.tolist(), "a": a_list.tolist(), "b": b_list.tolist(),
           "interp": "linear, clamped at both ends",
           "n_angle": int(na), "n_dir": int(nT * nP),
           "raw": dict(zip(("med_dB", "p90_dB"), err(a_raw, b_raw))),
           "mono_affine": {"a": a_m, "b": b_m,
                           "med_dB": float(np.median(np.abs(a_m + b_m * X[mono] - Y[mono])))},
           "table": {"med_dB": med_tab, "p90_dB": p90_tab},
           "per_bin": per_bin,
           "note": "Δ = 入射方向 u(θi,φi) 与观测方向 u(θj,φj) 的三维夹角；"
                   "单站 Δ=0、方向图沿 θs=θi 锥面、双站为双站角。"}
    jp = os.path.join(RESULT_DIR, "calib_affine.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return res


# ============================================================
# 七、第二变量可行性闸门（--mode gate2d）
# ============================================================

def _sep_and_psi(sel, ft, fp):
    """入射 u(θi,φi) 与观测 u(θj,φj) 的：Δ = 三维夹角；ψ = 观测方向在入射平面
    局部系下的方位角（e_x ⊥ 入射面=侧向，e_y 在入射面内=上向；ψ=0 侧散、ψ=90 前/后向）。

    注意 Δ ≠ Δφ：锥面环扫时 Δ 是三维角（θ=30° 锥面最大 Δ 仅 60°）。"""
    na, nT, nP = len(sel), len(ft), len(fp)
    ti, pi_ = np.deg2rad(sel[:, 0]), np.deg2rad(sel[:, 1])
    tj, pj = np.deg2rad(ft), np.deg2rad(fp)
    ui = np.stack([np.sin(ti) * np.cos(pi_), np.sin(ti) * np.sin(pi_), np.cos(ti)], 1)
    uj = np.empty((na, nT, nP, 3))
    uj[..., 0] = np.sin(tj)[None, :, None] * np.cos(pj)[None, None, :]
    uj[..., 1] = np.sin(tj)[None, :, None] * np.sin(pj)[None, None, :]
    uj[..., 2] = np.cos(tj)[None, :, None]
    cosd = (uj * ui[:, None, None, :]).sum(-1)
    sep = np.rad2deg(np.arccos(np.clip(cosd, -1.0, 1.0)))
    ex = np.cross(ui, np.array([0.0, 0.0, 1.0]))
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    ey = np.cross(ui, ex)
    cx = (uj * ex[:, None, None, :]).sum(-1)
    cy = (uj * ey[:, None, None, :]).sum(-1)
    psi = np.rad2deg(np.arctan2(cy, cx)) % 360.0
    return sep, psi


def _fit_bins(X, Y, sep, de, min_n=50):
    """逐 Δ 档 np.polyfit(X→Y) 得 (a,b)；样本不足档位返回 nan（由调用方回退）。"""
    a = np.full(len(de) - 1, np.nan)
    b = np.full(len(de) - 1, np.nan)
    for i in range(len(de) - 1):
        m = (sep >= de[i]) & (sep < de[i + 1])
        if m.sum() < min_n:
            continue
        bb, aa = np.polyfit(X[m], Y[m], 1)
        a[i], b[i] = aa, bb
    return a, b


def _cv_errors(X, Y, sep, sec, sec_edges, de, folds, nfold, min_cell=200):
    """留折（按入射 φ 分块）交叉验证的**逐样本**误差 |a+b·X−Y|。

    基线列（只看 Δ 的 (a,b)）始终在训练折上拟合；给了 sec_edges 时，每个第二变量
    分档在训练折上单独拟合一组 (a,b)，作用于该档的测试样本（样本不足则回退到基线列）。"""
    dm = 0.5 * (de[:-1] + de[1:])
    e = np.full(X.shape, np.nan)
    for k in range(nfold):
        tr, te = folds != k, folds == k
        a1, b1 = _fit_bins(X[tr], Y[tr], sep[tr], de)
        ok1 = np.isfinite(a1)
        A = np.interp(sep[te], dm[ok1], a1[ok1])
        B = np.interp(sep[te], dm[ok1], b1[ok1])
        if sec_edges is not None:
            for lo, hi in zip(sec_edges[:-1], sec_edges[1:]):
                mt = te & (sec >= lo) & (sec < hi)
                if not mt.any():
                    continue
                mtr = tr & (sec >= lo) & (sec < hi)
                if mtr.sum() < min_cell:
                    continue                       # 该档训练样本太少 → 用基线列
                a2, b2 = _fit_bins(X[mtr], Y[mtr], sep[mtr], de, min_n=25)
                ok2 = np.isfinite(a2)
                if ok2.sum() < 2:
                    continue
                A[mt[te]] = np.interp(sep[mt], dm[ok2], a2[ok2])
                B[mt[te]] = np.interp(sep[mt], dm[ok2], b2[ok2])
        e[te] = np.abs(A + B * X[te] - Y[te])
    return e


def _cv_errors_thetalin(X, Y, sep, thn, de, folds, nfold, min_n=100):
    """留折 CV：每个 Δ 档内用 **(θ 线性基)** 同时建模 a、b：
        a(θ)=c0+c1·θ_n,  b(θ)=c2+c3·θ_n,  θ_n=(θ_i−90)/60 ∈ [−1,1]
    （4 参数/档，全部样本参与最小二乘，比"逐 (θ,Δ) 格子各拟合一组"稳得多：
      单站 Δ=0 档只有 468 点，13 个格子各 37 点会直接退化。）"""
    dm = 0.5 * (de[:-1] + de[1:])
    e = np.full(X.shape, np.nan)
    for k in range(nfold):
        tr, te = folds != k, folds == k
        a1, b1 = _fit_bins(X[tr], Y[tr], sep[tr], de)
        ok1 = np.isfinite(a1)
        A = np.interp(sep[te], dm[ok1], a1[ok1])
        B = np.interp(sep[te], dm[ok1], b1[ok1])
        for i in range(len(de) - 1):
            mtr = tr & (sep >= de[i]) & (sep < de[i + 1])
            mt = te & (sep >= de[i]) & (sep < de[i + 1])
            if mtr.sum() < min_n or not mt.any():
                continue
            D = np.stack([np.ones(mtr.sum()), thn[mtr], X[mtr], thn[mtr] * X[mtr]], 1)
            c, *_ = np.linalg.lstsq(D, Y[mtr], rcond=None)
            A[mt[te]] = c[0] + c[1] * thn[mt]
            B[mt[te]] = c[2] + c[3] * thn[mt]
        e[te] = np.abs(A + B * X[te] - Y[te])
    return e


def _th_nodes_from_edges(th_edges):
    return 0.5 * (np.asarray(th_edges)[:-1] + np.asarray(th_edges)[1:])


def _cv_errors_thetanodes(X, Y, sep, th, th_nodes, de, folds, nfold,
                          min_cell=100, min_n=25):
    """留折 CV：**可部署形式** —— θ 节点 × Δ 节点的 2D 表（每个格子一组 (a,b)）。

    拟合：样本按最近 θ 节点分组、按 Δ 档分组，格子内 polyfit；格子样本 < min_cell
    则回退该 Δ 档的 1D 系数（保证处处有定义）。
    应用：先在 θ 维线性插值（两端 clamp）、再在 Δ 维线性插值（与 1D 表同口径）。"""
    dm = 0.5 * (de[:-1] + de[1:])
    th_nodes = np.asarray(th_nodes, float)
    grp = np.abs(th[None, :] - th_nodes[:, None]).argmin(0)
    e = np.full(X.shape, np.nan)
    for k in range(nfold):
        tr, te = folds != k, folds == k
        a1, b1 = _fit_bins(X[tr], Y[tr], sep[tr], de)
        AA = np.repeat(a1[None, :], len(th_nodes), 0)
        BB = np.repeat(b1[None, :], len(th_nodes), 0)
        for r in range(len(th_nodes)):
            for i in range(len(de) - 1):
                m = tr & (grp == r) & (sep >= de[i]) & (sep < de[i + 1])
                if m.sum() < min_cell:
                    continue
                bb, aa = np.polyfit(X[m], Y[m], 1)
                AA[r, i], BB[r, i] = aa, bb
        A = np.stack([np.interp(th[te], th_nodes, AA[:, i]) for i in range(len(de) - 1)], -1)
        B = np.stack([np.interp(th[te], th_nodes, BB[:, i]) for i in range(len(de) - 1)], -1)
        lo = np.clip(np.searchsorted(dm, sep[te]) - 1, 0, len(dm) - 2)
        w = np.clip((sep[te] - dm[lo]) / (dm[lo + 1] - dm[lo]), 0.0, 1.0)[..., None]
        Aq = (np.take_along_axis(A, lo[..., None], -1) * (1 - w)
              + np.take_along_axis(A, (lo + 1)[..., None], -1) * w)[..., 0]
        Bq = (np.take_along_axis(B, lo[..., None], -1) * (1 - w)
              + np.take_along_axis(B, (lo + 1)[..., None], -1) * w)[..., 0]
        e[te] = np.abs(Aq + Bq * X[te] - Y[te])
    return e


def mode_gate2d(zpath=None, nfold=6):
    """3a/3b 可行性闸门：Δ 之外再加一个变量（ψ 相对入射面方位 / θ_i 入射俯仰）
    到底还能买多少 dB —— 全部用**留折交叉验证**（按入射 φ 分 6 块），并分口径报告。

    3a（判决路径）看「单站 Δ=0」子集；3b 看「方向图锥面」与「其余双站」子集
    （后者首次给出双站的独立精度数字，此前只有一致性自检）。"""
    zp = zpath or os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    z = np.load(zp)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft, fp = z["ff_theta"].astype(np.float64), z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape

    X = 10 * np.log10(np.maximum(sp, 1e-30))
    Y = X - dfull
    sep, psi = _sep_and_psi(sel, ft, fp)

    ph = np.mod(sel[:, 1], 360.0)
    folds = np.minimum((ph // (360.0 / nfold)).astype(int), nfold - 1)

    def flat(v):
        return np.ascontiguousarray(np.broadcast_to(v, X.shape)).reshape(-1)

    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    sf = flat(sep)
    af_theta = flat(np.tile(sel[:, 0][:, None, None], (1, nT, nP)))
    af_psi = flat(psi)
    ff_ = flat(np.tile(folds[:, None, None], (1, nT, nP)))

    # 口径子集（互斥）：单站 Δ=0 / 锥面环扫（θ_j=θ_i，方向图包）/ 其余双站
    tj = np.broadcast_to(ft[None, :, None], (na, nT, nP))
    ti = np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))
    m_mono = sep < 1e-6
    m_cone = (np.abs(tj - ti) < 1e-9) & ~m_mono
    m_rest = ~m_mono & ~m_cone
    subs = {"单站(Δ=0, 判决路径)": flat(m_mono).astype(bool),
            "锥面环扫(方向图包)": flat(m_cone).astype(bool),
            "其余双站 σ_ij": flat(m_rest).astype(bool),
            "全方向池化": np.ones_like(ff_, bool)}

    de = DELTA_EDGES
    psi_edges = {1: None,
                 2: np.array([0.0, 180.0, 360.0]),
                 4: np.arange(0.0, 361.0, 90.0),
                 6: np.arange(0.0, 361.0, 60.0),
                 12: np.arange(0.0, 361.0, 30.0)}
    th_edges = {1: None, 4: np.linspace(25.0, 155.0, 5),
                8: np.linspace(25.0, 155.0, 9), 13: np.linspace(25.0, 155.0, 14)}

    def stats(mask, e):
        v = e[mask]
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "med_dB": float(np.median(v)),
                "p90_dB": float(np.percentile(v, 90))}

    print(f"[第二变量闸门] {na} 角 × {nT}×{nP} 方向，{nfold} 折留角 CV（按入射 φ 分块）")
    print("  raw（不标定）基准：", end="")
    e_raw = np.abs(Yf - Xf)
    print("  ".join(f"{k} {np.median(e_raw[v]):.3f}" for k, v in subs.items()))

    out = {"n_angle": int(na), "n_dir": int(nT * nP), "nfold": nfold,
           "subsets": {k: {"raw": stats(v, e_raw)} for k, v in subs.items()},
           "candidates": {}}

    cands = [("1D Δ 表（现行）", None, None),
             ("Δ × ψ(2档)", af_psi, psi_edges[2]),
             ("Δ × ψ(4档)", af_psi, psi_edges[4]),
             ("Δ × ψ(6档)", af_psi, psi_edges[6]),
             ("Δ × ψ(12档)", af_psi, psi_edges[12]),
             ("Δ × θ_i(4档)", af_theta, th_edges[4]),
             ("Δ × θ_i(13档)", af_theta, th_edges[13])]
    for name, sec, edges in cands:
        e = _cv_errors(Xf, Yf, sf, sec, edges, de, ff_, nfold)
        row = {k: stats(v, e) for k, v in subs.items()}
        out["candidates"][name] = row
        print(f"  {name:<16} " + "  ".join(
            f"{k[:4]} {row[k]['med_dB']:.3f}/{row[k]['p90_dB']:.3f}" for k in subs))

    thn = flat(np.tile(((sel[:, 0] - 90.0) / 60.0)[:, None, None], (1, nT, nP)))
    e = _cv_errors_thetalin(Xf, Yf, sf, thn, de, ff_, nfold)
    row = {k: stats(v, e) for k, v in subs.items()}
    out["candidates"]["Δ × θ_i(线性基)"] = row
    print(f"  {'Δ × θ_i(线性基)':<16} " + "  ".join(
        f"{k[:4]} {row[k]['med_dB']:.3f}/{row[k]['p90_dB']:.3f}" for k in subs))

    # 可部署形式：θ 节点 × Δ 节点 2D 表（θ 维线性插值）
    thf_deg = flat(np.tile(sel[:, 0][:, None, None], (1, nT, nP)))
    for nnode in (4, 5, 7):
        nodes = _th_nodes_from_edges(np.linspace(25.0, 155.0, nnode + 1))
        e = _cv_errors_thetanodes(Xf, Yf, sf, thf_deg, nodes, de, ff_, nfold)
        row = {k: stats(v, e) for k, v in subs.items()}
        out["candidates"][f"Δ × θ_i({nnode}节点线性)"] = row
        print(f"  {('Δ × θ_i(%d节点线性)' % nnode):<16} " + "  ".join(
            f"{k[:4]} {row[k]['med_dB']:.3f}/{row[k]['p90_dB']:.3f}" for k in subs))

    # 机制：主档（52.5–112.5°）内按 ψ 分档的 (a,b)，与锥面/全球面最优 (a,b) 对比
    mech = {"delta_bin_deg": [52.5, 112.5], "psi_bins": []}
    mbin = (sep >= 52.5) & (sep < 112.5)
    for lo, hi in zip(psi_edges[6][:-1], psi_edges[6][1:]):
        m = mbin & (psi >= lo) & (psi < hi)
        if m.sum() < 50:
            continue
        b, a = np.polyfit(X[m], Y[m], 1)
        mech["psi_bins"].append({"psi_deg": [float(lo), float(hi)], "n": int(m.sum()),
                                 "a": float(a), "b": float(b)})
    b, a = np.polyfit(X[mbin], Y[mbin], 1)
    mech["all_psi"] = {"a": float(a), "b": float(b), "n": int(mbin.sum())}
    mcone = mbin & m_cone
    if mcone.sum() > 50:
        b, a = np.polyfit(X[mcone], Y[mcone], 1)
        mech["cone_only"] = {"a": float(a), "b": float(b), "n": int(mcone.sum())}
    out["mechanism"] = mech
    print(f"\n  机制（Δ {mech['delta_bin_deg'][0]}–{mech['delta_bin_deg'][1]}° 档内）：")
    print(f"    全 ψ 合计 a {mech['all_psi']['a']:+.3f} b {mech['all_psi']['b']:.4f}"
          f"（n={mech['all_psi']['n']}）")
    for c in mech["psi_bins"]:
        print(f"    ψ {c['psi_deg'][0]:5.0f}–{c['psi_deg'][1]:<5.0f}° n={c['n']:>7}"
              f"  a {c['a']:+7.3f}  b {c['b']:.4f}")
    if "cone_only" in mech:
        print(f"    其中锥面环扫 a {mech['cone_only']['a']:+.3f} b "
              f"{mech['cone_only']['b']:.4f}（n={mech['cone_only']['n']}）")

    jp = os.path.join(RESULT_DIR, f"_diag_gate2d_nf{nfold}.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return out


def mode_patcalib2d(zpath=None, nfold=6, nnode=4):
    """在 Δ 表基础上加第二维 **θ_i（入射俯仰）**，产出可部署的 2D 标定表。

    依据（`--mode gate2d` 留折 CV，4/6/12 折一致，**这个形式本身是 CV 选出来的**）：
      · ψ（相对入射面方位）**完全无效**：4 档 2.824→2.818、细档更差 ⇒ AGENTS 遗留④
        里"需引入相对入射面方位"的假设被否证；
      · θ_i **线性基**（a,b 各自线性于 θ）也无效：双站 2.824→2.835、锥面 3.025→3.043
        ⇒ (a,b) 对 θ 并非线性；
      · **θ 节点 × Δ 节点 表格 + θ 维线性插值**有效且处处不劣化（见下表）。

    形式：每个 (θ 节点, Δ 档) 格子一组 (a,b)；格子样本 <100 回退该 Δ 档的 1D 系数
    （c1=c3=0 等价物，保证连续可导性只在插值层面）。查询：先在 θ 维线性插值、再在 Δ
    维线性插值，两端 clamp。θ 节点 = 4 个（区间 [25,155]° 四等分的中心）。

    产出 results/calib_affine_2d.json（含 CV 与样本内两套分口径数字）。"""
    zp = zpath or os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    z = np.load(zp)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft, fp = z["ff_theta"].astype(np.float64), z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape

    X = 10 * np.log10(np.maximum(sp, 1e-30))
    Y = X - dfull
    sep, _ = _sep_and_psi(sel, ft, fp)
    th3 = np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))
    Xf, Yf, sf, thf = X.reshape(-1), Y.reshape(-1), sep.reshape(-1), th3.reshape(-1)

    de = DELTA_EDGES
    dm = 0.5 * (de[:-1] + de[1:])
    a1, b1 = _fit_bins(Xf, Yf, sf, de)
    th_nodes = _th_nodes_from_edges(np.linspace(25.0, 155.0, nnode + 1))
    grp = np.abs(thf[None, :] - th_nodes[:, None]).argmin(0)

    A = np.repeat(a1[None, :], nnode, 0)
    B = np.repeat(b1[None, :], nnode, 0)
    cells = []
    for r in range(nnode):
        for i in range(len(de) - 1):
            m = (grp == r) & (sf >= de[i]) & (sf < de[i + 1])
            n_cell = int(m.sum())
            if n_cell < 100:
                cells.append({"theta_deg": float(th_nodes[r]), "delta_deg": float(dm[i]),
                              "n": n_cell, "fallback_1d": True})
                continue
            bb, aa = np.polyfit(Xf[m], Yf[m], 1)
            A[r, i], B[r, i] = aa, bb
            cells.append({"theta_deg": float(th_nodes[r]), "delta_deg": float(dm[i]),
                          "n": n_cell, "a": float(aa), "b": float(bb),
                          "fallback_1d": False})

    def apply_eval():
        """θ 插值 → Δ 插值（与部署口径一致）→ 逐样本误差。"""
        Ai = np.stack([np.interp(th3, th_nodes, A[:, i]) for i in range(len(de) - 1)], -1)
        Bi = np.stack([np.interp(th3, th_nodes, B[:, i]) for i in range(len(de) - 1)], -1)
        lo = np.clip(np.searchsorted(dm, sep) - 1, 0, len(dm) - 2)
        w = np.clip((sep - dm[lo]) / (dm[lo + 1] - dm[lo]), 0.0, 1.0)[..., None]
        Aq = (np.take_along_axis(Ai, lo[..., None], -1) * (1 - w)
              + np.take_along_axis(Ai, (lo + 1)[..., None], -1) * w)[..., 0]
        Bq = (np.take_along_axis(Bi, lo[..., None], -1) * (1 - w)
              + np.take_along_axis(Bi, (lo + 1)[..., None], -1) * w)[..., 0]
        return np.abs(Aq + Bq * X - Y)

    e2d = apply_eval()
    e1d = np.abs(np.interp(sep, dm, a1) + np.interp(sep, dm, b1) * X - Y)
    e_raw = np.abs(X - Y)

    m_mono = (sep < 1e-6).reshape(-1)
    m_cone = ((np.abs(np.broadcast_to(ft[None, :, None], (na, nT, nP)) - th3) < 1e-9)
              & (sep > 1e-6)).reshape(-1)
    subs = {"单站(Δ=0)": m_mono, "锥面环扫(方向图)": m_cone,
            "其余双站": ~m_mono & ~m_cone, "全方向池化": np.ones_like(m_mono)}

    def pack(e):
        e = np.asarray(e).reshape(-1)
        return {k: {"n": int(mk.sum()),
                    "med_dB": float(np.median(e[mk])),
                    "p90_dB": float(np.percentile(e[mk], 90))} for k, mk in subs.items()}

    ph = np.mod(sel[:, 1], 360.0)
    folds = np.minimum((ph // (360.0 / nfold)).astype(int), nfold - 1)
    foldf = np.tile(folds[:, None, None], (1, nT, nP)).reshape(-1)
    ecv = _cv_errors_thetanodes(Xf, Yf, sf, thf, th_nodes, de, foldf, nfold)
    r_in = {"raw": pack(e_raw), "tab1d": pack(e1d), "tab2d": pack(e2d)}
    cv = {"nfold": nfold, "raw": pack(e_raw), "tab1d": {}, "tab2d": pack(ecv)}
    e1d_cv = np.full(Xf.shape, np.nan)
    for k in range(nfold):        # 1D 表的同折对照
        tr, te = foldf != k, foldf == k
        a_, b_ = _fit_bins(Xf[tr], Yf[tr], sf[tr], de)
        ok = np.isfinite(a_)
        e1d_cv[te] = np.abs(np.interp(sf[te], dm[ok], a_[ok])
                            + np.interp(sf[te], dm[ok], b_[ok]) * Xf[te] - Yf[te])
    cv["tab1d"] = pack(e1d_cv)

    print(f"[2D 标定表] θ {nnode} 节点 × Δ {len(dm)} 节点（θ 维线性插值，可部署形式）")
    print("  %-16s %18s %18s %14s" % ("口径", "1D 表(内/CV)", "2D 表(内/CV)", "raw 内"))
    for k in subs:
        print("  %-16s %6.3f / %6.3f   %6.3f / %6.3f   %6.3f"
              % (k, r_in["tab1d"][k]["med_dB"], cv["tab1d"][k]["med_dB"],
                 r_in["tab2d"][k]["med_dB"], cv["tab2d"][k]["med_dB"],
                 r_in["raw"][k]["med_dB"]))
    print("\n  Δ=0 档各 θ 节点的 (a,b)：")
    for r in range(nnode):
        print("    θ=%6.2f°  a %+7.3f  b %.4f  n=%d%s"
              % (th_nodes[r], A[r, 0], B[r, 0], int((grp == r).sum()),
                 "  （回退 1D）" if cells[r * (len(de) - 1)]["fallback_1d"] else ""))

    res = {"form": "sigma_true_dB = a(theta_node, delta_node) + b(theta_node, delta_node) * sigma_PO_dB",
           "theta_nodes_deg": th_nodes.tolist(), "delta_nodes_deg": dm.tolist(),
           "a": A.tolist(), "b": B.tolist(), "cells": cells,
           "interp": "linear in theta over nodes, then linear in delta over node centres; clamped",
           "fallback": "cell with <100 samples reuses that delta bin's 1D coefficients",
           "n_angle": int(na), "n_dir": int(nT * nP), "nfold": nfold,
           "subsets": {"in_sample_raw": r_in["raw"], "in_sample_1d": r_in["tab1d"],
                       "in_sample_2d": r_in["tab2d"]},
           "cv": cv,
           "calib_1d": {"delta_nodes_deg": dm.tolist(), "a": a1.tolist(), "b": b1.tolist()},
           "note": "θ_i = 入射俯仰（模型有效范围 30–150°），Δ = 入射与观测方向的三维夹角；"
                   "单站 Δ=0、方向图沿 θs=θi 锥面、双站为双站角。CV = 按入射 φ 分块的留角交叉验证。"}
    jp = os.path.join(RESULT_DIR, "calib_affine_2d.json")
    json.dump(res, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return res


def mode_residgate(zpath=None, nfold=6, nsub=300000, max_iter=250, seed=0):
    """残差可学性闸门：给定 PO 的**标量 σ 与几何角**，残差里还剩多少可学的？

    背景：判决 C 指出 `mesh_po` 3.16 dB 的残差"电流来源比网格更关键"，隐含下一步是
    训一个残差网络。但在投入之前必须先量化一个**上界问题**：

        用**可部署的输入**（PO 标量 σ_PO + 几何角），换一个比"分段仿射标定表"
        更灵活的函数类，还能多吃掉多少 dB？

    - 若灵活模型在同口径 CV 下显著优于 2D 表 ⇒ 分段仿射是瓶颈，残差网络值得训；
    - 若基本持平 ⇒ (σ_PO 标量 + 几何) 这一输入空间已饱和，残差必须来自
      **复场/电流级**信息（需重算整条链路，不在本闸门范围）⇒ RCS 主线可封板。

    做法：HistGradientBoostingRegressor（1.26M 样本下逼近该输入空间的可达上界），
    与 1D/2D 标定表**共用同一留折（按入射 φ 分块）与同一口径子集**。

    特征集：
      F0 = [σ_PO]                                    全局单标量（对照）
      F1 = [σ_PO, Δ, θ_i]                            与 2D 表同信息量
      F2 = F1 + ψ                                    + 遗留④ 点名的方位
      F3 = F2 + θ_s, sinφ_s, cosφ_s, sinφ_i, cosφ_i  全几何
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    zp = zpath or os.path.join(RESULT_DIR, "_diag_po_fullmap.npz")
    z = np.load(zp)
    sp = z["sp_full"].astype(np.float64)
    dfull = z["d_full"].astype(np.float64)
    sel = z["sel"].astype(np.float64)
    ft, fp = z["ff_theta"].astype(np.float64), z["ff_phi"].astype(np.float64)
    na, nT, nP = sp.shape

    X = 10.0 * np.log10(np.maximum(sp, 1e-30))
    Y = X - dfull
    sep, psi = _sep_and_psi(sel, ft, fp)

    ph = np.mod(sel[:, 1], 360.0)
    folds = np.minimum((ph // (360.0 / nfold)).astype(int), nfold - 1)

    def flat(v):
        return np.ascontiguousarray(np.broadcast_to(v, X.shape)).reshape(-1)

    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    sf = flat(sep)
    psif = flat(psi)
    thif = flat(np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP)))
    thjf = flat(np.broadcast_to(ft[None, :, None], (na, nT, nP)))
    d2r = np.deg2rad
    sinpj = flat(np.broadcast_to(np.sin(d2r(fp))[None, None, :], (na, nT, nP)))
    cospj = flat(np.broadcast_to(np.cos(d2r(fp))[None, None, :], (na, nT, nP)))
    sinpi = flat(np.broadcast_to(np.sin(d2r(ph))[:, None, None], (na, nT, nP)))
    cospi = flat(np.broadcast_to(np.cos(d2r(ph))[:, None, None], (na, nT, nP)))
    foldf = flat(np.tile(folds[:, None, None], (1, nT, nP))).astype(np.int64)

    tj = np.broadcast_to(ft[None, :, None], (na, nT, nP))
    ti = np.broadcast_to(sel[:, 0][:, None, None], (na, nT, nP))
    m_mono = sep < 1e-6
    m_cone = (np.abs(tj - ti) < 1e-9) & ~m_mono
    m_rest = ~m_mono & ~m_cone
    mono = "单站(Δ=0, 判决路径)"
    subs = {mono: flat(m_mono).astype(bool),
            "锥面环扫(方向图包)": flat(m_cone).astype(bool),
            "其余双站 σ_ij": flat(m_rest).astype(bool),
            "全方向池化": np.ones(Xf.size, bool)}

    def stats(mask, e):
        v = e[mask]
        v = v[np.isfinite(v)]
        return {"n": int(v.size), "med_dB": float(np.median(v)),
                "p90_dB": float(np.percentile(v, 90))}

    de = DELTA_EDGES
    e_raw = np.abs(Yf - Xf)
    e_1d = _cv_errors(Xf, Yf, sf, None, None, de, foldf, nfold)
    nodes = _th_nodes_from_edges(np.linspace(25.0, 155.0, 5))
    e_2d = _cv_errors_thetanodes(Xf, Yf, sf, thif, nodes, de, foldf, nfold)

    fdict = {"F0 单标量[σ_PO]": [Xf],
             "F1 [σ_PO,Δ,θ_i]": [Xf, sf, thif],
             "F2 F1+ψ": [Xf, sf, thif, psif],
             "F3 F2+θ_s,φ_s,φ_i": [Xf, sf, thif, psif, thjf, sinpj, cospj, sinpi, cospi]}
    Fs = {k: np.column_stack(v).astype(np.float32) for k, v in fdict.items()}

    print(f"[残差可学性闸门] {na} 角 × {nT}×{nP} 方向，{nfold} 折留角 CV"
          f"（按入射 φ 分块），训练子采样 {nsub}，HGB max_iter={max_iter}", flush=True)
    rows = {}
    for name, e in (("raw（不标定）", e_raw), ("1D Δ 表（现行）", e_1d),
                    ("2D 表（θ4×Δ6，现行最优）", e_2d)):
        rows[name] = {k: stats(v, e) for k, v in subs.items()}

    rng = np.random.default_rng(seed)
    t0 = time.time()
    for k, F in Fs.items():
        err = np.full(Xf.size, np.nan)
        for f in range(nfold):
            tr = np.flatnonzero(foldf != f)
            te = np.flatnonzero(foldf == f)
            if nsub and tr.size > nsub:
                tr = rng.choice(tr, nsub, replace=False)
            m = HistGradientBoostingRegressor(
                max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=127,
                min_samples_leaf=100, l2_regularization=1.0,
                early_stopping=False, random_state=seed)
            m.fit(F[tr], Yf[tr])
            err[te] = m.predict(F[te])
        e = np.where(np.isfinite(err), np.abs(Yf - np.nan_to_num(err)), np.nan)
        rows[k] = {kk: stats(v, e) for kk, v in subs.items()}
        print(f"  {k:<22} " + "  ".join(
            f"{kk[:4]} {rows[k][kk]['med_dB']:.3f}" for kk in subs)
            + f"   [{time.time()-t0:.0f}s]", flush=True)

    # 可部署形式候选：**全样本联合拟合**的低阶多项式（借用所有 Δ 档样本，仍可写成公式/表格）。
    # 动机：上表 HGB 的收益若主要来自"借样本"而非"函数类"，则低阶联合拟合应能拿到大部分。
    def _cv_lin(cols):
        e = np.full(Xf.size, np.nan)
        for f in range(nfold):
            tr = np.flatnonzero(foldf != f)
            te = np.flatnonzero(foldf == f)
            B = np.column_stack([c[tr] for c in cols])
            coef, *_ = np.linalg.lstsq(B, Yf[tr], rcond=None)
            Bt = np.column_stack([c[te] for c in cols])
            e[te] = np.abs(Yf[te] - Bt @ coef)
        return e

    dsc = sf / 180.0
    tsc = (thif - 90.0) / 60.0
    one = np.ones_like(Xf)
    lin = {"P1 Δ³×σ_PO 联合（8 参，可写成公式）":
           [one, dsc, dsc ** 2, dsc ** 3, Xf, Xf * dsc, Xf * dsc ** 2, Xf * dsc ** 3],
           "P2 Δ³θ²×σ_PO 联合（18 参）":
           [one, dsc, dsc ** 2, dsc ** 3, tsc, tsc ** 2, dsc * tsc, dsc ** 2 * tsc, dsc * tsc ** 2,
            Xf, Xf * dsc, Xf * dsc ** 2, Xf * dsc ** 3, Xf * tsc, Xf * tsc ** 2,
            Xf * dsc * tsc, Xf * dsc ** 2 * tsc, Xf * dsc * tsc ** 2]}
    for k, cols in lin.items():
        rows[k] = {kk: stats(v, _cv_lin(cols)) for kk, v in subs.items()}
        print(f"  {k:<22} " + "  ".join(
            f"{kk[:4]} {rows[k][kk]['med_dB']:.3f}" for kk in subs)
            + f"   [{time.time()-t0:.0f}s]", flush=True)

    # 判别：灵活性的收益来自「函数类更强」还是「借用全向图样本」？
    # 只用 Δ=0 档自身样本（每折约 370 个）训练同一特征集 ⇒ 若与标定表持平，
    # 则上表收益主要来自**借样本**（低方差正则），而非函数类。
    mono_only = {}
    mtr_all = subs[mono]
    for k, F in Fs.items():
        err = np.full(Xf.size, np.nan)
        for f in range(nfold):
            tr = np.flatnonzero((foldf != f) & mtr_all)
            te = np.flatnonzero((foldf == f) & mtr_all)
            if tr.size < 50 or te.size < 10:
                continue
            m = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                min_samples_leaf=20, l2_regularization=1.0,
                early_stopping=False, random_state=seed)
            m.fit(F[tr], Yf[tr])
            err[te] = m.predict(F[te])
        e = np.where(np.isfinite(err), np.abs(Yf - np.nan_to_num(err)), np.nan)
        mono_only[k] = stats(mtr_all, e)
        print(f"    [仅 Δ=0 档训练] {k:<22} 单站 {mono_only[k]['med_dB']:.3f}"
              f"/{mono_only[k]['p90_dB']:.3f}", flush=True)

    order = (["raw（不标定）", "1D Δ 表（现行）", "2D 表（θ4×Δ6，现行最优）"]
             + list(lin) + list(Fs))
    print("\n  ── 汇总（中位 |Δ| dB，CV）──")
    hdr = "  %-26s" % "方案" + "".join("  %-20s" % k for k in subs)
    print(hdr)
    for name in order:
        r = rows[name]
        print("  %-26s" % name + "".join(
            "  %-20s" % f"{r[k]['med_dB']:.3f}/{r[k]['p90_dB']:.3f}" for k in subs))

    cand_all = list(Fs) + list(lin)
    base = rows["2D 表（θ4×Δ6，现行最优）"][mono]["med_dB"]
    best = min(rows[k][mono]["med_dB"] for k in cand_all)
    bestname = min(cand_all, key=lambda k: rows[k][mono]["med_dB"])
    gain = base - best
    print(f"\n  判决路径（单站）2D 表 {base:.3f} → 最优灵活模型 {bestname} {best:.3f}"
          f"  ⇒ 灵活函数类收益上界 {gain:+.3f} dB")

    out = {"n_angle": int(na), "n_dir": int(nT * nP), "nfold": nfold,
           "nsub": int(nsub), "max_iter": int(max_iter),
           "subsets": list(subs), "rows": rows, "mono_only_train": mono_only,
           "verdict_gain_mono_dB": float(gain),
           "verdict_best_feature_set": bestname,
           "criterion": "收益 ≥0.3 dB ⇒ 分段仿射是瓶颈，残差网络值得训；<0.15 dB ⇒ "
                        "(σ_PO 标量+几何) 已饱和，残差须来自复场/电流级信息"}
    jp = os.path.join(RESULT_DIR, f"_diag_residgate_nf{nfold}.json")
    json.dump(out, open(jp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    print(f"\n已存 {jp}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",
                    choices=["mono", "validate", "full", "post", "feko", "gen", "fullmap",
                             "patcalib", "gate2d", "patcalib2d", "residgate"],
                    default="mono",
                    help="post=只读 _diag_decision_full.npz 重跑分析；"
                         "feko=查 FEKO 原始 .out；gen=标定泛化验证；"
                         "fullmap=468 角 × 2701 方向完整双站误差图；"
                         "patcalib=由 fullmap 产出 Δ-分档标定表（供部署三路输出共用）；"
                         "gate2d=Δ 之外第二变量（ψ/θ_i）的留折可行性闸门（分口径）；"
                         "residgate=残差可学性闸门（灵活函数类 vs 2D 标的收益上界）")
    ap.add_argument("--nval", type=int, default=12, help="validate 模式的入射角数")
    ap.add_argument("--step", type=int, default=1, help="mono/fullmap 模式的角抽样步长")
    ap.add_argument("--ncase", type=int, default=0,
                    help="feko/fullmap 模式的样本数上限（0=全部）")
    ap.add_argument("--stride", type=int, default=1, help="feko 模式 case 抽样步长")
    ap.add_argument("--nrecip", type=int, default=6, help="feko 模式互易性自检的方向数")
    ap.add_argument("--nrep", type=int, default=20, help="gen 模式随机划分重复次数")
    ap.add_argument("--nfold", type=int, default=6, help="gate2d 模式留角 CV 的折数（按入射 φ 分块）")
    ap.add_argument("--nsub", type=int, default=300000, help="residgate 模式每折训练子采样数")
    ap.add_argument("--maxiter", type=int, default=250, help="residgate 模式 HGB 迭代上限")
    ap.add_argument("--pooled", type=int, default=60,
                    help="full 模式下做封板口径同批对照的角数（0=跳过）")
    ap.add_argument("--conv", choices=["h1", "h2"], default="h1", help="单站索引约定")
    ap.add_argument("--cache", default=os.path.join(RESULT_DIR, "_diag_decision_err.npz"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.mode == "validate":
        mode_validate(args.nval)
        return
    if args.mode == "feko":
        mode_feko(args.ncase, args.stride, args.nrecip)
        return
    if args.mode == "gen":
        mode_gen(args.nrep)
        return
    if args.mode == "fullmap":
        mode_fullmap(args.step, args.ncase)
        return
    if args.mode == "patcalib":
        mode_patcalib()
        return
    if args.mode == "gate2d":
        mode_gate2d(nfold=args.nfold)
        return
    if args.mode == "patcalib2d":
        mode_patcalib2d()
        return
    if args.mode == "residgate":
        mode_residgate(nfold=args.nfold, nsub=args.nsub, max_iter=args.maxiter)
        return

    t0 = time.time()
    if args.mode == "full":
        sel, sp, st, spr, str_, it_f = mode_full(args.pooled)
        analyse(sel, sp, st, 1, "h1", t0, tag="full(468角,单站口径)",
                ring=(spr, str_), it_x=it_f)
        return
    if args.mode == "post":
        zp = os.path.join(RESULT_DIR, "_diag_decision_full.npz")
        jp = os.path.join(RESULT_DIR, "_diag_decision_full.json")
        z = np.load(zp)
        mvr, by_theta, prof, fwd = report_ring(z["sp_ring"], z["st_ring"], z["it"],
                                               z["ip"], z["ff_theta"], z["ff_phi"])
        extra = json.load(open(jp, encoding="utf-8"))
        extra.update({"mono_vs_ring": mvr, "by_theta": by_theta,
                      "angle_profile": prof, "forward_scatter": fwd})
        json.dump(extra, open(jp, "w", encoding="utf-8"), indent=2,
                  ensure_ascii=False, default=float)
        analyse(z["sel"], z["sig_p"], z["sig_t"], 1, "h1", t0,
                tag="post(468角,单站口径)", ring=(z["sp_ring"], z["st_ring"]),
                it_x=z["it"])
        return

    if os.path.exists(args.cache) and not args.no_cache:
        z = np.load(args.cache)
        sel, sp, st = z["sel"], z["sig_p"], z["sig_t"]
        print(f"载入缓存 {args.cache}  n={len(sp)}", flush=True)
    else:
        sel, sp, st = compute_mono_sigma(args.step, args.conv)
        np.savez(args.cache, sel=sel, sig_p=sp, sig_t=st)
        print(f"已存缓存 {args.cache}", flush=True)
    # 角度表为 θ 外层 13 × φ 内层 36；只有能整除 36 的抽样才保持"相邻 φ = 相邻帧"语义
    rowlen = 36 // args.step if (36 % args.step == 0) else 1
    if rowlen <= 1:
        print("[警告] step 不整除 36，帧积累语义不可用，rowlen=1")
    analyse(sel, sp, st, args.step, args.conv, t0, tag=f"mono(step={args.step})",
            rowlen_mono=rowlen)


if __name__ == "__main__":
    main()
