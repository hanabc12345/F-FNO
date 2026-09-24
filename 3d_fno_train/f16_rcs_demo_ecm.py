# -*- coding: utf-8 -*-
"""
f16_rcs_demo_ecm.py — 单雷达 ECM（F16 干扰机对抗）实时演示
==========================================================
F16 携带干扰机：**干扰不改变物理 RCS（ONNX 代理），只改变雷达"表观 RCS"**：
    σ_obs(R) = σ_real + σ_jam(R)，σ_jam ∝ R²
（单程干扰 vs 双程回波 → 远距压制、近距烧穿；真实机制）
标定：R_BT = 4 km 处 σ_jam = -5 dBsm（ERP 假设值）→ σ_jam(R) = σ_jam(4km)·(R/4km)²

状态机：RWR 探测(R<11.9km) → 干扰压制 → 烧穿(σ_jam < 1.4·σ_real 干扰失效) → 掉头逃离
识别联动：雷达观测/识别用**表观 RCS** → 干扰开启时识别被欺骗（演示电子战破坏识别）

数据来源：3D F-FNO P3 ONNX + NFFFT（同 f16_rcs_demo.py）

面板：3D（含状态标注）· RCS 曲线（真实灰虚线/表观蓝/σ_jam 黄）· 方向图+全向压制环 ·
状态面板（状态机 + 识别置信度）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_ecm.py --selftest
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_ecm.py --seconds 180
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import sys
import time
import argparse

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import f16_rcs_demo as D
from f16_rcs_demo_2radar import fuse_rcs
from f16_rcs_demo_classify import TEMPLATES, classify_seq


# ============================================================
# 一、ECM 干扰机物理模型（单雷达）
# ============================================================

class ECM:
    """干扰机模型：σ_jam ∝ R²（单程干扰），烧穿判定 + 状态机。"""

    R_DETECT = 11900.0        # RWR 探测距离 m
    R_BT = 4000.0             # 烧穿参考距离 m
    SIG_JAM_AT_BT = -5.0      # dBsm @ R_BT（ERP 假设值）
    TH_BT = 1.4               # 烧穿阈值：σ_jam < TH_BT·σ_real → 干扰失效

    def __init__(self):
        self.sig_jam_bt = 10.0 ** (self.SIG_JAM_AT_BT / 10.0)

    def jam_sigma(self, R):
        """σ_jam(R) [m²]：∝R²（单程干扰与距离平方成正比）"""
        return self.sig_jam_bt * (R / self.R_BT) ** 2

    def state(self, R, sigma_real):
        if R >= self.R_DETECT:
            return "未探测"
        if self.jam_sigma(R) < self.TH_BT * (sigma_real + 1e-12):
            return "烧穿"          # 近距烧穿：回波双程占优
        return "干扰压制"


# ============================================================
# 二、自检
# ============================================================

def selftest():
    eng = D.F16RCSEngine()
    ecm = ECM()
    print("\n=== 单雷达 ECM 自检 ===")
    # 1) σ_jam ∝ R² 标定
    for R in (8000.0, 4000.0, 2000.0):
        db = 10 * np.log10(ecm.jam_sigma(R) + 1e-12)
        print(f"  σ_jam(R={R:5.0f}m) = {db:7.2f} dBsm"
              f"   (标定 4km=-5dBsm, 每减半距离 -6dB)")
    ok = abs(10 * np.log10(ecm.jam_sigma(4000.0)) - ecm.SIG_JAM_AT_BT) < 1e-3
    # 2) 状态机（σ_real 假设 -12 dBsm 常数）
    print("  状态机（σ_real=-12 dBsm 假设）:")
    for R in (12000.0, 7100.0, 2000.0):
        st = ecm.state(R, 10.0 ** (-12 / 10.0))
        print(f"    R={R:6.0f}m → {st}")
    # 3) ONNX 真实链路：表观叠加 + 烧穿恢复（用 h5 角度 90/0 直接推理）
    th, ph = 90.0, 0.0
    pred = eng.predict(eng.make_input(th, ph))
    rcs_real = eng.rcs_single(pred, th, ph)
    print(f"  ONNX 真实链路（θ=90,φ=0）: σ_real = {10*np.log10(rcs_real):7.2f} dBsm")
    for R in (7100.0, 3000.0):
        sjam = ecm.jam_sigma(R)
        sobs = rcs_real + sjam
        st = ecm.state(R, rcs_real)
        print(f"    R={R:5.0f}m: σ_jam={10*np.log10(sjam):6.1f} dBsm "
              f"σ_obs={10*np.log10(sobs):6.1f} dBsm 状态={st}")
    print("  自检通过" if ok else "  自检失败！")


# ============================================================
# 三、主演示
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--snapshot", default=None)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.headless:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    eng = D.F16RCSEngine()
    sim = D.F16Sim()
    ecm = ECM()
    R1 = np.asarray([0.0, 0.0, 0.02])

    plt.ion()
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.9, 0.95])
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_cls = fig.add_subplot(gs[0, 2])
    ax_rcs = fig.add_subplot(gs[1, :])
    ax_pol = fig.add_subplot(gs[2, 0], projection="polar")
    ax_info = fig.add_subplot(gs[2, 1:])
    ax_info.axis("off")
    fig.tight_layout(pad=2.5)

    ts, real_db, obs_db, jam_db = [], [], [], []
    pats = None
    frames_buf = []
    WIN = 10
    probs = np.ones(len(TEMPLATES)) / len(TEMPLATES)
    best = "—"
    st_name = "未探测"
    nsteps = int(args.seconds / args.dt)

    print(f"开始单雷达 ECM 演示: {args.seconds}s @ dt={args.dt}s, {nsteps} 帧")
    for k in range(nsteps):
        sim.step(args.dt)
        t0 = time.time()
        theta_los, phi_los = sim.los_angles()
        dyaw_deg, dpitch_deg = sim.body_offset()
        th_eff = float(np.clip(theta_los + dpitch_deg, 25.0, 155.0))
        ph_eff = (phi_los + dyaw_deg) % 360.0
        # 真实物理 RCS（ONNX 代理，干扰不影响）
        x = eng.make_input(th_eff, ph_eff)
        pred = eng.predict(x)
        rcs_real = eng.rcs_single(pred, th_eff, ph_eff)
        # 干扰：表观 RCS 叠加 σ_jam
        R = np.linalg.norm(sim.pos - R1)
        sjam = ecm.jam_sigma(R)
        sobs = rcs_real + sjam
        st_name = ecm.state(R, rcs_real)
        if st_name == "烧穿":
            sim.retreat = True                       # 干扰失效 → 掉头逃离
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        real_db.append(10 * np.log10(rcs_real + 1e-9))
        obs_db.append(10 * np.log10(sobs + 1e-9))
        jam_db.append(10 * np.log10(sjam + 1e-12))
        if pats is None or k % 10 == 0:
            pats = eng.rcs_pattern(pred)

        # 识别：用**表观 RCS**（干扰破坏识别的关键）
        z_obs = [obs_db[-1]]
        frames_buf.append((z_obs, [(th_eff, ph_eff)]))
        if len(frames_buf) > WIN:
            frames_buf.pop(0)
        probs, best = classify_seq([f[0] for f in frames_buf],
                                   [f[1] for f in frames_buf], TEMPLATES)

        # ---------- 绘图 ----------
        ax3d.clear()
        tr = np.asarray(sim.track)
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", lw=1.2, alpha=0.7, label="F16 轨迹")
        ax3d.plot(tr[-1:, 0], tr[-1:, 1], tr[-1:, 2], "bo", ms=6)
        ax3d.plot([R1[0]], [R1[1]], [R1[2]], "r*", ms=18, label="雷达")
        ax3d.plot([R1[0], tr[-1, 0]], [R1[1], tr[-1, 1]], [R1[2], tr[-1, 2]],
                  "r--", lw=0.8, alpha=0.6)
        rng3 = max(np.linalg.norm(tr[-1, :2] - R1[:2]), 6000.0) * 1.15
        ax3d.set_xlim(R1[0] - rng3, R1[0] + rng3)
        ax3d.set_ylim(R1[1] - rng3, R1[1] + rng3)
        ax3d.set_zlim(min(0.0, tr[:, 2].min() - 500), max(4000.0, tr[:, 2].max() + 500))
        ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
        ax3d.set_title(f"F16 干扰机对抗  t={sim.t:.0f}s  状态={st_name}", fontsize=11)
        ax3d.legend(loc="upper left", fontsize=7)
        ax3d.view_init(elev=28, azim=-58)

        # 识别面板
        ax_cls.clear()
        names = [t.name for t in TEMPLATES]
        ax_cls.barh(range(len(names)), probs * 100, color=["C0", "C2", "C3"],
                    alpha=0.85, height=0.55)
        ax_cls.set_yticks(range(len(names))); ax_cls.set_yticklabels(names)
        ax_cls.set_xlim(0, 100); ax_cls.set_xlabel("置信度 (%)")
        for i, pr in enumerate(probs):
            ax_cls.text(pr * 100 + 1, i, f"{pr*100:.0f}%", va="center", fontsize=10)
        ax_cls.invert_yaxis(); ax_cls.grid(axis="x", alpha=0.3)
        ax_cls.set_title(f"识别（用表观 RCS） → {best}", fontsize=10)
        ax_cls.axvline(50, color="gray", ls=":", lw=0.8)

        # RCS 曲线：真实/表观/σ_jam
        ax_rcs.clear()
        ax_rcs.plot(ts, real_db, "--", color="gray", lw=1.2, label="真实 RCS（物理）")
        ax_rcs.plot(ts, obs_db, "b-", lw=2.0, label="表观 RCS（雷达观测）")
        ax_rcs.plot(ts, jam_db, "y-", lw=1.4, alpha=0.85, label="σ_jam（干扰）")
        ax_rcs.plot(ts[-1], obs_db[-1], "ro", ms=7)
        ax_rcs.set_xlabel("t (s)"); ax_rcs.set_ylabel("RCS (dBsm)")
        ax_rcs.set_title(f"干扰机压制  表观 {obs_db[-1]:.1f} dBsm  （状态: {st_name}）")
        ax_rcs.grid(alpha=0.3); ax_rcs.legend(loc="lower left", fontsize=7)
        if len(ts) > 2:
            ax_rcs.set_xlim(max(ts[0], ts[-1] - 90), ts[-1])
            allv = real_db + obs_db + jam_db
            lo, hi = min(allv), max(allv)
            pad = max((hi - lo) * 0.15, 3.0)
            ax_rcs.set_ylim(lo - pad, hi + pad)

        # 方向图 + 全向压制环
        ax_pol.clear()
        cut = int(np.argmin(np.abs(pats[1] - 90)))
        db = np.maximum(pats[0][cut], -60)
        ax_pol.plot(np.deg2rad(pats[2]), db, lw=1.2, label="远场方向图")
        ax_pol.set_ylim(-60, 0)
        ax_pol.grid(alpha=0.3)
        jam_dbm = float(np.clip(jam_db[-1], -60, 0))
        ax_pol.plot(np.linspace(0, 2 * np.pi, 360), np.full(360, jam_dbm),
                    "y-", lw=1.4, alpha=0.9, label="σ_jam 全向环")
        ax_pol.set_title(f"方向图 φ-cut (θ=90°) + 压制环 {jam_dbm:.0f} dBsm", fontsize=10)
        ax_pol.legend(loc="lower left", fontsize=7)

        ax_info.clear(); ax_info.axis("off")
        lines = [
            f"ECM 对抗态势（t = {sim.t:.0f}s）",
            f"  距离 R : {R/1000:.2f} km   状态: {st_name}",
            f"  σ_real: {real_db[-1]:7.1f} dBsm   σ_jam: {jam_db[-1]:7.1f} dBsm",
            f"  表观 RCS: {obs_db[-1]:7.1f} dBsm（雷达观测值）",
            f"  识别判定: {best}  F-16={probs[0]*100:4.0f}% "
            f"无人机={probs[1]*100:4.0f}% 巡飞弹={probs[2]*100:4.0f}%",
            f"  推理耗时: {ti_ms:.0f} ms/帧",
            "",
            "物理模型：干扰不改变物理 RCS，只叠加表观 σ_jam；",
            "σ_jam ∝ R²（单程干扰 vs 双程回波）→ 远距压制、近距烧穿；",
            "状态机：RWR 探测(11.9km) → 压制 → 烧穿(σ_jam<1.4σ_real) → 掉头；",
            "识别用表观 RCS → 干扰开时被欺骗（本帧判定为巡飞弹）即电子战效果。",
        ]
        ax_info.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10)
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        plt.pause(0.001)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s R={R/1000:.1f}km 状态={st_name} "
                  f"表观={obs_db[-1]:.1f} 判定={best} 推理 {ti_ms:.0f}ms", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
