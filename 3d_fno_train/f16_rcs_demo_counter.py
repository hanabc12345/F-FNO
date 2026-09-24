# -*- coding: utf-8 -*-
"""
f16_rcs_demo_counter.py — 双雷达协同反制干扰机（电子攻防闭环）实时演示
===================================================================
回答"两雷达能否同时压制干扰机"：**不能对等功率压制**（干扰单程 R² vs
回波双程 R⁴，干扰机天然占优），但**协同反制可行**。四阶段对抗闭环
（时间线驱动，参数可调）：

| 阶段 | 时间 | 机制 | σ_jam 系数 |
|:---|:---|:---|:---|
| ① 干扰压制 | <t_counter | 主波束压雷达1 | ×1 |
| ② 协同反制 | t_counter–t_sat | 功率稀释(-3dB)+频率分集(-6dB) | ×1/8 |
| ③ 干扰饱和 | t_sat–t_kill | 干扰机备用功率×2 → 功率/散热上限 | ×2/8 |
| ④ 反辐射摧毁 | >t_kill | 双站交叉定位 → 命中 → 干扰消失 | ×0 |

实现：`ECM3.jam_coef(t) ∈ {1, 1/8, 2/8, 0}`；
面板 RCS 曲线加**阶段竖线**；3D 显示反辐射命中标记（橙色 X）；
状态面板显示当前阶段 + σ_jam 系数 + 联合识别。

数据来源：3D F-FNO P3 ONNX（batch=2）+ NFFFT（同 f16_rcs_demo_2radar.py）。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_counter.py --selftest
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_counter.py --seconds 180
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_counter.py --t-counter 30 --t-sat 60 --t-kill 90
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
from f16_rcs_demo_2radar import F16RCSEngine2, fuse_rcs, pattern_at
from f16_rcs_demo_classify import TEMPLATES, classify_seq
from f16_rcs_demo_ecm_2radar import ECM2


# ============================================================
# 一、协同反制干扰机模型（四阶段）
# ============================================================

class ECM3(ECM2):
    """在 ECM2（主波束+旁瓣）基础上叠加时间线驱动的协同反制阶段。"""

    def __init__(self, sll_db=15.0, t_counter=30.0, t_sat=60.0, t_kill=90.0):
        super().__init__(sll_db)
        self.t_counter = t_counter
        self.t_sat = t_sat
        self.t_kill = t_kill

    def jam_coef(self, t):
        """σ_jam 阶段系数：1 → 1/8 → 2/8 → 0"""
        if t < self.t_counter:
            return 1.0
        if t < self.t_sat:
            return 1.0 / 8.0      # 功率稀释 -3dB + 频率分集 -6dB = -9dB ≈ ×1/8
        if t < self.t_kill:
            return 2.0 / 8.0      # 干扰机备用功率×2（+3dB）→ 触及功率/散热上限，净 -6dB
        return 0.0                # 反辐射摧毁，干扰消失

    def phase(self, t):
        if t < self.t_counter:
            return 1, "① 干扰压制"
        if t < self.t_sat:
            return 2, "② 协同反制"
        if t < self.t_kill:
            return 3, "③ 干扰饱和"
        return 4, "④ 反辐射摧毁"

    def jam_sigma(self, R, sll=0.0, t=0.0):
        return super().jam_sigma(R, sll) * self.jam_coef(t)


# ============================================================
# 二、自检
# ============================================================

def selftest():
    ecm = ECM3(sll_db=15.0, t_counter=30.0, t_sat=60.0, t_kill=90.0)
    print("\n=== 协同反制自检（四阶段 σ_jam 系数）===")
    ok = True
    for t, want in ((10.0, 1.0), (45.0, 1 / 8), (75.0, 2 / 8), (100.0, 0.0)):
        got = ecm.jam_coef(t)
        ok &= abs(got - want) < 1e-9
        ph = ecm.phase(t)[1]
        print(f"  t={t:5.0f}s  jam_coef={got:.3f}  阶段={ph}")
    # 表观序列（σ_real=-12 dBsm, R=7100m 常数）随阶段递减
    s_real = 10.0 ** (-12 / 10.0)
    print("  表观 RCS 随阶段（σ_real=-12dBsm, R=7100m 常数）:")
    for t in (10.0, 45.0, 75.0, 100.0):
        sjam = ecm.jam_sigma(7100.0, 0.0, t)
        sobs = s_real + sjam
        print(f"    t={t:5.0f}s: σ_jam={10*np.log10(sjam+1e-12):6.1f} dBsm → "
              f"表观={10*np.log10(sobs):6.1f} dBsm")
    print("  自检通过" if ok else "  自检失败！")


# ============================================================
# 三、主演示
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--fuse", choices=["mean", "max"], default="mean")
    ap.add_argument("--sll-db", type=float, default=15.0)
    ap.add_argument("--t-counter", type=float, default=30.0, help="协同反制开始时刻 s")
    ap.add_argument("--t-sat", type=float, default=60.0, help="干扰饱和反扑时刻 s")
    ap.add_argument("--t-kill", type=float, default=90.0, help="反辐射摧毁时刻 s")
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

    eng = F16RCSEngine2()
    sim = D.F16Sim()
    ecm = ECM3(sll_db=args.sll_db, t_counter=args.t_counter,
               t_sat=args.t_sat, t_kill=args.t_kill)
    R1 = np.asarray([0.0, 0.0, 0.02])
    R2 = np.asarray([7000.0, -5000.0, 0.02])

    plt.ion()
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.9, 0.95])
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_cls = fig.add_subplot(gs[0, 2])
    ax_rcs = fig.add_subplot(gs[1, :])
    ax_pol1 = fig.add_subplot(gs[2, 0], projection="polar")
    ax_pol2 = fig.add_subplot(gs[2, 1], projection="polar")
    ax_pol3 = fig.add_subplot(gs[2, 2], projection="polar")
    fig.tight_layout(pad=2.5)

    ts, real_db, o1_db, o2_db, fz_db, jam_db = [], [], [], [], [], []
    pats = None
    frames_buf = []
    WIN = 10
    probs = np.ones(len(TEMPLATES)) / len(TEMPLATES)
    best = "—"
    phase_no, phase_name = 1, "① 干扰压制"
    coef_now = 1.0
    nsteps = int(args.seconds / args.dt)

    print(f"开始协同反制演示: {args.seconds}s @ dt={args.dt}s, "
          f"阶段切换 {args.t_counter}/{args.t_sat}/{args.t_kill}s, "
          f"sll={args.sll_db}dB, {nsteps} 帧")
    for k in range(nsteps):
        sim.step(args.dt)
        t0 = time.time()
        dyaw_deg, dpitch_deg = sim.body_offset()
        angs, los = [], []
        for R in (R1, R2):
            d = sim.pos - R
            d /= np.linalg.norm(d)
            th_los = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
            ph_los = np.degrees(np.arctan2(d[1], d[0]))
            angs.append((float(np.clip(th_los + dpitch_deg, 25.0, 155.0)),
                         (ph_los + dyaw_deg) % 360.0))
            los.append((th_los, ph_los))
        # 批量 ONNX 推理 → 两雷达真实物理 RCS
        x = eng.make_inputs([a[0] for a in angs], [a[1] for a in angs])
        preds = eng.predict_batch(x)
        r_real = [eng.rcs_single(preds[i], angs[i][0], angs[i][1]) for i in range(2)]
        # 干扰：阶段系数 × (主波束压 R1 / 旁瓣弱压 R2)
        R1d = np.linalg.norm(sim.pos - R1); R2d = np.linalg.norm(sim.pos - R2)
        coef_now = ecm.jam_coef(sim.t)
        sjam = [ecm.jam_sigma(R1d, 0.0, sim.t), ecm.jam_sigma(R2d, args.sll_db, sim.t)]
        sobs = [r_real[0] + sjam[0], r_real[1] + sjam[1]]
        phase_no, phase_name = ecm.phase(sim.t)
        if coef_now == 0.0:
            sim.retreat = True            # 干扰被摧毁后 F16 脱离
        fused = fuse_rcs(sobs, args.fuse)
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        real_db.append(10 * np.log10((r_real[0] + r_real[1]) / 2 + 1e-12))
        o1_db.append(10 * np.log10(sobs[0] + 1e-9))
        o2_db.append(10 * np.log10(sobs[1] + 1e-9))
        fz_db.append(10 * np.log10(fused + 1e-9))
        jam_db.append(10 * np.log10(sjam[0] + 1e-12))
        if pats is None or k % 10 == 0:
            p1 = eng.rcs_pattern(preds[0])
            p2 = eng.rcs_pattern(preds[1])
            pwr = 0.5 * (10.0 ** (p1[0] / 10.0) + 10.0 ** (p2[0] / 10.0))
            pats = (p1, p2, (10.0 * np.log10(np.maximum(pwr, 1e-9)), p1[1], p1[2]))

        # 识别：双雷达表观特征
        z_obs = [10 * np.log10(sobs[0] + 1e-9), 10 * np.log10(sobs[1] + 1e-9)]
        frames_buf.append((z_obs, list(angs)))
        if len(frames_buf) > WIN:
            frames_buf.pop(0)
        probs, best = classify_seq([f[0] for f in frames_buf],
                                   [f[1] for f in frames_buf], TEMPLATES)

        # ---------- 绘图 ----------
        ax3d.clear()
        tr = np.asarray(sim.track)
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", lw=1.2, alpha=0.7, label="F16 轨迹")
        ax3d.plot(tr[-1:, 0], tr[-1:, 1], tr[-1:, 2], "bo", ms=6)
        for R, c, nm in ((R1, "r", "雷达1"), (R2, "m", "雷达2")):
            ax3d.plot([R[0]], [R[1]], [R[2]], c + "*", ms=16, label=nm)
            ax3d.plot([R[0], tr[-1, 0]], [R[1], tr[-1, 1]], [R[2], tr[-1, 2]],
                      c + "--", lw=0.8, alpha=0.6)
        if coef_now == 0.0:
            # 反辐射摧毁命中标记（橙色 X，位于双站交叉定位指向的干扰机位置）
            ax3d.plot([tr[-1, 0]], [tr[-1, 1]], [tr[-1, 2]],
                      marker="X", ms=24, color="orange", mec="k", mew=1.2,
                      label="反辐射命中")
        rng3 = max(np.linalg.norm(tr[-1, :2] - R1[:2]), 6000.0) * 1.15
        ax3d.set_xlim(R1[0] - rng3, R1[0] + rng3)
        ax3d.set_ylim(R1[1] - rng3, R1[1] + rng3)
        ax3d.set_zlim(min(0.0, tr[:, 2].min() - 500), max(4000.0, tr[:, 2].max() + 500))
        ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
        ax3d.set_title(f"协同反制对抗  t={sim.t:.0f}s  {phase_name}", fontsize=11)
        ax3d.legend(loc="upper left", fontsize=7)
        ax3d.view_init(elev=28, azim=-58)

        ax_cls.clear()
        names = [t.name for t in TEMPLATES]
        ax_cls.barh(range(len(names)), probs * 100, color=["C0", "C2", "C3"],
                    alpha=0.85, height=0.55)
        ax_cls.set_yticks(range(len(names))); ax_cls.set_yticklabels(names)
        ax_cls.set_xlim(0, 100); ax_cls.set_xlabel("置信度 (%)")
        for i, pr in enumerate(probs):
            ax_cls.text(pr * 100 + 1, i, f"{pr*100:.0f}%", va="center", fontsize=10)
        ax_cls.invert_yaxis(); ax_cls.grid(axis="x", alpha=0.3)
        ax_cls.set_title(f"联合识别 → {best}", fontsize=10)
        ax_cls.axvline(50, color="gray", ls=":", lw=0.8)

        ax_rcs.clear()
        ax_rcs.plot(ts, real_db, "--", color="gray", lw=1.2, label="真实 RCS（均）")
        ax_rcs.plot(ts, o1_db, "b-", lw=1.5, alpha=0.85, label="R1 表观")
        ax_rcs.plot(ts, o2_db, "m-", lw=1.5, alpha=0.85, label="R2 表观")
        ax_rcs.plot(ts, fz_db, "r-", lw=2.2, label=f"联合合成({args.fuse})")
        ax_rcs.plot(ts, jam_db, "y-", lw=1.2, alpha=0.7, label="σ_jam 系数×")
        ax_rcs.plot(ts[-1], fz_db[-1], "ro", ms=7)
        # 阶段竖线
        ax_rcs.axvline(args.t_counter, color="g", ls=":", lw=1.5)
        ax_rcs.axvline(args.t_sat, color="y", ls=":", lw=1.5)
        ax_rcs.axvline(args.t_kill, color="r", ls=":", lw=1.5)
        ax_rcs.text(args.t_counter, 0.02, "反制", ha="center", fontsize=8, color="g")
        ax_rcs.text(args.t_sat, 0.02, "饱和", ha="center", fontsize=8, color="orange")
        ax_rcs.text(args.t_kill, 0.02, "摧毁", ha="center", fontsize=8, color="r")
        ax_rcs.set_xlabel("t (s)"); ax_rcs.set_ylabel("RCS (dBsm)")
        ax_rcs.set_title(f"协同反制四阶段  当前 {phase_name}  jam系数×{coef_now}")
        ax_rcs.grid(alpha=0.3); ax_rcs.legend(loc="lower left", fontsize=6)
        if len(ts) > 2:
            ax_rcs.set_xlim(max(ts[0], ts[-1] - 90), ts[-1])
            allv = real_db + o1_db + o2_db + fz_db + jam_db
            lo, hi = min(allv), max(allv)
            pad = max((hi - lo) * 0.15, 3.0)
            ax_rcs.set_ylim(lo - pad, hi + pad)

        for ax, pat, R, c, nm in ((ax_pol1, pats[0], R1, "r", "雷达1"),
                                  (ax_pol2, pats[1], R2, "m", "雷达2"),
                                  (ax_pol3, pats[2], None, "k", "合成")):
            ax.clear()
            cut = int(np.argmin(np.abs(pat[1] - 90)))
            db = np.maximum(pat[0][cut], -60)
            ax.plot(np.deg2rad(pat[2]), db, lw=1.2)
            ax.set_ylim(-60, 0)
            ax.grid(alpha=0.3)
            if R is not None:
                dv = sim.pos - R
                azi = np.deg2rad(np.degrees(np.arctan2(dv[1], dv[0])))
                ax.plot([azi, azi], [-60, -35], c, lw=2.5)
                ax.text(azi, -55, nm, ha="center", color=c, fontsize=9)
            else:
                for Rr, cr, lbl in ((R1, "r", "R1"), (R2, "m", "R2")):
                    dv = sim.pos - Rr
                    azi = np.deg2rad(np.degrees(np.arctan2(dv[1], dv[0])))
                    ax.plot([azi, azi], [-60, -35], cr, lw=1.5)
                    ax.text(azi, -55, lbl, ha="center", color=cr, fontsize=8)
        ax_pol1.set_title(f"双站方向图（入射=雷达1）", fontsize=9)
        ax_pol2.set_title(f"双站方向图（入射=雷达2）", fontsize=9)
        ax_pol3.set_title("功率平均示意（非物理观测）", fontsize=9)

        fig.canvas.draw_idle(); fig.canvas.flush_events()
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        plt.pause(0.001)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s {phase_name} "
                  f"系数={coef_now:.3f} 表观1={o1_db[-1]:.1f} 表观2={o2_db[-1]:.1f} "
                  f"联合={fz_db[-1]:.1f} 判定={best} 推理 {ti_ms:.0f}ms", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
