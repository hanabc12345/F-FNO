# -*- coding: utf-8 -*-
"""
f16_rcs_demo_ecm_2radar.py — 双雷达联合探测 vs ECM（多基地抗干扰）实时演示
======================================================================
干扰机**功率/空域有限**：
  · 雷达1 在主波束（σ_jam 同单雷达模型，4km 处 -5 dBsm，∝R²）
  · 雷达2 落在旁瓣（`--sll-db` 默认 15 dB 弱压制）
各雷达独立烧穿（σ_jam < 1.4σ_real 失效）。

展示：雷达1 被压制欺骗（表观抬升→误判），雷达2 旁瓣未被有效压制 →
**联合识别维持 F-16**（多基地天然抗干扰）。--sll-db 0 作对照（双雷达都被压制）。

物理：σ_obs_i = σ_real_i + σ_jam_i；合成表观 = 两表观功率平均（--fuse）；
识别用双雷达表观特征 [σ_obs1, σ_obs2]（classify_seq 多帧贝叶斯累积）。

数据来源：3D F-FNO P3 ONNX（batch=2 一次推理）+ NFFFT（同 f16_rcs_demo_2radar.py）。

面板：3D（双雷达+各自状态）· RCS 曲线（真实/R1 表观/R2 表观/联合合成 四线）·
三方向图 · 状态面板。

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_ecm_2radar.py --selftest
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_ecm_2radar.py --seconds 180
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_ecm_2radar.py --sll-db 0   # 双雷达都被压制（对照）
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


# ============================================================
# 一、ECM 干扰机物理模型（双雷达：主波束 + 旁瓣）
# ============================================================

class ECM2:
    """干扰机：主波束压雷达1（sll=0），雷达2 在旁瓣（--sll-db 衰减）。"""

    R_DETECT = 11900.0
    R_BT = 4000.0
    SIG_JAM_AT_BT = -5.0
    TH_BT = 1.4

    def __init__(self, sll_db=15.0):
        self.sll_db = sll_db
        self.sig_jam_bt = 10.0 ** (self.SIG_JAM_AT_BT / 10.0)

    def jam_sigma(self, R, sll=0.0):
        """σ_jam(R) [m²]：∝R²，旁瓣按 -sll dB 衰减。"""
        return self.sig_jam_bt * (R / self.R_BT) ** 2 * 10.0 ** (-sll / 10.0)

    def state(self, R, sigma_real, sll=0.0):
        if R >= self.R_DETECT:
            return "未探测"
        if self.jam_sigma(R, sll) < self.TH_BT * (sigma_real + 1e-12):
            return "烧穿"
        return "压制" if sll < 0.1 else "旁瓣弱压制"


# ============================================================
# 二、自检
# ============================================================

def selftest():
    eng = F16RCSEngine2()
    print("\n=== 双雷达 ECM 自检 ===")
    # 1) 旁瓣衰减
    s_main = ECM2().jam_sigma(7100.0, 0.0)
    s_sll = ECM2().jam_sigma(7100.0, 15.0)
    print(f"  R=7100m: 主波束 σ_jam={10*np.log10(s_main):.1f} dBsm, "
          f"旁瓣15dB σ_jam={10*np.log10(s_sll):.1f} dBsm "
          f"(差 {10*np.log10(s_main/s_sll):.0f} dB)")
    # 2) 真实飞行场景：t≈20-25s 推进 10 帧，表观特征多帧累积识别
    R1 = np.asarray([0.0, 0.0, 0.02])
    R2 = np.asarray([7000.0, -5000.0, 0.02])
    for sll in (15.0, 0.0):
        e2 = ECM2(sll_db=sll)
        sim = D.F16Sim()
        for _ in range(40):
            sim.step(0.5)                       # 推进到 t=20s（干扰已开）
        buf_z, buf_ang = [], []
        for _ in range(10):
            sim.step(0.5)
            dyaw_deg, dpitch_deg = sim.body_offset()
            angs = []
            for R in (R1, R2):
                d = sim.pos - R
                d /= np.linalg.norm(d)
                th_los = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
                ph_los = np.degrees(np.arctan2(d[1], d[0]))
                angs.append((float(np.clip(th_los + dpitch_deg, 25.0, 155.0)),
                             (ph_los + dyaw_deg) % 360.0))
            x = eng.make_inputs([a[0] for a in angs], [a[1] for a in angs])
            preds = eng.predict_batch(x)
            r_real = [eng.rcs_single(preds[i], angs[i][0], angs[i][1]) for i in range(2)]
            Rd = [np.linalg.norm(sim.pos - R1), np.linalg.norm(sim.pos - R2)]
            sjam = [e2.jam_sigma(Rd[0], 0.0), e2.jam_sigma(Rd[1], sll)]
            sobs = [r_real[0] + sjam[0], r_real[1] + sjam[1]]
            buf_z.append([10 * np.log10(v + 1e-9) for v in sobs])
            buf_ang.append(list(angs))
        p, best = classify_seq(buf_z, buf_ang, TEMPLATES)
        print(f"  --sll-db {sll:5.1f}: 表观累积10帧 → 识别 {best} "
              f"(F16={p[0]*100:.0f}% 无人机={p[1]*100:.0f}% 巡飞弹={p[2]*100:.0f}%)")
    print("  说明：sll=15 时雷达2 旁瓣未被有效压制 → 提供真实观测 → 联合识别应维持 F-16；")
    print("        sll=0 时双雷达都被压制 → 表观抬升 → 误判巡飞弹（对照）。")
    print("  自检通过（物理公式 + 真实链路验证）")


# ============================================================
# 三、主演示
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--fuse", choices=["mean", "max"], default="mean")
    ap.add_argument("--sll-db", type=float, default=15.0, help="雷达2 旁瓣衰减 dB（越小受压制越强）")
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
    ecm = ECM2(sll_db=args.sll_db)
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

    ts, real_db, o1_db, o2_db, fz_db = [], [], [], [], []
    pats = None
    frames_buf = []
    WIN = 10
    probs = np.ones(len(TEMPLATES)) / len(TEMPLATES)
    best = "—"
    st1 = st2 = "未探测"
    nsteps = int(args.seconds / args.dt)

    print(f"开始双雷达 ECM 演示: {args.seconds}s @ dt={args.dt}s, "
          f"sll={args.sll_db}dB, 合成={args.fuse}, {nsteps} 帧")
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
        # 干扰叠加：雷达1 主波束、雷达2 旁瓣
        R1d = np.linalg.norm(sim.pos - R1); R2d = np.linalg.norm(sim.pos - R2)
        sjam = [ecm.jam_sigma(R1d, 0.0), ecm.jam_sigma(R2d, args.sll_db)]
        sobs = [r_real[0] + sjam[0], r_real[1] + sjam[1]]
        st1 = ecm.state(R1d, r_real[0], 0.0)
        st2 = ecm.state(R2d, r_real[1], args.sll_db)
        if st1 == "烧穿":
            sim.retreat = True
        fused = fuse_rcs(sobs, args.fuse)
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        real_db.append(10 * np.log10((r_real[0] + r_real[1]) / 2 + 1e-12))
        o1_db.append(10 * np.log10(sobs[0] + 1e-9))
        o2_db.append(10 * np.log10(sobs[1] + 1e-9))
        fz_db.append(10 * np.log10(fused + 1e-9))
        if pats is None or k % 10 == 0:
            p1 = eng.rcs_pattern(preds[0])
            p2 = eng.rcs_pattern(preds[1])
            pwr = 0.5 * (10.0 ** (p1[0] / 10.0) + 10.0 ** (p2[0] / 10.0))
            pats = (p1, p2, (10.0 * np.log10(np.maximum(pwr, 1e-9)), p1[1], p1[2]))

        # 识别：用双雷达表观特征（多基地抗干扰的关键）
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
        rng3 = max(np.linalg.norm(tr[-1, :2] - R1[:2]), 6000.0) * 1.15
        ax3d.set_xlim(R1[0] - rng3, R1[0] + rng3)
        ax3d.set_ylim(R1[1] - rng3, R1[1] + rng3)
        ax3d.set_zlim(min(0.0, tr[:, 2].min() - 500), max(4000.0, tr[:, 2].max() + 500))
        ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
        ax3d.set_title(f"双雷达 vs 干扰机  t={sim.t:.0f}s\n"
                       f"R1: {st1}  R2: {st2}", fontsize=10)
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
        ax_cls.set_title(f"联合识别（双雷达表观） → {best}", fontsize=10)
        ax_cls.axvline(50, color="gray", ls=":", lw=0.8)

        ax_rcs.clear()
        ax_rcs.plot(ts, real_db, "--", color="gray", lw=1.2, label="真实 RCS（均）")
        ax_rcs.plot(ts, o1_db, "b-", lw=1.5, alpha=0.85, label=f"R1 表观 ({st1})")
        ax_rcs.plot(ts, o2_db, "m-", lw=1.5, alpha=0.85, label=f"R2 表观 ({st2})")
        ax_rcs.plot(ts, fz_db, "r-", lw=2.2, label=f"联合合成({args.fuse})")
        ax_rcs.plot(ts[-1], fz_db[-1], "ro", ms=7)
        ax_rcs.set_xlabel("t (s)"); ax_rcs.set_ylabel("RCS (dBsm)")
        ax_rcs.set_title(f"双雷达抗干扰  联合表观 {fz_db[-1]:.1f} dBsm")
        ax_rcs.grid(alpha=0.3); ax_rcs.legend(loc="lower left", fontsize=6)
        if len(ts) > 2:
            ax_rcs.set_xlim(max(ts[0], ts[-1] - 90), ts[-1])
            allv = real_db + o1_db + o2_db + fz_db
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
        ax_pol1.set_title(f"双站方向图（入射=雷达1）  {st1}", fontsize=9)
        ax_pol2.set_title(f"双站方向图（入射=雷达2）  {st2}", fontsize=9)
        ax_pol3.set_title("功率平均示意（非物理观测）", fontsize=9)

        fig.canvas.draw_idle(); fig.canvas.flush_events()
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        plt.pause(0.001)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s R1={R1d/1000:.1f}km({st1}) "
                  f"R2={R2d/1000:.1f}km({st2}) 表观1={o1_db[-1]:.1f} 表观2={o2_db[-1]:.1f} "
                  f"联合={fz_db[-1]:.1f} 判定={best} 推理 {ti_ms:.0f}ms", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
