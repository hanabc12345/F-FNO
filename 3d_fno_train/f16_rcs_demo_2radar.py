# -*- coding: utf-8 -*-
"""
f16_rcs_demo_2radar.py — 双雷达两点合成 RCS 实时演示
====================================================
两部雷达部署在不同位置同时探测同一架 F16，各自得到不同视角下的单站 RCS；
再将两个 RCS 合成为单一估计（默认功率平均，可 --fuse max），
降低单视角 RCS 闪烁对"型号判读"的干扰（合成曲线更平稳，利于型号比对）。

数据来源：3D F-FNO P3 模型（ONNX），每帧对两雷达角度批量推理一次（batch=2）：
  飞行模拟 → 各雷达视线入射角 + 机体姿态偏移 → 批量解析入射场(2,7,64,48,32)
  → ONNX 推理(2,12,...) → 各雷达 NFFFT 单站 RCS → 合成 → 实时图
  （3D 战场 / 双雷达+合成 RCS 曲线 / 方向图标注两雷达视角 / 状态面板）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_2radar.py --selftest
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_2radar.py --seconds 180 --fuse mean
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
import f16_rcs_demo as D            # 复用引擎 / 飞行模拟 / 中文字体配置

GRID = D.GRID
BETA0 = D.BETA0


class F16RCSEngine2(D.F16RCSEngine):
    """扩展：批量输入构造 + 批量 ONNX 推理（两雷达一次推理）。"""

    def make_inputs(self, theta_list, phi_list):
        """多入射角 → 标准化输入 (B,7,64,48,32)。khat=-k̂_std, e0=θ̂_std。"""
        t = np.deg2rad(np.asarray(theta_list, np.float64))
        p = np.deg2rad(np.asarray(phi_list, np.float64))
        st, ct = np.sin(t), np.cos(t)
        sp, cp = np.sin(p), np.cos(p)
        khat = -np.stack([st * cp, st * sp, ct], axis=1)     # (B,3)
        e0 = np.stack([ct * cp, ct * sp, -st], axis=1)       # (B,3)
        kx, ky, kz = khat[:, 0, None, None, None], khat[:, 1, None, None, None], khat[:, 2, None, None, None]
        phase = np.exp(-1j * BETA0 * (kx * self.X[None] + ky * self.Y[None] + kz * self.Z[None]))
        e_inc = e0[:, None, None, None, :] * phase[..., None]   # (B,64,48,32,3)
        B = len(theta_list)
        x = np.zeros((B, 7, *GRID), np.float32)
        x[:, 0] = self.eps_mask[None]
        x[:, 1:4] = e_inc.real.transpose(0, 4, 1, 2, 3)
        x[:, 4:7] = e_inc.imag.transpose(0, 4, 1, 2, 3)
        return (x - self.xm) / self.xs

    def predict_batch(self, x):
        """批量推理 → 反标准化 (B,12,64,48,32)。"""
        out = self.sess.run(None, {"input": x.astype(np.float32)})[0]
        return out * self.ys.reshape(1, 12, 1, 1, 1) + self.ym.reshape(1, 12, 1, 1, 1)


def fuse_rcs(rcs_lin_list, method="mean"):
    """多雷达线性 RCS (m²) → 合成。mean=功率平均；max=取最大（保守探测）。"""
    a = np.asarray(rcs_lin_list)
    if method == "max":
        return float(a.max())
    return float(a.mean())


def pattern_at(pat, th_deg, ph_deg):
    """在方向图 (θ,φ) 最近网格点取值 (dBsm)，φ 做环绕处理。"""
    i = int(np.argmin(np.abs(pat[1] - th_deg)))
    d = np.abs((pat[2] - (ph_deg % 360.0) + 180.0) % 360.0 - 180.0)
    j = int(np.argmin(d))
    return float(pat[0][i, j])


def selftest():
    eng = F16RCSEngine2()
    print("\n=== 双雷达批量推理自检（batch vs 单样本一致性）===")
    # 两组角度：单次逐样本 vs 批量，应逐位一致
    angs = [(90.0, 0.0), (60.0, 45.0)]
    x_b = eng.make_inputs(*zip(*angs))
    out_b = eng.predict_batch(x_b)
    for i, (th, ph) in enumerate(angs):
        x_s = eng.make_input(th, ph)
        out_s = eng.predict(x_s)
        rel = np.linalg.norm(out_b[i] - out_s) / (np.linalg.norm(out_s) + 1e-12)
        print(f"  角度 θ={th},φ={ph}: batch vs 单样本 rel={rel:.2e}  "
              f"RCS={10*np.log10(eng.rcs_single(out_b[i], th, ph)+1e-9):.2f} dBsm")
    print("（rel≈0 即批量推理正确）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--fuse", choices=["mean", "max"], default="mean", help="RCS 合成方式")
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
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    eng = F16RCSEngine2()
    sim = D.F16Sim()
    R1 = np.asarray([0.0, 0.0, 0.02])        # 雷达 1（原点，同单雷达 demo）
    R2 = np.asarray([7000.0, -5000.0, 0.02]) # 雷达 2（东北侧异位部署）

    plt.ion()
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.9, 0.95])
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_info = fig.add_subplot(gs[0, 2])
    ax_info.axis("off")
    ax_rcs = fig.add_subplot(gs[1, :])
    ax_pol1 = fig.add_subplot(gs[2, 0], projection="polar")
    ax_pol2 = fig.add_subplot(gs[2, 1], projection="polar")
    ax_pol3 = fig.add_subplot(gs[2, 2], projection="polar")
    fig.tight_layout(pad=2.5)

    ts, r1_db, r2_db, fz_db, tb12, b12_db = [], [], [], [], [], []
    pats = None
    sig12_db = sig21_db = float("nan")
    nsteps = int(args.seconds / args.dt)

    print(f"开始双雷达演示: {args.seconds}s @ dt={args.dt}s, 合成={args.fuse}, {nsteps} 帧")
    for k in range(nsteps):
        sim.step(args.dt)
        t0 = time.time()
        # 两雷达各自视线入射角 + 机体姿态偏移
        dyaw_deg, dpitch_deg = sim.body_offset()
        angs, los = [], []
        for R in (R1, R2):
            d = sim.pos - R
            d /= np.linalg.norm(d)
            th_los = np.degrees(np.arccos(np.clip(d[2], -1, 1)))
            ph_los = np.degrees(np.arctan2(d[1], d[0]))
            th_eff = float(np.clip(th_los + dpitch_deg, 25.0, 155.0))
            ph_eff = (ph_los + dyaw_deg) % 360.0
            angs.append((th_eff, ph_eff))
            los.append((th_los, ph_los))
        # 批量 ONNX 推理
        x = eng.make_inputs([a[0] for a in angs], [a[1] for a in angs])
        preds = eng.predict_batch(x)
        rcs1 = eng.rcs_single(preds[0], angs[0][0], angs[0][1])
        rcs2 = eng.rcs_single(preds[1], angs[1][0], angs[1][1])
        fused = fuse_rcs([rcs1, rcs2], args.fuse)
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        r1_db.append(10 * np.log10(rcs1 + 1e-9))
        r2_db.append(10 * np.log10(rcs2 + 1e-9))
        fz_db.append(10 * np.log10(fused + 1e-9))
        if pats is None or k % 10 == 0:
            p1 = eng.rcs_pattern(preds[0])
            p2 = eng.rcs_pattern(preds[1])
            pwr = 0.5 * (10.0 ** (p1[0] / 10.0) + 10.0 ** (p2[0] / 10.0))  # 方向图功率平均
            p3 = (10.0 * np.log10(np.maximum(pwr, 1e-9)), p1[1], p1[2])
            pats = (p1, p2, p3)
            # 真双站 RCS：σ12 = R1 发 → R2 方向接收（p1 在 R2 视线方向的散射）
            sig12_db = pattern_at(p1, angs[1][0], angs[1][1])
            sig21_db = pattern_at(p2, angs[0][0], angs[0][1])
            tb12.append(sim.t)
            b12_db.append(sig12_db)
        # ---------- 绘图 ----------
        ax3d.clear()
        tr = np.asarray(sim.track)
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", lw=1.2, alpha=0.7, label="F16 轨迹")
        ax3d.plot(tr[-1:, 0], tr[-1:, 1], tr[-1:, 2], "bo", ms=6)
        for R, c, nm in ((R1, "r", "雷达1"), (R2, "m", "雷达2")):
            ax3d.plot([R[0]], [R[1]], [R[2]], c + "*", ms=16, label=nm)
            ax3d.plot([R[0], tr[-1, 0]], [R[1], tr[-1, 1]], [R[2], tr[-1, 2]],
                      c + "--", lw=0.8, alpha=0.6)
        rng = max(np.linalg.norm(tr[-1, :2] - R1[:2]), 6000.0) * 1.15
        ax3d.set_xlim(R1[0] - rng, R1[0] + rng)
        ax3d.set_ylim(R1[1] - rng, R1[1] + rng)
        ax3d.set_zlim(min(0.0, tr[:, 2].min() - 500), max(4000.0, tr[:, 2].max() + 500))
        ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
        ax3d.set_title(f"双雷达同时探测  t={sim.t:.0f}s")
        ax3d.legend(loc="upper left", fontsize=7)
        ax3d.view_init(elev=28, azim=-58)

        ax_rcs.clear()
        ax_rcs.plot(ts, r1_db, "b-", lw=1.2, label="雷达1 单站")
        ax_rcs.plot(ts, r2_db, "m-", lw=1.2, alpha=0.85, label="雷达2 单站")
        ax_rcs.plot(ts, fz_db, "r-", lw=2.2, label=f"合成({args.fuse})")
        ax_rcs.plot(tb12, b12_db, "y.-", lw=1.5, alpha=0.9, label="真双站 σ12 (R1发→R2收)")
        ax_rcs.plot(ts[-1], fz_db[-1], "ro", ms=7)
        ax_rcs.set_xlabel("t (s)"); ax_rcs.set_ylabel("RCS (dBsm)")
        ax_rcs.set_title(f"双雷达两点合成 RCS  合成 {fz_db[-1]:.1f} dBsm")
        ax_rcs.grid(alpha=0.3); ax_rcs.legend(loc="lower left", fontsize=7)
        if len(ts) > 2:
            ax_rcs.set_xlim(max(ts[0], ts[-1] - 90), ts[-1])
            allv = r1_db + r2_db + fz_db + b12_db
            lo, hi = min(allv), max(allv)
            pad = max((hi - lo) * 0.15, 3.0)
            ax_rcs.set_ylim(lo - pad, hi + pad)

        # ---------- 三个方向图：入射=雷达1 / 入射=雷达2 / 功率平均示意 ----------
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
                # 本雷达视线方位
                dv = sim.pos - R
                azi = np.deg2rad(np.degrees(np.arctan2(dv[1], dv[0])))
                ax.plot([azi, azi], [-60, -35], c, lw=2.5)
                ax.text(azi, -55, nm, ha="center", color=c, fontsize=9)
            else:
                # 合成（功率平均示意）：标注两雷达观测方位
                for Rr, cr, lbl in ((R1, "r", "R1"), (R2, "m", "R2")):
                    dv = sim.pos - Rr
                    azi = np.deg2rad(np.degrees(np.arctan2(dv[1], dv[0])))
                    ax.plot([azi, azi], [-60, -35], cr, lw=1.5)
                    ax.text(azi, -55, lbl, ha="center", color=cr, fontsize=8)
        ax_pol1.set_title(f"双站方向图（入射=雷达1）  ★=σ12", fontsize=10)
        ax_pol2.set_title(f"双站方向图（入射=雷达2）  ★=σ21", fontsize=10)
        ax_pol3.set_title("功率平均示意（非物理观测）", fontsize=10)
        # ★ 标出真双站点：σ12 位于 p1 的 R2 散射方位；σ21 位于 p2 的 R1 散射方位
        for ax, ph_deg, dbv in ((ax_pol1, angs[1][1], sig12_db),
                                (ax_pol2, angs[0][1], sig21_db)):
            azi = np.deg2rad(ph_deg % 360.0)
            rv = max(float(dbv), -58)
            ax.plot([azi, azi], [-60, rv], "y:", lw=1.0)
            ax.scatter([azi], [rv], marker="*", s=220, color="y", edgecolors="k", zorder=5)

        ax_info.clear(); ax_info.axis("off")
        d1 = np.linalg.norm(sim.pos - R1); d2 = np.linalg.norm(sim.pos - R2)
        lines = [
            f"双雷达态势（t = {sim.t:.0f} s）",
            f"  距离 R1 : {d1/1000:.2f} km   距离 R2 : {d2/1000:.2f} km",
            f"  视线角 R1: θ={angs[0][0]:5.1f}° φ={angs[0][1]:6.1f}°",
            f"  视线角 R2: θ={angs[1][0]:5.1f}° φ={angs[1][1]:6.1f}°",
            f"  航向/俯仰/滚转: {np.degrees(sim.yaw)%360:5.1f}° / "
            f"{np.degrees(sim.pitch):+5.1f}° / {sim.roll:+5.1f}°",
            "",
            "各雷达单站 RCS（ONNX 实时推理）",
            f"  R1 : {r1_db[-1]:7.1f} dBsm",
            f"  R2 : {r2_db[-1]:7.1f} dBsm",
            f"  合成: {fz_db[-1]:7.1f} dBsm  ({args.fuse})",
            "",
            "真双站 RCS（一发一收，每 10 帧更新）",
            f"  σ12 (R1发→R2收): {sig12_db:7.1f} dBsm",
            f"  σ21 (R2发→R1收): {sig21_db:7.1f} dBsm",
            f"  互易差 |σ12−σ21|: {abs(sig12_db - sig21_db):6.2f} dB",
            f"  推理耗时: {ti_ms:.0f} ms/帧 (batch=2)",
            "",
            "说明：单站 RCS 闪烁不同→合成平滑视角依赖；",
            "真双站 σ12≈σ21 为互易性物理自洽检验；",
            "右图=方向图功率平均示意，非物理观测。",
        ]
        ax_info.text(0.02, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10)
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        plt.pause(0.001)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s "
                  f"RCS R1={r1_db[-1]:.1f} R2={r2_db[-1]:.1f} 合成={fz_db[-1]:.1f} dBsm "
                  f"推理 {ti_ms:.0f}ms", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
