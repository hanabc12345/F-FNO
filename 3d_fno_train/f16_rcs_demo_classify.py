# -*- coding: utf-8 -*-
"""
f16_rcs_demo_classify.py — 双雷达 RCS 特征 + 模板库 型号识别演示
================================================================
在双雷达两点观测基础上加入"型号识别"：
  特征 = 双雷达单站 RCS 向量 z = [σ₁, σ₂] (dBsm) + 已知视线角
  模板 = F-16（FEKO 真值方向图 h5）/ 无人机（解析弱目标）/ 巡飞弹（解析中强目标）
  判定 = 高斯负对数似然（chi2）→ softmax 置信度 → argmax 识别

模板与观测不同源（观测=ONNX 代理预测或模板抽样），识别误差真实反映
"代理精度 + 视角闪烁"下的多目标区分能力。

--target 切换被探测目标：
  f16    : 观测 = ONNX 代理实时推理（真实链路）
  drone  : 观测 = 无人机模板 + Swerling 起伏
  missile: 观测 = 巡飞弹模板 + Swerling 起伏

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_classify.py --selftest
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo_classify.py --seconds 180 --target f16
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import sys
import time
import argparse

import numpy as np
import h5py

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import f16_rcs_demo as D
from f16_rcs_demo_2radar import F16RCSEngine2, fuse_rcs, pattern_at

H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"


# ============================================================
# 一、目标模板库
# ============================================================

class F16Template:
    """F-16 模板：FEKO 真值 468 角度单站 RCS 方向图（主极化 θ-pol）。
    观测视角 (θ,φ) → 最近入射角 → 该方向图在 (θ,φ) 的散射 RCS。"""
    name = "F-16"

    def __init__(self, h5path=H5):
        with h5py.File(h5path, "r") as f:
            self.rcs = f["rcs"][:]            # (468,37,73) 单站 RCS m²（入射=行，方向图=列）
            self.ang = f["angles"][:]         # (468,2) θ,φ
        self.th_g = np.arange(0, 181, 5)
        self.ph_g = np.arange(0, 360, 5)

    def _inc_idx(self, th, ph):
        i = int(np.clip(round((th - 30) / 10), 0, 12))
        j = int(round(ph / 10)) % 36
        return i * 36 + j

    def mu_single(self, th, ph):
        i = self._inc_idx(th, ph)
        ti = int(np.clip(round(th / 5), 0, 36))
        pi = int(round(ph / 5)) % 72
        return 10.0 * np.log10(self.rcs[i, ti, pi] + 1e-9)

    def std(self, th, ph):
        return 8.0                          # 代理误差(P4 RCS P90≈10dB) + 视角闪烁


class DroneTemplate:
    """小型无人机（四旋翼 ~1m）：弱、近各向同性。"""
    name = "无人机"
    MU = -12.0
    STD = 3.0

    def mu_single(self, th, ph):
        return self.MU

    def std(self, th, ph):
        return self.STD


class MissileTemplate:
    """巡飞弹（~2m 细长体）：中强目标，平均 RCS 高于无人机。"""
    name = "巡飞弹"
    MU = -2.0
    STD = 4.0

    def mu_single(self, th, ph):
        return self.MU

    def std(self, th, ph):
        return self.STD


TEMPLATES = [F16Template(), DroneTemplate(), MissileTemplate()]
TBY = {t.name: t for t in TEMPLATES}


def classify(z_db, view_angles, templates):
    """单帧分类：双雷达 RCS 特征 z_db=[σ1,σ2] (dBsm) + 视角 → (各模板概率, 最优名)。"""
    loglik = []
    for T in templates:
        chi2 = 0.0
        for i, (th, ph) in enumerate(view_angles):
            mu, s = T.mu_single(th, ph), T.std(th, ph)
            chi2 += ((z_db[i] - mu) / s) ** 2
        loglik.append(-0.5 * chi2)
    loglik = np.asarray(loglik)
    mx = loglik.max()
    p = np.exp(loglik - mx)
    p = p / p.sum()
    return p, templates[int(loglik.argmax())].name


def classify_seq(z_list, ang_list, templates):
    """多帧累积分类（视角自适应）：逐帧用该帧视角匹配模板求和（贝叶斯累积），
    抑制单帧闪烁且不破坏视角信息——比"特征先取中位数再比当前视角"物理正确。"""
    acc = np.zeros(len(templates))
    for z, angs in zip(z_list, ang_list):
        for ti, T in enumerate(templates):
            for i, (th, ph) in enumerate(angs):
                mu, s = T.mu_single(th, ph), T.std(th, ph)
                acc[ti] += ((z[i] - mu) / s) ** 2
    loglik = -0.5 * acc
    mx = loglik.max()
    p = np.exp(loglik - mx)
    p = p / p.sum()
    return p, templates[int(loglik.argmax())].name


# ============================================================
# 二、自检：识别器对三种目标应正确判定
# ============================================================

def selftest():
    print("\n=== 型号识别自检（模板加载 + 三目标判定）===")
    print(f"  F-16 模板载入: rcs{np.shape(TEMPLATES[0].rcs)} 角度{len(TEMPLATES[0].ang)}")
    rng = np.random.default_rng(0)
    views = [(80.0, 30.0), (100.0, 120.0)]
    ok = True
    for tgt in ("f16", "drone", "missile"):
        if tgt == "f16":
            T = TBY["F-16"]
            # 用真值加观测噪声模拟代理误差
            z = [T.mu_single(*v) + float(rng.normal(0, 4.0)) for v in views]
        else:
            T = TBY["无人机" if tgt == "drone" else "巡飞弹"]
            z = [T.mu_single(*v) + float(rng.normal(0, 1.5)) for v in views]
        p, best = classify(z, views, TEMPLATES)
        good = best == ("F-16" if tgt == "f16" else "无人机" if tgt == "drone" else "巡飞弹")
        ok &= good
        print(f"  目标={tgt:8s} 观测 z={z[0]:6.1f},{z[1]:6.1f} dBsm → "
              f"F16={p[0]*100:4.0f}% 无人机={p[1]*100:4.0f}% 巡飞弹={p[2]*100:4.0f}% → {best} {'✓' if good else '✗'}")
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
    ap.add_argument("--target", choices=["f16", "drone", "missile"], default="f16",
                    help="被探测目标：f16=ONNX 代理链路；drone/missile=模板抽样")
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
    R1 = np.asarray([0.0, 0.0, 0.02])
    R2 = np.asarray([7000.0, -5000.0, 0.02])
    tgt = TBY["F-16" if args.target == "f16" else "无人机" if args.target == "drone" else "巡飞弹"]
    rng = np.random.default_rng(1)

    plt.ion()
    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.05, 0.9, 0.95, 0.8])
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax_cls = fig.add_subplot(gs[0, 2])
    ax_rcs = fig.add_subplot(gs[1, :])
    ax_pol1 = fig.add_subplot(gs[2, 0], projection="polar")
    ax_pol2 = fig.add_subplot(gs[2, 1], projection="polar")
    ax_pol3 = fig.add_subplot(gs[2, 2], projection="polar")
    ax_info = fig.add_subplot(gs[3, :])
    ax_info.axis("off")
    fig.tight_layout(pad=2.5)

    ts, r1_db, r2_db, fz_db, tb12, b12_db = [], [], [], [], [], []
    pats = None
    sig12_db = sig21_db = float("nan")
    frames_buf = []
    WIN = 10                          # 识别累积窗（帧）
    probs = np.ones(len(TEMPLATES)) / len(TEMPLATES)
    best = "—"
    nsteps = int(args.seconds / args.dt)

    print(f"开始型号识别演示: {args.seconds}s @ dt={args.dt}s 目标={args.target} {nsteps} 帧")
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
        # 观测 RCS：f16=代理推理；drone/missile=模板抽样
        if args.target == "f16":
            x = eng.make_inputs([a[0] for a in angs], [a[1] for a in angs])
            preds = eng.predict_batch(x)
            rcs1 = eng.rcs_single(preds[0], angs[0][0], angs[0][1])
            rcs2 = eng.rcs_single(preds[1], angs[1][0], angs[1][1])
        else:
            z_t = [tgt.mu_single(a[0], a[1]) + float(rng.normal(0, tgt.std(a[0], a[1]) * 0.5))
                   for a in angs]
            rcs1, rcs2 = 10.0 ** (z_t[0] / 10.0), 10.0 ** (z_t[1] / 10.0)
        fused = fuse_rcs([rcs1, rcs2], args.fuse)
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        r1_db.append(10 * np.log10(rcs1 + 1e-9))
        r2_db.append(10 * np.log10(rcs2 + 1e-9))
        fz_db.append(10 * np.log10(fused + 1e-9))

        # 型号识别（每帧，视角自适应多帧似然累积抑制闪烁）
        z_obs = [10 * np.log10(rcs1 + 1e-9), 10 * np.log10(rcs2 + 1e-9)]
        frames_buf.append((z_obs, list(angs)))
        if len(frames_buf) > WIN:
            frames_buf.pop(0)
        probs, best = classify_seq([f[0] for f in frames_buf],
                                   [f[1] for f in frames_buf], TEMPLATES)

        if args.target == "f16" and (pats is None or k % 10 == 0):
            p1 = eng.rcs_pattern(preds[0])
            p2 = eng.rcs_pattern(preds[1])
            pwr = 0.5 * (10.0 ** (p1[0] / 10.0) + 10.0 ** (p2[0] / 10.0))
            p3 = (10.0 * np.log10(np.maximum(pwr, 1e-9)), p1[1], p1[2])
            pats = (p1, p2, p3)
            sig12_db = pattern_at(p1, angs[1][0], angs[1][1])
            sig21_db = pattern_at(p2, angs[0][0], angs[0][1])
            tb12.append(sim.t)
            b12_db.append(sig12_db)

        # ---------- 绘图 ----------
        ax3d.clear()
        tr = np.asarray(sim.track)
        ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", lw=1.2, alpha=0.7, label="轨迹")
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
        ax3d.set_title(f"双雷达同时探测  t={sim.t:.0f}s")
        ax3d.legend(loc="upper left", fontsize=7)
        ax3d.view_init(elev=28, azim=-58)

        # 识别面板
        ax_cls.clear()
        names = [t.name for t in TEMPLATES]
        cols = ["C0", "C2", "C3"]
        ax_cls.barh(range(len(names)), probs * 100, color=cols, alpha=0.85, height=0.55)
        ax_cls.set_yticks(range(len(names)))
        ax_cls.set_yticklabels(names)
        ax_cls.set_xlim(0, 100)
        ax_cls.set_xlabel("置信度 (%)")
        for i, pr in enumerate(probs):
            ax_cls.text(pr * 100 + 1, i, f"{pr*100:.0f}%", va="center", fontsize=10)
        ax_cls.invert_yaxis()
        ax_cls.grid(axis="x", alpha=0.3)
        ax_cls.set_title(f"型号识别 → {best}\n目标设定: {args.target}", fontsize=10)
        ax_cls.axvline(50, color="gray", ls=":", lw=0.8)

        ax_rcs.clear()
        ax_rcs.plot(ts, r1_db, "b-", lw=1.2, label="雷达1 单站")
        ax_rcs.plot(ts, r2_db, "m-", lw=1.2, alpha=0.85, label="雷达2 单站")
        ax_rcs.plot(ts, fz_db, "r-", lw=2.2, label=f"合成({args.fuse})")
        if args.target == "f16" and b12_db:
            ax_rcs.plot(tb12, b12_db, "y.-", lw=1.5, alpha=0.9, label="真双站 σ12")
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

        # 三个方向图（仅 f16 有真实方向图；其他目标跳过）
        for ax, pat, R, c, nm in ((ax_pol1, None, R1, "r", "雷达1"),
                                  (ax_pol2, None, R2, "m", "雷达2"),
                                  (ax_pol3, None, None, "k", "合成")):
            ax.clear()
            ax.set_ylim(-60, 0)
            ax.grid(alpha=0.3)
        if args.target == "f16" and pats is not None:
            for ax, pat, R, c, nm in ((ax_pol1, pats[0], R1, "r", "雷达1"),
                                      (ax_pol2, pats[1], R2, "m", "雷达2"),
                                      (ax_pol3, pats[2], None, "k", "合成")):
                cut = int(np.argmin(np.abs(pat[1] - 90)))
                db = np.maximum(pat[0][cut], -60)
                ax.plot(np.deg2rad(pat[2]), db, lw=1.2)
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
            ax_pol1.set_title(f"双站方向图（入射=雷达1）  ★=σ12", fontsize=9)
            ax_pol2.set_title(f"双站方向图（入射=雷达2）  ★=σ21", fontsize=9)
            ax_pol3.set_title("功率平均示意（非物理观测）", fontsize=9)
            for ax, ph_deg, dbv in ((ax_pol1, angs[1][1], sig12_db),
                                    (ax_pol2, angs[0][1], sig21_db)):
                azi = np.deg2rad(ph_deg % 360.0)
                rv = max(float(dbv), -58)
                ax.plot([azi, azi], [-60, rv], "y:", lw=1.0)
                ax.scatter([azi], [rv], marker="*", s=200, color="y", edgecolors="k", zorder=5)
        else:
            ax_pol1.set_title("方向图（目标非 F-16，仅显示）", fontsize=9)

        ax_info.clear(); ax_info.axis("off")
        d1 = np.linalg.norm(sim.pos - R1); d2 = np.linalg.norm(sim.pos - R2)
        lines = [
            f"型号识别（t = {sim.t:.0f}s）  目标设定: {args.target}  判定: {best}",
            f"  观测 z=[σ1,σ2] = {z_obs[0]:6.1f}, {z_obs[1]:6.1f} dBsm (累积{len(frames_buf)}帧)   "
            f"F-16={probs[0]*100:4.0f}%  无人机={probs[1]*100:4.0f}%  巡飞弹={probs[2]*100:4.0f}%",
            f"  距离 R1={d1/1000:.2f}km R2={d2/1000:.2f}km  合成RCS={fz_db[-1]:6.1f} dBsm",
            f"  真双站 σ12={sig12_db:6.1f}  σ21={sig21_db:6.1f}  互易差={abs(sig12_db-sig21_db):4.1f} dB"
            if args.target == "f16" else
            f"  模板目标观测（Swerling 起伏抽样）  μ={tgt.mu_single(0,0):.1f} dBsm",
            "",
            "识别原理：观测 z=[σ1,σ2] 逐帧按该帧视线角匹配模板，",
            "多帧高斯负对数似然累积（贝叶斯）→ softmax 置信度，抑制闪烁；",
            "F-16 模板=FEKO 真值 468 角度方向图（观测=ONNX 代理，不同源→含真实误差）；",
            "无人机/巡飞弹=解析模型（弱各向同性 / 中强目标）。",
        ]
        ax_info.text(0.01, 0.97, "\n".join(lines), va="top", ha="left", fontsize=10)
        fig.canvas.draw_idle(); fig.canvas.flush_events()
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        plt.pause(0.001)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s 目标={args.target} 判定={best} "
                  f"F16={probs[0]*100:.0f}% 无人机={probs[1]*100:.0f}% 巡飞弹={probs[2]*100:.0f}% "
                  f"推理 {ti_ms:.0f}ms", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
