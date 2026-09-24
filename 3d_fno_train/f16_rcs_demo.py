# -*- coding: utf-8 -*-
"""
f16_rcs_demo.py — F16 对雷达机动逼近的实时 RCS 模拟演示
======================================================
数据来源：3D F-FNO P3 模型（ONNX: results/fno_f16_3d_p3.onnx），端到端实时链路：
  飞行模拟（转弯/滚转/俯仰机动）→ 视线入射角(θ,φ) + 机体姿态偏移
  → 解析入射场(7 通道) → ONNX 推理(E+H 12 通道) → 表面等效流 NFFFT 单站 RCS
  → 实时图像（3D 战场 / RCS 时间曲线 / 远场方向图 / 状态面板）

角度约定（k̂ / 相位 / 观测方向与训练数据 FEKO 一致；e0 符号见下 ⚠）：
  入射角 (θ,φ) → k̂ = -[sinθcosφ, sinθsinφ, cosθ]（FEKO BETA0 方向余弦）
  e0 = θ̂ 极化 = [cosθcosφ, cosθsinφ, -sinθ]
      ⚠ 与训练表 incidence_table.npz 的关系：**e0_ana = −e0_tab**（实测 max|e0_tab+e0_ana| = 4.8e-07、
        max|e0_tab−e0_ana| = 2.0，468/468 角）⇒ 本文件用的是 θ̂，训练用的是 FEKO .out 的 −θ̂。
        因 RCS 取模方、本应无影响，但模型含非线性 ⇒ 逐方向有 2.4 dB 中位系统偏差（AGENTS.md item 6 结果二）。
        属**需授权**的部署链路事项，未改；khat 约定双方一致（max diff 6e-07）。
  相位 exp(-jβ·k̂·r)，β = 62.8754 rad/m（3.0 GHz）
  单站 RCS：观测方向 r̂ = +[sinθcosφ, sinθsinφ, cosθ]（散射回波朝入射反向，数据集单站点）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo.py --selftest   # 数值自检（vs h5 真值）
  & "F:/miniconda3/envs/isaac311/python.exe" f16_rcs_demo.py --seconds 180 --dt 0.5
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import sys
import time
import argparse

import numpy as np
import h5py
import torch
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
matplotlib.rcParams["axes.unicode_minus"] = False

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
from fno_f16_3d_p4_nffft import surface_parts, nffft, ETA0

RESULT_DIR = os.path.join(BASE, "results")
ONNX_PATH = os.path.join(RESULT_DIR, "fno_f16_3d_p3.onnx")
H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"
BETA0 = 62.8754                     # 3.0 GHz
GRID = (64, 48, 32)


# ============================================================
# 一、静态几何与模型加载
# ============================================================

class F16RCSEngine:
    """一次加载的推理引擎：eps 几何 + ONNX session + 标准化参数 + 表面面元。"""

    def __init__(self, onnx_path=ONNX_PATH):
        t0 = time.time()
        # 1) eps 体素场与网格坐标（只读 h5 小字段）
        with h5py.File(H5, "r") as f:
            self.eps = f["eps_field"][:]                      # (64,48,32)
            self.gx = f["grid_x"][:].astype(np.float64)
            self.gy = f["grid_y"][:].astype(np.float64)
            self.gz = f["grid_z"][:].astype(np.float64)
        self.eps_mask = (self.eps > 1.5).astype(np.float32)   # (64,48,32)
        self.X, self.Y, self.Z = np.meshgrid(self.gx, self.gy, self.gz, indexing="ij")
        self.X = self.X.astype(np.float32); self.Y = self.Y.astype(np.float32)
        self.Z = self.Z.astype(np.float32)
        h = float(np.median(np.diff(self.gx)))
        # 2) 表面面元（固定，只算一次）
        self.idxs, self.rsurf, self.dS = surface_parts(self.eps, self.gx, self.gy, self.gz, h)
        self.n = self.dS / (np.linalg.norm(self.dS, axis=1, keepdims=True) + 1e-12)
        # 3) 标准化参数（来自训练 ckpt）
        ckpt = torch.load(os.path.join(RESULT_DIR, "ckpt_full_p3.pt"),
                          map_location="cpu", weights_only=False)
        st = ckpt["stats"]
        self.xm, self.xs = np.asarray(st["x_inc_mean"], np.float32), np.asarray(st["x_inc_std"], np.float32)
        self.ym, self.ys = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        # 4) ONNX session（CUDA → CPU 回退）
        import onnxruntime as ort
        try:
            self.sess = ort.InferenceSession(onnx_path,
                                             providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        except Exception:
            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        print(f"引擎就绪: provider={self.provider} 表面体素={len(self.idxs)} "
              f"载入耗时 {time.time()-t0:.1f}s")

    # ---------- 入射场构造（与训练 load_data 完全一致） ----------
    def make_input(self, theta_deg, phi_deg):
        """入射角 (θ,φ)° → 标准化输入 (1,7,64,48,32)。khat=-k̂_std, e0=θ̂_std。"""
        t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
        st, ct, sp, cp = np.sin(t), np.cos(t), np.sin(p), np.cos(p)
        khat = -np.array([st * cp, st * sp, ct], np.float32)   # FEKO BETA0 方向余弦
        e0 = np.array([ct * cp, ct * sp, -st], np.float32)     # θ 极化（主极化）
        phase = np.exp(-1j * BETA0 * (khat[0] * self.X + khat[1] * self.Y + khat[2] * self.Z))
        e_inc = e0[None, None, None, :] * phase[..., None]     # (64,48,32,3)
        x = np.zeros((1, 7, *GRID), np.float32)
        x[0, 0] = self.eps_mask
        x[0, 1:4] = e_inc.real.transpose(3, 0, 1, 2)
        x[0, 4:7] = e_inc.imag.transpose(3, 0, 1, 2)
        return (x - self.xm) / self.xs

    def predict(self, x):
        """ONNX 推理 → 反标准化 12 通道场 (12,64,48,32)。"""
        out = self.sess.run(None, {"input": x.astype(np.float32)})[0][0]
        return out * self.ys.reshape(12, 1, 1, 1) + self.ym.reshape(12, 1, 1, 1)

    # ---------- 单站 RCS（入射角 = 观测角，雷达回波） ----------
    def rcs_single(self, pred, theta_deg, phi_deg, e0mag=1.0):
        t, p = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
        st, ct, sp, cp = np.sin(t), np.cos(t), np.sin(p), np.cos(p)
        rhat = np.array([st * cp, st * sp, ct], np.float64)      # 散射回波方向
        th = np.array([ct * cp, ct * sp, -st], np.float64)
        ph = np.array([-sp, cp, 0.0], np.float64)
        E_s, H_s = self.pred_to_surface(pred)
        J = np.cross(self.n, H_s)                                # n̂×H_tot（PEC: M=0）
        E_ff = nffft(J, np.zeros_like(J), self.rsurf, self.dS, rhat[None, :], BETA0)
        E_th = (E_ff * th).sum(); E_ph = (E_ff * ph).sum()      # 复数（保留虚部）
        return 4.0 * np.pi * (abs(E_th) ** 2 + abs(E_ph) ** 2) / e0mag ** 2

    # ---------- 全网格远场方向图（用于方向图子图，慢更新） ----------
    def rcs_pattern(self, pred, e0mag=1.0):
        th_grid = np.arange(0, 181, 5)
        ph_grid = np.arange(0, 360, 5)
        T, P = np.meshgrid(np.deg2rad(th_grid), np.deg2rad(ph_grid), indexing="ij")
        T = T.ravel(); P = P.ravel()
        st, ct = np.sin(T), np.cos(T); sp, cp = np.sin(P), np.cos(P)
        rhat = np.stack([st * cp, st * sp, ct], axis=1)
        E_s, H_s = self.pred_to_surface(pred)
        J = np.cross(self.n, H_s)
        E_ff = nffft(J, np.zeros_like(J), self.rsurf, self.dS, rhat, BETA0)
        th_v = np.stack([ct * cp, ct * sp, -st], axis=1)
        ph_v = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
        Eth = (E_ff * th_v).sum(axis=1); Eph = (E_ff * ph_v).sum(axis=1)
        rcs = 4.0 * np.pi * (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / e0mag ** 2
        return 10 * np.log10(np.maximum(rcs, 1e-9)).reshape(len(th_grid), len(ph_grid)), th_grid, ph_grid

    def pred_to_surface(self, pred):
        i0, i1, i2 = self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]
        E3 = pred[0:3] + 1j * pred[3:6]
        H3 = pred[6:9] + 1j * pred[9:12]
        return E3[:, i0, i1, i2].T, H3[:, i0, i1, i2].T


# ============================================================
# 二、F16 飞行模拟器（转弯 / 滚转 / 俯仰机动）
# ============================================================

def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class F16Sim:
    """F16 朝雷达机动逼近：航向持续转向雷达（限制最大转向率），
    转弯时滚转倾斜联动，加入小幅俯仰起伏机动；逼近后掉头脱离。"""

    def __init__(self, radar=(0.0, 0.0, 0.02), start=(8200.0, 9500.0, 3000.0),
                 speed=260.0, max_turn_deg=14.0):
        self.radar = np.asarray(radar, np.float64)
        self.pos = np.asarray(start, np.float64)
        self.speed = speed                       # m/s
        self.yaw = np.deg2rad(-65.0)             # 初始航向明显偏离目标方向（产生转弯）
        self.pitch = 0.0
        self.roll = 0.0
        self.max_turn = np.deg2rad(max_turn_deg) # 最大转向率 rad/s
        self.t = 0.0
        self.retreat = False
        self.track = [self.pos.copy()]

    def step(self, dt):
        d = self.radar - self.pos
        psi_t = np.arctan2(d[1], d[0])           # 指向雷达的目标航向
        if self.retreat:
            psi_t = psi_t + np.pi                # 掉头脱离
        dyaw = _wrap(psi_t - self.yaw)
        self.yaw += np.clip(dyaw, -self.max_turn * dt, self.max_turn * dt)
        # 滚转倾斜联动（转弯越急滚转越大，限幅 ±55°）
        self.roll = float(np.clip(-np.degrees(dyaw) * 1.5, -55.0, 55.0))
        # 俯仰起伏机动（模拟操纵）
        self.t += dt
        target_pitch = 0.10 * np.sin(0.35 * self.t) + 0.05 * np.sin(0.11 * self.t)
        self.pitch = float(0.9 * self.pitch + 0.1 * target_pitch)
        # 位置积分（沿航向/俯仰）
        v = self.speed * np.array([np.cos(self.yaw) * np.cos(self.pitch),
                                   np.sin(self.yaw) * np.cos(self.pitch),
                                   np.sin(self.pitch)])
        self.pos = self.pos + v * dt
        self.track.append(self.pos.copy())
        if len(self.track) > 400:
            self.track.pop(0)
        if np.linalg.norm(d) < 2600.0:
            self.retreat = True
        # 确保高度在地面上方
        self.pos[2] = max(self.pos[2], 100.0)
        return self.pos.copy()

    def los_angles(self):
        """视线入射角：d̂=雷达→F16。返回 (θ_los, φ_los) 度。"""
        d = self.pos - self.radar
        d /= np.linalg.norm(d)
        return float(np.degrees(np.arccos(np.clip(d[2], -1, 1)))), float(np.degrees(np.arctan2(d[1], d[0])))

    def body_offset(self):
        """机体姿态 → 有效入射角偏移（度）：航向差放大 2×，俯仰/滚转联动。"""
        d = self.radar - self.pos
        psi_los = np.arctan2(d[1], d[0])
        yaw_off = np.degrees(_wrap(self.yaw - psi_los))   # 机体相对视线偏航
        return 2.0 * yaw_off, 2.0 * np.degrees(self.pitch) + 0.35 * self.roll


# ============================================================
# 三、实时可视化
# ============================================================

def setup_figure():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    plt.ion()
    fig = plt.figure(figsize=(17, 9))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_rcs = fig.add_subplot(2, 2, 2)
    ax_pol = fig.add_subplot(2, 2, 3, projection="polar")
    ax_info = fig.add_subplot(2, 2, 4)
    ax_info.axis("off")
    fig.tight_layout(pad=2.5)
    return fig, ax3d, ax_rcs, ax_pol, ax_info


def draw_frame(fig, ax3d, ax_rcs, ax_pol, ax_info, sim, ts, rcs_db,
               phi_disp, theta_disp, pat_db, pat_th, pat_ph, ti_ms):
    import matplotlib.pyplot as plt
    # ---- 3D 战场 ----
    ax3d.clear()
    tr = np.asarray(sim.track)
    ax3d.plot(tr[:, 0], tr[:, 1], tr[:, 2], "b-", lw=1.2, alpha=0.7, label="F16 轨迹")
    ax3d.plot(tr[-1:, 0], tr[-1:, 1], tr[-1:, 2], "bo", ms=6)
    ax3d.plot([sim.radar[0]], [sim.radar[1]], [sim.radar[2]], "r*", ms=18, label="雷达")
    # 机体姿态箭头（航向 + 俯仰）
    d = sim.pos - sim.radar
    s = 1200.0 / max(np.linalg.norm(d) / 1000.0, 2.0)     # 箭头长度随距离缩放
    vx = np.cos(sim.yaw) * np.cos(sim.pitch)
    vy = np.sin(sim.yaw) * np.cos(sim.pitch)
    vz = np.sin(sim.pitch)
    ax3d.quiver(tr[-1, 0], tr[-1, 1], tr[-1, 2], vx, vy, vz, color="k", length=s, label="机体朝向")
    # 视线
    ax3d.plot([sim.radar[0], tr[-1, 0]], [sim.radar[1], tr[-1, 1]],
              [sim.radar[2], tr[-1, 2]], "r--", lw=0.8, alpha=0.6)
    rng = max(np.linalg.norm(tr[-1, :2] - sim.radar[:2]), 6000.0) * 1.15
    ax3d.set_xlim(sim.radar[0] - rng, sim.radar[0] + rng)
    ax3d.set_ylim(sim.radar[1] - rng, sim.radar[1] + rng)
    z0, z1 = min(0.0, tr[:, 2].min() - 500), max(4000.0, tr[:, 2].max() + 500)
    ax3d.set_zlim(z0, z1)
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title(f"F16 对雷达机动逼近  t={sim.t:.0f}s  距离={np.linalg.norm(d):.1f} km")
    ax3d.legend(loc="upper left", fontsize=7)
    ax3d.view_init(elev=28, azim=-58)

    # ---- RCS 时间曲线 ----
    ax_rcs.clear()
    ax_rcs.plot(ts, rcs_db, "g-", lw=1.4)
    ax_rcs.plot(ts[-1], rcs_db[-1], "ro", ms=7)
    ax_rcs.set_xlabel("t (s)"); ax_rcs.set_ylabel("RCS (dBsm)")
    ax_rcs.set_title(f"单站 RCS（ONNX 实时推理）  当前 {rcs_db[-1]:.1f} dBsm")
    ax_rcs.grid(alpha=0.3)
    if len(ts) > 2:
        ax_rcs.set_xlim(max(ts[0], ts[-1] - 90), ts[-1])
        lo = min(rcs_db); hi = max(rcs_db)
        pad = max((hi - lo) * 0.15, 3.0)
        ax_rcs.set_ylim(lo - pad, hi + pad)

    # ---- 远场方向图（当前入射俯仰切面） ----
    ax_pol.clear()
    cut = int(np.argmin(np.abs(pat_th - 90)))      # θ=90° 俯仰切面
    db = pat_db[cut]
    ang = np.deg2rad(pat_ph)
    dbc = np.maximum(db, -60)                       # 显示下限
    ax_pol.plot(ang, dbc, lw=1.2)
    ax_pol.set_ylim(-60, 0)
    ax_pol.set_title(f"远场 RCS 方向图 φ-cut (θ=90°, 最近更新)")
    ax_pol.grid(alpha=0.3)

    # ---- 状态面板 ----
    ax_info.clear(); ax_info.axis("off")
    d = sim.pos - sim.radar
    lines = [
        f"F16 状态（t = {sim.t:.0f} s）",
        f"  位置    : ({sim.pos[0]:.0f}, {sim.pos[1]:.0f}, {sim.pos[2]:.0f}) m",
        f"  距离    : {np.linalg.norm(d)/1000:.2f} km",
        f"  速度    : {sim.speed:.0f} m/s",
        f"  航向 ψ  : {np.degrees(sim.yaw) % 360:6.1f}°",
        f"  俯仰 θa : {np.degrees(sim.pitch):+5.1f}°",
        f"  滚转 φr : {sim.roll:+5.1f}°",
        "",
        "RCS 推理（3D F-FNO P3 ONNX）",
        f"  视线角  : θ={theta_disp:5.1f}°  φ={phi_disp:6.1f}°",
        f"  单站 RCS: {rcs_db[-1]:7.1f} dBsm",
        f"  推理耗时: {ti_ms:6.0f} ms/帧",
        f"  Provider: {ENGINE.provider}",
        "",
        "说明：有效入射角 = 视线角 + 机体姿态偏移",
        "（转弯/滚转/俯仰直接影响 RCS 波形）",
    ]
    ax_info.text(0.02, 0.97, "\n".join(lines), va="top", ha="left",
                 fontsize=10)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()


ENGINE = None   # 全局引擎引用（状态面板显示 provider）


# ============================================================
# 四、自检（vs h5 真实单站 RCS）
# ============================================================

def selftest():
    global ENGINE
    eng = F16RCSEngine()
    ENGINE = eng
    with h5py.File(H5, "r") as f:
        rcs_true = f["rcs"][:]
        ff_t = f["ff_theta"][:]; ff_p = f["ff_phi"][:]
        h_ang = f["angles"][:]
    print("\n=== 单站 RCS 自检（ONNX+NFFFT vs FEKO h5）===")
    for i in [0, 216, 300]:
        t0 = time.time()
        theta, phi = float(h_ang[i, 0]), float(h_ang[i, 1])
        x = eng.make_input(theta, phi)
        pred = eng.predict(x)
        rcs = eng.rcs_single(pred, theta, phi)
        # 单站参考：观测角 = 入射角（最近 5° 网格）
        it = int(np.argmin(np.abs(ff_t - theta))); ip = int(np.argmin(np.abs(ff_p - phi)))
        ref = float(rcs_true[i, it, ip])
        db = 10 * np.log10(rcs + 1e-9); dbref = 10 * np.log10(ref + 1e-9)
        print(f"  case_{i:03d} (θ={theta:.0f},φ={phi:.0f}): "
              f"ONNX {db:7.2f} dBsm vs FEKO {dbref:7.2f} dBsm  |Δ|={abs(db-dbref):.2f} dB  "
              f"({(time.time()-t0)*1000:.0f} ms)")
    print("（P4 参考：模型+管线 |ΔRCS| 中位 ~4 dB）")


# ============================================================
# 五、主循环
# ============================================================

def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=180.0, help="模拟秒数")
    ap.add_argument("--dt", type=float, default=0.5, help="模拟步长 s")
    ap.add_argument("--cut-every", type=int, default=10, help="方向图更新间隔（帧）")
    ap.add_argument("--headless", action="store_true", help="无窗口（Agg backend，测试用）")
    ap.add_argument("--snapshot", default=None, help="每 30 帧覆盖保存 PNG 快照")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    if args.headless:
        import matplotlib
        matplotlib.use("Agg")

    eng = F16RCSEngine()
    ENGINE = eng
    sim = F16Sim()

    fig, ax3d, ax_rcs, ax_pol, ax_info = setup_figure()
    ts, rcs_db, phis, thetas = [], [], [], []
    pat_db, pat_th, pat_ph = None, None, None
    nsteps = int(args.seconds / args.dt)

    print(f"开始演示: {args.seconds}s 模拟 @ dt={args.dt}s, {nsteps} 帧")
    t_prev = time.time()
    for k in range(nsteps):
        sim.step(args.dt)
        theta_los, phi_los = sim.los_angles()
        dyaw_deg, dpitch_deg = sim.body_offset()
        # 有效入射角（限制在数据集范围 30-150°）
        theta_eff = float(np.clip(theta_los + dpitch_deg, 25.0, 155.0))
        phi_eff = (phi_los + dyaw_deg) % 360.0
        t0 = time.time()
        x = eng.make_input(theta_eff, phi_eff)
        pred = eng.predict(x)
        rcs = eng.rcs_single(pred, theta_eff, phi_eff)
        ti_ms = (time.time() - t0) * 1000
        ts.append(sim.t)
        rcs_db.append(10 * np.log10(rcs + 1e-9))
        phis.append(phi_eff); thetas.append(theta_eff)
        if pat_db is None or k % args.cut_every == 0:
            pat_db, pat_th, pat_ph = eng.rcs_pattern(pred)
        draw_frame(fig, ax3d, ax_rcs, ax_pol, ax_info, sim, ts, rcs_db,
                   phi_eff, theta_eff, pat_db, pat_th, pat_ph, ti_ms)
        if args.snapshot and (k % 30 == 0):
            fig.savefig(args.snapshot, dpi=80)
        # 节流：确保界面可响应（近似 1× 实时或更慢）
        plt_pause = 0.001
        import matplotlib.pyplot as plt
        plt.pause(plt_pause)
        if (k + 1) % 50 == 0:
            print(f"  帧 {k+1}/{nsteps} t={sim.t:.0f}s RCS={rcs_db[-1]:.1f} dBsm "
                  f"推理 {ti_ms:.0f}ms 累计 {(time.time()-t_prev):.0f}s", flush=True)
    print("演示结束。窗口保持，关闭以退出。")
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
