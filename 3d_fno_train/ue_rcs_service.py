# -*- coding: utf-8 -*-
"""
ue_rcs_service.py — FNO-RCS 实时推理 TCP 服务（Python 端）
===========================================================
拓扑：本机 Python 作为 TCP 服务端，监听 0.0.0.0:9001；UE 客户端主动连接
      （UE 连接地址 = 本机 LAN IP + 端口，启动时打印，如 172.22.43.18:9001）。
      连接建立后按请求-响应循环工作（小端二进制，支持 1~4 站点 + 每站压制干扰，详见下方协议）：
        UE → Python : 33+17×N B 请求（ts + pos + rot(deg) + N 个站点：radar/jammer_on/jam_db）
        Python → UE : 单站包（表观RCS/干扰状态/双站σ_ij）+ 可选方向图包
      数据接口见 docs/飞机数据.md。

二进制协议（全部小端）：
  请求（33+17×N B，N=1~4 个站点）:
              头 33 B：int64 timestamp(ms) + aircraft_pos[3] + aircraft_rot[3](pitch,yaw,roll,deg)
                        + uint8 n_stations
              每站 17 B：radar_pos[3] + uint8 jammer_on(0/1) + float32 jam_db_at4km
                        （4km 处干扰标定 dBsm，σ_jam∝R²，参考距离 4km）
  响应·单站包（16+24N+4N(N-1) B，Type=0 仅单点 / Type=1 其后附方向图）：
              int32   Type + int64 timestamp(ms) + int32 n_stations
              每站 16 B：los_theta, los_phi, eff_theta, eff_phi（float32）
              每站 4 B： rcs_apparent_db（表观 RCS=物理单站+σ_jam，dBsm）
              每站 4 B： jam_state（0=未探测 1=压制 2=烧穿/正常跟踪）
              N(N-1)×4 B：σ_ij_db（i发j收的有序双站 RCS，行主序；N<2 为空）
  响应·方向图包（24+16×n B，Type=1 时紧跟单站包之后；基于站点1的近场合成，
             θ 切面随站点1 的 LOS 有效入射角动态变化）：
              int32 Type(=2) + int64 ts(ms) + theta_deg(double) + n(int32)
              + PhiDegArr/RcsDbArr(double×n)
  异常约定：帧内任一站点角度越界/退化 → 整帧无效（所有 apparent/σ_ij 返回 -100，state=0，
             UE 以 apparent ≤ -99 作无效过滤）。

角度/坐标系约定（与 UE 一致）：
  - UE 坐标：X 前 / Y 右 / Z 上（左手系），旋转 FRotator(yaw, pitch, roll) rad
  - 模型训练坐标系：F16 机头 +X、右翼 +Y、上 +Z（与 UE 机体系一致）
  - 有效入射角 = 雷达→飞机方向矢量，经"UE 机体→世界旋转矩阵"转置变换到机体系后的 (θ,φ)
  - 模型训练范围 θ∈[30°,150°]，越界不做推理，valid=0 且 RCS 置 0

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" ue_rcs_service.py --selftest                    # 自检（不需 UE）
  & "F:/miniconda3/envs/isaac311/python.exe" ue_rcs_service.py                                # 监听 0.0.0.0:9001
  & "F:/miniconda3/envs/isaac311/python.exe" ue_rcs_service.py --host 127.0.0.1 --port 9001  # 仅本机联调
  UE 客户端连接地址：启动时打印的本机 LAN IP:端口（默认端口 9001）
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import sys
import time
import struct
import socket
import queue
import threading
import argparse

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from fno_f16_3d_p4_nffft import surface_parts, nffft, ETA0  # noqa: E402

RESULT_DIR = os.path.join(BASE, "results")
ONNX_PATH = os.path.join(RESULT_DIR, "fno_f16_3d_p3.onnx")
CKPT_PATH = os.path.join(RESULT_DIR, "ckpt_full_p3.pt")
H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"

BETA0 = 62.8754            # 3.0 GHz 波数 rad/m
GRID = (64, 48, 32)
THETA_MIN, THETA_MAX = 30.0, 150.0   # 模型有效入射角 θ 训练范围（越界 valid=0）

DEFAULT_HOST = "0.0.0.0"             # Python 服务端监听全部网卡（UE 连接本机 LAN IP）
DEFAULT_PORT = 9001

# ---- 二进制协议（小端，可变长度，支持 1~4 站点 + 每站压制干扰） ----
# 请求：头 33 B（<q3f3fB）= int64 ts(ms) + pos[3] + rot[3](deg) + uint8 n_stations
REQ_HEAD = struct.Struct("<q3f3fB")
#       每站 17 B（<3fBf）= radar_pos[3] + uint8 jammer_on + float32 jam_db_at4km
REQ_STATION = struct.Struct("<3fBf")
# 单站包响应：头 16 B（<iqi）= Type(int32) + ts(ms) + n_stations(int32)
RESP_HEAD = struct.Struct("<iqi")
RESP_STATION_ANGLES = struct.Struct("<4f")   # los_theta,los_phi,eff_theta,eff_phi
RESP_STATION_FLOAT = struct.Struct("<f")     # 表观 RCS / σ_ij
RESP_STATION_STATE = struct.Struct("<i")     # jam_state
# 方向图包 24+16×n B: Type(int32=2) + int64 ts(ms) + theta_deg(double) + n(int32) + PhiDegArr/RcsDbArr(double×n)
PAT_HEAD = struct.Struct("<iqdi")

TYPE_SINGLE_ONLY = 0         # 单站包 Type：仅单点图
TYPE_SINGLE_WITH_PATTERN = 1 # 单站包 Type：其后附带方向图包
TYPE_PATTERN = 2             # 方向图包自身的 Type

PAT_STEP = 10.0      # 方向图方位角步长 °（0~360 → 37 点）
MAX_STATIONS = 4     # 单帧最大站点数
MAX_BATCH = 8        # 每批最多帧数（帧×站=样本数，见 MAX_SAMPLES）
MAX_SAMPLES = 8      # 每批最多 ONNX 样本数（帧内各站各 1 样本；实测样本 batch>8 时 ScatterND 灾难性退化）

# ---- 压制干扰模型（与 f16_rcs_demo_ecm 一致）----
ECM_R_BT = 4000.0      # 干扰标定参考距离 m（4km 处 σ_jam = jam_db_at4km 标定值）
ECM_R_DETECT = 11.9e3  # RWR 探测距离 m（未探测阈值，同 demo）
ECM_BURN_RATIO = 1.4   # 烧穿阈值：σ_jam < 1.4·σ_real 视为烧穿
JAM_STATE_NONE = 0     # 未探测
JAM_STATE_SUPPRESS = 1 # 压制
JAM_STATE_BURN = 2     # 烧穿 / 正常跟踪（干扰无效或无干扰）
INVALID_RCS_DB = -100.0   # 无效帧哨兵值（UE 以 apparent ≤ -99 作异常过滤）


# ============================================================
# 一、RCS 推理引擎（复用 f16_rcs_demo 验证过的 ONNX+NFFFT 管线）
# ============================================================

class RcsEngine:
    """一次加载的推理引擎：eps 几何 + ONNX session + 标准化参数 + 表面面元。"""

    def __init__(self, onnx_path=ONNX_PATH, device_id=0):
        """device_id：CUDA 物理 GPU 序号（>=0 用该卡 CUDA，CUDA 不可用自动回退 CPU；
        多卡并行：每卡一个 RcsEngine 实例，各自独占 ONNX session）。"""
        t0 = time.time()
        # 1) eps 体素场与网格坐标（只读 h5 小字段）
        with h5py.File(H5, "r") as f:
            eps = f["eps_field"][:]
            gx = f["grid_x"][:].astype(np.float64)
            gy = f["grid_y"][:].astype(np.float64)
            gz = f["grid_z"][:].astype(np.float64)
        self.eps_mask = (eps > 1.5).astype(np.float32)          # (64,48,32)
        self.X, self.Y, self.Z = np.meshgrid(gx, gy, gz, indexing="ij")
        self.X = self.X.astype(np.float32)
        self.Y = self.Y.astype(np.float32)
        self.Z = self.Z.astype(np.float32)
        # 2) 表面面元（固定，只算一次）
        h = float(np.median(np.diff(gx)))
        self.idxs, self.rsurf, self.dS = surface_parts(eps, gx, gy, gz, h)
        self.n = self.dS / (np.linalg.norm(self.dS, axis=1, keepdims=True) + 1e-12)
        # 3) 标准化参数（训练 ckpt stats）
        ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        st = ckpt["stats"]
        self.xm, self.xs = np.asarray(st["x_inc_mean"], np.float32), np.asarray(st["x_inc_std"], np.float32)
        self.ym, self.ys = np.asarray(st["y_mean"], np.float32), np.asarray(st["y_std"], np.float32)
        # 4) ONNX session（指定 GPU 序号；CUDA 初始化失败自动回退 CPU）
        import onnxruntime as ort
        self.device_id = device_id
        try:
            providers = ([("CUDAExecutionProvider", {"device_id": int(device_id)})]
                         if device_id is not None and device_id >= 0
                         else ["CPUExecutionProvider"])
            self.sess = ort.InferenceSession(onnx_path, providers=providers)
        except Exception:
            self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.provider = self.sess.get_providers()[0]
        dev_tag = f"  [GPU{self.device_id}]" if "CUDA" in self.provider else "  [CPU]"
        print(f"引擎就绪: provider={self.provider}{dev_tag} 表面体素={len(self.idxs)} "
              f"载入耗时 {time.time()-t0:.1f}s", flush=True)

    # ---------- 入射场构造（与训练 load_data 完全一致） ----------
    def make_input(self, theta_deg, phi_deg):
        """入射角 (θ,φ)° → 标准化输入 (1,7,64,48,32)。khat=-[st cp, st sp, ct]。"""
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

    def predict_batch(self, xs):
        """批量 ONNX 推理：xs 为 (1,7,64,48,32) 列表 → (N,12,64,48,32)。
        模型输入 batch 维为动态（实测 batch=8 吞吐约 2.1 倍）。"""
        xb = np.concatenate(xs, axis=0).astype(np.float32)      # (N,7,64,48,32)
        out = self.sess.run(None, {"input": xb})[0]
        return out * self.ys.reshape(1, 12, 1, 1, 1) + self.ym.reshape(1, 12, 1, 1, 1)

    # ---------- 单站 RCS（入射角 = 观测角，雷达回波） ----------
    def rcs_single(self, pred, theta_deg, phi_deg, e0mag=1.0):
        return float(self.rcs_directions(pred, [theta_deg], [phi_deg], e0mag)[0])

    def rcs_directions(self, pred, th_deg, ph_deg, e0mag=1.0):
        """在任意一组方向（机体系球坐标 θ,φ，等长数组）上求远场 RCS，一次 NFFFT 多方向。
        用于双站 σ_ij（站 i 的近场在站 j 的方向上取值）。返回 (N,) 线性值。"""
        t = np.deg2rad(np.asarray(th_deg, np.float64))
        p = np.deg2rad(np.asarray(ph_deg, np.float64))
        st, ct = np.sin(t), np.cos(t)
        sp, cp = np.sin(p), np.cos(p)
        rhat = np.stack([st * cp, st * sp, ct], axis=1)      # 散射回波方向 (N,3)
        th_v = np.stack([ct * cp, ct * sp, -st], axis=1)
        ph_v = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
        i0, i1, i2 = self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]
        E_s = (pred[0:3] + 1j * pred[3:6])[:, i0, i1, i2].T
        H_s = (pred[6:9] + 1j * pred[9:12])[:, i0, i1, i2].T
        J = np.cross(self.n, H_s)                            # n̂×H_tot（PEC: M=0）
        E_ff = nffft(J, np.zeros_like(J), self.rsurf, self.dS, rhat, BETA0)
        Eth = (E_ff * th_v).sum(axis=1); Eph = (E_ff * ph_v).sum(axis=1)
        return 4.0 * np.pi * (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / e0mag ** 2

    def pattern_sweep(self, theta_deg, phi_deg, pred, phi_step=PAT_STEP, e0mag=1.0):
        """合成方向图（1次推理 + 1次NFFFT多方向，随入射角动态变化）：
        复用当前入射角 (theta_deg, phi_deg) 已算好的近场 pred，在 θ=theta_deg 圆锥面上
        扫 φ=0..360° 求各方向远场 RCS。后向散射方向（φ=phi_deg）处与 rcs_single 完全一致。
        返回 (phis (N,), rcs_db (N,))。"""
        phis = np.arange(0.0, 360.0 + 1e-6, phi_step, dtype=np.float64)
        t = np.deg2rad(theta_deg); pv = np.deg2rad(phis)
        st, ct = np.sin(t), np.cos(t)
        sp, cp = np.sin(pv), np.cos(pv)
        rhat = np.stack([st * cp, st * sp, np.full_like(sp, ct)], axis=1)   # 后向散射方向 (N,3)
        th_v = np.stack([ct * cp, ct * sp, np.full_like(sp, -st)], axis=1)
        ph_v = np.stack([-sp, cp, np.zeros_like(sp)], axis=1)
        i0, i1, i2 = self.idxs[:, 0], self.idxs[:, 1], self.idxs[:, 2]
        E_s = (pred[0:3] + 1j * pred[3:6])[:, i0, i1, i2].T
        H_s = (pred[6:9] + 1j * pred[9:12])[:, i0, i1, i2].T
        J = np.cross(self.n, H_s)                              # n̂×H_tot（PEC: M=0）
        E_ff = nffft(J, np.zeros_like(J), self.rsurf, self.dS, rhat, BETA0)
        Eth = (E_ff * th_v).sum(axis=1); Eph = (E_ff * ph_v).sum(axis=1)
        rcs = 4.0 * np.pi * (np.abs(Eth) ** 2 + np.abs(Eph) ** 2) / e0mag ** 2
        return phis, 10.0 * np.log10(rcs + 1e-9)


# ============================================================
# 二、几何：LOS 角 + 机体系有效入射角
# ============================================================

def ue_body_to_world(yaw, pitch, roll):
    """UE FRotator 约定（左手法则，X 前 / Y 右 / Z 上）机体→世界旋转矩阵。
    参考 UE FMatrix::Rotator：v_world = R @ v_body。"""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cp * cy,              cp * sy,              sp],
        [sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp],
        [-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp],
    ], np.float64)


def compute_angles(aircraft_pos, rot, radar_pos):
    """由 飞机位置/姿态(yaw,pitch,roll rad)/雷达位置 计算角度。
    返回 (los_theta_elev, los_phi_azim, eff_theta, eff_phi) 单位度；
    零距离等退化情况返回 None。"""
    d = np.asarray(aircraft_pos, np.float64) - np.asarray(radar_pos, np.float64)
    dist = np.linalg.norm(d)
    if dist < 1e-6:
        return None
    u_world = d / dist                             # 雷达→飞机 单位矢量
    # LOS 俯仰角/方位角（UI 显示：俯仰角为与水平面的夹角，上正）
    los_theta = float(np.degrees(np.arcsin(np.clip(u_world[2], -1.0, 1.0))))
    los_phi = float(np.degrees(np.arctan2(u_world[1], u_world[0])) % 360.0)
    # 机体系有效入射角：视线方向变换到机体坐标
    yaw, pitch, roll = float(rot[0]), float(rot[1]), float(rot[2])
    u_body = ue_body_to_world(yaw, pitch, roll).T @ u_world
    eff_theta = float(np.degrees(np.arccos(np.clip(u_body[2], -1.0, 1.0))))   # 0-180
    eff_phi = float(np.degrees(np.arctan2(u_body[1], u_body[0])) % 360.0)     # 0-360
    return los_theta, los_phi, eff_theta, eff_phi


def unpack_request(req):
    """可变长请求 → (timestamp_ms, pos[3], rot[3](yaw,pitch,roll,rad), stations)。
    stations = [(radar_pos[3], jammer_on, jam_db_at4km), ...]，1~4 站。
    头 33B: int64 ts(ms) + pos[3] + [pitch,yaw,roll]° + uint8 n_stations；每站 17B。"""
    head = REQ_HEAD.unpack_from(req)
    timestamp = head[0]
    aircraft_pos = np.asarray(head[1:4], np.float64)
    rot = np.deg2rad([head[5], head[4], head[6]])          # UE [pitch,yaw,roll]° → (yaw,pitch,roll) rad
    n = min(max(int(head[7]), 1), MAX_STATIONS)
    stations = []
    off = REQ_HEAD.size
    for _ in range(n):
        f = REQ_STATION.unpack_from(req, off)
        stations.append((np.asarray(f[0:3], np.float64), int(f[3]), float(f[4])))
        off += REQ_STATION.size
    return timestamp, aircraft_pos, rot, stations


def ecm_analysis(dist, sigma_real, jammer_on, jam_db_at4km):
    """压制干扰分析（每站）→ (rcs_apparent_db, jam_state)。
    σ_jam(R) = σ_jam(4km标定) × (R/4km)²（单程干扰 vs 双程回波，同 demo）；
    状态：R≥R_DETECT→未探测；干扰开且σ_jam≥1.4σ_real→压制；否则烧穿/正常跟踪。"""
    sigma_real = max(float(sigma_real), 0.0)
    if jammer_on:
        jam_sigma = 10 ** (jam_db_at4km / 10.0) * (dist / ECM_R_BT) ** 2
    else:
        jam_sigma = 0.0
    rcs_apparent_db = 10 * np.log10(sigma_real + jam_sigma + 1e-9)
    if dist >= ECM_R_DETECT:
        state = JAM_STATE_NONE
    elif jammer_on and jam_sigma >= ECM_BURN_RATIO * sigma_real:
        state = JAM_STATE_SUPPRESS
    else:
        state = JAM_STATE_BURN
    return rcs_apparent_db, state


def pack_single_response(typ, timestamp, st_angles, apparent_db, states, sigma_db):
    """单站包：Type + ts + n_stations + [每站 los_t,los_p,eff_t,eff_p] + [每站 apparent]
    + [每站 jam_state] + [σ_ij 行主序]。长度 16+24N+4N(N-1) B。"""
    n = len(st_angles)
    out = bytearray(RESP_HEAD.pack(typ, timestamp, n))
    for lo_t, lo_p, ef_t, ef_p in st_angles:
        out += RESP_STATION_ANGLES.pack(lo_t, lo_p, ef_t, ef_p)
    for v in apparent_db:
        out += RESP_STATION_FLOAT.pack(v)
    for s in states:
        out += RESP_STATION_STATE.pack(s)
    for v in sigma_db:
        out += RESP_STATION_FLOAT.pack(v)
    return bytes(out)


def pack_pattern(timestamp, theta_deg, phi_deg, rcs_db):
    """方向图包：Type(2)+ts+theta_deg(double)+n(int32)+PhiDegArr[RcsDbArr](double×n)。"""
    phi = np.asarray(phi_deg, "<f8")
    rcs = np.asarray(rcs_db, "<f8")
    return (PAT_HEAD.pack(TYPE_PATTERN, timestamp, theta_deg, len(phi))
            + phi.tobytes() + rcs.tobytes())


def handle_batch(engine, fields_list):
    """批量处理多帧（每帧 1~4 站）→ (响应字节列表, pred_map)。
    pred_map: {帧下标: {站下标: 近场预测}}，供方向图合成（站0）复用。
    fields=(timestamp_ms, pos, rot, stations)；全部站的输入合并为一次批量推理。
    帧内任一站点越界/退化 → 整帧无效（apparent/σ_ij 置 -100，state=0）。"""
    results = [None] * len(fields_list)
    pred_map = {}
    valid = []          # (frame_idx, station_idx, ts, lo_t, lo_p, ef_t, ef_p, dist, jam_on, jam_db)
    base_xs = []
    for i, (timestamp, aircraft_pos, rot, stations) in enumerate(fields_list):
        n = len(stations)
        sentinel = pack_single_response(TYPE_SINGLE_ONLY, timestamp,
                                        [(0.0, 0.0, 0.0, 0.0)] * n,
                                        [INVALID_RCS_DB] * n, [JAM_STATE_NONE] * n,
                                        [INVALID_RCS_DB] * (n * (n - 1)))
        if (not np.isfinite(aircraft_pos).all() or not np.isfinite(rot).all()
                or not stations):
            results[i] = (sentinel, 0.0)
            continue
        st_ang = []
        ok = True
        for radar_pos, jam_on, jam_db in stations:
            if not np.isfinite(radar_pos).all():
                ok = False
                break
            ang = compute_angles(aircraft_pos, rot, radar_pos)
            if ang is None:
                ok = False
                break
            los_theta, los_phi, eff_theta, eff_phi = ang
            if not (THETA_MIN <= eff_theta <= THETA_MAX):
                ok = False
                break
            dist = float(np.linalg.norm(aircraft_pos - radar_pos))
            st_ang.append((los_theta, los_phi, eff_theta, eff_phi, dist, jam_on, jam_db))
        if not ok:
            results[i] = (sentinel, 0.0)
            continue
        for s, (lo_t, lo_p, ef_t, ef_p, dist, jam_on, jam_db) in enumerate(st_ang):
            valid.append((i, s, timestamp, lo_t, lo_p, ef_t, ef_p, dist, jam_on, jam_db))
            base_xs.append(engine.make_input(ef_t, ef_p))

    if base_xs:
        t0 = time.perf_counter()
        preds = engine.predict_batch(base_xs)          # (K,12,64,48,32)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # 每站物理 RCS + 干扰分析 + 双站 σ_ij
        per_st = {}          # frame_idx → list[(angles, apparent_db, state)]
        sig = {}             # frame_idx → {站i: rcs_directions(...)}
        for k, (i, s, ts, lo_t, lo_p, ef_t, ef_p, dist, jam_on, jam_db) in enumerate(valid):
            rcs = engine.rcs_single(preds[k], ef_t, ef_p)
            app_db, state = ecm_analysis(dist, rcs, jam_on, jam_db)
            per_st.setdefault(i, []).append(((lo_t, lo_p, ef_t, ef_p), app_db, state))
            sig.setdefault(i, {})[s] = preds[k]
            pred_map.setdefault(i, {})[s] = preds[k]
        for i, st_list in per_st.items():
            n = len(st_list)
            sigma_db = []
            # σ_ij：站 i 的近场在站 j 的方向上取值（i 发 j 收，行主序）
            for a in range(n):
                _, _, ef_t_a, ef_p_a = st_list[a][0]
                for b in range(n):
                    if a == b:
                        continue
                    _, _, ef_t_b, ef_p_b = st_list[b][0]
                    v = engine.rcs_directions(sig[i][a], [ef_t_b], [ef_p_b])[0]
                    sigma_db.append(float(10 * np.log10(v + 1e-9)))
            results[i] = (pack_single_response(TYPE_SINGLE_ONLY, fields_list[i][0],
                                               [s[0] for s in st_list],
                                               [s[1] for s in st_list],
                                               [s[2] for s in st_list],
                                               sigma_db), dt_ms)
    return results, pred_map


# ============================================================
# 三、TCP 服务端主循环（Python 监听，UE 客户端主动连接）
# ============================================================

class FrameWorker(threading.Thread):
    """单 GPU 推理 worker：阻塞取首帧 → 尽取积压（≤MAX_BATCH 帧）→ 一次批量推理 →
    打包(单站包+可选方向图包) → 输出。多 worker 各绑一 GPU(session.run 释放 GIL 真并行)；
    CPU 输入组装与上一批 GPU 推理由多线程自然重叠（流水线）。"""

    def __init__(self, engine, inbox, outbox, wid=0, max_samp=MAX_SAMPLES, shutdown=None):
        super().__init__(daemon=True)
        self.engine = engine
        self.inbox = inbox       # Queue[(seq, raw_frame, pat_tick)]；None=退出
        self.outbox = outbox     # Queue[(seq, blob)]；None=退出
        self.wid = wid
        self.max_samp = max_samp   # 每批 ONNX 样本上限（显存不足可调小）
        self.shutdown = shutdown or threading.Event()   # 连接断开即置位：丢弃积压退出，释放 GPU
        self.n_batches = 0
        self.dt_ms_sum = 0.0

    def run(self):
        eng = self.engine
        try:
            while not self.shutdown.is_set():
                item = self.inbox.get()
                if item is None:
                    return
                # 攒批：同时受 帧数≤MAX_BATCH 与 样本数≤MAX_SAMPLES 约束
                # （样本=batch 的 ONNX 输入数；实测样本 batch>8 时 ScatterND 灾难性退化，须压住）
                batch = [item]
                n_samp = len(unpack_request(item[1])[3])
                while len(batch) < MAX_BATCH and not self.shutdown.is_set():
                    try:
                        nxt = self.inbox.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:                 # 退出哨兵：直接结束，不再续攒
                        return
                    ns = len(unpack_request(nxt[1])[3])
                    if n_samp + ns > self.max_samp:
                        self.inbox.put(nxt)          # 放回队首，留待下一批
                        break
                    batch.append(nxt)
                    n_samp += ns
                fields = [unpack_request(f) for _, f, _ in batch]
                try:
                    results, pred_map = handle_batch(eng, fields)
                except Exception as e:                       # 推理/后处理异常：整批回送无效哨兵，不中断服务
                    import traceback
                    print(f"[worker{self.wid}] 处理 {len(batch)} 帧异常: {e}", flush=True)
                    results, pred_map = [], {}
                    for (_, f0, _) in batch:
                        fl = unpack_request(f0)
                        n_st = len(fl[3])
                        results.append((pack_single_response(
                            TYPE_SINGLE_ONLY, fl[0], [(0.0, 0.0, 0.0, 0.0)] * n_st,
                            [INVALID_RCS_DB] * n_st, [JAM_STATE_NONE] * n_st,
                            [INVALID_RCS_DB] * (n_st * (n_st - 1))), 0.0))
                self.n_batches += 1
                self.dt_ms_sum += results[0][1] if results else 0.0
                for k, ((seq, _, pat_tick), (resp, _)) in enumerate(zip(batch, results)):
                    typ, ts, n = RESP_HEAD.unpack_from(resp)
                    blob = resp
                    # 方向图触发帧（主线程已按 pattern_every 打标）且站点1 有效才附方向图
                    pat_pred = pred_map.get(k, {}).get(0)
                    st0_ok = (n >= 1 and RESP_STATION_FLOAT.unpack_from(
                        resp, RESP_HEAD.size + 16 * n)[0] > INVALID_RCS_DB + 1)
                    if pat_tick and pat_pred is not None and st0_ok:
                        blob = (RESP_HEAD.pack(TYPE_SINGLE_WITH_PATTERN, ts, n)
                                + resp[RESP_HEAD.size:])
                        _, _, ef_t0, ef_p0 = RESP_STATION_ANGLES.unpack_from(resp, RESP_HEAD.size)
                        phis, rcs_db = eng.pattern_sweep(ef_t0, ef_p0, pat_pred)
                        blob += pack_pattern(ts, ef_t0, phis, rcs_db)
                    self.outbox.put((seq, blob))
        except (ConnectionError, OSError):
            return


class FrameSender(threading.Thread):
    """按全局 seq 保序回发：多 worker 乱序产出先入暂存，序号连续即 sendall，
    保证单站包与方向图包字节不交错、响应顺序与请求一致。"""

    def __init__(self, conn, outbox):
        super().__init__(daemon=True)
        self.conn = conn
        self.outbox = outbox
        self.pending = {}
        self.next_seq = 0

    def run(self):
        try:
            while True:
                item = self.outbox.get()
                if item is None:
                    return
                seq, blob = item
                self.pending[seq] = blob
                while self.next_seq in self.pending:
                    self.conn.sendall(self.pending.pop(self.next_seq))
                    self.next_seq += 1
        except (ConnectionError, OSError):
            return


def get_lan_ip():
    """探测本机主 LAN IP（不实际发包，离线可用）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def req_frame_size(buf):
    """由请求头解析单帧总长（33+17×n）。buf 不足头长返回 None。"""
    if len(buf) < REQ_HEAD.size:
        return None
    n = min(max(int(REQ_HEAD.unpack_from(buf)[7]), 1), MAX_STATIONS)
    return REQ_HEAD.size + n * REQ_STATION.size


def serve(engines, host, port, pattern_every=5, max_samp=MAX_SAMPLES):
    """Python 作为 TCP 服务端（多卡并行 + 流水线）：监听 host:port，接受 UE 客户端连接，
    断线后继续等待新连接。每个 engine（一张 GPU）起一个 FrameWorker 推理线程，GPU 间真并行；
    帧到即按 seq 轮流分发，worker 自行攒批(≤MAX_BATCH)批量推理；FrameSender 按 seq 保序回发。
    UE 协议不变：每帧单站包 + 可选方向图包（Type 打标）。max_samp=每批样本上限（显存不足调小）。"""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(1)
    n_workers = len(engines)
    print(f"[监听] Python 服务端 {host}:{port}，等待 UE 客户端连接"
          f"（{n_workers} 卡并行 worker，Ctrl+C 退出）...", flush=True)
    try:
        while True:
            conn, addr = srv.accept()
            conn.settimeout(60.0)                 # 单帧等待超时
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # 禁用Nagle，响应小包立即发出
            print(f"[已连接] UE 客户端 {addr}，进入请求-响应循环", flush=True)
            inbox, outbox = queue.Queue(), queue.Queue()
            evt_shutdown = threading.Event()
            workers = [FrameWorker(eng, inbox, outbox, wid=i, max_samp=max_samp,
                                   shutdown=evt_shutdown)
                       for i, eng in enumerate(engines)]
            sender = FrameSender(conn, outbox)
            for w in workers:
                w.start()
            sender.start()
            seq = 0
            pat_cnt = 0
            t_log = time.time()
            try:
                buf = b""
                while True:
                    # 阻塞等满一帧（可变长：先头 33B 得 n_stations，再等满 33+17n）
                    fsize = req_frame_size(buf)
                    while fsize is None or len(buf) < fsize:
                        need = (REQ_HEAD.size if fsize is None
                                else fsize - len(buf))
                        chunk = conn.recv(need)
                        if not chunk:
                            raise ConnectionError("UE 关闭连接")
                        buf += chunk
                        fsize = req_frame_size(buf)
                    if fsize is None or len(buf) < fsize:
                        break
                    frame = buf[:fsize]
                    buf = buf[fsize:]
                    # 方向图触发打标（每 pattern_every 帧一个；pattern_every=0 不发送）
                    pat_tick = False
                    if pattern_every > 0:
                        pat_cnt += 1
                        if pat_cnt >= pattern_every:
                            pat_tick = True
                            pat_cnt = 0
                    inbox.put((seq, frame, pat_tick))          # 轮流分发（多 worker 自攒批）
                    seq += 1
                    if seq % 60 == 0:                          # 低频日志，不拖吞吐
                        now = time.time()
                        span = max(now - t_log, 1e-3)          # 防 Windows 时钟节拍导致除零
                        avg = {w.wid: (w.dt_ms_sum / w.n_batches if w.n_batches else 0.0)
                               for w in workers}
                        print(f"[主线程] 输入 {60/span:.1f} fps，累计 {seq} 帧，"
                              f"批次 {[w.n_batches for w in workers]}，"
                              f"onnx 均值 {[f'{avg[w.wid]:.0f}ms' for w in workers]}", flush=True)
                        t_log = now
            except (socket.timeout, ConnectionError, OSError) as e:
                print(f"[连接异常] {addr}: {e}", flush=True)
            finally:
                evt_shutdown.set()                # 先置位：worker 丢弃剩余积压立即退出，释放 GPU
                for w in workers:
                    inbox.put(None)               # 通知 worker 退出
                outbox.put(None)                  # 通知 sender 退出
                try:
                    conn.close()
                except OSError:
                    pass
    except KeyboardInterrupt:
        print("\n收到中断，退出。", flush=True)
    finally:
        srv.close()


# ============================================================
# 四、自检（vs h5 真实单站 RCS）
# ============================================================

def selftest():
    eng = RcsEngine()
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
        it = int(np.argmin(np.abs(ff_t - theta))); ip = int(np.argmin(np.abs(ff_p - phi)))
        ref = float(rcs_true[i, it, ip])
        db = 10 * np.log10(rcs + 1e-9); dbref = 10 * np.log10(ref + 1e-9)
        print(f"  case_{i:03d} (θ={theta:.0f},φ={phi:.0f}): "
              f"ONNX {db:7.2f} dBsm vs FEKO {dbref:7.2f} dBsm  |Δ|={abs(db-dbref):.2f} dB  "
              f"({(time.time()-t0)*1000:.0f} ms)", flush=True)
    print("（P4 参考：模型+管线 |ΔRCS| 中位 ~4 dB）\n")

    # 几何角度自检：零姿态应等价于 LOS 极角
    apos = np.array([8000.0, 6000.0, 2000.0]); radar = np.array([0.0, 0.0, 0.0])
    rot = np.array([0.0, 0.0, 0.0])
    lo_t, lo_p, ef_t, ef_p = compute_angles(apos, rot, radar)
    d = apos - radar
    polar = np.degrees(np.arccos(d[2] / np.linalg.norm(d)))
    print(f"=== 角度自检 ===")
    print(f"  零姿态: los_theta(elev)={lo_t:.2f}° los_phi(azim)={lo_p:.2f}°")
    print(f"  eff_theta={ef_t:.2f}°（应≈LOS极角 {polar:.2f}°），eff_phi={ef_p:.2f}°（应≈LOS方位 {lo_p:.2f}°）")
    assert abs(ef_t - polar) < 1e-6 and abs(ef_p - lo_p) < 1e-6
    # 偏航 90°：机头转向，有效方位应变化
    rot2 = np.array([np.deg2rad(90.0), 0.0, 0.0])
    _, _, ef_t2, ef_p2 = compute_angles(apos, rot2, radar)
    print(f"  偏航90°: eff_theta={ef_t2:.2f}° eff_phi={ef_p2:.2f}°（应随姿态变化）")
    print("  角度自检通过\n")


def main():
    ap = argparse.ArgumentParser(description="FNO-RCS 实时推理 TCP 服务（Python 服务端，UE 客户端主动连接）")
    ap.add_argument("--host", default=DEFAULT_HOST, help=f"监听地址（默认 {DEFAULT_HOST}，UE 连接本机 LAN IP）")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"监听端口（默认 {DEFAULT_PORT}）")
    ap.add_argument("--selftest", action="store_true", help="仅运行自检后退出（不需 UE）")
    ap.add_argument("--pattern-every", type=int, default=5,
                    help="方向图包发送周期（每 N 帧发一次，默认 5；0=不发送方向图）")
    ap.add_argument("--devices", default="0",
                    help="CUDA GPU 序号列表（逗号分隔），每个序号一个并行推理引擎/线程；"
                         "默认 0；双卡机器用 0,1 可显著提升吞吐（如 20fps）")
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES,
                    help=f"每批最多 ONNX 样本数（默认 {MAX_SAMPLES}；"
                         f"显存不足/GUI 占用大时可调小，如 4/2，吞吐相应下降）")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return

    devices = [int(x) for x in str(args.devices).split(",") if x.strip() != ""] or [0]
    engines = [RcsEngine(device_id=d) for d in devices]
    if args.pattern_every > 0:
        print(f"[方向图] 动态合成已启用：每 {args.pattern_every} 帧发送一次，"
              f"θ 切面随当前 LOS 有效入射角变化，φ 步长 {PAT_STEP:.0f}°（{int(360/PAT_STEP)+1} 点）", flush=True)
    lan_ip = get_lan_ip() or "127.0.0.1"
    print(f"[多卡] 启用 {len(engines)} 个推理引擎: devices={devices} "
          f"（GPU 序号 {[e.provider + (f'/{e.device_id}' if 'CUDA' in e.provider else '') for e in engines]}）", flush=True)
    print(f"[提示] UE 客户端请连接: {lan_ip}:{args.port}  （本机监听 {args.host}:{args.port}）", flush=True)
    serve(engines, args.host, args.port, pattern_every=args.pattern_every,
          max_samp=max(1, args.max_samples))


if __name__ == "__main__":
    main()
