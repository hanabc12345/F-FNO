# -*- coding: utf-8 -*-
"""
mock_ue_client.py — 模拟 UE 端 TCP 客户端（多站点 + 压制干扰，无 UE 环境联调用）
==================================================================================
协议与 ue_rcs_service.py 一致（小端二进制，可变长度，支持 1~4 站 + 每站干扰）。
模拟 UE 客户端：主动连接 Python 推理服务端，按帧发送请求并打印响应。

请求（33+17×N B）:  头 33B(ts ms + pos[3] + rot[3](°pitch,yaw,roll) + n_stations)
                    + 每站 17B(radar_pos[3] + jammer_on + jam_db_at4km)
单站包（16+24N+4N(N-1) B）: Type(0/1) + ts(ms) + n_stations
                    + 每站(los_t,los_p,eff_t,eff_p) + 每站表观RCS + 每站jam_state
                    + σ_ij 行主序（N≥2）
方向图包（Type=1 时紧跟单站包之后）: Type(2) + ts + theta_deg(double) + n(int32)
                    + PhiDegArr/RcsDbArr(double×n)

两种运行模式：
  a) --csv：回放 docs/F16_demo_trajectory.csv 的 90s 对抗场景（默认自动加载该文件）
  b) 无 --csv：内联双站快速冒烟（飞机直线逼近，t 越过 11.9km 后开干扰）

用法（先起 service，再起 mock client）：
  终端1: & "F:/miniconda3/envs/isaac311/python.exe" ue_rcs_service.py --port 9001
  终端2: & "F:/miniconda3/envs/isaac311/python.exe" mock_ue_client.py --host 127.0.0.1 --port 9001
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import time
import struct
import socket
import argparse
import csv

import numpy as np

# ---- 与 ue_rcs_service.py 一致的协议结构 ----
REQ_HEAD = struct.Struct("<q3f3fB")       # 33 B: ts(ms)+pos[3]+rot[3](°)+n_stations
REQ_STATION = struct.Struct("<3fBf")      # 17 B: radar[3]+jammer_on+jam_db_at4km
RESP_HEAD = struct.Struct("<iqi")         # 16 B: Type+ts(ms)+n_stations
RESP_STATION_ANGLES = struct.Struct("<4f")
RESP_STATION_FLOAT = struct.Struct("<f")
RESP_STATION_STATE = struct.Struct("<i")
PAT_HEAD = struct.Struct("<iqdi")         # 24 B: Type(2)+ts+theta(double)+n(int32)

TYPE_SINGLE_ONLY = 0
TYPE_SINGLE_WITH_PATTERN = 1
TYPE_PATTERN = 2
STATE_NAME = {0: "未探测", 1: "压制", 2: "烧穿", 3: "正常跟踪"}

R1 = (0.0, 0.0, 20.0)        # 前向火控雷达
R2 = (0.0, -6000.0, 20.0)    # 侧向监视雷达
JAM_DB = -11.5               # 4km 标定 dBsm（烧穿点约 4km）
CSV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "F16_demo_trajectory.csv")


def build_request(ts_ms, pos, rot_deg, stations):
    """编码可变长请求：头 + N 个站点块。stations=[(radar_pos[3], jam_on, jam_db), ...]"""
    head = REQ_HEAD.pack(int(ts_ms), *[float(v) for v in pos],
                         *[float(v) for v in rot_deg], len(stations))
    body = b"".join(REQ_STATION.pack(*(float(v) for v in rp), int(on), float(db))
                    for rp, on, db in stations)
    return head + body


def recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def parse_response(resp):
    """解析单站包 → (typ, ts, angles, apparent, states, sigs)"""
    typ, ts, n = RESP_HEAD.unpack_from(resp)
    angles = [RESP_STATION_ANGLES.unpack_from(resp, RESP_HEAD.size + 16 * i)
              for i in range(n)]
    app = [RESP_STATION_FLOAT.unpack_from(resp, RESP_HEAD.size + 16 * n + 4 * i)[0]
           for i in range(n)]
    states = [RESP_STATION_STATE.unpack_from(resp, RESP_HEAD.size + 20 * n + 4 * i)[0]
              for i in range(n)]
    sigs = [RESP_STATION_FLOAT.unpack_from(resp, RESP_HEAD.size + 24 * n + 4 * q)[0]
            for q in range(n * (n - 1))]
    return typ, ts, angles, app, states, sigs


def recv_pattern(conn):
    head = recv_exact(conn, PAT_HEAD.size)
    if head is None:
        return None
    typ, ts, theta, n = PAT_HEAD.unpack(head)
    if typ != TYPE_PATTERN:
        return f"[异常] 期望方向图包(Type=2)，实际 Type={typ}"
    body = recv_exact(conn, 16 * n)
    if body is None:
        return None
    phi = np.frombuffer(body[:8 * n], "<f8")
    rcs = np.frombuffer(body[8 * n:], "<f8")
    return f"方向图 ts={ts}ms θ={theta:.0f}° n={n} " \
           f"φ∈[{phi.min():.0f},{phi.max():.0f}]° RCS∈[{rcs.min():.1f},{rcs.max():.1f}] dBsm"


def run_frames(sock, args, frames_iter):
    """逐帧发送/接收，打印解析结果。frames_iter 产出 (ts_ms, pos, rot_deg, stations)。"""
    n = 0
    for ts_ms, pos, rot_deg, stations in frames_iter:
        sock.sendall(build_request(ts_ms, pos, rot_deg, stations))
        size = RESP_HEAD.size + 24 * len(stations) + 4 * len(stations) * (len(stations) - 1)
        resp = recv_exact(sock, size)
        if resp is None:
            print("[mock UE] Python 服务端断开。", flush=True)
            break
        typ, ts, angles, app, states, sigs = parse_response(resp)
        parts = [f"t={ts}ms n={len(stations)}站"]
        for s, ((lo_t, lo_p, ef_t, ef_p), a, st) in enumerate(zip(angles, app, states)):
            parts.append(f"站{s+1} eff=({ef_t:5.1f}°,{ef_p:6.1f}°) "
                         f"表观={a:6.1f} {STATE_NAME.get(st, st)}")
        if sigs:
            parts.append("σij=" + "/".join(f"{v:6.1f}" for v in sigs))
        print("[mock UE] " + " | ".join(parts), flush=True)
        if typ == TYPE_SINGLE_WITH_PATTERN:
            pat = recv_pattern(sock)
            print(f"[mock UE] {pat}", flush=True)
        elif typ != TYPE_SINGLE_ONLY:
            print(f"[mock UE] [异常] 未知 Type={typ}", flush=True)
        n += 1
        if args.dt > 0:
            time.sleep(args.dt)
    print(f"[mock UE] 完成 {n} 帧，关闭连接。", flush=True)


def iter_csv(args, path):
    """回放 CSV 场景：按行产出帧（含干扰开关/标定）。"""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    start = int(args.start * 1000 / 200 + 0.5)          # 秒 → CSV 行（200ms/帧）
    end = len(rows) if args.frames <= 0 else min(len(rows), start + args.frames)
    for r in rows[start:end]:
        ts_ms = int(r["time_ms"])
        pos = (float(r["x"]), float(r["y"]), float(r["z"]))
        rot = (float(r["pitch_deg"]), float(r["yaw_deg"]), float(r["roll_deg"]))
        stations = [
            (R1, int(r["jam1_on"]), float(r["jam_db1"])),
            (R2, int(r["jam2_on"]), float(r["jam_db2"])),
        ]
        yield ts_ms, pos, rot, stations


def iter_inline(args):
    """无 CSV 时的双站冒烟场景：直线逼近，距 R1<11.9km 后开干扰。"""
    t = 0.0
    pos = np.array([16000.0, 0.0, 1000.0])
    for i in range(args.frames):
        pos = pos + np.array([-220.0, 0.0, 0.0]) * args.dt
        t += args.dt
        d1 = np.linalg.norm(pos - np.array(R1))
        jam1 = 1 if d1 < 11.9e3 else 0
        rot = (0.0, 270.0, 0.0)
        stations = [(R1, jam1, JAM_DB), (R2, 0, JAM_DB)]
        yield int(t * 1000), pos.tolist(), rot, stations


def main():
    ap = argparse.ArgumentParser(description="模拟 UE 端 TCP 客户端（多站点+压制干扰）")
    ap.add_argument("--host", default="127.0.0.1", help="Python 推理服务端 IP（默认 127.0.0.1）")
    ap.add_argument("--port", type=int, default=9001, help="Python 推理服务端端口（默认 9001）")
    ap.add_argument("--dt", type=float, default=0.0, help="帧间隔 s（默认 0=全速发送）")
    ap.add_argument("--frames", type=int, default=40, help="发送帧数（默认 40；CSV 模式下 0=全部）")
    ap.add_argument("--start", type=float, default=0.0, help="CSV 起始时间 s（默认 0）")
    ap.add_argument("--csv", default=CSV_DEFAULT, help=f"场景CSV路径（默认 {CSV_DEFAULT}）")
    ap.add_argument("--no-csv", action="store_true", help="不使用 CSV，用内联冒烟场景")
    args = ap.parse_args()

    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    sock.settimeout(60.0)
    print(f"[mock UE 客户端] 已连接 {args.host}:{args.port}", flush=True)

    if args.no_csv:
        it = iter_inline(args)
    else:
        it = iter_csv(args, args.csv)
    run_frames(sock, args, it)
    sock.close()


if __name__ == "__main__":
    main()
