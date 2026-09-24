# -*- coding: utf-8 -*-
"""
probe_throughput.py — 临时吞吐探针（不入部署包）
固定速率持续发包，模拟 UE 以指定 fps 发送，统计服务端端到端处理能力：
  - 输入速率（实际发送帧/s）
  - 输出速率（收到完整响应帧/s）
  - RTT p50 / p95 / max（发送到收到完整响应）
  - sendall 阻塞占比（TCP 背压 = 服务端跟不上，发送被卡住）
场景：回放 F16_demo_trajectory.csv 双站（从 --start 秒起循环），与 UE 联调场景一致。

用法：
  python probe_throughput.py --port 9011 --fps 10 --dur 12 --start 30
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import socket
import argparse
import threading
from types import SimpleNamespace

import numpy as np
import mock_ue_client as mock

RESP_HEAD = mock.RESP_HEAD
PAT_HEAD = mock.PAT_HEAD
TYPE_SINGLE_WITH_PATTERN = mock.TYPE_SINGLE_WITH_PATTERN


def build_rows(args):
    """加载 CSV 双站帧并循环复用（从 start 秒起）。"""
    ns = SimpleNamespace(start=args.start, frames=0)
    return list(mock.iter_csv(ns, mock.CSV_DEFAULT))


def recv_frame(sock):
    """读一帧单站包，Type=1 时顺带读完方向图包。返回 (typ, n)。"""
    head = mock.recv_exact(sock, RESP_HEAD.size)
    if head is None:
        return None
    typ, _ts, n = RESP_HEAD.unpack(head)
    n = max(int(n), 1)
    body_sz = 24 * n + 4 * n * (n - 1)
    if mock.recv_exact(sock, body_sz) is None:
        return None
    if typ == TYPE_SINGLE_WITH_PATTERN:
        ph = mock.recv_exact(sock, PAT_HEAD.size)
        if ph is None:
            return None
        p_typ, _t, _theta, pn = PAT_HEAD.unpack(ph)
        if p_typ != mock.TYPE_PATTERN or mock.recv_exact(sock, 16 * pn) is None:
            return None
    return typ, n


def main():
    ap = argparse.ArgumentParser(description="固定速率吞吐探针（模拟 UE 持续发包）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9011)
    ap.add_argument("--fps", type=float, required=True, help="目标发送帧率")
    ap.add_argument("--dur", type=float, default=12.0, help="测试时长 s")
    ap.add_argument("--start", type=float, default=30.0, help="CSV 起始秒（默认 30=干扰压制段）")
    args = ap.parse_args()

    rows = build_rows(args)
    if not rows:
        print("[probe] 无场景帧可用", flush=True)
        return
    interval = 1.0 / args.fps

    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    sock.settimeout(3.0)          # 3s 无响应视为处理完毕（正常批次远快于此）
    print(f"[probe] 已连接 {args.host}:{args.port}  目标 {args.fps:.1f} fps"
          f"（{interval*1000:.0f} ms/帧）  双站场景循环 {len(rows)} 帧", flush=True)

    ev_send = threading.Event()
    ev_recv = threading.Event()
    send_ts = []
    sent = [0]
    recv_n = [0]
    recv_times = []
    send_block = [0.0]
    lag_skip = [0]
    err = []
    win = {}                     # (send_start, send_end) 实际发送窗口

    def sender():
        next_t = time.perf_counter()
        k = 0
        win["start"] = next_t
        while not ev_send.is_set():
            cur = time.perf_counter()
            if cur < next_t:
                time.sleep(max(min(next_t - cur, 0.002), 0.0))
                continue
            next_t += interval
            if next_t < cur - 0.05:            # 落后 50ms 以上：跳过一个节拍，防止追发积压
                lag_skip[0] += 1
                next_t = cur + interval
            ts_ms, pos, rot, sts = rows[k % len(rows)]
            k += 1
            req = mock.build_request(ts_ms, pos, rot, sts)
            t0 = time.perf_counter()
            try:
                sock.sendall(req)
            except OSError as e:
                err.append(f"send: {e}")
                break
            send_block[0] += time.perf_counter() - t0
            send_ts.append(t0)
            sent[0] += 1
        win["end"] = time.perf_counter()

    def receiver():
        while not ev_recv.is_set():
            try:
                got = recv_frame(sock)
            except socket.timeout:                 # 3s 无新数据=服务端处理完毕，正常结束
                break
            except OSError as e:
                err.append(f"recv: {e}")
                break
            if got is None:
                err.append("连接被服务端关闭")
                break
            recv_n[0] += 1
            recv_times.append(time.perf_counter())

    th_s = threading.Thread(target=sender, daemon=True)
    th_r = threading.Thread(target=receiver, daemon=True)
    th_s.start()
    th_r.start()
    time.sleep(args.dur)
    ev_send.set()                 # 停发
    th_s.join(timeout=5)
    time.sleep(2.5)               # 让已发请求的响应尽量回补（服务端满负荷时需排队）
    ev_recv.set()                 # 停收
    th_r.join(timeout=5)
    try:
        sock.close()
    except OSError:
        pass

    n_sent, n_recv = sent[0], recv_n[0]
    print("\n=== 结果 ===", flush=True)
    if n_sent >= 2:
        span_in = send_ts[-1] - send_ts[0]
        in_fps = (n_sent - 1) / max(span_in, 1e-9)
        print(f"  已发送请求: {n_sent}  实际输入 {in_fps:.2f} fps"
              f"（目标 {args.fps:.1f}）", flush=True)
        if n_recv >= 2:
            span_out = recv_times[-1] - recv_times[0]
            out_fps = (n_recv - 1) / max(span_out, 1e-9)
            print(f"  收到完整响应: {n_recv}  实际输出 {out_fps:.2f} fps"
                  f"（含收尾仍在回补则偏低）", flush=True)
        else:
            print(f"  收到完整响应: {n_recv}  → 几乎没有响应返回", flush=True)
    rtt = None
    if n_recv >= 2 and n_recv <= n_sent:
        rtt = np.asarray(recv_times[:n_recv]) - np.asarray(send_ts[:n_recv])
        print(f"  RTT(发→收完整): p50={np.percentile(rtt,50)*1000:.0f}ms  "
              f"p95={np.percentile(rtt,95)*1000:.0f}ms  max={rtt.max()*1000:.0f}ms", flush=True)
    print(f"  sendall 阻塞占比: {send_block[0]/max(send_ts[-1]-send_ts[0],1e-9)*100:.1f}%"
          f"（阻塞>5% 说明服务端吃不下，被 TCP 背压）", flush=True)
    if lag_skip[0]:
        print(f"  节拍追赶跳过: {lag_skip[0]} 次（发送端自身来不及，非服务端瓶颈）", flush=True)
    miss = n_sent - n_recv
    if miss > 2:
        print(f"  !! 有 {miss} 帧未收到响应（服务端跟不上或仍在队列）", flush=True)
    else:
        print(f"  帧完整率: {n_recv/max(n_sent,1)*100:.1f}% （算得过来）", flush=True)
    for e in err[:5]:
        print(f"  [错误] {e}", flush=True)
    # 响应滞后趋势：收尾是否仍堆积（看后半段每帧平均RTT，按已收序号对齐）
    if rtt is not None and n_recv >= 20:
        n2 = n_recv
        half = n2 // 2
        rtt0 = np.asarray(recv_times[:half]) - np.asarray(send_ts[:half])
        rtt1 = np.asarray(recv_times[half:n2]) - np.asarray(send_ts[half:n2])
        print(f"  响应滞后趋势: 前半段均RTT={rtt0.mean()*1000:.0f}ms  "
              f"后半段均RTT={rtt1.mean()*1000:.0f}ms"
              f"（后半明显更大 = 积压增长，跟不上）", flush=True)


if __name__ == "__main__":
    main()
