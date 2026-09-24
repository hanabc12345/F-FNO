# -*- coding: utf-8 -*-
"""
monitor_progress.py — FEKO 批量数据生成进度监控（配合 gen_feko_batch.py v3）

功能：
  1. 完成角度计数 / 总进度 / 当前求解 case
  2. 失败与重试统计（解析 feko_run.log）
  3. 生成器存活检测（FEKO 进程数 + 日志新鲜度）
  4. 速率 / ETA 估算

用法：
  python monitor_progress.py            # 循环监控（默认 60s 刷新）
  python monitor_progress.py --once     # 单次快照后退出
  python monitor_progress.py --interval 120   # 自定义刷新间隔（秒）
"""
import os
import re
import sys
import time
import argparse
import glob

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

RUN_DIR = r"f:\MyWorkSpace\UAVGame\3d_feko_run"
LOGFILE = os.path.join(RUN_DIR, "feko_run.log")
TOTAL = 468  # θ 30..150/10 × φ 0..350/10


def scan_completed():
    """返回已完成 case 序号集合（.out 尾部含 Finished:）"""
    done = set()
    for p in glob.glob(os.path.join(RUN_DIR, "case_[0-9][0-9][0-9].out")):
        idx = int(os.path.basename(p)[5:8])
        try:
            with open(p, "rb") as f:
                f.seek(-4096, 2)
                tail = f.read().decode("utf-8", "ignore")
            if "Finished:" in tail:
                done.add(idx)
        except OSError:
            pass
    return done


def current_case(done):
    """找到当前正在求解的 case：最近更新的未完成 .status.txt"""
    best, best_mtime = None, 0.0
    for p in glob.glob(os.path.join(RUN_DIR, "case_[0-9][0-9][0-9].status.txt")):
        idx = int(os.path.basename(p)[5:8])
        if idx in done:
            continue
        m = os.path.getmtime(p)
        if m > best_mtime:
            best, best_mtime = idx, m
    return best


def parse_log():
    """从 feko_run.log 解析：开始时间、失败次数、重试次数、失败角度清单、完成时间戳列表"""
    start_ts = None
    fails = 0
    retries = 0
    failed_idx = set()
    ok_times = []
    if os.path.exists(LOGFILE):
        with open(LOGFILE, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\]", line)
                if not m:
                    continue
                ts = time.mktime(time.strptime(
                    f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S"))
                if "批量生成启动" in line:
                    start_ts = ts
                elif "失败跳过" in line or "批量结束" in line:
                    for mm in re.finditer(r"case_(\d{3})", line):
                        failed_idx.add(int(mm.group(1)))
                elif " 失败:" in line and "第" in line:
                    fails += 1
                    mm = re.search(r"case_(\d{3})", line)
                    if mm:
                        failed_idx.add(int(mm.group(1)))
                elif " OK " in line and "case_" in line:
                    ok_times.append(ts)
                mm = re.search(r"第 (\d)/\d 次", line)
                if mm and int(mm.group(1)) > 1:
                    retries += 1
    return start_ts, fails, retries, failed_idx, ok_times


def feko_alive():
    """FEKO 求解进程是否存活"""
    if HAVE_PSUTIL:
        n = sum(1 for pr in psutil.process_iter(["name"])
                if pr.info["name"] and re.search(r"feko", pr.info["name"], re.I))
        return n
    return None


def fmt_eta(seconds):
    if seconds is None or seconds < 0:
        return "--"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h{m:02d}m"


def snapshot():
    done = scan_completed()
    total = TOTAL
    cur = current_case(done)
    start_ts, fails, retries, failed_idx, ok_times = parse_log()
    alive = feko_alive()

    lines = []
    lines.append("=" * 56)
    lines.append("FEKO 数据生成监控   " + time.strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("-" * 56)
    lines.append(f"进度: {len(done)}/{total}  ({len(done)/total*100:.1f}%)")
    lines.append(f"当前: " + (f"case_{cur:03d} 求解中" if cur is not None
                             else ("全部完成" if len(done) >= total else "等待下一 case")))
    lines.append(f"失败: {fails} 次  重试: {retries} 次" +
                 (f"  失败角度: {sorted(failed_idx)}" if failed_idx else ""))
    if alive is None:
        lines.append("生成器: 未检测(psutil 不可用)")
    elif alive > 0:
        lines.append(f"生成器: 运行中 (FEKO 进程 {alive} 个)")
    else:
        # 无 FEKO 进程：可能正在 case 间隔，或已结束/挂掉
        log_new = os.path.exists(LOGFILE) and (time.time() - os.path.getmtime(LOGFILE)) < 600
        lines.append("生成器: 无 FEKO 进程" + ("" if log_new else " 且日志 10 分钟内未更新 → 可能已中断!"))

    now = time.time()
    NOMINAL_PER_CASE = 8.4 * 60  # 已知单例耗时（秒），事件不足时兜底
    if ok_times:
        gaps = sorted(b - a for a, b in zip(ok_times, ok_times[1:]) if b > a)
        if gaps:
            per_case = gaps[len(gaps) // 2]  # 中位数
        elif start_ts:
            per_case = max(ok_times[0] - start_ts, NOMINAL_PER_CASE * 0.5)
        else:
            per_case = NOMINAL_PER_CASE
    else:
        per_case = NOMINAL_PER_CASE
    rate = 60.0 / per_case

    if start_ts:
        elapsed = max(now - start_ts, 1.0)
        lines.append(f"已用: {fmt_eta(elapsed)}  速率: {rate:.3f} case/min  "
                     f"(单例 {fmt_eta(per_case)}/case)")
        remain = (total - len(done)) / rate * 60.0
        eta = time.strftime("%m-%d %H:%M", time.localtime(now + remain))
        lines.append(f"预计剩余: {fmt_eta(remain)}  预计完成: {eta}")
    lines.append("=" * 56)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="单次快照后退出")
    ap.add_argument("--interval", type=int, default=60, help="刷新间隔秒数（默认 60）")
    args = ap.parse_args()
    while True:
        print(snapshot(), flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
