# -*- coding: utf-8 -*-
"""
bench_batch.py — 验证 ONNX 模型 batch>1 推理的可行性与加速效果
================================================================
做法：复用 ue_rcs_service.Engine，取多个不同角度的入射场按 batch 轴拼接，
     实测 session.run 的耗时与吞吐，判断 UE 多帧能否批量加速。

用法：& "F:/miniconda3/envs/isaac311/python.exe" bench_batch.py
"""
import os
import sys
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ue_rcs_service import RcsEngine  # noqa: E402


def main():
    eng = RcsEngine()
    print(f"provider = {eng.provider}")

    # 8 个不同的有效入射角（θ 30-150，φ 0-360），模拟 UE 多帧
    angles = [(60.0, 30.0), (90.0, 90.0), (120.0, 150.0), (45.0, 210.0),
              (135.0, 300.0), (75.0, 60.0), (105.0, 240.0), (60.0, 180.0)]
    xs = [eng.make_input(t, p).astype(np.float32) for t, p in angles]  # 各 (1,7,64,48,32)

    print(f"{'batch':>5} | {'每批ms':>7} | {'每帧ms':>7} | {'吞吐帧/s':>8} | out shape")
    for B in (1, 2, 4, 8):
        xb = np.concatenate(xs[:B], axis=0)          # (B,7,64,48,32)
        eng.sess.run(None, {"input": xb})            # warmup
        reps = 8
        t0 = time.perf_counter()
        out = None
        for _ in range(reps):
            out = eng.sess.run(None, {"input": xb})[0]
        dt = (time.perf_counter() - t0) / reps
        print(f"{B:>5} | {dt*1000:7.0f} | {dt/B*1000:7.0f} | {1/(dt/B):8.1f} | {out.shape}")


if __name__ == "__main__":
    main()
