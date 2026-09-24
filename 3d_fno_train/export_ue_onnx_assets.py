# -*- coding: utf-8 -*-
"""导出 UE 端使用 ONNX 所需的辅助固定数据（一次生成，UE 端加载一次）

产出：results/ue_fno_assets.bin —— UE ONNX 插件推理时构造输入所需的全部固定数据：
  eps_mask（699 金属体素掩膜）+ 网格坐标 gx/gy/gz + 标准化/反标准化系数（47 个 float）。

文件布局（全部小端）：
  [4B]int32  64          # nx
  [4B]int32  48          # ny
  [4B]int32  32          # nz
  gx:     64 × float32   # x 坐标 (m)
  gy:     48 × float32   # y 坐标 (m)
  gz:     32 × float32   # z 坐标 (m)
  eps_mask: 64×48×32 × float32   # 0=空气 1=金属（C 序，索引 [ix*48+iy]*32+iz）
  x_inc_mean:  7 × float32
  x_inc_std:   7 × float32
  y_mean:     12 × float32
  y_std:      12 × float32

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" export_ue_onnx_assets.py
验证：
  & "F:/miniconda3/envs/isaac311/python.exe" export_ue_onnx_assets.py --verify
"""
import os
import sys
import struct
import argparse

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE, "results")
OUT_PATH = os.path.join(RESULT_DIR, "ue_fno_assets.bin")
H5 = r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_feko_data\f16_3d_rcs_dataset_v2.h5"
CKPT_PATH = os.path.join(RESULT_DIR, "ckpt_full_p3.pt")

GRID = (64, 48, 32)


def build():
    # 1) eps 体素场与网格坐标（h5 小字段）
    with h5py.File(H5, "r") as f:
        eps = f["eps_field"][:]
        gx = f["grid_x"][:].astype(np.float32)
        gy = f["grid_y"][:].astype(np.float32)
        gz = f["grid_z"][:].astype(np.float32)
    eps_mask = (eps > 1.5).astype(np.float32)

    # 2) 标准化/反标准化系数（训练 ckpt stats）
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    st = ckpt["stats"]
    x_inc_mean = np.asarray(st["x_inc_mean"], np.float32).reshape(-1)
    x_inc_std = np.asarray(st["x_inc_std"], np.float32).reshape(-1)
    y_mean = np.asarray(st["y_mean"], np.float32).reshape(-1)
    y_std = np.asarray(st["y_std"], np.float32).reshape(-1)
    assert gx.shape == (64,) and gy.shape == (48,) and gz.shape == (32,)
    assert eps_mask.shape == GRID and int(eps_mask.sum()) == 699
    assert x_inc_mean.shape == (7,) and x_inc_std.shape == (7,)
    assert y_mean.shape == (12,) and y_std.shape == (12,)

    # 3) 打包写盘（全小端）
    buf = struct.pack("<3i", *GRID)
    buf += gx.tobytes() + gy.tobytes() + gz.tobytes()
    buf += eps_mask.tobytes()
    buf += x_inc_mean.tobytes() + x_inc_std.tobytes()
    buf += y_mean.tobytes() + y_std.tobytes()
    with open(OUT_PATH, "wb") as f:
        f.write(buf)
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"已导出: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  grid={GRID} 金属体素={int(eps_mask.sum())}")
    print(f"  gx[0..-1]={gx[0]:.3f}..{gx[-1]:.3f}  gy={gy[0]:.3f}..{gy[-1]:.3f}  gz={gz[0]:.3f}..{gz[-1]:.3f}")
    return dict(eps_mask=eps_mask, gx=gx, gy=gy, gz=gz,
                x_inc_mean=x_inc_mean, x_inc_std=x_inc_std,
                y_mean=y_mean, y_std=y_std)


def verify():
    data = build()
    with open(OUT_PATH, "rb") as f:
        raw = f.read()
    assert len(raw) == 4 * 3 + 64 * 4 + 48 * 4 + 32 * 4 + 64 * 48 * 32 * 4 + (7 + 7 + 12 + 12) * 4
    pos = 0
    nx, ny, nz = struct.unpack_from("<3i", raw, pos); pos += 12
    gx = np.frombuffer(raw, np.float32, 64, pos); pos += 64 * 4
    gy = np.frombuffer(raw, np.float32, 48, pos); pos += 48 * 4
    gz = np.frombuffer(raw, np.float32, 32, pos); pos += 32 * 4
    eps_mask = np.frombuffer(raw, np.float32, 64 * 48 * 32, pos).reshape(64, 48, 32); pos += 64 * 48 * 32 * 4
    x_inc_mean = np.frombuffer(raw, np.float32, 7, pos); pos += 7 * 4
    x_inc_std = np.frombuffer(raw, np.float32, 7, pos); pos += 7 * 4
    y_mean = np.frombuffer(raw, np.float32, 12, pos); pos += 12 * 4
    y_std = np.frombuffer(raw, np.float32, 12, pos); pos += 12 * 4
    assert pos == len(raw), f"EOF 残余 {len(raw) - pos} 字节"
    ok = True
    for name, a, b in [("gx", gx, data["gx"]), ("gy", gy, data["gy"]), ("gz", gz, data["gz"]),
                       ("eps_mask", eps_mask, data["eps_mask"]),
                       ("x_inc_mean", x_inc_mean, data["x_inc_mean"]),
                       ("x_inc_std", x_inc_std, data["x_inc_std"]),
                       ("y_mean", y_mean, data["y_mean"]),
                       ("y_std", y_std, data["y_std"])]:
        same = np.array_equal(a, b)
        ok &= same
        print(f"  {name:10s} shape={a.shape} 读回一致={same}")
    assert ok, "读回校验失败！"
    assert int(eps_mask.sum()) == 699
    print(f"读回校验全部通过（{pos} 字节，无 EOF 残余）")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="生成后读回校验")
    args = ap.parse_args()
    if args.verify:
        verify()
    else:
        build()
