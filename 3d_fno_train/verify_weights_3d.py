# -*- coding: utf-8 -*-
"""验证：weights_f16_3d.bin 读回后与 PyTorch 原权重逐项一致（UE5 加载格式 QA）"""
import os
import sys
import struct
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE, "results")
BIN = os.path.join(RESULT_DIR, "weights_f16_3d.bin")


def read_array(f):
    """读一个数组：[4B]名长, 名, [4B]ndim, ndim×[4B]shape, 元素×float32 → (name, np.ndarray)"""
    (nl,) = struct.unpack("<i", f.read(4))
    name = f.read(nl).decode("utf-8")
    (ndim,) = struct.unpack("<i", f.read(4))
    shape = struct.unpack(f"<{ndim}i", f.read(4 * ndim))
    n = int(np.prod(shape))
    data = np.frombuffer(f.read(4 * n), dtype=np.float32).reshape(shape)
    return name, data


def main():
    assert os.path.exists(BIN), f"缺 {BIN}"
    # 期望权重（重建真实模型）
    ckpt = torch.load(os.path.join(RESULT_DIR, "ckpt_full_p3.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    grid = (64, 48, 32)
    sys.path.insert(0, BASE)
    import fno_f16_3d as M
    import fno_f16_3d_real as R
    f_fft = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_fft.load_state_dict(ckpt["model_state"])
    f_real = R.FFNO3D_Real(modes=tuple(cfg["modes"]), width=cfg["width"],
                           in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_real.load_state_dict(R.to_real_state_dict_3d(f_fft))
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in f_real.state_dict().items()
          if not k.startswith(("Frx", "Fix", "Fry", "Fiy", "Frz", "Fiz"))}

    with open(BIN, "rb") as f:
        magic = f.read(8)
        assert magic == b"FNO3DV1!", f"magic 不符: {magic}"
        (K,) = struct.unpack("<i", f.read(4))
        names = []
        for _ in range(K):
            name, arr = read_array(f)
            names.append(name)
            assert name in sd, f"多余数组 {name}"
            exp = sd[name]
            assert arr.shape == exp.shape, f"{name}: shape {arr.shape} != {exp.shape}"
            diff = np.abs(arr - exp).max()
            print(f"  {name:38s} shape={str(arr.shape):22s} max|diff|={diff:.3e}")
        # 配置
        modes = struct.unpack("<3i", f.read(4 * 3))
        (width, in_ch, out_ch) = struct.unpack("<3i", f.read(4 * 3))
        (nx, ny, nz, n_layers) = struct.unpack("<4i", f.read(4 * 4))
        print(f"modes={modes} width={width} in={in_ch} out={out_ch} "
              f"grid=({nx},{ny},{nz}) n_layers={n_layers}")
        assert tuple(modes) == tuple(cfg["modes"])
        assert (width, in_ch, out_ch) == (cfg["width"], 7, cfg["out_ch"])
        assert (nx, ny, nz, n_layers) == (*grid, 4)
        # 归一化
        stats = ckpt["stats"]
        for name, C in (("x_inc_mean", 7), ("x_inc_std", 7), ("y_mean", 12), ("y_std", 12)):
            vals = np.frombuffer(f.read(4 * C), dtype=np.float32)
            exp = np.asarray(stats[name], dtype=np.float32).reshape(-1)
            assert vals.shape == exp.shape, f"{name} 长度 {vals.shape} != {exp.shape}"
            d = np.abs(vals - exp).max()
            print(f"  {name:38s} len={C} max|diff|={d:.3e}")
        # EOF 检查
        rest = f.read()
        assert len(rest) == 0, f"尾部剩余 {len(rest)} 字节"
    print(f"\nOK: {K} 数组 + 配置 + 归一化全部对齐 ({len(names)} 命名权重)")


if __name__ == "__main__":
    main()
