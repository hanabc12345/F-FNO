# -*- coding: utf-8 -*-
"""导出 UE5 可直接加载的 3D F-FNO P3 权重文件 weights_f16_3d.bin
格式（全部小端 float32，对齐 2D export_weights.py 惯例）：
  [8B]   magic "FNO3DV1!"
  [4B]   int32 数组个数 K
  K × ( [4B]名字长度 N, N×[1B]名字, [4B]ndim, ndim×[4B]shape, 元素数×[4B]float32 )
  [4B]   modes[0], modes[1], modes[2]（int32）
  [4B]   width, in_ch, out_ch（int32）
  [4B]   nx, ny, nz, n_layers（int32）
  [4B]   x_inc_mean ×7, x_inc_std ×7, y_mean ×12, y_std ×12（float32，47 个）
  注：归一化参数按原形状 (1,C,1,1,1) 顺序展平写入。
用法：
  & "F:/miniconda3/envs/isaac311/python.exe" export_weights_3d.py
产出：results/weights_f16_3d.bin
"""
import os
import sys
import struct
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_real as R

RESULT_DIR = os.path.join(BASE, "results")


def main():
    ckpt = torch.load(os.path.join(RESULT_DIR, "ckpt_full_p3.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    grid = (64, 48, 32)

    f_fft = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_fft.load_state_dict(ckpt["model_state"])
    f_real = R.FFNO3D_Real(modes=tuple(cfg["modes"]), width=cfg["width"],
                           in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_real.load_state_dict(R.to_real_state_dict_3d(f_fft))

    # 实数参数（复数谱权重已拆 _r/_i）
    sd = {}
    for k, v in f_real.state_dict().items():
        if not k.startswith(("Frx", "Fix", "Fry", "Fiy", "Frz", "Fiz")):
            sd[k] = v
    stats = ckpt["stats"]
    out_path = os.path.join(RESULT_DIR, "weights_f16_3d.bin")
    with open(out_path, "wb") as f:
        f.write(b"FNO3DV1!")
        names = list(sd.keys())
        f.write(struct.pack("<i", len(names)))
        for name in names:
            arr = sd[name].detach().cpu().numpy().astype(np.float32)
            nb = name.encode("utf-8")
            f.write(struct.pack("<i", len(nb)))
            f.write(nb)
            f.write(struct.pack("<i", arr.ndim))
            f.write(struct.pack("<%di" % arr.ndim, *arr.shape))
            f.write(arr.tobytes())
        # 配置
        f.write(struct.pack("<3i", *cfg["modes"]))
        f.write(struct.pack("<3i", cfg["width"], 7, cfg["out_ch"]))
        f.write(struct.pack("<4i", *grid, 4))
        # 归一化参数（按通道序展平）
        for name, C in (("x_inc_mean", 7), ("x_inc_std", 7), ("y_mean", 12), ("y_std", 12)):
            arr = np.asarray(stats[name], dtype=np.float32).reshape(-1)
            assert arr.shape[0] == C, f"{name} 长度 {arr.shape[0]} != {C}"
            f.write(struct.pack(f"<{C}f", *arr))
    size = os.path.getsize(out_path)
    print(f"已导出: {out_path} ({size/1e6:.1f} MB)")
    print(f"数组数: {len(names)}, 配置: modes={cfg['modes']} width={cfg['width']} "
          f"out={cfg['out_ch']} grid={grid}")
    print(f"归一化: x_inc_mean[0]={stats['x_inc_mean'].reshape(-1)[0]:.4f} "
          f"y_std[0]={stats['y_std'].reshape(-1)[0]:.4f}")


if __name__ == "__main__":
    main()
