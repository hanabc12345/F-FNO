# -*- coding: utf-8 -*-
"""导出 3D F-FNO P3 模型（E+H 12 通道）为 ONNX + 数值/推理验证
产出：results/fno_f16_3d_p3.onnx（含 eps 掩膜 + E_inc Re/Im ×3 输入，7 通道）
验证：
  1. onnx.checker 结构校验
  2. ORT 与 PyTorch 前向一致（rel < 1e-3）
  3. 真实数据推理误差 vs FEKO 目标（与 P3 metrics 对照）
用法：
  & "F:/miniconda3/envs/isaac311/python.exe" export_onnx_3d.py
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import time
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_real as R
import fno_f16_3d_p3 as P3

RESULT_DIR = os.path.join(BASE, "results")
ONNX_PATH = os.path.join(RESULT_DIR, "fno_f16_3d_p3.onnx")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(RESULT_DIR, "ckpt_full_p3.pt"),
                      map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    grid = (64, 48, 32)

    # 构建实算模型并加载（复数权重 → _r/_i）
    f_fft = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"],
                     in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_fft.load_state_dict(ckpt["model_state"])
    f_real = R.FFNO3D_Real(modes=tuple(cfg["modes"]), width=cfg["width"],
                           in_ch=7, out_ch=cfg["out_ch"], grid=grid)
    f_real.load_state_dict(R.to_real_state_dict_3d(f_fft))
    f_real.cpu().eval()
    print(f"模型: width={cfg['width']} modes={cfg['modes']} out={cfg['out_ch']} "
          f"params={f_real.count_params():,}")

    # ---- 1. ONNX 导出 ----
    dummy = torch.randn(1, 7, *grid)
    torch.onnx.export(
        f_real, dummy, ONNX_PATH,
        input_names=["input"],
        output_names=["pred"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}},
    )
    import onnx
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX 导出 + 结构校验通过: {ONNX_PATH} "
          f"({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")

    # ---- 2. ORT vs PyTorch 前向一致 ----
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    x_np = np.random.randn(1, 7, *grid).astype(np.float32)
    out_ort = sess.run(None, {"input": x_np})[0]
    with torch.no_grad():
        out_pt = f_real(torch.tensor(x_np)).numpy()
    rel = np.linalg.norm(out_ort - out_pt) / np.linalg.norm(out_pt)
    print(f"ORT vs PyTorch: rel={rel:.3e}")
    assert rel < 1e-3, "ONNX 推理与 PyTorch 不一致！"

    # ---- 3. 真实数据推理（30 个角度抽样，对比 FEKO 目标） ----
    x_all, y_all, idx_all, angles, eps_mask = P3.load_p3()
    y_all_c = P3.clip12(y_all)
    stats = ckpt["stats"]
    ym, ys = stats["y_mean"], stats["y_std"]
    xm, xs = stats["x_inc_mean"], stats["x_inc_std"]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(x_all), 30, replace=False)
    t0 = time.time()
    errs_e, errs_h, corrs_e, corrs_h = [], [], [], []
    for i in sel:
        xin = (x_all[i:i+1] - xm) / xs
        out = sess.run(None, {"input": xin.astype(np.float32)})[0][0]
        out = out * ys.reshape(12, 1, 1, 1) + ym.reshape(12, 1, 1, 1)
        yt = y_all_c[i]
        errs_e.append(np.linalg.norm(out[:6] - yt[:6]) / (np.linalg.norm(yt[:6]) + 1e-12))
        errs_h.append(np.linalg.norm(out[6:] - yt[6:]) / (np.linalg.norm(yt[6:]) + 1e-12))
        pm = np.sqrt(np.sum(out[:6].reshape(3, 2, -1) ** 2, axis=(1, 2)))
        tm = np.sqrt(np.sum(yt[:6].reshape(3, 2, -1) ** 2, axis=(1, 2)))
        corrs_e.append(np.corrcoef(pm, tm)[0, 1])
        pm = np.sqrt(np.sum(out[6:].reshape(3, 2, -1) ** 2, axis=(1, 2)))
        tm = np.sqrt(np.sum(yt[6:].reshape(3, 2, -1) ** 2, axis=(1, 2)))
        corrs_h.append(np.corrcoef(pm, tm)[0, 1])
    dt = (time.time() - t0) / len(sel) * 1000
    print(f"ONNX 真实数据验证 (30 样本, CPU 单样本 {dt:.0f} ms):")
    print(f"  E: rel_med={np.median(errs_e)*100:.1f}% corr={np.median(corrs_e):.3f} "
          f"(P3 训练版 E_rel=54.2% corr=0.926)")
    print(f"  H: rel_med={np.median(errs_h)*100:.1f}% corr={np.median(corrs_h):.3f} "
          f"(P3 训练版 H_rel=53.6% corr=0.924)")


if __name__ == "__main__":
    main()
