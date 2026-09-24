# -*- coding: utf-8 -*-
"""
fig_errprop_law.py — 相干增益定律图（修正口径版，英文标注，供 AWPL 正文 Fig.3）
================================================================================
数据源：results/_diag_errprop_exact.json（修正口径扫描）
  · 蓝实线：median|ΔRCS| vs **干净估计**（同一 NFFFT 链路）→ 拟合 K=4.50±0.06, b=−0.05, R²=0.999
  · 红点线：median|ΔRCS| vs FEKO 真值（旧口径）→ 被 3.4 dB 阶梯面 floor 掩蔽
  · 黑虚线：一阶理论 2.39·ε/√G（G 中位 0.29）→ 4.44 dB/ε
  · 星标：三个真实模型检查点（x 取 ε=H_rel，y 取 E_sur 中位，与"vs 干净参考"同口径）

用法：
  & "F:/miniconda3/envs/isaac311/python.exe" fig_errprop_law.py
产出：results/fig_errprop.png
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE, "results")

# 模型检查点：x=ε=H_rel（J=n̂×H_tot 决定远场），y=E_sur 中位（=vs 干净参考，与扫描口径一致）
# 前两点取 exp_seeds_*.json 的三 seed 均值（468 角 in-sample / 234 留出角），
# 第三点（正则化负结果）为单次运行，取 exp_nffft_anglesets.json
MODEL_PTS = [("F-FNO w128 in-sample\n(53.4%, 4.32 dB)", 0.5343, 4.322),
             ("F-FNO w64 unseen\n(85.8%, 6.16 dB)", 0.8579, 6.165),
             ("structured-error\n(96.3%, 16.07 dB)", 0.9628, 16.066)]


def main():
    with open(os.path.join(RESULT_DIR, "_diag_errprop_exact.json"), encoding="utf-8") as f:
        d = json.load(f)
    sw = d["sweep"]
    eps = np.array([r["eps"] for r in sw])
    med = np.array([r["med_vs_clean"] for r in sw])
    sem = np.array([r["med_vs_clean_sem"] for r in sw])
    rms = np.array([r["rms_vs_clean"] for r in sw])
    feko = np.array([r["med_vs_FEKO"] for r in sw])
    G = d["G_med"]
    fm, fr = d["fit_median_vs_clean"], d["fit_rms_vs_clean"]
    K_th = 2.392 / np.sqrt(G)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    xs = np.linspace(0, eps.max(), 100)
    ax.plot(xs, K_th * xs, "--", color="k", lw=1.3,
            label=r"1st-order theory  $2.39\,\varepsilon/\sqrt{G}$"
                  f"  ($G$={G:.2f}) = {K_th:.2f} dB/$\\varepsilon$")
    ax.plot(xs, fm["K"] * xs + fm["b"], "-", color="tab:blue", lw=1.6,
            label=f"white-noise scan vs clean ref.\nfitted {fm['K']:.2f}$\\varepsilon$"
                  f"$-$({abs(fm['b']):.2f}) dB,  $R^2$={fm['r2']:.4f}")
    ax.errorbar(eps, med, yerr=sem, fmt="s", ms=5, color="tab:blue", capsize=3,
                label="median$|\\Delta$RCS$|$ vs clean reference")
    ax.plot(eps, rms, "^-", ms=5, lw=1.1, color="tab:green",
            label=f"rms($\\Delta$RCS)  (slope {fr['K']:.2f} dB/$\\varepsilon$)")
    ax.plot(eps, feko, "o:", ms=4.5, lw=1.1, color="tab:red",
            label="median$|\\Delta$RCS$|$ vs solver truth (old\ncalibration: masked by the 3.4 dB pipeline floor)")
    for lab, xe, ye in MODEL_PTS:
        ax.plot([xe], [ye], "*", ms=15, color="tab:orange", zorder=5)
        ax.annotate(lab, (xe, ye), textcoords="offset points", xytext=(-6, 8),
                    fontsize=7.5, ha="right", color="tab:orange")
    ax.set_xlabel(r"near-field relative error  $\varepsilon$")
    ax.set_ylabel(r"$|\Delta$RCS$|$  (dB)")
    ax.set_xlim(0, 1.06)
    ax.set_ylim(0, 18)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.2, loc="upper left")
    fig.tight_layout()
    png = os.path.join(RESULT_DIR, "fig_errprop.png")
    fig.savefig(png, dpi=200)
    print(f"已存: {png}")
    print(f"K_th={K_th:.3f} dB/eps (G={G:.4f})  K_meas={fm['K']:.3f}  ratio="
          f"{fm['K']/K_th:.3f}  b={fm['b']:.3f} dB  rms slope={fr['K']:.3f}")


if __name__ == "__main__":
    main()
