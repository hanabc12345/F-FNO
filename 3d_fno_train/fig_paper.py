# -*- coding: utf-8 -*-
"""
fig_paper.py - generate paper figures Fig.1 (overview), Fig.2 (architecture), Fig.3 (near-field slices)

Fig.1: (a) PEC airframe voxel cloud, (b) incident wave midplane, (c) far-field RCS pattern
Fig.2: F-FNO architecture schematic (lift -> n x [factorized spectral conv + 1x1 + GELU] -> project)
Fig.3: P3 (w128, 12ch) near-field prediction vs truth, midplane |E| and |H|, 2 cases
Outputs (results/): fig1_overview.png, fig2_arch.png, fig3_nearfield.png
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import time
import numpy as np
import h5py
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_p3 as P3

RESULT_DIR = os.path.join(BASE, "results")
CKPT = os.path.join(RESULT_DIR, "ckpt_full_p3.pt")
H5 = M.H5


def fig2_arch(out_path):
    """F-FNO architecture schematic (matplotlib boxes)."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")

    def box(x, w, y, h, text, fc="#eaf2fb", fs=9, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                           fc=fc, ec="#2b4a6f", lw=1.1)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#12305b")

    def arr(x1, y1, x2, y2):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                            mutation_scale=13, lw=1.3, color="#2b4a6f")
        ax.add_patch(a)

    # flow: input -> lift -> N x spectral blocks -> project -> output -> NFFFT
    box(0.2, 1.5, 1.3, 1.2, "Input\n7\xd7 64\xd7 48\xd7 32\n(mask + E_inc)", fs=8)
    arr(1.5, 1.8, 1.95, 1.8)
    box(1.95, 1.45, 1.4, 1.2, "Lift\n1\xd71\xd71 Conv", fs=9)
    arr(3.35, 1.8, 3.8, 1.8)
    # spectral blocks x4
    box(3.8, 2.3, 2.1, 1.2, "Spectral Block  \u00d7 4\nS(x) + C(x) + GELU\n[W = Wx + Wy + Wz]", fs=8.5, bold=True)
    arr(5.9, 1.8, 6.35, 1.8)
    box(6.35, 1.5, 1.35, 1.2, "Project\n1\xd71\xd71 Conv", fs=9)
    arr(7.7, 1.8, 8.15, 1.8)
    box(8.15, 1.7, 1.55, 1.2, "Output\n12 \xd7 64\xd7 48\xd7 32\n(E + H, Re/Im)", fs=8)
    arr(9.7, 1.8, 10.15, 1.8)
    box(10.15, 1.6, 1.6, 1.2, "NFFFT\nHuygens surface\n\u2192 RCS(θ, φ)", fs=8, fc="#f6f0e0")

    # spectral conv detail callout
    ax.text(4.95, 0.35, "Factorized spectral conv: rfftn \u2192 modes box (20,16,10) \u2192 Wx+Wy+Wz \u2192 zero-pad \u2192 irfftn",
            ha="center", fontsize=8, color="#5a6b7f")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Fig.2 -> {out_path}", flush=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fig_paper] device={device}", flush=True)
    t0 = time.time()
    x_all, y_all, idx_all, angles, eps_mask = P3.load_p3()
    y_all_c = P3.clip12(y_all)
    _, xm, xs = M.standardize(x_all, idx_all)
    _, ym, ys = M.standardize(y_all_c, idx_all)
    print(f"[fig_paper] data ready {time.time()-t0:.0f}s", flush=True)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"], in_ch=7, out_ch=12, gain=0.05).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"[fig_paper] P3 loaded width={cfg['width']} modes={cfg['modes']}", flush=True)

    # ---------------- Fig.1 ----------------
    print("[fig_paper] Fig.1 ...", flush=True)
    with h5py.File(H5, "r") as f:
        rcs_true = f["rcs"][:]
    metal = np.argwhere(eps_mask)
    step = max(1, len(metal) // 400)
    pts = metal[::step]

    i_fig = 0  # theta=30, phi=0
    th_idx = 18  # theta = 90 deg
    ph_deg = np.arange(73) * 5.0
    rcs_db = 10 * np.log10(np.maximum(rcs_true[i_fig, th_idx, :], 1e-12))

    fig1 = plt.figure(figsize=(13, 3.8))
    ax = fig1.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="k", alpha=0.6)
    ax.set_xlabel("x (voxel)"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("(a) PEC airframe (voxelized)")
    ax.view_init(elev=22, azim=-60)

    ax = fig1.add_subplot(1, 3, 2)
    zc = y_all.shape[3] // 2  # y-midplane
    inc = x_all[i_fig, 1]     # Re E_x of incident field
    im = ax.imshow(inc[:, zc, :].T, origin="lower", cmap="RdBu_r", aspect="auto")
    ax.set_title(f"(b) Incident wave Re(E_x), y-midplane\n(θ={int(angles[i_fig,0])}°, φ={int(angles[i_fig,1])}°)", fontsize=9)
    fig1.colorbar(im, ax=ax, fraction=0.046)

    ax = fig1.add_subplot(1, 3, 3)
    ax.plot(ph_deg, rcs_db, "k-", lw=1.2)
    ax.set_xlabel("φ (deg)"); ax.set_ylabel("RCS (dBsm)")
    ax.set_title("(c) Far-field RCS (θ=90°)", fontsize=9)
    ax.grid(alpha=0.3)
    fig1.tight_layout()
    out1 = os.path.join(RESULT_DIR, "fig1_overview.png")
    fig1.savefig(out1, dpi=200); plt.close(fig1)
    print(f"  Fig.1 -> {out1}", flush=True)

    # ---------------- Fig.2 ----------------
    print("[fig_paper] Fig.2 ...", flush=True)
    fig2_arch(os.path.join(RESULT_DIR, "fig2_arch.png"))

    # ---------------- Fig.3 ----------------
    print("[fig_paper] Fig.3 ...", flush=True)
    ym_t = torch.tensor(ym, device=device, dtype=torch.float32)
    ys_t = torch.tensor(ys, device=device, dtype=torch.float32)
    xm_t = torch.tensor(xm, device=device, dtype=torch.float32)
    xs_t = torch.tensor(xs, device=device, dtype=torch.float32)
    sels = [0]                       # 单案例紧凑版（AWPL 4 页限制，不再整页铺 4 行）
    fields = [("|E|", slice(0, 6)), ("|H|", slice(6, 12))]
    with h5py.File(H5, "r") as f:
        gx = f["grid_x"][:]; gy = f["grid_y"][:]; gz = f["grid_z"][:]
    xc = y_all_c.shape[4] // 2          # 横截面所在 x 体素（沿机身中部）
    nrow = len(fields)
    fig3, axes = plt.subplots(nrow, 3, figsize=(7.4, 5.2))
    ext = [gy[0], gy[-1], gz[0], gz[-1]]   # 横轴 y(m)，纵轴 z(m)
    i = sels[0]
    Xn = (torch.from_numpy(x_all[i:i + 1]).to(device) - xm_t) / xs_t
    with torch.no_grad():
        pb = (model(Xn) * ys_t + ym_t).cpu().numpy()[0]  # clipped space
    tb = y_all_c[i]                                      # clipped truth
    for r, (fname, slc) in enumerate(fields):
        tmag = np.sqrt(np.sum(tb[slc][:3] ** 2 + tb[slc][3:] ** 2, axis=0))[xc]
        pmag = np.sqrt(np.sum(pb[slc][:3] ** 2 + pb[slc][3:] ** 2, axis=0))[xc]
        emag = np.abs(pmag - tmag)
        for col, (mag, title) in enumerate(zip((tmag, pmag, emag), ("Truth", "Prediction", "Error"))):
            ax = axes[r, col]
            im = ax.imshow(mag.T, origin="lower", cmap="jet", aspect="auto", extent=ext)
            if r == 0:
                ax.set_title(title, fontsize=10, pad=6)
            if col == 0:
                ax.set_ylabel(f"{fname}\nθ={int(angles[i,0])}°, φ={int(angles[i,1])}°", fontsize=9)
            if r == nrow - 1:
                ax.set_xlabel("y (m)", fontsize=9)
            ax.tick_params(labelsize=7)
            cb = fig3.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cb.ax.tick_params(labelsize=7)
    fig3.suptitle(f"Near-field prediction vs truth, x = {gx[xc]:.2f} m (y–z plane)", fontsize=10)
    fig3.tight_layout(rect=(0, 0, 1, 0.955))
    out3 = os.path.join(RESULT_DIR, "fig3_nearfield.png")
    fig3.savefig(out3, dpi=200); plt.close(fig3)
    print(f"  Fig.3 -> {out3}", flush=True)
    print(f"[fig_paper] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
