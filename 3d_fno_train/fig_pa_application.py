# -*- coding: utf-8 -*-
"""
fig_pa_application.py - paper figure: one forward pass -> arbitrary observation configuration
=============================================================================================
Demonstration figure for the narrative "RCS is only one projection of the learned field":
for each of several incidence angles, a SINGLE P3 forward pass predicts the 12-channel
scattering near field, and the NFFFT then synthesizes the FULL 37x73 far-field RCS
panorama over all observation directions (i.e., an arbitrary bistatic/monostatic surface).

Panels per case: (a) FEKO truth panorama, (b) P3 prediction panorama, (c) |ΔRCS| dB panorama.
Numbers per case (median/P90/corr over the masked panorama) are printed and saved to JSON.

Reuses _exp_common.nffft_eval (truth = NFFFT of exact solver fields, pred = NFFFT of P3 field).
Outputs (results/): fig_pa_panorama.png, panorama_metrics.json
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_p3 as P3
import _exp_common as EC

RESULT_DIR = os.path.join(BASE, "results")
CKPT = os.path.join(RESULT_DIR, "ckpt_full_p3.pt")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[fig_pa] device={device}", flush=True)
    t0 = time.time()

    # ---- load P3 checkpoint + standardization (same as fig_paper.py) ----
    x_all, y_all, idx_all, angles, _ = P3.load_p3()
    y_all_c = P3.clip12(y_all)
    _, xm, xs = M.standardize(x_all, idx_all)
    _, ym, ys = M.standardize(y_all_c, idx_all)
    print(f"[fig_pa] data ready {time.time()-t0:.0f}s", flush=True)

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = M.FFNO3D(modes=tuple(cfg["modes"]), width=cfg["width"], in_ch=7, out_ch=12, gain=0.05).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    print(f"[fig_pa] P3 loaded width={cfg['width']} modes={tuple(cfg['modes'])}", flush=True)

    # ---- select representative incidence angles (search the 468-angle table) ----
    def find(th_des, ph_des):
        d = np.abs(angles[:, 0] - th_des) + np.abs(angles[:, 1] - ph_des)
        return int(np.argmin(d))
    sel = [find(90, 0), find(90, 90), find(150, 180)]   # broadside / oblique / tail-on cone
    print(f"[fig_pa] selected cases: {[(i, int(angles[i,0]), int(angles[i,1])) for i in sel]}", flush=True)

    # ---- full-panorama NFFFT: truth and prediction (reuses experiment pipeline) ----
    stats, rcs_all = EC.nffft_eval(model, sel, xm, xs, ym, ys, device=device,
                                   tag="P3", save_png=None)

    # ---- panorama figure ----
    with EC.h5py.File(M.H5, "r") as f:
        ff_theta = f["ff_theta"][:]
        ff_phi = f["ff_phi"][:]
    true_rcs = rcs_all["truth"]      # (n, 37, 73)  NFFFT of exact fields
    pred_rcs = rcs_all["pred"]       # (n, 37, 73)  NFFFT of P3 fields
    db_true = 10 * np.log10(np.maximum(true_rcs, 1e-12))
    db_pred = 10 * np.log10(np.maximum(pred_rcs, 1e-12))
    mask = true_rcs > 1e-6
    ddb = np.abs(db_pred - db_true); ddb[~mask] = np.nan
    db_true[~mask] = np.nan

    vmin, vmax = -35.0, 20.0
    n = len(sel)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n + 1.0, 10.5))
    for c, (i, r_) in enumerate(zip(sel, range(n))):
        for row, (arr, cmap, label) in enumerate(
                ((db_true[r_], "jet", "FEKO truth"),
                 (db_pred[r_], "jet", "P3 prediction"),
                 (ddb[r_], "hot_r", "|ΔRCS| (dB)"))):
            ax = axes[row, c]
            im = ax.imshow(arr, origin="lower", cmap=cmap, aspect="auto",
                           vmin=(vmin if row < 2 else 0.0),
                           vmax=(vmax if row < 2 else np.nanmax(np.nanpercentile(ddb, 99))))
            ax.set_xticks(np.arange(0, 73, 12), [f"{p:.0f}" for p in ff_phi[::12]])
            ax.set_yticks(np.arange(0, 37, 6), [f"{p:.0f}" for p in ff_theta[::6]])
            if row == 0:
                ax.set_title(f"case #{i}  θ_inc={int(angles[i,0])}°  φ_inc={int(angles[i,1])}°", fontsize=10)
            if row == 2 and c == n - 1:
                cb = fig.colorbar(im, ax=ax, fraction=0.046)
                cb.set_label(label)
            elif c == 0:
                ax.set_ylabel(f"{label}\nθ_obs (deg)")
            if row == 2:
                ax.set_xlabel("φ_obs (deg)")
    fig.suptitle("One forward pass of P3 -> validated NFFFT -> full 37×73 far-field RCS panorama "
                 "(each pixel = RCS at that observation direction)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.join(RESULT_DIR, "fig_pa_panorama.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[fig_pa] -> {out_png}", flush=True)

    # ---- numeric summary table ----
    rows = []
    for a, i in enumerate(sel):
        st = {"case": int(i), "theta_inc": float(angles[i, 0]), "phi_inc": float(angles[i, 1]),
              "theta_obs_grid": [float(ff_theta[0]), float(ff_theta[-1])],
              "phi_obs_grid": [float(ff_phi[0]), float(ff_phi[-1])],
              "median_dB": float(np.nanmedian(np.abs(ddb[a]))),
              "p90_dB": float(np.nanpercentile(np.abs(ddb[a]), 90))}
        mm = mask[a].ravel(); d1 = db_pred[a].ravel()[mm]; d2 = db_true[a].ravel()[mm]
        st["corr"] = float(np.corrcoef(d1, d2)[0, 1]) if d1.std() > 1e-9 else float("nan")
        rows.append(st)
        print(f"  case#{i} (θ={int(angles[i,0])}°,φ={int(angles[i,1])}°): "
              f"panorama |ΔRCS| median {st['median_dB']:.2f} dB, P90 {st['p90_dB']:.2f} dB, "
              f"corr {st['corr']:.3f}", flush=True)
    with open(os.path.join(RESULT_DIR, "panorama_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"[fig_pa] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
