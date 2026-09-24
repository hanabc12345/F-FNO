# -*- coding: utf-8 -*-
"""
exp_seeds.py — 多 seed 训练 + 角度级不确定性（回应审稿意见 #4/#5）
=================================================================
目的：
  ① 主模型（F-FNO w128/300ep）与基线（3D U-Net）各训练 ≥3 个 seed，
     使 Table I 的比较在**相同训练轮数**下成立（U-Net 也跑 300 ep）；
  ② 每个 seed 在**完整角度集**上跑 NFFFT 远场 RCS（full→468 角，interp→234 角），
     落盘 (n,37,73) 数组供复算；
  ③ angle-level bootstrap（对入射角有放回重采样）给出 median|ΔRCS| 的 95% 置信区间，
     并报告 seed 间波动 —— 明确"实验单位=入射角，角内观察像素高度相关，不做像素级 bootstrap"；
  ④ 同时给出可归因于代理的直接误差 E_sur = error[NFFFT(J_pred), NFFFT(J_true)]。

关键实现点：
  · 模型初始化必须发生在 manual_seed 之前（exp_baselines.train_model 只在训练前设种子，
    若模型先建则 seed 不控制初始化 → 本脚本先设种子再建模型）；
  · P = exp(jk r̂·r′) 相位核只算一次；(θ,φ) 观测网格 37×73；
  · 真值链路（floor）每个角度只算一次，所有 seed 共用。

用法：
  # 训练（GPU 由 CUDA_VISIBLE_DEVICES 或 --gpu 决定）
  & "F:/miniconda3/envs/isaac311/python.exe" exp_seeds.py --stage train \
      --arch ffno --split full --seeds 0,1,2 --epochs 300 --width 128
  & "F:/miniconda3/envs/isaac311/python.exe" exp_seeds.py --stage train \
      --arch unet --split full --seeds 0,1,2 --epochs 300 --width 64
  # 评估（载入已有 ckpt，跑 NFFFT + bootstrap）
  & "F:/miniconda3/envs/isaac311/python.exe" exp_seeds.py --stage eval \
      --arch ffno --split full --seeds 0,1,2
  # 推理成本基准（参数/延迟/FLOPs）
  & "F:/miniconda3/envs/isaac311/python.exe" exp_seeds.py --stage bench

产出（results/）：
  seed_{arch}_{split}_s{k}.pt                 每个 seed 的 ckpt
  exp_seeds_{arch}_{split}.json               近场指标 + 远场统计 + bootstrap CI
  exp_seeds_{arch}_{split}_rcs.npz            rcs_true / floor / pred_s{k} (n,37,73)
  exp_seeds_bench.json                        参数/延迟/FLOPs
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import sys
import json
import time
import argparse

import numpy as np
import h5py
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import fno_f16_3d as M
import fno_f16_3d_p3 as P3
import exp_baselines as EB
from fno_f16_3d_p4_nffft import (surface_parts, direction_grid, pred_to_surface)
from _exp_common import _rcs_stats, build_x_single

RESULT_DIR = os.path.join(BASE, "results")
ETA0 = 119.9169832 * np.pi
N_BOOT = 1000


# ============================================================
# 一、训练
# ============================================================

def train_seed(args, arch, split, seed, device, x_all, y_all_c, angles, ym, ys, xm, xs):
    """建模型（先设种子）→ 训练 → 存 ckpt + 近场指标"""
    idx_tr, idx_te = EB.make_split(split, angles)
    modes = tuple(int(m) for m in args.modes.split(","))
    torch.manual_seed(seed); np.random.seed(seed)
    model = EB.build_model(arch, args.width, modes, 7, 12).to(device)
    n_par = model.count_params()
    print(f"\n=== [{arch}/{split} seed={seed}] params={n_par:,} epochs={args.epochs} ===",
          flush=True)
    x_tr, _, _ = M.standardize(x_all, idx_tr)
    EB.train_model(arch, model, x_tr, y_all_c, idx_tr, ym, ys, device,
                   args.epochs, 8 if arch in ("ffno", "unet") else 1,
                   3e-3, 4096, seed)

    y_tr, _, _ = M.standardize(y_all_c, idx_tr)
    metrics = {}
    for tag, idx in (("train", idx_tr), ("test", idx_te)):
        if len(idx) == 0:
            continue
        rE, rH, mE, mH, cE, cH = P3.evaluate12(model, x_tr[idx], y_tr[idx],
                                               ym, ys, device, batch=1)
        metrics[tag] = {"rel_E": rE, "rel_H": rH, "med_E": mE, "med_H": mH,
                        "corr_E": cE, "corr_H": cH}
        print(f"  [{tag:5s}] E_rel={rE*100:.2f}% H_rel={rH*100:.2f}% "
              f"E_corr={cE:.3f}", flush=True)

    ck = {"model_state": model.state_dict(), "arch": arch, "split": split, "seed": seed,
          "config": {"modes": list(modes), "width": args.width, "epochs": args.epochs},
          "stats": {"ym": ym, "ys": ys, "xm": xm, "xs": xs}, "metrics": metrics}
    cpath = os.path.join(RESULT_DIR, f"seed_{arch}_{split}_s{seed}.pt")
    torch.save(ck, cpath)
    print(f"  ckpt: {cpath}", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"seed": seed, "params": n_par, "metrics": metrics}


# ============================================================
# 二、NFFFT 评估（floor 共用，逐 seed 预测）
# ============================================================

def rcs_of(H_s, P, dA, nvec, rhat, th, ph, k, e0mag):
    J = np.cross(nvec, H_s)                      # PEC: J = n̂×H_tot, M = 0
    dA_dS = dA[:, None]
    N = P.T @ (J * dA_dS)
    Np = N - (N * rhat).sum(axis=1, keepdims=True) * rhat
    E_ff = (1j * k / (4.0 * np.pi)) * (-ETA0 * Np)
    return 4.0 * np.pi * (np.abs((E_ff * th).sum(axis=1)) ** 2
                          + np.abs((E_ff * ph).sum(axis=1)) ** 2) / e0mag ** 2


def angle_set(split, angles, angle_set_opt):
    if angle_set_opt == "interp234" or (angle_set_opt == "auto" and split == "interp"):
        return np.where(angles[:, 1] % 20 == 10)[0]
    return np.arange(len(angles))


def load_seed_model(arch, split, seed, device):
    ck = torch.load(os.path.join(RESULT_DIR, f"seed_{arch}_{split}_s{seed}.pt"),
                    map_location="cpu", weights_only=False)
    st = ck["stats"]
    model = EB.build_model(arch, ck["config"]["width"],
                          tuple(ck["config"]["modes"]), 7, 12).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    return (model,
            np.asarray(st["ym"], np.float32).reshape(12, 1, 1, 1),
            np.asarray(st["ys"], np.float32).reshape(12, 1, 1, 1),
            np.asarray(st["xm"], np.float32).reshape(1, 7, 1, 1, 1),
            np.asarray(st["xs"], np.float32).reshape(1, 7, 1, 1, 1),
            ck)


def eval_rcs(arch, split, seeds, device, angle_set_opt="auto"):
    angles, e0, khat, beta = M.build_incidence_table()
    with h5py.File(M.H5, "r") as f:
        eps = f["eps_field"][:]
        ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
        gx = f["grid_x"][:].astype(np.float64)
        gy = f["grid_y"][:].astype(np.float64)
        gz = f["grid_z"][:].astype(np.float64)
        rcs_true_all = f["rcs"][:]
    k = float(beta[0])
    idxs, rsurf, dS = surface_parts(eps, gx, gy, gz)
    rhat, th, ph, shape = direction_grid(ff_theta, ff_phi)
    dA = np.linalg.norm(dS, axis=1)
    nvec = dS / dA[:, None]
    P = np.exp(1j * k * (rsurf @ rhat.T))
    sel = angle_set(split, angles, angle_set_opt)
    rcs_true = rcs_true_all[sel]
    print(f"[eval {arch}/{split}] 角度 {len(sel)}（interp-test {int(np.isin(sel, np.where(angles[:,1]%20==10)[0]).sum())}）"
          f" 面元 {len(dA)} 观测 {rhat.shape[0]}", flush=True)

    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    X = X.astype(np.float32); Y = Y.astype(np.float32); Z = Z.astype(np.float32)

    models = [load_seed_model(arch, split, s, device) for s in seeds]
    n_seed = len(models)
    floor = np.zeros((len(sel), *shape), dtype=np.float64)
    preds = [np.zeros((len(sel), *shape), dtype=np.float64) for _ in range(n_seed)]

    fh = h5py.File(M.H5, "r")
    t0 = time.time()
    for a, i in enumerate(sel):
        Hs_s = fh["H_scat"][i]
        phase = np.exp(-1j * beta[i] * (khat[i, 0] * X + khat[i, 1] * Y + khat[i, 2] * Z))
        e_inc = e0[i][None, None, None, :] * phase[..., None]
        h_inc = np.cross(khat[i][None, None, None, :],
                         np.broadcast_to(e_inc, e_inc.shape)) / ETA0
        H_tot = h_inc + Hs_s
        e0mag = float(np.linalg.norm(e0[i]))
        floor[a] = rcs_of(H_tot[idxs[:, 0], idxs[:, 1], idxs[:, 2]],
                          P, dA, nvec, rhat, th, ph, k, e0mag).reshape(shape)
        x = build_x_single(i, eps, e0, khat, beta, gx, gy, gz)
        for s_i, (model, ym, ys, xm, xs, _) in enumerate(models):
            with torch.no_grad():
                pred = model(torch.from_numpy((x - xm) / xs).to(device))[0].cpu().numpy()
            pred = pred * ys + ym
            _, H_s = pred_to_surface(pred, idxs)
            preds[s_i][a] = rcs_of(H_s, P, dA, nvec, rhat, th, ph, k,
                                   e0mag).reshape(shape)
        if (a + 1) % 20 == 0:
            print(f"  [{a+1}/{len(sel)}] {time.time()-t0:.0f}s", flush=True)
    fh.close()
    return {"angles_idx": sel, "rcs_true": rcs_true, "floor": floor,
            "preds": {s: preds[j] for j, s in enumerate(seeds)}}


# ============================================================
# 三、统计 + angle-level bootstrap
# ============================================================

def bootstrap_median_ci(diff_abs, mask, n_boot=N_BOOT, seed=0):
    """对**入射角**有放回重采样，重算 pooled median|ΔRCS| 的 95% CI。
    diff_abs/mask: (n_angle, 37, 73)。角内像素整体重采样（不做像素级 bootstrap）。"""
    n = diff_abs.shape[0]
    blocks = [diff_abs[a][mask[a]] for a in range(n)]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = np.median(np.concatenate([blocks[j] for j in idx]))
    return {"ci_low": float(np.percentile(vals, 2.5)),
            "ci_high": float(np.percentile(vals, 97.5)),
            "boot_median": float(np.median(vals)), "n_boot": n_boot,
            "n_angle": int(n)}


def decompose(res, seeds):
    """E_ref = error[floor, solver]; E_sur = error[pred, floor]; E_end = error[pred, solver]"""
    out = {"E_ref": _rcs_stats(res["floor"], res["rcs_true"]), "seeds": {}}
    mask = res["rcs_true"] > 1e-6
    dB_s = 10 * np.log10(np.maximum(res["rcs_true"], 1e-9))
    dB_f = 10 * np.log10(np.maximum(res["floor"], 1e-9))
    out["E_ref"]["bootstrap"] = bootstrap_median_ci(np.abs(dB_f - dB_s), mask)
    for s in seeds:
        p = res["preds"][s]
        dB_p = 10 * np.log10(np.maximum(p, 1e-9))
        e_end = _rcs_stats(p, res["rcs_true"])
        e_sur = _rcs_stats(p, res["floor"])
        e_end["bootstrap"] = bootstrap_median_ci(np.abs(dB_p - dB_s), mask)
        e_sur["bootstrap"] = bootstrap_median_ci(np.abs(dB_p - dB_f), mask)
        out["seeds"][str(s)] = {"E_end": e_end, "E_sur": e_sur}
    e_end_med = [out["seeds"][str(s)]["E_end"]["rcs_dB_err_median"] for s in seeds]
    e_sur_med = [out["seeds"][str(s)]["E_sur"]["rcs_dB_err_median"] for s in seeds]
    out["seed_variation"] = {
        "E_end_median_list": e_end_med,
        "E_end_mean": float(np.mean(e_end_med)), "E_end_std": float(np.std(e_end_med, ddof=1)),
        "E_sur_median_list": e_sur_med,
        "E_sur_mean": float(np.mean(e_sur_med)), "E_sur_std": float(np.std(e_sur_med, ddof=1))}
    return out


# ============================================================
# 四、推理成本基准
# ============================================================

def bench(args, device):
    import torch.nn as nn
    rows = []
    # U-Net 用 base=64，与 Table I 的 22.4M 基线一致（base=width）
    specs = [("ffno", 128, (20, 16, 10)), ("unet", 64, (20, 16, 10))]
    x = torch.randn(1, 7, 64, 48, 32, device=device)
    for arch, width, modes in specs:
        torch.manual_seed(0)
        model = EB.build_model(arch, width, modes, 7, 12).to(device).eval()
        n_par = model.count_params()
        with torch.no_grad():
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(30):
                model(x)
            torch.cuda.synchronize()
            ms = (time.time() - t0) / 30 * 1e3
        flops = None
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fc = FlopCounterMode(display=False)
            with torch.no_grad(), fc:
                model(x)
            flops = int(fc.get_total_flops())
        except Exception as e:                      # FFT 等算子可能不计入
            print(f"  [warn] FLOPs 统计失败（{arch}）: {e}")
        rows.append({"arch": arch, "width": width, "params": n_par,
                     "inference_ms_per_sample": ms, "flops": flops,
                     "flops_note": "FlopCounterMode; FFT ops may be uncounted"})
        print(f"  {arch} w{width}: params={n_par:,}  {ms:.1f} ms/sample  FLOPs={flops}")
        del model
        torch.cuda.empty_cache()
    jp = os.path.join(RESULT_DIR, "exp_seeds_bench.json")
    with open(jp, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  已存: {jp}")


# ============================================================
# 主流程
# ============================================================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="train", choices=["train", "eval", "bench"])
    ap.add_argument("--arch", default="ffno", choices=["ffno", "unet"])
    ap.add_argument("--split", default="full", choices=["full", "interp"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--modes", default="20,16,10")
    ap.add_argument("--angle-set", default="auto", choices=["auto", "full", "interp234"])
    ap.add_argument("--gpu", default="")
    args = ap.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"设备: {device}  arch={args.arch} split={args.split} seeds={seeds} "
          f"epochs={args.epochs}")
    os.makedirs(RESULT_DIR, exist_ok=True)

    if args.stage == "bench":
        bench(args, device)
        return

    if args.stage == "train":
        x_all, y_all, _, angles, eps = EB.load_p3_cached(use_cache=True)
        y_all_c = P3.clip12(y_all)
        idx_tr, _ = EB.make_split(args.split, angles)
        _, xm, xs = M.standardize(x_all, idx_tr)
        _, ym, ys = M.standardize(y_all_c, idx_tr)
        out = []
        for s in seeds:
            out.append(train_seed(args, args.arch, args.split, s, device,
                                  x_all, y_all_c, angles, ym, ys, xm, xs))
        jp = os.path.join(RESULT_DIR, f"exp_seeds_train_{args.arch}_{args.split}.json")
        with open(jp, "w") as f:
            json.dump({"arch": args.arch, "split": args.split, "epochs": args.epochs,
                       "width": args.width, "modes": args.modes, "runs": out}, f, indent=2)
        print(f"  已存: {jp}")
    else:                                          # eval
        res = eval_rcs(args.arch, args.split, seeds, device, args.angle_set)
        dec = decompose(res, seeds)
        out = {"arch": args.arch, "split": args.split, "angle_set": args.angle_set,
               "n_angle": int(len(res["angles_idx"])), "decomposition": dec,
               "angle_idx": [int(x) for x in res["angles_idx"]]}
        jp = os.path.join(RESULT_DIR, f"exp_seeds_{args.arch}_{args.split}.json")
        with open(jp, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=float)
        npz = os.path.join(RESULT_DIR, f"exp_seeds_{args.arch}_{args.split}_rcs.npz")
        np.savez_compressed(npz, rcs_true=res["rcs_true"].astype(np.float32),
                            floor=res["floor"].astype(np.float32),
                            **{f"pred_s{s}": res["preds"][s].astype(np.float32)
                               for s in seeds})
        print(f"  已存: {jp}\n  已存: {npz}")
        print(f"  E_ref 中位 {dec['E_ref']['rcs_dB_err_median']:.2f} dB "
              f"[{dec['E_ref']['bootstrap']['ci_low']:.2f}, "
              f"{dec['E_ref']['bootstrap']['ci_high']:.2f}]")
        for s in seeds:
            e1 = dec["seeds"][str(s)]["E_end"]; e2 = dec["seeds"][str(s)]["E_sur"]
            print(f"  seed{s}: E_end {e1['rcs_dB_err_median']:.2f} dB "
                  f"[{e1['bootstrap']['ci_low']:.2f}, {e1['bootstrap']['ci_high']:.2f}]  "
                  f"E_sur {e2['rcs_dB_err_median']:.2f} dB "
                  f"[{e2['bootstrap']['ci_low']:.2f}, {e2['bootstrap']['ci_high']:.2f}]")


if __name__ == "__main__":
    main()
