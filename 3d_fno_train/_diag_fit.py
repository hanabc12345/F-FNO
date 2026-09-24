# -*- coding: utf-8 -*-
"""快速诊断：单样本过拟合 + 高学习率，判定是网络结构问题还是优化/数据问题"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"f:\MyWorkSpace\UAVGame\03_FNO-RCS工作区\3d_fno_train")
from fno_f16_3d import load_data, standardize, clip_mag, FFNO3D

x_all, y_all, idx, angles = load_data()
y_all_c = clip_mag(y_all)
i = idx["interp_train"][0]
x1 = (x_all[i:i+1] - x_all[idx["interp_train"]].mean(axis=(0,2,3,4), keepdims=True)) \
     / (x_all[idx["interp_train"]].std(axis=(0,2,3,4), keepdims=True) + 1e-8)
y1 = (y_all_c[i:i+1] - y_all_c[idx["interp_train"]].mean(axis=(0,2,3,4), keepdims=True)) \
     / (y_all_c[idx["interp_train"]].std(axis=(0,2,3,4), keepdims=True) + 1e-8)
xt = torch.from_numpy(x1).cuda()
yt = torch.from_numpy(y1).cuda()
y_std = y_all_c[idx["interp_train"]].std(axis=(0,2,3,4), keepdims=True) + 1e-8
y_mean = y_all_c[idx["interp_train"]].mean(axis=(0,2,3,4), keepdims=True)

torch.manual_seed(0)
for width in (32, 64):
    m = FFNO3D(modes=(20,16,10), width=width, gain=0.05).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    niter = 2000 if width == 32 else 1000
    t0 = time.time()
    print(f"--- width={width} ---")
    for it in range(niter):
        loss = nn.MSELoss()(m(xt), yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if (it+1) % 200 == 0:
            with torch.no_grad():
                p = (m(xt)*torch.tensor(y_std, device='cuda') + torch.tensor(y_mean, device='cuda'))
                y_raw = torch.from_numpy(y_all_c[i:i+1]).cuda()
                rel = (p-y_raw).pow(2).sum().sqrt().item() / y_raw.pow(2).sum().sqrt().item()
            print(f"  width={width} it{it+1:5d} loss={loss.item():.4f} rel={rel*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)

