# -*- coding: utf-8 -*-
"""P4 调试：诊断 NFFFT 各环节量级"""
import os, sys
import numpy as np
import h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fno_f16_3d as M
import fno_f16_3d_p4_nffft as P

angles, e0, khat, beta = M.build_incidence_table()
with h5py.File(M.H5, "r") as f:
    E_scat = f["E_scat"][:]
    H_scat = f["H_scat"][:]
    eps = f["eps_field"][:]
    ff_theta = f["ff_theta"][:]; ff_phi = f["ff_phi"][:]
    gx = f["grid_x"][:].astype(np.float64)
    gy = f["grid_y"][:].astype(np.float64)
    gz = f["grid_z"][:].astype(np.float64)
    eff_h5 = f["E_ff"][0]
    rcs_h5 = f["rcs"][0]

i = 0
idxs, rsurf, dS = P.surface_parts(eps, gx, gy, gz)
print("rsurf range:", rsurf.min(0), rsurf.max(0), "dS[0]:", dS[0], "|dS| sum:", np.abs(dS).sum())

Etot, Htot = P.build_tot_fields(angles, e0, khat, beta, gx, gy, gz, E_scat, H_scat, i)
print("E_scat[i] abs max:", np.abs(E_scat[i]).max(), "H_scat[i] abs max:", np.abs(H_scat[i]).max())
print("Etot abs max:", np.abs(Etot).max(), "Htot abs max:", np.abs(Htot).max())

E_s = P.sample_surface_field(Etot, idxs)
H_s = P.sample_surface_field(Htot, idxs)
print("E_s abs max:", np.abs(E_s).max(), "H_s abs max:", np.abs(H_s).max())

n = dS / (np.linalg.norm(dS, axis=1, keepdims=True) + 1e-12)
J = np.cross(n, H_s)
Mm = -np.cross(n, E_s)
print("J abs max:", np.abs(J).max(), "M abs max:", np.abs(Mm).max())

rhat, th, ph, shape = P.direction_grid(ff_theta, ff_phi)
k = float(beta[0])

# 手工展开 nffft 逐步打印
phase = np.exp(1j * k * (rsurf @ rhat.T))
print("phase abs min/max:", np.abs(phase).min(), np.abs(phase).max())
print("phase dtype:", phase.dtype, "phase.T shape:", phase.T.shape)
JdS = J * dS
print("J dtype:", J.dtype, "dS dtype:", dS.dtype, "JdS abs max:", np.abs(JdS).max())
N = phase.T @ JdS
L = phase.T @ (Mm * dS)
print("N dtype:", N.dtype, "N shape:", N.shape, "N abs max:", np.abs(N).max(), "L abs max:", np.abs(L).max())
Nrhat = (N * rhat).sum(axis=1, keepdims=True) * rhat
cross = np.cross(rhat, L)
print("Nrhat abs max:", np.abs(Nrhat).max(), "cross abs max:", np.abs(cross).max())
E_ff = P.nffft(J, Mm, rsurf, dS, rhat, k)
print("E_ff abs max:", np.abs(E_ff).max())
E_theta = (E_ff * th).sum(axis=1)
E_phi = (E_ff * ph).sum(axis=1)
print("E_theta abs max:", np.abs(E_theta).max(), "E_phi abs max:", np.abs(E_phi).max())
rcs = 4*np.pi*(np.abs(E_theta)**2 + np.abs(E_phi)**2)/np.linalg.norm(e0[i])**2
print("rcs min/max:", rcs.min(), rcs.max())

# J-only（PEC 物理正确：M=0）对比
E_ffJ = P.nffft(J, np.zeros_like(Mm), rsurf, dS, rhat, k)
rcsJ = 4*np.pi*(np.abs((E_ffJ*th).sum(axis=1))**2 + np.abs((E_ffJ*ph).sum(axis=1))**2)/np.linalg.norm(e0[i])**2
rcsJ = rcsJ.reshape(shape)
rcs = rcs.reshape(shape)
rcs_h5 = rcs_h5.reshape(shape)
m = rcs_h5 > 1e-6
for name, arr in [("J+M", rcs), ("J-only", rcsJ)]:
    d = np.abs(10*np.log10(arr[m]+1e-9) - 10*np.log10(rcs_h5[m]+1e-9))
    print(f"{name}: |ΔdB| med={np.median(d):.2f} p90={np.percentile(d,90):.2f} "
          f"corr={np.corrcoef(10*np.log10(arr[m]+1e-9), 10*np.log10(rcs_h5[m]+1e-9))[0,1]:.3f}")

print("h5 E_ff abs max:", np.abs(eff_h5).max(), "h5 rcs min/max:", rcs_h5.min(), rcs_h5.max())
print("|E0|:", np.linalg.norm(e0[i]), "e0:", e0[i])
