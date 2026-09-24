# -*- coding: utf-8 -*-
"""
gen_feko_batch.py — FEKO 批量生成 F-16 三维近场散射场 + 远场 RCS 数据集（F-FNO 训练用，v2）

v2 相对 v1 的变更：
  1. 入射角网格扩展：θ∈{30..150 步长10} × φ∈{0..350 步长10} = 468 组（全向覆盖）
  2. 每个角度一次求解输出 双极化 数据（CalculateOrthogonalPolarisationsEnabled）：
       - 主极化（θ 极化，PolarisationAngle=0）：近场散射 E + H、远场 RCS
       - 正交极化（φ 极化，自动生成）：近场散射 E + H、远场 RCS
     → 完整 2×2 RCS 极化矩阵 (σθθ,σφθ,σθφ,σφφ) 一次求解可得
  3. 近场同时输出 E 与 H（CalculateMagneticFields=true，六分量）

流程：
  1. 对每个入射角 (theta, phi) 生成 Lua 脚本（导入 f16_refined.stl → 3GHz 平面波(正交极化) →
     近场笛卡尔网格（仅散射 E+H 场）+ 远场 RCS）
  2. 调 cadfeko.exe --non-interactive --run-script 运行（串行）
  3. 解析 .out 中两套近场散射 E/H 场（64×48×32 复数）与两套远场 RCS
  4. 体素化网格 → epsilon_r 场（FNO 输入）
  5. 组装 HDF5 数据集（v2，含 H_scat 与正交极化数据）

用法：
  python gen_feko_batch.py --pilot           # 只跑第 1 个角度（θ=30,φ=0）验证管线
  python gen_feko_batch.py --only 0,1,2      # 只跑指定角度序号
  python gen_feko_batch.py --resume          # 跳过已完成(.out 含 Finished:)的角度，断点续跑
  python gen_feko_batch.py --build-h5        # 跳过 FEKO，仅从已有 .out 组装 HDF5
  python gen_feko_batch.py --verify-total 0  # 散射场一致性验证
  python gen_feko_batch.py --resume --max-retries 2   # 失败/卡死自动重试（默认 2 次）

日志监控（v3）：
  - 运行日志：3d_feko_run/feko_run.log（时间戳 + 事件：启动/完成/失败/重试/停滞）
  - 停滞检测：psutil 采样 FEKO 进程树累计 CPU，--stall-window 秒内累计 CPU 低于
    --stall-cpu-min 秒 → 判定卡死，记录日志并杀进程树后重试
  - 失败处理：ERROR/.out 无 Finished/超时/停滞 → 记录报错摘要 → 重试（默认 2 次）
    → 仍失败则跳过该角度并继续（不中断整批）
"""
import os
import re
import sys
import time
import argparse
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import h5py

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("[warn] psutil 不可用，停滞检测退化为 .out mtime 判定", flush=True)

BASE = os.path.dirname(os.path.abspath(__file__))
RUN_DIR = r"f:\MyWorkSpace\UAVGame\3d_feko_run"
CADFEKO = r"F:\Program Files\Altair\2026\feko\bin\cadfeko.exe"
STL = os.path.join(RUN_DIR, "f16_refined.stl")
OUT_H5 = os.path.join(BASE, "f16_3d_rcs_dataset_v2.h5")
LOGFILE = os.path.join(RUN_DIR, "feko_run.log")

STALL_SAMPLE = 30.0  # 停滞检测 CPU 采样间隔（秒）

FREQ_HZ = 3e9
LAMBDA = 2.99792458e8 / FREQ_HZ

# ---------------- 近场网格（与体素化共用，均匀 0.03125 m ≈ λ/3.2） ----------------
NX, NY, NZ = 64, 48, 32
X0, Y0, Z0 = -0.75, -0.75, -0.60
X1, Y1, Z1 = 1.21875, 0.71875, 0.36875
DX = (X1 - X0) / (NX - 1)
DY = (Y1 - Y0) / (NY - 1)
DZ = (Z1 - Z0) / (NZ - 1)

# ---------------- 入射角网格：468 组（θ 30..150 step10 × φ 0..350 step10） ----------------
THETAS = list(range(30, 151, 10))
PHIS = list(range(0, 360, 10))


def angle_list():
    return [(t, p) for t in THETAS for p in PHIS]


# ---------------- Lua 脚本模板 ----------------
# 单平面波源 + CalculateOrthogonalPolarisationsEnabled（自动产出正交极化第二激励）
LUA_TEMPLATE = """-- {name}.lua  theta={theta} phi={phi} (near E+H, ortho-polarisation, scattered={scattered})
local status = io.open([[{run_dir}\\{name}.status.txt]], "w")
local function log(m)
    if status then status:write(m .. "\\n") status:flush() end
end
local ok, err = pcall(function()
    application = cf.Application.GetInstance()
    project = application:NewProject()

    local meshes = project.Importer.MeshImporter:Import([[{stl}]])
    log("import ok count=" .. tostring(#meshes))

    project.Contents.SolutionConfigurations.GlobalFrequency.Start = "{freq}"
    log("freq ok")

    local pw = project.Contents.SolutionConfigurations.GlobalSources:AddPlaneWave({theta}, {phi})
    pw:SetProperties({{
        PolarisationAngle = 0.0,
        CalculateOrthogonalPolarisationsEnabled = {ortho},
    }})
    log("pw ok " .. tostring(pw.Label))

    local config = project.Contents.SolutionConfigurations[1]
    local nf = config.NearFields:AddCartesian({x0}, {y0}, {z0}, {x1}, {y1}, {z1}, {nx}, {ny}, {nz})
    nf:SetProperties({{
        Advanced = {{
            CalculateMagneticFields = {calc_h},
            OnlyScatteredPartCalculationEnabled = {scattered},
        }}
    }})
    log("nf ok " .. tostring(nf.Label))

    local ff = config.FarFields:Add(0, 0, 180, 360, 5, 5)
    log("ff ok " .. tostring(ff.Label))

    application:SaveAs([[{run_dir}\\{name}.cfx]])
    log("save ok")

    local result = application.Launcher:RunFEKO()
    log("runfeko ok succeeded=" .. tostring(result.Succeeded))
end)
if not ok then log("FAIL: " .. tostring(err)) end
status:close()
"""


def make_lua(idx, theta, phi, scattered=True, calc_h=True, ortho=True, name=None):
    if name is None:
        name = f"case_{idx:03d}"
    lua = os.path.join(RUN_DIR, f"{name}.lua")
    content = LUA_TEMPLATE.format(
        name=name, theta=theta, phi=phi, stl=STL, run_dir=RUN_DIR,
        freq=f"{FREQ_HZ:.0f}", x0=X0, y0=Y0, z0=Z0, x1=X1, y1=Y1, z1=Z1,
        nx=NX, ny=NY, nz=NZ,
        scattered="true" if scattered else "false",
        calc_h="true" if calc_h else "false",
        ortho="true" if ortho else "false",
    )
    with open(lua, "w", encoding="utf-8") as f:
        f.write(content)
    return lua


# ---------------- 运行单个角度（v3：日志 + 停滞检测 + 自动重试） ----------------
def log_event(msg):
    """统一监控日志：控制台 + feko_run.log 双写，带时间戳"""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _proc_tree_cpu_seconds(root_pid):
    """root 进程及其全部后代进程的累计 CPU 秒数（user+kernel）。
    求解主力是 feko.csv.impi/runfeko 等子进程，cadfeko 本身 CPU 极低，
    因此必须按进程树聚合，否则无法区分"卡死"与"正常等待"。"""
    if not HAVE_PSUTIL:
        return 0.0
    total = 0.0
    if not psutil.pid_exists(root_pid):
        return total
    stack = [psutil.Process(root_pid)]
    while stack:
        p = stack.pop()
        try:
            t = p.cpu_times()
            total += t.user + t.system
            stack.extend(p.children(recursive=False))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _kill_tree(proc):
    """强杀 cadfeko 整棵进程树，避免残留 feko.csv.impi 孤儿进程占用 CPU/内存"""
    try:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, timeout=30)
    except Exception:
        pass
    if HAVE_PSUTIL:
        try:
            p = psutil.Process(proc.pid)
            for kid in p.children(recursive=True):
                try:
                    kid.kill()
                except Exception:
                    pass
            p.kill()
        except Exception:
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def _error_excerpt(text, maxlen=400):
    """从 .out 提取与错误相关的行摘要（含 error/fatal/exception），便于日志记录"""
    lines = []
    for m in re.finditer(r"(?im)^.*\b(error|fatal|exception)\b.*$", text):
        s = m.group(0).strip()
        if not s or "error estimate" in s.lower():
            continue  # 跳过 FEKO 摘要表里的良性"error estimates"行
        if s not in lines:
            lines.append(s)
        if len(lines) >= 5:
            break
    if not lines:
        lines = text.strip().splitlines()[-3:]
    return " | ".join(lines)[:maxlen]


def _solve_once(idx, theta, phi, timeout, calc_h, ortho, stall_window, stall_cpu_min):
    """单次求解尝试。返回 (idx, theta, phi, ok, dt, msg)"""
    out_path = os.path.join(RUN_DIR, f"case_{idx:03d}.out")
    for p in (os.path.join(RUN_DIR, f"case_{idx:03d}.lua"),
              out_path,
              os.path.join(RUN_DIR, f"case_{idx:03d}.status.txt")):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    lua = make_lua(idx, theta, phi, scattered=True, calc_h=calc_h, ortho=ortho)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            [CADFEKO, "--non-interactive", "--run-script", lua],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        return idx, theta, phi, False, 0.0, f"launch error: {e}"

    last_cpu, last_t = None, None
    while time.time() - t0 < timeout:
        # cadfeko 退出且 .out 含 Finished 标记 → 完成
        done_file = False
        if os.path.exists(out_path):
            try:
                with open(out_path, "rb") as f:
                    f.seek(-4096, 2)
                    tail = f.read().decode("utf-8", "ignore")
                done_file = "Finished:" in tail
            except OSError:
                done_file = False
        if proc.poll() is not None and done_file:
            # 校验有无 ERROR
            try:
                with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                if re.search(r"\bERROR\b", txt):
                    _kill_tree(proc)
                    return (idx, theta, phi, False, time.time() - t0,
                            "ERROR in .out: " + _error_excerpt(txt))
            except OSError:
                pass
            return idx, theta, phi, True, time.time() - t0, "ok"
        if proc.poll() is not None and not done_file:
            # 进程已退出但 .out 无 Finished 标记 → 直接判失败
            _kill_tree(proc)
            return idx, theta, phi, False, time.time() - t0, "cadfeko 提前退出，.out 无 Finished:"
        # 停滞检测：窗口内进程树累计 CPU 无进展 → 卡死
        if proc.poll() is None and HAVE_PSUTIL:
            now = time.time()
            cpu = _proc_tree_cpu_seconds(proc.pid)
            if last_cpu is not None:
                wall = now - last_t
                dcpu = cpu - last_cpu
                if wall >= stall_window and dcpu < stall_cpu_min:
                    _kill_tree(proc)
                    return (idx, theta, phi, False, now - t0,
                            f"STALL: {wall:.0f}s 内进程树累计 CPU 仅 {dcpu:.1f}s")
            last_cpu, last_t = cpu, now
        time.sleep(STALL_SAMPLE)
    if proc.poll() is None:
        _kill_tree(proc)
    return idx, theta, phi, False, time.time() - t0, "timeout"


def run_case(idx, theta, phi, timeout=5400.0, calc_h=True, ortho=True,
             max_retries=2, stall_window=360.0, stall_cpu_min=15.0):
    """带自动重试的 case 求解：失败/超时/停滞 → 记录日志 → 重试 → 仍失败则跳过"""
    attempts = 0
    res = None
    while attempts <= max_retries:
        attempts += 1
        log_event(f"case_{idx:03d} θ={theta:3.0f} φ={phi:3.0f} 启动求解 (第 {attempts} 次尝试)")
        res = _solve_once(idx, theta, phi, timeout, calc_h, ortho,
                          stall_window, stall_cpu_min)
        _, _, _, ok, dt, msg = res
        if ok:
            log_event(f"case_{idx:03d} θ={theta:3.0f} φ={phi:3.0f} OK (尝试 {attempts} 次) "
                      f"{dt/60:.1f} min")
            return res
        log_event(f"case_{idx:03d} θ={theta:3.0f} φ={phi:3.0f} 失败: {msg} "
                  f"(第 {attempts}/{max_retries+1} 次)")
        if attempts <= max_retries:
            time.sleep(5)  # 短暂等待后重试
    return res


# ---------------- 解析 .out ----------------
def parse_plane_wave(text, occurrence=0):
    """入射平面波信息：theta, phi, E0 复数(3,), 传播方向 khat(3,), beta0。
    occurrence=0 主极化(θ)，occurrence=1 正交极化(φ，仅当 CalculateOrthogonalPolarisationsEnabled)。"""
    pat = r"Direction of incidence:\s+THETA =\s+([\d.]+)\s+PHI =\s+([\d.]+)"
    pos = -1
    for _ in range(occurrence + 1):
        m = re.search(pat, text[pos + 1:])
        if not m:
            raise ValueError(f"plane wave incidence #{occurrence} not found")
        pos = pos + 1 + m.start()
    theta, phi = float(m.group(1)), float(m.group(2))
    e0 = np.zeros(3, dtype=complex)
    seg = text[pos:]
    for c in "XYZ":
        m = re.search(rf"\|E0{c}\| =\s*([\d.Ee+-]+)\s+ARG\(E0{c}\) =\s*([\d.Ee+-]+)", seg)
        e0["XYZ".index(c)] = float(m.group(1)) * np.exp(1j * np.deg2rad(float(m.group(2))))
    khat = np.array([float(re.search(rf"BETA0{c} =\s*([\d.Ee+-]+)", seg).group(1)) for c in "XYZ"])
    m = re.search(r"Wave number:\s+BETA0\s+=\s*\(\s*([\d.Ee+-]+)", seg)
    beta0 = float(m.group(1))
    return theta, phi, e0, khat, beta0


_NF_ROW = re.compile(
    r"^\s*0\s+(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s+"
    r"(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s+"
    r"(-?[\d.Ee+-]+)\s+(-?[\d.Ee+-]+)\s*$"
)

MARKER_E = "VALUES OF THE ELECTRIC FIELD STRENGTH in V/m"
MARKER_H = "VALUES OF THE MAGNETIC FIELD STRENGTH in A/m"


def parse_nearfield(text, marker=MARKER_E, occurrence=0, nx=NX, ny=NY, nz=NZ):
    """解析近场 E 或 H 场（总场或散射场，取决于请求设置），返回复数数组 (nx, ny, nz, 3)。
    marker: MARKER_E / MARKER_H；occurrence: 0=主极化，1=正交极化。"""
    pos = -1
    for _ in range(occurrence + 1):
        pos = text.find(marker, pos + 1)
        if pos < 0:
            raise ValueError(f"near field section #{occurrence} ({marker[:30]}) not found")
    seg = text[pos:]
    rows = []
    in_data = False
    for line in seg.splitlines():
        if line.startswith("medium"):
            in_data = True
            continue
        if not in_data:
            continue
        m = _NF_ROW.match(line)
        if m:
            rows.append([float(x) for x in m.groups()])
        elif (line.strip().startswith("Near field request") or
                line.strip().startswith("Far field request") or
                line.strip().startswith("VALUES OF THE") or
                "read from buffer:" in line):
            break
    n = nx * ny * nz
    if len(rows) != n:
        raise ValueError(f"near field rows {len(rows)} != {n} (occurrence {occurrence})")
    a = np.asarray(rows, dtype=np.float64)  # (n,9): x,y,z,Fxm,Fxp,Fym,Fyp,Fzm,Fzp
    reim = a[:, 3:].reshape(nz, ny, nx, 3, 2)  # 行序: x 最快
    F = reim[..., 0] * np.exp(1j * np.deg2rad(reim[..., 1]))
    return F.transpose(2, 1, 0, 3)  # (nx,ny,nz,3)


_FF_ROW = re.compile(
    r"^\s*([\d.]+)\s+([\d.]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+"
    r"([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+"
    r"([\d.Ee+-]+)\s+\w+\s*$"
)


def parse_farfield(text, occurrence=0):
    """解析远场 RCS 表（θ 0-180/5° × φ 0-360/5°），返回 dict:
    theta(nt,), phi(np,), Etheta(nt,np), Ephi(nt,np), rcs(nt,np) (m²)。
    occurrence=0 主极化，1 正交极化。"""
    marker = "VALUES OF THE SCATTERED ELECTRIC FIELD STRENGTH IN THE FAR FIELD"
    pos = -1
    for _ in range(occurrence + 1):
        pos = text.find(marker, pos + 1)
        if pos < 0:
            raise ValueError(f"far field section #{occurrence} not found")
    seg = text[pos:]
    rows = []
    for line in seg.splitlines():
        # 段终止：遇到下一段起始标记即停止
        if ("read from buffer:" in line or
                line.strip().startswith("EXCITATION BY INCIDENT") or
                line.strip().startswith("Near field request") or
                line.strip().startswith("Far field request")):
            break
        m = _FF_ROW.match(line)
        if m:
            rows.append([float(x) for x in m.groups()])
    a = np.asarray(rows, dtype=np.float64)  # (n,9): theta,phi,Etm,Etp,Epm,Epp,scs,ar,angle
    thetas = np.unique(a[:, 0])
    phis = np.unique(a[:, 1])
    ntheta, nphi = len(thetas), len(phis)
    if a.shape[0] != ntheta * nphi:
        raise ValueError(f"far field rows {a.shape[0]} != {ntheta}*{nphi}")
    et = a[:, 2] * np.exp(1j * np.deg2rad(a[:, 3]))
    ep = a[:, 4] * np.exp(1j * np.deg2rad(a[:, 5]))
    rcs = a[:, 6]  # scattering cross section in m²
    # 表中 phi 为外层、theta 为内层 → reshape(nphi, ntheta) 后转置
    return dict(
        theta=thetas, phi=phis,
        Etheta=et.reshape(nphi, ntheta).T,
        Ephi=ep.reshape(nphi, ntheta).T,
        rcs=rcs.reshape(nphi, ntheta).T,
    )


def parse_out(idx):
    """解析 case_<idx>.out → 主/正交极化 的 E/H 近场 + 远场 RCS"""
    out_path = os.path.join(RUN_DIR, f"case_{idx:03d}.out")
    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    theta, phi, e0, khat, beta0 = parse_plane_wave(text, occurrence=0)
    _, _, e0_ortho, _, _ = parse_plane_wave(text, occurrence=1)
    E_scat = parse_nearfield(text, MARKER_E, occurrence=0)
    H_scat = parse_nearfield(text, MARKER_H, occurrence=0)
    E_scat_ortho = parse_nearfield(text, MARKER_E, occurrence=1)
    H_scat_ortho = parse_nearfield(text, MARKER_H, occurrence=1)
    ff = parse_farfield(text, occurrence=0)
    ff_ortho = parse_farfield(text, occurrence=1)
    return dict(theta=theta, phi=phi, e0=e0, e0_ortho=e0_ortho,
                khat=khat, beta0=beta0,
                E_scat=E_scat, H_scat=H_scat,
                E_scat_ortho=E_scat_ortho, H_scat_ortho=H_scat_ortho,
                ff=ff, ff_ortho=ff_ortho)


# ---------------- 散射场一致性验证（总场 - 入射场 vs 散射场） ----------------
def run_total_verify(idx, timeout=5400.0):
    """对角度 angles[idx] 用总场(OnlyScatteredPart=false)重跑一遍，
    验证 E_scat ≈ E_total - E_inc（解析入射场），返回 (max_abs_err, max_rel_err)。"""
    theta, phi = angle_list()[idx]
    name = f"case_{idx:03d}_total"
    lua = os.path.join(RUN_DIR, f"{name}.lua")
    out_path = os.path.join(RUN_DIR, f"{name}.out")
    make_lua(idx, theta, phi, scattered=False, calc_h=False, ortho=False, name=name)
    for p in (out_path, os.path.join(RUN_DIR, f"{name}.status.txt")):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass
    print(f"  [verify-total] 运行 θ={theta} φ={phi} 总场近场求解 …")
    t0 = time.time()
    proc = subprocess.Popen(
        [CADFEKO, "--non-interactive", "--run-script", lua],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    while time.time() - t0 < timeout:
        done_file = False
        if os.path.exists(out_path):
            try:
                with open(out_path, "rb") as f:
                    f.seek(-4096, 2)
                    tail = f.read().decode("utf-8", "ignore")
                done_file = "Finished:" in tail
            except OSError:
                done_file = False
        if proc.poll() is not None and done_file:
            break
        time.sleep(10)
    if proc.poll() is None:
        proc.kill()
        raise RuntimeError(f"{name} timeout after {timeout/60:.0f} min")
    if not os.path.exists(out_path):
        raise RuntimeError(f"{name}.out not found")

    with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
        txt_total = f.read()
    _, _, e0, khat, beta0 = parse_plane_wave(txt_total, occurrence=0)
    E_total = parse_nearfield(txt_total, MARKER_E, occurrence=0)
    E_scat = parse_out(idx)["E_scat"]  # 既有散射场（主极化）

    # 解析入射场 E_inc(r) = E0 * exp(-j beta0 (khat·r))，相位参考点 (0,0,0)
    xs = X0 + DX * np.arange(NX)
    ys = Y0 + DY * np.arange(NY)
    zs = Z0 + DZ * np.arange(NZ)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    phase = -beta0 * (pts @ khat)              # (n,)
    E_inc = (e0 * np.exp(1j * phase)).reshape(NX, NY, NZ, 3)

    E_diff = E_total - E_inc
    num = np.abs(E_diff - E_scat)
    den = np.abs(E_scat)
    rel = num / np.maximum(den, 1e-12)
    print(f"  [verify-total] θ={theta} φ={phi} 结果:")
    print(f"    |E_total|max = {np.abs(E_total).max():.4f}  |E_scat|max = {den.max():.4f}")
    print(f"    E_total - E_inc vs E_scat: max 绝对误差 = {num.max():.3e}, "
          f"mean 绝对误差 = {num.mean():.3e}")
    print(f"    相对误差: max = {rel.max():.3e}, mean = {rel.mean():.3e}")
    return float(num.max()), float(rel.max())


# ---------------- 体素化（FNO 输入：epsilon_r 场） ----------------
_EPS_CACHE = None


def voxelize_eps(force=False):
    global _EPS_CACHE
    cache = os.path.join(BASE, "f16_eps_3d.npy")
    if not force and os.path.exists(cache):
        _EPS_CACHE = np.load(cache)
        return _EPS_CACHE
    import trimesh
    mesh = trimesh.load(STL, process=False)
    mesh.merge_vertices()  # 消除重复顶点，保证 contains() 可靠
    print(f"  voxelize: {len(mesh.faces)} faces, watertight={mesh.is_watertight}")
    xs = X0 + DX * np.arange(NX)
    ys = Y0 + DY * np.arange(NY)
    zs = Z0 + DZ * np.arange(NZ)
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    inside = mesh.contains(pts)
    eps = np.ones(NX * NY * NZ, dtype=np.float32)
    eps[inside] = 1e6
    eps = eps.reshape(NX, NY, NZ)
    np.save(cache, eps)
    _EPS_CACHE = eps
    return eps


# ---------------- 组装 HDF5 ----------------
def build_h5(idx_list, results):
    angles = angle_list()
    E_all, H_all, Eo_all, Ho_all = [], [], [], []
    ang_all, rcs_all, E_ff_all = [], [], []
    rcs_o_all, E_ff_o_all = [], []
    ff_theta = ff_phi = None
    meta = []
    used = []

    for idx in idx_list:
        theta, phi = angles[idx]
        try:
            d = parse_out(idx)
        except Exception as e:
            print(f"  [warn] case_{idx:03d} parse failed: {e}")
            continue
        used.append(idx)
        ang_all.append((theta, phi))
        E_all.append(d["E_scat"].astype(np.complex64))
        H_all.append(d["H_scat"].astype(np.complex64))
        Eo_all.append(d["E_scat_ortho"].astype(np.complex64))
        Ho_all.append(d["H_scat_ortho"].astype(np.complex64))
        rcs_all.append(d["ff"]["rcs"].astype(np.float32))
        E_ff_all.append(np.stack([d["ff"]["Etheta"], d["ff"]["Ephi"]], axis=-1).astype(np.complex64))
        rcs_o_all.append(d["ff_ortho"]["rcs"].astype(np.float32))
        E_ff_o_all.append(np.stack([d["ff_ortho"]["Etheta"], d["ff_ortho"]["Ephi"]], axis=-1).astype(np.complex64))
        if ff_theta is None:
            ff_theta = d["ff"]["theta"]
            ff_phi = d["ff"]["phi"]
        meta.append(dict(idx=idx, theta=theta, phi=phi,
                         e0=d["e0"], e0_ortho=d["e0_ortho"],
                         khat=d["khat"], beta0=d["beta0"]))
        print(f"  case_{idx:03d} θ={theta} φ={phi}: "
              f"|E_scat|max={np.abs(d['E_scat']).max():.3f} "
              f"|H_scat|max={np.abs(d['H_scat']).max():.3f} "
              f"RCSmax={d['ff']['rcs'].max():.3f} m² "
              f"RCSmax(φ-pol)={d['ff_ortho']['rcs'].max():.3f} m²")

    n = len(used)
    if n == 0:
        print("  无有效 .out，跳过 HDF5 组装")
        return []
    E_all = np.asarray(E_all)
    H_all = np.asarray(H_all)
    Eo_all = np.asarray(Eo_all)
    Ho_all = np.asarray(Ho_all)
    ang_all = np.asarray(ang_all)
    rcs_all = np.asarray(rcs_all)
    E_ff_all = np.asarray(E_ff_all)
    rcs_o_all = np.asarray(rcs_o_all)
    E_ff_o_all = np.asarray(E_ff_o_all)
    ntheta, nphi = len(ff_theta), len(ff_phi)

    eps = voxelize_eps()
    with h5py.File(OUT_H5, "w") as f:
        f.attrs["freq_hz"] = FREQ_HZ
        f.attrs["lambda_m"] = LAMBDA
        f.attrs["mesh"] = os.path.basename(STL)
        f.attrs["n_angles"] = len(used)
        f.attrs["version"] = 2
        f.attrs["dual_polarisation"] = True
        f.create_dataset("eps_field", data=eps, compression="gzip")
        f.create_dataset("grid_x", data=X0 + DX * np.arange(NX))
        f.create_dataset("grid_y", data=Y0 + DY * np.arange(NY))
        f.create_dataset("grid_z", data=Z0 + DZ * np.arange(NZ))
        f["grid_x"].attrs["x0"] = X0
        f["grid_x"].attrs["dx"] = DX
        f["grid_y"].attrs["y0"] = Y0
        f["grid_y"].attrs["dy"] = DY
        f["grid_z"].attrs["z0"] = Z0
        f["grid_z"].attrs["dz"] = DZ
        f.create_dataset("angles", data=ang_all)  # [n,2] theta,phi (deg)
        # 主极化（θ 极化入射）
        f.create_dataset("E_scat", data=E_all, compression="gzip")      # [n,nx,ny,nz,3] complex64
        f.create_dataset("H_scat", data=H_all, compression="gzip")      # [n,nx,ny,nz,3] complex64
        f.create_dataset("rcs", data=rcs_all, compression="gzip")       # m² [n,nt,np]
        f.create_dataset("E_ff", data=E_ff_all, compression="gzip")     # [n,nt,np,2] Eθ/Eφ
        # 正交极化（φ 极化入射）
        f.create_dataset("E_scat_ortho", data=Eo_all, compression="gzip")
        f.create_dataset("H_scat_ortho", data=Ho_all, compression="gzip")
        f.create_dataset("rcs_ortho", data=rcs_o_all, compression="gzip")
        f.create_dataset("E_ff_ortho", data=E_ff_o_all, compression="gzip")
        f.create_dataset("ff_theta", data=ff_theta)
        f.create_dataset("ff_phi", data=ff_phi)
    print(f"\n  HDF5 已保存: {OUT_H5}")
    print(f"  E_scat shape: {E_all.shape} dtype={E_all.dtype}")
    print(f"  H_scat shape: {H_all.shape} dtype={H_all.dtype}")
    print(f"  rcs shape:    {rcs_all.shape}")
    print(f"  eps shape:    {eps.shape}  (1=自由空间, 1e6=金属)")
    return meta


def main():
    # 独立进程(Start-Process)启动时 stdout 可能为 GBK，打印 θ/φ/² 会 UnicodeEncodeError
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="只运行第 1 个角度（θ=30,φ=0）")
    ap.add_argument("--only", type=str, default=None, help="逗号分隔的角度序号，如 0,1,2")
    ap.add_argument("--resume", action="store_true",
                    help="跳过 .out 中已含 Finished: 的角度（断点续跑）")
    ap.add_argument("--max-parallel", type=int, default=1, help="并发 FEKO 求解数（默认 1）")
    ap.add_argument("--build-h5", action="store_true", help="跳过 FEKO，仅从 .out 组装 HDF5")
    ap.add_argument("--no-ortho", action="store_true", help="关闭正交极化（仅主极化，节约一半时间）")
    ap.add_argument("--no-h", action="store_true", help="不计算 H 场（仅 E 场）")
    ap.add_argument("--verify-total", type=int, default=-1,
                    help="散射场一致性验证：对指定角度序号用总场重跑，对比 E_scat 与 E_total-E_inc")
    ap.add_argument("--max-timeout", type=float, default=5400.0, help="单角度超时秒数")
    ap.add_argument("--max-retries", type=int, default=2,
                    help="失败/超时/停滞时重试次数（默认 2）")
    ap.add_argument("--stall-window", type=float, default=360.0,
                    help="停滞判定窗口秒数：窗口内进程树累计 CPU 无进展即判定卡死（默认 360）")
    ap.add_argument("--stall-cpu-min", type=float, default=15.0,
                    help="窗口内最小累计 CPU 秒数，低于则判定停滞（默认 15）")
    args = ap.parse_args()

    if args.verify_total >= 0:
        run_total_verify(args.verify_total, args.max_timeout)
        return

    angles = angle_list()
    if args.pilot:
        idx_list = [0]
    elif args.only:
        idx_list = [int(i) for i in args.only.split(",")]
    else:
        idx_list = list(range(len(angles)))

    if args.resume and not args.build_h5:
        idx_list = [i for i in idx_list
                    if not (os.path.exists(os.path.join(RUN_DIR, f"case_{i:03d}.out"))
                            and "Finished:" in open(os.path.join(RUN_DIR, f"case_{i:03d}.out"),
                                                    "r", encoding="utf-8", errors="ignore").read()[-4000:])]
        print(f"  [resume] 剩余 {len(idx_list)} 个角度未完成")

    if args.build_h5:
        build_h5(idx_list, None)
        return

    ortho = not args.no_ortho
    calc_h = not args.no_h
    log_event(f"FEKO 批量生成启动: {len(idx_list)} 个角度, 并发 {args.max_parallel}, "
              f"重试 {args.max_retries} 次, 停滞窗口 {args.stall_window:.0f}s/CPU {args.stall_cpu_min:.0f}s")
    print(f"  网格: {NX}×{NY}×{NZ}, 近场盒 x[{X0},{X1}] y[{Y0},{Y1}] z[{Z0},{Z1}] m")
    print(f"  频率: {FREQ_HZ/1e9:.1f} GHz (λ={LAMBDA*100:.1f} cm)")
    print(f"  近场: 散射 E+H({('开' if calc_h else '关')}) 双极化({('开' if ortho else '关')})")
    print(f"  预估单角度耗时: {8.8 if (calc_h and ortho) else (6.0 if ortho else (6.5 if calc_h else 5.5))} min")
    results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_case, idx, *angles[idx], args.max_timeout, calc_h, ortho,
                          args.max_retries, args.stall_window, args.stall_cpu_min): idx
                for idx in idx_list}
        for fut in as_completed(futs):
            idx, theta, phi, ok, dt, msg = fut.result()
            results[idx] = (ok, dt, msg)
            status = "OK " if ok else "FAIL"
            log_event(f"case_{idx:03d} θ={theta:3.0f} φ={phi:3.0f} {status} "
                      f"{dt/60:.1f} min  {msg}")

    n_ok = sum(1 for v in results.values() if v[0])
    log_event(f"批量结束: 完成 {n_ok}/{len(idx_list)}，总耗时 {(time.time()-t0)/60:.1f} min")
    fail_idx = [idx for idx in idx_list if not results[idx][0]]
    if fail_idx:
        log_event(f"失败跳过: {len(fail_idx)} 个角度 -> {fail_idx}")
    ok_idx = [idx for idx in idx_list if results[idx][0]]
    if ok_idx:
        build_h5(ok_idx, results)


if __name__ == "__main__":
    main()
