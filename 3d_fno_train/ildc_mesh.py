# -*- coding: utf-8 -*-
"""
ildc_mesh.py — Mitzner ILDC（增量长度绕射系数）边缘绕射库
================================================================================
理论来源与移植说明
--------------------------------------------------------------------------------
本模块是 **Mitzner (1974) AFAL-TR-73-296《Incremental Length Diffraction
Coefficients》** 中并矢绕射系数（eq. 3-46A / 3-55 ~ 3-74）的 Python 移植。

移植所依据的"活体参照"是开源实现 **cl-rcs**（Tobias Rautenkranz, GPLv3,
https://tobias.rautenkranz.ch/assets/cl-rcs.tar.bz2）：该实现在源码里逐行标注了
Mitzner 报告的公式编号，其作者用 Gordon(1988) 的纯边缘贡献算例与 Mitzner 报告
Figure 9-11 做过数值验证。本模块忠于该实现的**符号约定与特例分支**，不自行改动。

必须继承的两处已知勘误（cl-rcs 作者 + Shore & Yaghjian 1988 一致确认）：
  · eq. 3-46A 末项中的 sin[(βs+βi)/2] 应替换为 sin[(βs-βi)/2]（见 `_d_cross`）
  · eq. 3-73 的 g 特例**不应**含 1/ν 因子（见 `g_all` 的 3-73 分支）

关键物理链条（PO + FW = PTD）：
    E_total = E_PO(受照面物理光学积分)  +  E_fringe(沿棱边的 ILDC 线积分)
ILDC 给出的是**条纹波（fringe wave）**，即"精确电流 − PO 电流"对应的场，因此必须
与 PO 相干叠加，而不是替换 PO。

坐标与角度约定（与 Mitzner / cl-rcs 一致，勿凭直觉改）：
  · e_i = 入射波**传播方向的反向**（由目标指向源）；单站时 e_i = e_s = r̂
  · e_s = 散射波传播方向（由目标指向观察者）
  · β_i = -asin(e_i·l̂)，β_s = +asin(e_s·l̂)（l̂ 为棱边单位切向）——两者符号约定
    不同是参考实现的既定写法，已由 test_beta_i/test_beta_s 逐例核对
  · φ 为绕 l̂ 从 e-x 起算的方位角，e-x = -normalize(n̂₁+n̂₂)（劈内角平分线反向）
  · α 为劈半角，ν = π/(2(π-α)) 为劈因子，外劈角 = π/ν = 2(π-α)
  · 绕射系数 D 的量纲使 σ = (4π/k²)|p̂_r·D·p̂_i|²；本模块返回的是
    **F = D/k**，满足 E_s = F·E_i·e^{ikr}/r（r=1 时直接是场）

用法：
    from ildc_mesh import selftest, build_edges, fringe_far_field
    selftest()          # 直角楔/刀口解析校验（必须先过）
"""
import numpy as np

CIS45 = np.exp(1j * np.pi / 4)
S2PI = np.sqrt(2.0 * np.pi)
EPS_BETA = 1e-6          # 掠入射保护（|β| < π/2 - EPS 才参与）
N_STAT = {"nan": 0, "nonfinite": 0}


# ============================================================
# 一、Mitzner 绕射系数的标量核（f / g 及其特例分支）
# ============================================================

def synth_sinc(x):
    """(未归一化) sinc：sin(x)/x，x=0 处取 1。复数可用。"""
    x = np.asarray(x, dtype=np.complex128)
    out = np.ones_like(x)
    nz = x != 0
    out[nz] = np.sin(x[nz]) / x[nz]
    return out


def cot(x):
    """1/tan（参考实现刻意用 1/tan 而非 cos/sin，见 cl-rcs utils.lisp）"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 / np.tan(x)


def float_equal(a, b, max_rel=1.1920929e-7, max_diff=1e-4):
    """cl-rcs `float-equal` 的逐元素版：绝对差或相对差任一满足即视为相等。
    浮点"表观奇点"的特例分支全靠它切换，容差取值必须与参考实现一致。"""
    d = np.abs(a - b)
    return (d <= max_diff) | (d <= np.maximum(np.abs(a), np.abs(b)) * max_rel)


def small_v(V):
    """eq. 3-50：v = arccos(V)，|V|>1 时按 -j·log(V+√(V²-1)) 解析延拓到复角。"""
    V = np.asarray(V, dtype=np.float64)
    out = np.empty(V.shape, dtype=np.complex128)
    m = np.abs(V) <= 1.0
    out[m] = np.arccos(np.clip(V[m], -1.0, 1.0))
    m = V > 1.0
    out[m] = 1j * np.arccosh(V[m])
    m = V < -1.0
    out[m] = np.pi - 1j * np.arccosh(-V[m])
    return out


def U(phi):
    """eq. 3-67 阶跃函数：1 当 0 < φ ≤ π"""
    return ((phi > 0) & (phi <= np.pi)).astype(np.float64)


def U_plus(phi_i, alpha):
    """eq. 2-102：'plus' 面被照明的开关"""
    return ((phi_i >= alpha) & (phi_i < np.pi + alpha)).astype(np.float64)


def U_minus(phi_i, alpha):
    """eq. 2-103：'minus' 面被照明的开关"""
    return ((phi_i > np.pi - alpha) & (phi_i <= 2 * np.pi + alpha)).astype(np.float64)


def wedge_factor(alpha):
    """劈因子 ν = π/[2(π-α)]；外劈角 = π/ν = 2(π-α)"""
    return np.pi / (2.0 * (np.pi - alpha))


def _f_base(V, psi, nu):
    """eq. 3-65 一般形式"""
    v = small_v(V)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = nu * np.sin(nu * v) / (np.cos(nu * v) - np.cos(nu * psi))
        t2 = np.where(U(np.pi - psi) > 0, np.sin(v) / (np.cos(v) - np.cos(psi)), 0.0)
    return (CIS45 / S2PI) / np.sin(v) * (t1 - t2)


def _g_base(V, psi, nu):
    """eq. 3-66 一般形式"""
    v = small_v(V)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = nu * np.sin(nu * psi) / (np.cos(nu * v) - np.cos(nu * psi))
        t2 = np.where(U(np.pi - psi) > 0, np.sin(psi) / (np.cos(v) - np.cos(psi)), 0.0)
    return -1.0 * (CIS45 / S2PI) * (t1 - t2)


def f_all(V, psi, nu):
    """f 的**可去奇点**处理版。分支优先级（与 cl-rcs `ildc-f-all` 的 cond 一致）：
    3-72 > 3-71 > V≈1(3-70 邻域) > 3-65 一般式。

    ★ 已知未正则化的奇点（与参考实现同源，勿误判为移植 bug）：
    V → -1（等价于 ψ → π，即入射或散射方向掠过某一劈面，属几何光学边界）时
    f 有真极点（ψ→π 处退化为 ~1/sin²ψ，单条棱实测可达 1e10）。cl-rcs 在
    `ildc-f-all` 开头就把该分支注释掉了并留 FIXME：
        ;; ((float-equal V-big -1) ;; f goes to infinity as V -> -1
        ;;  FIXME what to do??
    3-71 特例式本身在 ψ→π 也发散，因此"特例分支"并不能救这一支。
    在真网格上做全角度方向图时该极点会被反复命中，必须外部做一致化
    （UTD 过渡函数 / 极点距离截断）后才可用。
    """
    V = np.asarray(V, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    nu = np.maximum(np.asarray(nu, dtype=np.float64), 0.5)
    out = _f_base(V, psi, nu)
    # 3-70 邻域：V≈1
    m = float_equal(V, 1.0, max_diff=1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = (CIS45 / (2 * S2PI)) * (
            nu ** 2 / np.sin(0.5 * nu * psi) ** 2
            - np.where((np.pi - psi > 0) & (np.pi - psi < np.pi),
                       1.0 / np.sin(0.5 * psi) ** 2, 0.0))
    out = np.where(m, val, out)
    # 3-71：V≈cos ψ 且 0 ≤ ψ ≤ π
    m = float_equal(V, np.cos(psi), max_diff=1e-3) & (psi >= 0) & (psi <= np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        val = -1.0 * (CIS45 / (2 * S2PI)) * (
            (nu * cot(nu * psi) - cot(psi)) / np.sin(psi))
    out = np.where(m, val, out)
    # 3-72：V≈1 且 ψ≈0
    m = float_equal(V, 1.0) & float_equal(psi, 0.0)
    out = np.where(m, -1.0 * (CIS45 / (6 * S2PI)) * (1 - nu ** 2), out)
    return out


def g_all(V, psi, nu):
    """g 的可去奇点处理版。分支优先级：3-74 > 3-73 > 3-66 一般式。"""
    V = np.asarray(V, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    nu = np.maximum(np.asarray(nu, dtype=np.float64), 0.5)
    out = _g_base(V, psi, nu)
    # 3-73：V≈cos ψ 且 0 ≤ ψ ≤ π（**无 1/ν 因子**，见模块 docstring 勘误）
    m = float_equal(V, np.cos(psi), max_diff=1e-2) & (psi >= 0) & (psi <= np.pi)
    out = np.where(m, -1.0 * (CIS45 / (2 * S2PI)) * (nu * cot(nu * psi) - cot(psi)), out)
    # 3-74：V≈1 且 ψ≈0 → 0
    m = float_equal(V, 1.0, max_rel=1e-6, max_diff=1e-2) & \
        float_equal(psi, 0.0, max_rel=1e-6, max_diff=1e-2)
    out = np.where(m, 0.0, out)
    return out


# ============================================================
# 二、并矢绕射系数 D_U（eq. 3-46A + 3-55 ~ 3-61）
# ============================================================

def d_U(beta_s, beta_i, phi_s, phi_i, alpha, nu, a_perp, a_par):
    """返回 (c_perp, c_par)：满足 D_U·ê_i = c_perp·ê_⊥^s + c_par·ê_∥^s。

    直接返回"作用于入射极化后的两个标量分量"，避免构造 3×3 并矢，省 9 倍内存。
        D_U = D_⊥ (ê_⊥^s ⊗ ê_⊥^i) + D_∥ (ê_∥^s ⊗ ê_∥^i) + D_x (ê_∥^s ⊗ ê_⊥^i)
        a_⊥ = ê_⊥^i·ê_i, a_∥ = ê_∥^i·ê_i  （入射极化在 ⊥/∥ 基上的分解）
    """
    cbs = np.cos(beta_s) / np.cos(beta_i)                 # eq. 3-34
    V_p = -cbs * np.cos(phi_s - alpha)                    # eq. 3-47
    V_m = -cbs * np.cos(phi_s + alpha)                    # eq. 3-48
    psi_p = phi_i - alpha                                 # eq. 3-47
    psi_m = 2 * np.pi + alpha - phi_i                     # eq. 3-48
    f_p, f_m = f_all(V_p, psi_p, nu), f_all(V_m, psi_m, nu)
    g_p, g_m = g_all(V_p, psi_p, nu), g_all(V_m, psi_m, nu)

    D_perp = f_p * np.sin(phi_s - alpha) - f_m * np.sin(phi_s + alpha)   # 3-56/57/46A
    D_par = (g_p + g_m) * cbs                                            # 3-58/59/46A
    D_x_star = -1.0 * (f_p * np.cos(phi_s - alpha)
                       - f_m * np.cos(phi_s + alpha)) * cbs              # 3-60
    D_x_ss = -1.0 * (CIS45 / S2PI) * (U_plus(phi_i, alpha) - U_minus(phi_i, alpha)) \
        * np.cos(beta_s) * np.tan(beta_i)                                # 3-61
    # 3-46A：末项用 sin((βs-βi)/2)，非 (+)
    D_x = (2.0 * D_x_star
           * (1 + np.sin(beta_s) * np.sin(beta_i))
           / (np.cos(beta_s) * np.cos(beta_i))
           * np.cos(0.5 * (beta_s + beta_i))
           * np.sin(0.5 * (beta_s - beta_i)) + D_x_ss)

    return D_perp * a_perp, D_par * a_par + D_x * a_perp


# ============================================================
# 三、几何小工具
# ============================================================

def angle_ccw(v1, v2, n):
    """绕 n（单位）从 v1 到 v2 的逆时针方位角，∈ (-π, π]"""
    det = (n * np.cross(v1, v2)).sum(-1)
    return np.arctan2(det, (v1 * v2).sum(-1))


def wrap_angle(a):
    return np.mod(a, 2 * np.pi)


def e_x_dir(l_n, n1, n2):
    """eq. 2-60：方位角参考。e-x = -normalize(n̂₁+n̂₂)，退化时取 -normalize(l̂×n̂₁)"""
    s = n1 + n2
    ln = np.linalg.norm(s, axis=-1)
    alt = np.cross(l_n, n1)
    alt = alt / np.maximum(np.linalg.norm(alt, axis=-1, keepdims=True), 1e-300)
    good = (ln > 1e-6)[:, None]
    return -1.0 * np.where(good, s / np.maximum(ln, 1e-300)[:, None], alt)


def beta_i_of(e_i, l_n):
    """test_beta_i 逐例核对：β_i = -asin(e_i·l̂)"""
    return -np.arcsin(np.clip((e_i * l_n).sum(-1), -1.0, 1.0))


def beta_s_of(e_s, l_n):
    """test_beta_s 逐例核对：β_s = +asin(e_s·l̂)"""
    return np.arcsin(np.clip((e_s * l_n).sum(-1), -1.0, 1.0))


def e_perp_par_i(k_hat, l_n):
    """eq. 2-5/2-6 入射极化基（k_hat 为传播方向）。perpendicular 带负号，见 cl-rcs"""
    v = np.cross(l_n, k_hat)
    perp = -1.0 * v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
    par = np.cross(perp, k_hat)
    return perp, par


def e_perp_par_s(k_hat, l_n):
    """eq. 2-14/2-15 散射极化基"""
    v = np.cross(l_n, k_hat)
    perp = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)
    par = np.cross(k_hat, perp)
    return perp, par


def e_tau_vec(n, e_s, e_i):
    """eq. 4-4：把 (e_s+e_i) 投影到面内并归一，返回 (e, tau)。
    τ = |n̂×(e_s+e_i)|/2 ∈ [0,1]，τ=0 表示镜面方向（掠射/正交退化）。

    e 是**单位**矢量 = proj_n(e_s+e_i)/|proj_n(e_s+e_i)|（cl-rcs test `e-tau` 断言
    `vec-normalized-p`）。注意 v = v_n×n 恒等于 proj_n(e_s+e_i) 且 |v| = 2τ，
    因此除法必须是 /(2τ) 而不是乘 2τ —— 后者会让 Y_n 与 acc 都偏 2τ 倍，
    表现为"τ=0 的算例全过、τ≠0 的算例全错"（零点被填平）。
    """
    v_n = np.cross(n, e_s + e_i)
    mag = np.linalg.norm(v_n, axis=-1)
    tau = np.clip(0.5 * mag, 0.0, 1.0)
    v = np.cross(v_n, n)
    vm = np.linalg.norm(v, axis=-1, keepdims=True)
    safe = vm[:, 0] > 1e-300
    e = np.zeros_like(v)
    e[safe] = v[safe] / (2.0 * tau[safe, None])
    return e, tau


# ============================================================
# 四、半边/唯一棱边提取
# ============================================================

def build_edges(tri, nvm, vert_tol=1e-6):
    """从三角网格提取**唯一棱边**表（每条物理棱边一个代表半边 + 对偶面）。

    返回 dict：
      r      (m,3) 代表半边的起点坐标（Mitzner 相位参考点）
      Cvec   (m,3) 有向棱矢量（起点→终点）
      Cn     (m,)  棱长
      ln     (m,3) 单位切向 l̂
      f1,f2  (m,)  代表半边所属面、对偶面索引
      n1,n2  (m,3) 两面外法向
      alpha  (m,)  劈半角 α = (π - angle(n1,n2))/2
      nu     (m,)  劈因子 ν
      open   (m,)  布尔：无对偶（开放边，不参与绕射）
    """
    V = tri.reshape(-1, 3)
    key = np.round(V / vert_tol).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    F = inv.reshape(-1, 3)
    Q = uniq.astype(np.float64) * vert_tol
    nt = len(F)

    he_v0 = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    he_v1 = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    he_f = np.repeat(np.arange(nt), 3)

    order = {}
    rep, twin_of = [], {}
    for e in range(len(he_v0)):
        a, b = int(he_v0[e]), int(he_v1[e])
        if (b, a) in order:                       # 反向半边已登记 → e 是对偶
            twin_of[order[(b, a)]] = e
            continue
        order[(a, b)] = e
        rep.append(e)
    rep = np.array(rep, dtype=np.int64)

    f1 = he_f[rep]
    f2 = np.full(len(rep), -1, dtype=np.int64)
    ok = np.array([i in twin_of for i in rep])
    f2[ok] = he_f[[twin_of[int(rep[i])] for i in np.nonzero(ok)[0]]]

    r = Q[he_v0[rep]]
    Cvec = Q[he_v1[rep]] - Q[he_v0[rep]]
    Cn = np.linalg.norm(Cvec, axis=1)
    ln = Cvec / np.maximum(Cn, 1e-300)[:, None]

    n1 = nvm[f1]
    n2 = nvm[np.maximum(f2, 0)]
    s = n1 + n2
    ang = 2.0 * np.arctan2(np.linalg.norm(n1 - n2, axis=1),
                           np.linalg.norm(s, axis=1))
    alpha = 0.5 * (np.pi - ang)
    nu = wedge_factor(alpha)
    return dict(r=r, Cvec=Cvec, Cn=Cn, ln=ln, f1=f1, f2=f2,
                n1=n1, n2=n2, alpha=alpha, nu=nu, open=~ok)


# ============================================================
# 五、条纹波（fringe wave）远场：沿棱边的 ILDC 线积分
# ============================================================

def prepare_edges(E, e_i, ei_hat):
    """方向无关量的预计算（每个入射角做一次）。
    E   : build_edges 的输出
    e_i : (3,) 入射方向的反向（由目标指向源），单站时 = -k̂
    ei_hat : (3,) 单位入射电场方向（用于极化分解）
    """
    ln, n1, n2, alpha, nu = E["ln"], E["n1"], E["n2"], E["alpha"], E["nu"]
    lit1 = (n1 @ e_i) > 0                              # 面 1 受照？
    n = np.where(lit1[:, None], n1, n2)                # n = 受照面外法向
    ex = e_x_dir(ln, n1, n2)                           # eq. 2-60
    perp_i, par_i = e_perp_par_i(e_i, ln)              # eq. 2-5/2-6
    phi_i = wrap_angle(angle_ccw(ex, np.broadcast_to(e_i, ln.shape), ln))
    b_i = beta_i_of(e_i, ln)
    keep = (~E["open"]
            & ((n @ e_i) >= 0)                                     # 至少一面受照
            & (phi_i > alpha + 1e-6) & (phi_i < 2 * np.pi + alpha - 1e-6)
            & (np.abs(b_i) < np.pi / 2 - EPS_BETA))
    return dict(n=n, ex=ex, perp_i=perp_i, par_i=par_i,
                phi_i=phi_i, beta_i=b_i,
                a_perp=perp_i @ ei_hat, a_par=par_i @ ei_hat, keep=keep)


def fringe_far_field(E, P, k, e_i, e_s_list, ei_hat, keller_cone=False,
                     subset=None, d_max=None, verbose=False):
    """沿棱边积分求条纹波远场。

    E       : build_edges 输出
    P       : prepare_edges 输出
    k       : 波数
    e_i     : (3,) 入射方向反向（单位）
    e_s_list: (ndir,3) 观察方向（单位，= r̂）
    ei_hat  : (3,) 单位入射电场方向
    keller_cone : True 时只保留 β_s≈β_i 的棱（PTD 模式；单站下恒成立）
    subset  : 可选 bool 掩码，进一步限制参与的棱（例如只留锐棱）
    d_max   : 可选。**诊断开关，非常规路径**。把 (D_⊥, D_∥, D_x) 作用后的两个
              标量分量 (c_perp, c_par) 的合成模长按比例压到 ≤ d_max（保持相位）。
              用途：Mitzner 非一致系数在几何光学边界（V→-1 / ψ→π）有未正则化的
              极点，会把个别方向抬到 1e6~1e10。本开关给出"极点被压住之后 ILDC
              还剩多少正确信息"的上界估计；不是物理模型，不可当结果使用。
              None = 保持参考实现原样（默认）。

    返回 (ndir,3) 复矢量：E_fringe(r̂)，满足 E_s = E_fringe·e^{ikr}/r
    （即已把 Mitzner 的 D 除以 k 归一化）。
    """
    ln, Cn, Cvec, r = E["ln"], E["Cn"], E["Cvec"], E["r"]
    alpha, nu, n = E["alpha"], E["nu"], P["n"]
    ex, perp_i, par_i = P["ex"], P["perp_i"], P["par_i"]
    phi_i, b_i = P["phi_i"], P["beta_i"]
    a_perp, a_par = P["a_perp"], P["a_par"]

    act = P["keep"].copy()
    if subset is not None:
        act &= subset
    idx = np.nonzero(act)[0]
    if verbose:
        print(f"    参与绕射的棱边: {len(idx)} / {len(ln)}")

    ln_, Cn_, Cv_, r_ = ln[idx], Cn[idx], Cvec[idx], r[idx]
    al_, nu_, n_ = alpha[idx], nu[idx], n[idx]
    ex_, pi_, pa_ = ex[idx], perp_i[idx], par_i[idx]
    phii_, bi_ = phi_i[idx], b_i[idx]
    ap_, al_par = a_perp[idx], a_par[idx]
    r_cn = 0.5 * Cv_

    es_list = np.asarray(e_s_list, dtype=np.float64)
    out = np.zeros((len(es_list), 3), dtype=np.complex128)
    pref = CIS45.conjugate() / S2PI          # e^{-iπ/4}/√(2π)

    for d in range(len(es_list)):
        e_s = es_list[d]
        b_s = beta_s_of(e_s, ln_)
        if keller_cone:
            okc = float_equal(b_s, bi_, 1e-2)
        else:
            okc = np.ones(len(idx), dtype=bool)
        ok = okc & (np.abs(b_s) < np.pi / 2 - EPS_BETA)
        if not ok.any():
            continue
        s_ln, s_n, s_al, s_nu = ln_[ok], n_[ok], al_[ok], nu_[ok]
        s_bi, s_phii, s_bs = bi_[ok], phii_[ok], b_s[ok]
        s_Cn = Cn_[ok]
        s_r, s_rcn, s_ap, s_par = r_[ok], r_cn[ok], ap_[ok], al_par[ok]

        # 散射侧：方位角、极化基
        phi_s = wrap_angle(angle_ccw(ex_[ok], np.broadcast_to(e_s, s_ln.shape), s_ln))
        perp_s, par_s = e_perp_par_s(e_s, s_ln)

        c_perp, c_par = d_U(s_bs, s_bi, phi_s, s_phii, s_al, s_nu,
                            s_ap, s_par)
        if d_max is not None:
            cm = np.sqrt(np.abs(c_perp) ** 2 + np.abs(c_par) ** 2)
            sc = np.minimum(1.0, d_max / np.maximum(cm, 1e-300))
            c_perp, c_par = c_perp * sc, c_par * sc

        e_v, tau = e_tau_vec(s_n, e_s, e_i)
        Y_n = np.where(tau == 0, 0.0, tau * k * s_Cn * (e_v * s_ln).sum(-1))
        ph = k * ((s_r @ e_i) + (s_r @ e_s))
        acc = np.where(tau == 0, 1.0, np.exp(-2j * tau * k * (e_v * s_rcn).sum(-1)))
        coef = acc * s_Cn * np.exp(1j * ph) * synth_sinc(Y_n)
        out[d] = pref * np.sum(coef[:, None] * (c_perp[:, None] * perp_s
                                                + c_par[:, None] * par_s), axis=0)
    return out


# ============================================================
# 六、解析校验（直角楔 / 刀口 / 半平面）——动作 2 的"前置闸门"
# ============================================================

def selftest(verbose=True):
    """全部判据取自 cl-rcs 的单元测试（其本身引用 Mitzner 报告的公式号）。
    任何一条不过，都不应把 ILDC 结果当作可信结论。"""
    res = {}

    def chk(name, ok, info=""):
        res[name] = bool(ok)
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {info}")

    # --- 1. beta_i / beta_s 四例（test_beta_i / test_beta_s）---
    y = np.array([0.0, 1.0, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    d1 = np.array([0.0, 1 / np.sqrt(2), 1 / np.sqrt(2)])
    d2 = np.array([0.0, -1 / np.sqrt(2), 1 / np.sqrt(2)])
    L = np.stack([y, y, y, y])
    bi = beta_i_of(np.stack([z, y, d1, d2]), L)
    bs = beta_s_of(np.stack([z, y, d2, d1]), L)
    chk("beta_i(4例)", np.allclose(bi, [0.0, -np.pi / 2, -np.pi / 4, np.pi / 4], atol=1e-9),
        f"{np.round(bi, 4)}")
    chk("beta_s(4例)", np.allclose(bs, [0.0, np.pi / 2, -np.pi / 4, np.pi / 4], atol=1e-9),
        f"{np.round(bs, 4)}")

    # --- 2. e-x（test_e-x / test_e-x-knife-edge）---
    lx = np.array([[1.0, 0, 0]] * 2)
    n1_ = np.array([[0, 1.0, 0], [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]])
    n2_ = np.array([[0, -1.0, 0], [0, -1 / np.sqrt(2), 1 / np.sqrt(2)]])
    exv = e_x_dir(lx, n1_, n2_)
    chk("e-x(2例)", np.allclose(exv, [[0, 0, -1.0], [0, 0, -1.0]], atol=1e-9), f"{np.round(exv,4)}")

    # --- 3. phi-s 四例（test_phi-s）：棱沿 x，e-x = -ẑ ---
    edge = np.array([1.0, 0, 0])
    exd = np.array([0, 0, -1.0])
    vecs = [z, -z, np.array([0, -1.0, 0]), y]
    phs = [wrap_angle(angle_ccw(exd, v, edge)) for v in vecs]
    chk("phi-s(4例)", np.allclose(phs, [np.pi, 0.0, 3 * np.pi / 2, np.pi / 2], atol=1e-9),
        f"{np.round(phs, 4)}")

    # --- 4. 3-68 / 3-69：U(π-ψ±) ≡ U± ---
    #     注：3-69 仅在 φ_i,α ∈ [0,π/2] 上成立（与参考实现测试的生成域一致），
    #     超出该域时 φ_i 落入双面同时受照区，两者都按约定退化，不构成判据。
    rng = np.random.default_rng(0)
    phi_i = rng.uniform(0, np.pi / 2, 400)
    al = rng.uniform(0, np.pi / 2, 400)
    l1 = np.abs(U(np.pi - (phi_i - al)) - U_plus(phi_i, al)).max()
    l2 = np.abs(U(np.pi - (2 * np.pi + al - phi_i)) - U_minus(phi_i, al)).max()
    chk("3-68 U(π-ψ+)≡U+", l1 < 1e-12, f"max|Δ|={l1:.1e}")
    chk("3-69 U(π-ψ-)≡U-", l2 < 1e-12, f"max|Δ|={l2:.1e}")

    # --- 5. 刀口(α=0)背散射闭式：3-121 / 2-161 / 2-163 ---
    beta = rng.uniform(0.05, 1.5, 300)
    phi = rng.uniform(0.1, np.pi - 0.05, 300)
    D_perp = _d_scalar(-beta, beta, phi, phi, 0.0, "perp")
    D_par = _d_scalar(beta, -beta, phi, phi, 0.0, "par")
    D_x = _d_scalar(-beta, beta, phi, phi, 0.0, "x")
    ref_perp = -CIS45 / (2 * S2PI) * np.where(phi < np.pi,
                                            1 - np.tan(phi / 2 - np.pi / 4),
                                            1 + np.tan(phi / 2 + np.pi / 4))
    ref_par = -CIS45 / (2 * S2PI) * np.where(phi < np.pi,
                                           1 + np.tan(phi / 2 - np.pi / 4),
                                           1 - np.tan(phi / 2 + np.pi / 4))
    ref_x = -CIS45 / S2PI * cot(phi / 2) * np.sin(beta)
    e_perp = np.abs(D_perp - ref_perp) / np.maximum(np.abs(ref_perp), 1e-30)
    e_par = np.abs(D_par - ref_par) / np.maximum(np.abs(ref_par), 1e-30)
    e_x = np.abs(D_x - ref_x) / np.maximum(np.abs(ref_x), 1e-30)
    chk("2-161 刀口背散射 D⊥", np.median(e_perp) < 1e-3, f"中位相对误差 {np.median(e_perp):.2e}")
    chk("2-163 刀口背散射 D∥", np.median(e_par) < 1e-3, f"中位相对误差 {np.median(e_par):.2e}")
    chk("3-121 刀口背散射 Dx", np.median(e_x) < 1e-3, f"中位相对误差 {np.median(e_x):.2e}")

    # --- 6. 刀口(α=0)的 +/− 对称（3-102 / 3-104）---
    phi_s = rng.uniform(0.1, np.pi - 0.1, 300)
    Dpp = _d_scalar(beta, -beta, phi_s, phi_s, 0.0, "perp_plus")
    Dpm = _d_scalar(beta, -beta, phi_s, phi_s, 0.0, "perp_minus")
    e_kn = np.median(np.abs(Dpp - Dpm) / np.maximum(np.abs(Dpp), 1e-30))
    chk("3-102 α=0: D⊥+ ≡ D⊥-", e_kn < 1e-6, f"中位相对差 {e_kn:.1e}")

    # --- 6b. 2-36 任意劈角下的 (βs,βi)→(-βs,-βi) 奇偶性 ---
    bs = rng.uniform(-1.5, 1.5, 2000)
    bi = rng.uniform(-1.5, 1.5, 2000)
    ps = rng.uniform(0.1, np.pi - 0.1, 2000)
    pi_ = rng.uniform(0.1, np.pi - 0.1, 2000)
    a_ = rng.uniform(0.05, np.pi - 0.05, 2000)
    for nm, w, sgn in (("⊥⊥", "perp", 1), ("∥∥", "par", 1), ("∥⊥", "x", -1)):
        A = _d_scalar(bs, bi, ps, pi_, a_, w)
        B = _d_scalar(-bs, -bi, ps, pi_, a_, w)
        err = np.nanmedian(np.abs(A - sgn * B) / np.maximum(np.abs(A), 1e-30))
        chk(f"2-36 {nm} 奇偶性", err < 1e-6, f"中位相对差 {err:.1e}")

    # --- 7. 有限性扫描 ---
    nb = 20000
    bb_s = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, nb)
    bb_i = rng.uniform(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, nb)
    pp_s = rng.uniform(0, 2 * np.pi, nb)
    pp_i = rng.uniform(0, 2 * np.pi, nb)
    aa = rng.uniform(0.01, np.pi - 0.01, nb)
    Dv = _d_scalar(bb_s, bb_i, pp_s, pp_i, aa, "perp")
    Dv2 = _d_scalar(bb_s, bb_i, pp_s, pp_i, aa, "x")
    big = np.isfinite(Dv).mean() > 0.999 and np.isfinite(Dv2).mean() > 0.999
    chk("D 全域有限（无不可去奇点）", big,
        f"有限占比 ⊥={np.isfinite(Dv).mean():.4f} x={np.isfinite(Dv2).mean():.4f}")

    return res


def _d_scalar(beta_s, beta_i, phi_s, phi_i, alpha, which):
    """校验用：直接返回 D_U 的某一标量分量（取 ê⊥^s/ê∥^s 为常数基的系数）。"""
    beta_s = np.asarray(beta_s, dtype=np.float64)
    beta_i = np.asarray(beta_i, dtype=np.float64)
    phi_s = np.asarray(phi_s, dtype=np.float64)
    phi_i = np.asarray(phi_i, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64)
    nu = np.maximum(wedge_factor(alpha), 0.5)
    cbs = np.cos(beta_s) / np.cos(beta_i)
    V_p = -cbs * np.cos(phi_s - alpha)
    V_m = -cbs * np.cos(phi_s + alpha)
    f_p, f_m = f_all(V_p, phi_i - alpha, nu), f_all(V_m, 2 * np.pi + alpha - phi_i, nu)
    g_p, g_m = g_all(V_p, phi_i - alpha, nu), g_all(V_m, 2 * np.pi + alpha - phi_i, nu)
    if which == "perp":
        return f_p * np.sin(phi_s - alpha) - f_m * np.sin(phi_s + alpha)
    if which == "perp_plus":
        return f_p * np.sin(phi_s - alpha)
    if which == "perp_minus":
        return -f_m * np.sin(phi_s + alpha)
    if which == "par":
        return (g_p + g_m) * cbs
    if which == "x":
        D_x_star = -1.0 * (f_p * np.cos(phi_s - alpha)
                           - f_m * np.cos(phi_s + alpha)) * cbs
        D_x_ss = -1.0 * (CIS45 / S2PI) * (U_plus(phi_i, alpha) - U_minus(phi_i, alpha)) \
            * np.cos(beta_s) * np.tan(beta_i)
        return (2.0 * D_x_star * (1 + np.sin(beta_s) * np.sin(beta_i))
                / (np.cos(beta_s) * np.cos(beta_i))
                * np.cos(0.5 * (beta_s + beta_i))
                * np.sin(0.5 * (beta_s - beta_i)) + D_x_ss)
    raise ValueError(which)


if __name__ == "__main__":
    print("=== ILDC 解析校验（直角楔 / 刀口）===")
    r = selftest()
    print(f"\n通过 {sum(r.values())}/{len(r)}")
