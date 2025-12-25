import numpy as np
from numpy.lib.scimath import sqrt as csqrt  
import matplotlib.pyplot as plt
import itertools
import pandas as pd

I = 1j

def cexp_safe(x):

    xr = np.real(x)
    xi = np.imag(x)
    xr_clip = np.clip(xr, -700, 700)  
    return np.exp(xr_clip + 1j * xi)

def as_delta_beta(mat):
    """mat = (delta, beta)"""
    return float(mat[0]), float(mat[1])

def layers_extract_delta_beta(layers):
    """
    layers: [( (δ,β), thickness_nm, is_res ), ...]
    """
    d = np.array([as_delta_beta(L[0])[0] for L in layers], dtype=float)
    b = np.array([as_delta_beta(L[0])[1] for L in layers], dtype=float)
    t = np.array([float(L[1]) for L in layers], dtype=float)
    rflag = np.array([int(L[2]) for L in layers], dtype=int)
    return d, b, t, rflag

# ============================================================
# 1) Parratt 递推：单界面 r,t；多层 r
# ============================================================

def ParrattRSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower):
    
    n_upper = 1.0 - delta_upper + 1.0 * I * beta_upper
    n_lower = 1.0 - delta_lower + 1.0 * I * beta_lower
    kz_upper = csqrt((n_upper * k)**2 - (k**2 - kz**2))
    kz_lower = csqrt((n_lower * k)**2 - (k**2 - kz**2))
    Q_upper = 2.0 * kz_upper
    Q_lower = 2.0 * kz_lower
    return (Q_upper - Q_lower) / (Q_upper + Q_lower)

def ParrattTSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower):
    n_upper = 1.0 - delta_upper + 1.0 * I * beta_upper
    n_lower = 1.0 - delta_lower + 1.0 * I * beta_lower
    kz_upper = csqrt((n_upper * k)**2 - (k**2 - kz**2))
    kz_lower = csqrt((n_lower * k)**2 - (k**2 - kz**2))
    Q_upper = 2.0 * kz_upper
    Q_lower = 2.0 * kz_lower
    return (2.0 * Q_upper) / (Q_upper + Q_lower)

def ParrattRMulti(r_single_upper, r_multi_lower, d_lower, k, kz, delta_lower, beta_lower):
    n_lower = 1.0 - delta_lower + 1.0 * I * beta_lower
    kz_lower = csqrt((n_lower * k)**2 - (k**2 - kz**2))
    phase = cexp_safe(1.0 * I * 2.0 * kz_lower * d_lower)
    return (r_single_upper + r_multi_lower * phase) / (1.0 + r_single_upper * r_multi_lower * phase)

def ParrattR(k, kz, layers):
    d, b, t, _ = layers_extract_delta_beta(layers)
    d_upper = np.concatenate(([0.0], d[:-1]))
    b_upper = np.concatenate(([0.0], b[:-1]))
    d_lower = d.copy()
    b_lower = b.copy()

    prss = [ParrattRSingle(k, kz, d_upper[i], b_upper[i], d_lower[i], b_lower[i])
            for i in range(len(d_lower))]
    r = prss[-1]
    for i in range(len(layers) - 1, 0, -1):
        r = ParrattRMulti(prss[i-1], r, t[i-1], k, kz, d[i-1], b[i-1])
    return r

# ============================================================
# 2) 2×2 传输矩阵链：LSingle1/2, LN, LD, Ln, LZ
# ============================================================

def LSingle1(k, kz, z, delta_upper, beta_upper, delta_lower, beta_lower):
    n_lower = 1.0 - delta_lower + 1.0 * I * beta_lower
    kz_lower = csqrt((n_lower * k)**2 - (k**2 - kz**2))

    tul = ParrattTSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    tlu = ParrattTSingle(k, kz, delta_lower, beta_lower, delta_upper, beta_upper)
    rul = ParrattRSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    rlu = ParrattRSingle(k, kz, delta_lower, beta_lower, delta_upper, beta_upper)

    
    B = np.array([[cexp_safe( I * kz_lower * z), 0.0],
                  [0.0, cexp_safe(-I * kz_lower * z)]], dtype=complex)
    C = np.array([[1.0, rlu],
                  [rlu, 1.0]], dtype=complex)
    return (1.0 / tul) * (B @ C) * (1.0 / tlu)

def LSingle2(k, kz, z, delta_upper, beta_upper, delta_lower, beta_lower):

    n_lower = 1.0 - delta_lower + 1.0 * I * beta_lower
    kz_lower = csqrt((n_lower * k)**2 - (k**2 - kz**2))

    tul = ParrattTSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    tlu = ParrattTSingle(k, kz, delta_lower, beta_lower, delta_upper, beta_upper)
    rul = ParrattRSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    rlu = ParrattRSingle(k, kz, delta_lower, beta_lower, delta_upper, beta_upper)

    A = np.array([[1.0, rul],
                  [rul, 1.0]], dtype=complex)
    B = np.array([[cexp_safe( I * kz_lower * z), 0.0],
                  [0.0, cexp_safe(-I * kz_lower * z)]], dtype=complex)
    C = np.array([[1.0, rlu],
                  [rlu, 1.0]], dtype=complex)

 
    return A

def LN(k, kz, delta_upper, beta_upper, delta_lower, beta_lower):
    rN0 = ParrattRSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    tN0 = ParrattTSingle(k, kz, delta_upper, beta_upper, delta_lower, beta_lower)
    A = np.array([[0.0, 0.0],
                  [0.0, 1.0]], dtype=complex)
    B = np.array([[1.0, rN0],
                  [rN0, 1.0]], dtype=complex)
    return (A @ B) * (1.0 / tN0)

def LD(k, kz, layers):

    d, b, t, _ = layers_extract_delta_beta(layers)


    d_mid = d[:-1]
    b_mid = b[:-1]
    d_up = np.concatenate(([0.0], d_mid))
    b_up = np.concatenate(([0.0], b_mid))
    d_low = np.concatenate((d_mid, [0.0]))
    b_low = np.concatenate((b_mid, [0.0]))

    ld = np.eye(2, dtype=complex)
    # forward
    for i in range(0, len(layers)-1):
        ld = LSingle1(k, kz, t[i], d_up[i], b_up[i], d_low[i], b_low[i]) @ ld
    # backward
    for i in range(len(layers)-2, -1, -1):
        ld = LSingle2(k, kz, t[i], d_up[i], b_up[i], d_low[i], b_low[i]) @ ld

    return LN(k, kz, 0.0, 0.0, 0.0, 0.0) @ ld

def Ln(layers, z):
    
    t = np.array([float(L[1]) for L in layers], dtype=float)
    z1 = float(z)
    n = 0
    for i in range(0, len(layers)-1):
        if z1 - t[i] < 0:
            break
        n = i + 1
        z1 = z1 - t[i]
    return n, z1

def LZ(k, kz, layers, z):
    
    d, b, t, _ = layers_extract_delta_beta(layers)

    d_mid = d[:-1]
    b_mid = b[:-1]
    d_up = np.concatenate(([0.0], d_mid))
    b_up = np.concatenate(([0.0], b_mid))
    d_low = np.concatenate((d_mid, [0.0]))
    b_low = np.concatenate((b_mid, [0.0]))

    n, z1 = Ln(layers, z)

    ld = np.eye(2, dtype=complex)
    
    for i in range(0, n):
        ld = LSingle1(k, kz, t[i], d_up[i], b_up[i], d_low[i], b_low[i]) @ ld
   
    ld = LSingle1(k, kz, z1, d_up[n], b_up[n], d_low[n], b_low[n]) @ ld
    ld = LSingle2(k, kz, z1, d_up[n], b_up[n], d_low[n], b_low[n]) @ ld
    
    for i in range(n-1, -1, -1):
        ld = LSingle2(k, kz, t[i], d_up[i], b_up[i], d_low[i], b_low[i]) @ ld

    return ld

# ============================================================
# 3) 含核层的反射率：R / REIT / REIT1
# ============================================================

def R_total(k, kz, layers, z, d_thick, Delta):
    
    rho = 75.0
    fnuc = 0.0012
    f0 = 2.0 * np.pi * fnuc * rho / kz
    fn = f0 / (2.0 * Delta - 1.0 * I)

    ld = LD(k, kz, layers)
    R0 = -ld[1, 0] / ld[1, 1]
    lz = LZ(k, kz, layers, z)

    E0 = np.array([
        [(lz[0,0] + lz[1,0]) * (lz[1,1] + lz[0,1]) * fn,
         (lz[1,1] + lz[0,1])**2 * fn],
        [-(lz[0,0] + lz[1,0])**2 * fn,
         -(lz[0,0] + lz[1,0]) * (lz[1,1] + lz[0,1]) * fn]
    ], dtype=complex)

    num = -(ld[1,0] * (1.0 + 1.0*I*d_thick*E0[0,0]) + 1.0*I*d_thick*ld[1,1]*E0[1,0])
    den =  (ld[1,1] * (1.0 + 1.0*I*d_thick*E0[1,1]) + 1.0*I*d_thick*ld[1,0]*E0[0,1])
    return num / den

def REIT(k, kz, layers, z1, z2, d_thick, Delta):
   
    rho = 75.0
    fnuc = 0.0012
    f0 = 2.0 * np.pi * fnuc * rho / kz
    fn = f0 / (2.0 * Delta - 1.0 * I)

    ld = LD(k, kz, layers)
    lz1 = LZ(k, kz, layers, z1)
    lz2 = LZ(k, kz, layers, z2)

    def E_from_lz(lz):
        return np.array([
            [(lz[0,0] + lz[1,0]) * (lz[0,1] + lz[1,1]),
             (lz[0,1] + lz[1,1])**2],
            [-(lz[0,0] + lz[1,0])**2,
             -(lz[0,0] + lz[1,0]) * (lz[0,1] + lz[1,1])]
        ], dtype=complex)

    E1 = E_from_lz(lz1)
    E2 = E_from_lz(lz2)
    ES = d_thick * E1 + d_thick * E2
    EP = (d_thick**2) * (E2 @ E1)

    num = -(ld[1,0] * (1.0 + I*fn*ES[0,0] - (fn**2)*EP[0,0]) +
            ld[1,1] * (I*fn*ES[1,0] - (fn**2)*EP[1,0]))
    den =  (ld[1,1] * (1.0 + I*fn*ES[1,1] - (fn**2)*EP[1,1]) +
            ld[1,0] * (I*fn*ES[0,1] - (fn**2)*EP[0,1]))
    return num / den

def REIT1(k, kz, layers, z1, z2, d1, d2, Delta):
   
    rho = 75.0
    fnuc = 0.0012
    f0 = 2.0 * np.pi * fnuc * rho / kz
    fn = f0 / (2.0 * Delta - 1.0 * I)

    ld = LD(k, kz, layers)
    lz1 = LZ(k, kz, layers, z1)
    lz2 = LZ(k, kz, layers, z2)

    def E_from_lz(lz):
        return np.array([
            [(lz[0,0] + lz[1,0]) * (lz[0,1] + lz[1,1]),
             (lz[0,1] + lz[1,1])**2],
            [-(lz[0,0] + lz[1,0])**2,
             -(lz[0,0] + lz[1,0]) * (lz[0,1] + lz[1,1])]
        ], dtype=complex)

    E1 = E_from_lz(lz1)
    E2 = E_from_lz(lz2)
    ES = d1 * E1 + d2 * E2
    EP = (d1 * d2) * (E2 @ E1)

    num = -(ld[1,0] * (1.0 + I*fn*ES[0,0] - (fn**2)*EP[0,0]) +
            ld[1,1] * (I*fn*ES[1,0] - (fn**2)*EP[1,0]))
    den =  (ld[1,1] * (1.0 + I*fn*ES[1,1] - (fn**2)*EP[1,1]) +
            ld[1,0] * (I*fn*ES[0,1] - (fn**2)*EP[0,1]))
    return num / den

# ============================================================
# 4) Green 矩阵与反射：Refle2 / RGreen / RGreenMagnetic
# ============================================================

def Refle2(r0, G, Delta, gamma0, gvec):
    
    lam, V = np.linalg.eig(G)
    
    for j in range(V.shape[1]):
        vj = V[:, j]
        V[:, j] = vj / np.sqrt(np.vdot(vj, vj))

    J = np.real(lam)
    Gamma = 2.0 * np.imag(lam)

    s = 0.0 + 0.0j
    for j in range(len(lam)):
        vj = V[:, j]
        num = (np.dot(gvec, vj)) * (np.dot(vj, gvec))
        den = (Delta + J[j] + 1j * (gamma0 + Gamma[j]) / 2.0)
        s += num / den
    return r0 + 1j * s

def nd(layers):
    
    NL = len(layers)
    l = []
    tl = []
    nl = 0

   
    for i in range(NL):
        if int(layers[i][2]) == 1:
            nl += 1
            dl = 0.0
            for j in range(i):
                dl += float(layers[j][1])
            dl = dl + 0.5 * float(layers[i][1])
            l.append(dl)

    
    for i in range(NL):
        if int(layers[i][2]) == 1:
            tl.append(float(layers[i][1]))

    return nl, np.array(l, float), np.array(tl, float)

def BeamParameters_with_Delta(Delta):
   
    k0 = (4.66 * Delta + 1.44125e13) * 0.005067731 / 1e9
    return k0

def RGreen(theta0_mrad, layers, C0, Delta):
    
    theta0 = float(theta0_mrad) 
    n0 = C0 / theta0
    k = BeamParameters_with_Delta(0.0)
    kz = k * np.sin(theta0 / 1000.0)   

    r0 = ParrattR(k, kz, layers)
    nl, lpos, tl = nd(layers)

    lz_list = [LZ(k, kz, layers, lpos[i]) for i in range(nl)]
    p = np.array([lz_list[i][0,0] + lz_list[i][1,0] for i in range(nl)], dtype=complex)
    q = np.array([lz_list[i][0,1] + lz_list[i][1,1] for i in range(nl)], dtype=complex)

    G1 = np.eye(nl, dtype=complex)
    for i in range(nl):
        for j in range(i, nl):
            G1[i, j] = 1.0 * I * (p[j] + r0 * q[j]) * q[i] * n0 * np.sqrt(tl[i] * tl[j])
    for i in range(nl):
        for j in range(0, i):
            G1[i, j] = G1[j, i]

    qleft = 1.0
    g1 = np.array([1.0 * I * (p[i] + r0 * q[i]) * qleft * np.sqrt(n0 * tl[i]) for i in range(nl)], dtype=complex)

    return Refle2(r0, G1, Delta, 1.0, g1)

def RGreenMagnetic(B0, theta0_mrad, layers, C0, Delta):
    
    theta0 = float(theta0_mrad)
    ms = B0 / 33.0 * 32.0
    n0 = C0 / theta0
    k = BeamParameters_with_Delta(0.0)
    kz = k * np.sin(theta0 / 1000.0)

    r0 = ParrattR(k, kz, layers)

    nl0, l0, tl0 = nd(layers)


    l = np.repeat(l0, 2)
    tl = np.repeat(tl0 * 0.5, 2)
    nl = 2 * nl0

   
    MS = np.zeros((nl, nl), dtype=complex)
    sgn = 1.0
    for i in range(nl):
        MS[i, i] = sgn * ms
        sgn *= -1.0

    lz_list = [LZ(k, kz, layers, l[i]) for i in range(nl)]
    p = np.array([lz_list[i][0,0] + lz_list[i][1,0] for i in range(nl)], dtype=complex)
    q = np.array([lz_list[i][0,1] + lz_list[i][1,1] for i in range(nl)], dtype=complex)

    G1 = np.eye(nl, dtype=complex)
    for i in range(nl):
        for j in range(i, nl):
            G1[i, j] = 1.0 * I * (p[j] + r0 * q[j]) * q[i] * n0 * np.sqrt(tl[i] * tl[j])
    for i in range(nl):
        for j in range(0, i):
            G1[i, j] = G1[j, i]

    G1 = G1 + 1.0 * MS  

    qleft = 1.0
    g1 = np.array([1.0 * I * (p[i] + r0 * q[i]) * qleft * np.sqrt(n0 * tl[i]) for i in range(nl)], dtype=complex)
    return Refle2(r0, G1, Delta, 1.0, g1)

# C0 = const parameters: theta0_mrad Layers:list = length: 8 (-1:inf)
def GreenFun(theta0_mrad, Layers, C0):
   
    theta0 = float(theta0_mrad)
    n0 = C0 / theta0

    k  = BeamParameters_with_Delta(0.0)
    kz = k * np.sin(theta0 / 1000.0)

    r0 = ParrattR(k, kz, Layers)

    nl, l, tl = nd(Layers)  

    lz_list = [LZ(k, kz, Layers, l[i]) for i in range(nl)]

    p = np.array([lz_list[i][0,0] + lz_list[i][1,0] for i in range(nl)], dtype=complex)
    q = np.array([lz_list[i][0,1] + lz_list[i][1,1] for i in range(nl)], dtype=complex)

    G1 = np.eye(nl, dtype=complex)

    
    for i in range(nl):
        for j in range(i, nl):
            G1[i, j] = 1.0*I * (p[j] + r0*q[j]) * q[i] * n0 * np.sqrt(tl[i]*tl[j])
  
    for i in range(nl):
        for j in range(0, i):
            G1[i, j] = G1[j, i]

    O1 = np.array([1.0*(p[i] + r0*q[i]) * np.sqrt(n0 * tl[i]) for i in range(nl)], dtype=complex)

    return G1, O1

def GreenFunMagnetic(B0, theta0_mrad, Layers, C0, use_MS=False):
   
    theta0 = float(theta0_mrad)
    ms = B0/33.0*32.0
    n0 = C0 / theta0

    k  = BeamParameters_with_Delta(0.0)
    kz = k * np.sin(theta0 / 1000.0)
    r0 = ParrattR(k, kz, Layers)

    nl0, l0, tl0 = nd(Layers)

    
    l = np.repeat(l0, 2)
    tl = np.repeat(tl0 * 0.5, 2)
    nl = 2 * nl0

   
    MS = np.zeros((nl, nl), dtype=complex)
    s = 1.0
    for i in range(nl):
        MS[i, i] = s * ms
        s = -s

    lz_list = [LZ(k, kz, Layers, l[i]) for i in range(nl)]
    p = np.array([lz_list[i][0,0] + lz_list[i][1,0] for i in range(nl)], dtype=complex)
    q = np.array([lz_list[i][0,1] + lz_list[i][1,1] for i in range(nl)], dtype=complex)

    G1 = np.eye(nl, dtype=complex)
    for i in range(nl):
        for j in range(i, nl):
            G1[i, j] = 1.0*I * (p[j] + r0*q[j]) * q[i] * n0 * np.sqrt(tl[i]*tl[j])
    for i in range(nl):
        for j in range(0, i):
            G1[i, j] = G1[j, i]

   
    if use_MS:
        G1 = G1 + 1.0*MS
    else:
        G1 = G1 + 0.0*MS

    return G1

# ============================================================
# 5) 时间/频率变换：NTimeToFre / NFreToTime
# ============================================================

def NTimeToFre(Tdata, Tstep):
   
    Tdata = np.asarray(Tdata, dtype=complex)
    nL = len(Tdata)
    Lifetime = 141.0
    TstepL = Tstep / Lifetime
    Fstep = 2.0 * np.pi / (TstepL * nL)

    norm = np.sqrt(2.0*np.pi) / (Fstep * np.sqrt(nL))

  
    shift = int(np.round(nL/2))
    Td = np.roll(Tdata, shift)
    Fd = np.fft.fft(Td) * norm
    Fd = np.roll(Fd, shift)

    FreRange = (np.arange(-len(Fd)/2, len(Fd)/2) * Fstep)
    return np.column_stack([FreRange, np.abs(Fd)**2])

def NFreToTime(Fdata, Fstep):
    
    Fdata = np.asarray(Fdata, dtype=complex)
    nL = len(Fdata)
    norm = Fstep * np.sqrt(nL) / np.sqrt(2.0*np.pi)

    Td = np.fft.ifft(Fdata) * norm
    Tstep = 2.0*np.pi / (Fstep * nL) * 141.0
    Time = Tstep * np.arange(0, len(Td))
    return np.column_stack([Time, np.abs(Td)**2])



def demo_like_nb():
   
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)

   
    lam = 0.086
    CFe = 7.74 * 1.06 * 0.5

   
    Layers = [
        (Platinum, 3.00, 0),
        (Carbon,   6.15, 0),
        (Iron,     0.63, 1),
        (Carbon,   4.25, 0),
        (Iron,     1.24, 1),
        (Carbon,  16.98, 0),
        (Iron,     1.03, 1),
        (Carbon,  13.71, 0),
        (Platinum,10.00, 0),
    ]

    
    k = BeamParameters_with_Delta(0.0)
    thetas = np.arange(0.0, 6.0 + 1e-12, 0.01)  
    refl = []
    for th in thetas:
        kz = k * np.sin(th/1000.0)
        r = ParrattR(k, kz, Layers)
        refl.append(np.abs(r)**2)
    refl = np.array(refl)

    plt.figure()
    plt.plot(thetas, refl)
    plt.xlabel("Incidence angle θ (mrad)")
    plt.ylabel("Reflectivity |r|^2")
    plt.title("Bare cavity Parratt reflectivity")
    plt.grid(True)
    plt.show()
 #   df1 = pd.DataFrame({
#    "theta (mrad)": thetas,
#    "Reflectivity |r|^2": refl
#})

  
    thetas2 = np.arange(3.2, 3.6 + 1e-12, 0.001)
    refl2 = []
    for th in thetas2:
        kz = k * np.sin(th/1000.0)
        r = ParrattR(k, kz, Layers)
        refl2.append(np.abs(r)**2)
    refl2 = np.array(refl2)
    theta0 = thetas2[np.argmin(refl2)]
    print("theta0 (mrad) =", theta0)


    dex = 100
    Deltas = np.linspace(-dex, dex, 2001)
    Rcav = np.array([RGreenMagnetic(0.0, theta0, Layers, CFe, D) for D in Deltas])
    plt.figure()
    plt.plot(Deltas, np.abs(Rcav)**2)
    plt.xlabel("Δ")
    plt.ylabel("|R|^2")
    plt.title("RGreenMagnetic(B0=0) reflectivity vs Δ")
    plt.grid(True)
    plt.show()
    plot_eigs_scan_theta(Layers, CFe)
  #  df2 = pd.DataFrame({
 #   "Delta": Deltas,
  #  "Reflectivity |R|^2": np.abs(Rcav)**2
#})
#    with pd.ExcelWriter("reflectivity_data.xlsx", engine="openpyxl") as writer:
#        df1.to_excel(writer, sheet_name="Theta_vs_Reflectivity", index=False)
 #       df2.to_excel(writer, sheet_name="Delta_vs_Reflectivity", index=False)
 #   print("数据已导出为 reflectivity_data.xlsx")

def plot_eigs_scan_theta(Layers, CFe):


    def best_match_by_continuity(prev, curr):
        
        n = len(prev)
        best_perm = None
        best_cost = np.inf

        for perm in itertools.permutations(range(n)):
            cost = 0.0
            for j in range(n):
                cost += abs(curr[perm[j]] - prev[j])
            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        return curr[list(best_perm)]

    realPart = []
    imagPart = []
    thetaValues = []

    prev_eigs = None 

    for theta0 in np.arange(5.40, 5.45 + 1e-12, 0.001):
        G, _ = GreenFun(theta0, Layers, CFe)
        # G1->matrix 3 times 3
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

        eigvals = np.linalg.eigvals(G1)

        
        if prev_eigs is None:
            
            eigvals_tracked = eigvals[np.argsort(np.real(eigvals))]
        else:
            eigvals_tracked = best_match_by_continuity(prev_eigs, eigvals)

        prev_eigs = eigvals_tracked

        realPart.append(np.real(eigvals_tracked))
        imagPart.append(np.imag(eigvals_tracked))
        thetaValues.append(theta0)

    thetaValues = np.array(thetaValues)
    realPart = np.array(realPart)  
    imagPart = np.array(imagPart)  

    r1, r2, r3 = realPart[:, 0], realPart[:, 1], realPart[:, 2]
    i1, i2, i3 = imagPart[:, 0], imagPart[:, 1], imagPart[:, 2]


    plt.figure()
    plt.plot(thetaValues, r1, label="r1")
    plt.plot(thetaValues, r2, label="r2")
    plt.plot(thetaValues, r3, label="r3")
    plt.plot(thetaValues, i1, label="i1", linestyle="--")
    plt.plot(thetaValues, i2, label="i2", linestyle="--")
    plt.plot(thetaValues, i3, label="i3", linestyle="--")
    plt.xlabel("θ0 (mrad)")
    plt.ylabel("unit")
    plt.grid(True)
    plt.legend()
    plt.show()
    df = pd.DataFrame({
    "theta (mrad)": thetaValues,
    "r1": r1,
    "r2": r2,
    "r3": r3,
    "i1": i1,
    "i2": i2,
    "i3": i3
})

    df.to_excel("Eigenvalues_vs_theta.xlsx", index=False)


 
    plt.figure()
    plt.plot(thetaValues, r1, label="r1")
    plt.plot(thetaValues, r2, label="r2")
    plt.plot(thetaValues, r3, label="r3")
    plt.xlabel("θ0 (mrad)")
    plt.ylabel("unit")
    plt.grid(True)
    plt.legend()
    plt.show()

    
    plt.figure()
    plt.plot(thetaValues, i1, label="i1")
    plt.plot(thetaValues, i2, label="i2")
    plt.plot(thetaValues, i3, label="i3")
    plt.xlabel("θ0 (mrad)")
    plt.ylabel("unit")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    demo_like_nb()
