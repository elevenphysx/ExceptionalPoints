import numpy as np
from typing import List, Tuple
from scipy.optimize import differential_evolution, minimize
import Greenfun as Gn

Platinum = {"δ": 1.713e-5, "β": 2.518e-6}
Carbon   = {"δ": 2.257e-6, "β": 1.230e-9}
Iron     = {"δ": 7.298e-6, "β": 3.33e-7}
CFe = 7.74 * 1.06 * 0.5 * 1


# Bounds
THETA0_BOUNDS = (2.0, 8.0)
THICKNESS_BOUNDS = [
    (0.5, 4.0),    # Pt
    (1.0, 40.0),   # C
    (0.8, 3.0),    # Fe
    (1.0, 40.0),   # C
    (0.5, 3.0),    # Fe
    (1.0, 40.0),   # C
    (0.5, 3.0),    # Fe
    (1.0, 40.0),   # C
    (0.5, 3.0),    # Fe
    (1.0, 40.0),   # C
]
IMAG_MIN = 5.0
IMAG_PENALTY = 1e4


BOUNDS = [THETA0_BOUNDS] + THICKNESS_BOUNDS



def build_layer_template(thickness_bounds: List[Tuple[float, float]]):
    n_var = len(thickness_bounds)

    if n_var < 4 or (n_var % 2 != 0):
        raise ValueError("Invalid THICKNESS_BOUNDS length.")

    n_pair = (n_var - 2) // 2
    template = []

    template.append((Platinum, None, 0))
    for _ in range(n_pair):
        template.append((Carbon, None, 0))
        template.append((Iron,   None, 1))
    template.append((Carbon, None, 0))
    template.append((Platinum, -1.0, 0))

    return template


LAYER_TEMPLATE = build_layer_template(THICKNESS_BOUNDS)


def n_resonant_layers() -> int:
    return int(sum(flag == 1 for (_, _, flag) in LAYER_TEMPLATE))

def build_layers(thicknesses: np.ndarray):
    thicknesses = np.asarray(thicknesses, dtype=float).ravel()
    n_opt = sum(t is None for (_, t, _) in LAYER_TEMPLATE)

    if thicknesses.size != n_opt:
        raise ValueError("Thickness size mismatch.")

    layers = []
    i = 0
    for (mat, t, flag) in LAYER_TEMPLATE:
        if t is None:
            layers.append([mat, thicknesses[i], flag])
            i += 1
        else:
            layers.append([mat, t, flag])
    return layers


def eig_spread_of_G(theta0: float, thicknesses: np.ndarray):
    layers = build_layers(thicknesses)
    G, _ = Gn.GreenFun(theta0, layers, CFe)

    n_res = n_resonant_layers()
    if G.shape != (n_res, n_res):
        raise RuntimeError("Unexpected G shape.")

    lam = np.sort_complex(np.linalg.eigvals(G))
    mean_lam = np.mean(lam)
    diff = lam - mean_lam

    spread = np.sum(diff.real**2 + diff.imag**2)
    return float(spread), lam, mean_lam

def loss_all_eigs_equal(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    theta0 = x[0]
    t = x[1:]

    if not (THETA0_BOUNDS[0] <= theta0 <= THETA0_BOUNDS[1]):
        return 1e6
    for (lo, hi), ti in zip(THICKNESS_BOUNDS, t):
        if not (lo <= ti <= hi):
            return 1e6

    try:
        spread, eigs, _ = eig_spread_of_G(theta0, t)
    except Exception:
        return 1e6

    loss = spread
    min_im = np.min(eigs.imag)
    if min_im < IMAG_MIN:
        loss += IMAG_PENALTY * (IMAG_MIN - min_im) ** 2

    return float(loss)


def run_optimization(seed: int = 0):

    print("================================================================================")
    print("ARBITRARY-ORDER EIGENVALUE COLLAPSE OPTIMIZATION")
    print(f"Im(lambda_i) >= {IMAG_MIN}")
    print("Resonant layers =", n_resonant_layers())
    print("================================================================================")

    iter_counter = {"i": 0}

    def de_callback(xk, convergence):
        iter_counter["i"] += 1
        if iter_counter["i"] % 1000 == 0:
            fval = loss_all_eigs_equal(xk)
            print(
                f"[DE step {iter_counter['i']:5d}] "
                f"f(x) = {fval:.6e}, "
                f"convergence = {convergence:.3e}"
            )

    de = differential_evolution(
        loss_all_eigs_equal,
        bounds=BOUNDS,
        strategy="best1bin",
        popsize=25,
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=1e-12,
        updating="deferred",
        workers=-1,
        maxiter=100000,
        polish=False,
        seed=seed,
        disp=False,
        callback=de_callback,
    )

    x0 = de.x.copy()

    local = minimize(
        loss_all_eigs_equal,
        x0=x0,
        method="L-BFGS-B",
        bounds=BOUNDS,
        options=dict(
            maxiter=5000,
            ftol=1e-20,
            gtol=1e-14,
            disp=True,
        ),
    )

    xbest = local.x
    theta0 = xbest[0]
    thicknesses = xbest[1:]

    spread, eigs, mean_eig = eig_spread_of_G(theta0, thicknesses)

    print("\n=== Best solution ===")
    print("loss =", spread)
    print("theta0 =", theta0)
    print("thicknesses =", thicknesses.tolist())
    print("mean eig =", mean_eig)
    print("eigvals =", eigs)
    print("min Im(eig) =", np.min(eigs.imag))

    return xbest, eigs


if __name__ == "__main__":
    run_optimization(seed=0)
