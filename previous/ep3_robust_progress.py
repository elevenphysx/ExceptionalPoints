"""
Exceptional Point Finder for Nuclear Resonance Cavity
Structure: Pt-C-Fe-C-Fe-C-Fe-C-Pt(substrate, inf)
Target: Robust search for EP3 (allow EP2 basins but escape them)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import os
import importlib.util
from tqdm import tqdm


# ============================================================
# Import Green function
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "green_function_new",
    os.path.join(current_dir, "green function-new.py")
)
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun


# ============================================================
# Build multilayer structure
# ============================================================
def build_layers(params, fixed_materials):
    """
    params = [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
    """
    theta0 = float(params[0])
    thicknesses = params[1:9]
    C0 = 7.74 * 1.06 * 0.5

    Platinum, Carbon, Iron = fixed_materials

    Layers = [
        (Platinum, thicknesses[0], 0),
        (Carbon,   thicknesses[1], 0),
        (Iron,     thicknesses[2], 1),
        (Carbon,   thicknesses[3], 0),
        (Iron,     thicknesses[4], 1),
        (Carbon,   thicknesses[5], 0),
        (Iron,     thicknesses[6], 1),
        (Carbon,   thicknesses[7], 0),
        (Platinum, np.inf, 0),
    ]

    return theta0, Layers, C0


# ============================================================
# Objective function (Anti-stagnation EP2->EP3)
# ============================================================
def objective_function(
    params,
    fixed_materials,
    bounds=None,
    return_details=False,
    threshold=5.0,
    penalty_weight=2.0
):
    """
    Robust EP search objective:
      - EP2 -> EP3 staged degeneracy (closest pair + third-to-center)
      - Adaptive pull of the 3rd eigenvalue
      - Soft Re/Im threshold (avoid boundary traps)
      - Anti-EP2 plateau kick (enabled only when stagnating)
    """

    if not hasattr(objective_function, "best_loss"):
        objective_function.best_loss = np.inf
        objective_function.stagnation_counter = 0

    # Bounds check
    if bounds is not None:
        for p, (lo, hi) in zip(params, bounds):
            if not (lo <= p <= hi):
                if return_details:
                    return 1e12, None, None, None, None, None, None, None
                return 1e12

    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

        eigvals = np.linalg.eigvals(G1)
        if np.any(~np.isfinite(eigvals)):
            raise ValueError("Non-finite eigenvalues")

        z = np.asarray(eigvals[:3], dtype=np.complex128)

        # EP2 -> EP3 metrics
        d01 = np.abs(z[0] - z[1])
        d02 = np.abs(z[0] - z[2])
        d12 = np.abs(z[1] - z[2])

        if d01 <= d02 and d01 <= d12:
            i, j, k, d_pair = 0, 1, 2, d01
        elif d02 <= d01 and d02 <= d12:
            i, j, k, d_pair = 0, 2, 1, d02
        else:
            i, j, k, d_pair = 1, 2, 0, d12

        center = 0.5 * (z[i] + z[j])
        d_third = np.abs(z[k] - center)

        scale = np.mean(np.abs(z)) + 1e-9

        # Adaptive alpha: once EP2 is good, strongly pull the third
        alpha0 = 0.2
        beta = 2.0
        alpha_eff = alpha0 * (1.0 + beta / (d_pair + 1e-3))

        loss_degeneracy = (d_pair + alpha_eff * d_third) / scale

        # Threshold penalties (soft feasibility + edge repulsion)
        real_parts_3 = np.real(z)
        imag_parts_3 = np.imag(z)

        min_re = float(np.min(np.abs(real_parts_3)))
        min_im = float(np.min(np.abs(imag_parts_3)))

        pen_re = (np.exp(threshold - min_re) - 1.0) if min_re < threshold else 0.0
        pen_im = (np.exp(threshold - min_im) - 1.0) if min_im < threshold else 0.0

        # Edge repulsion to avoid |Re| ~ threshold trap
        margin = 1.0
        edge_gap = (threshold + margin) - min_re
        pen_edge = (np.exp(edge_gap) - 1.0) if edge_gap > 0 else 0.0

        penalty = penalty_weight * (pen_re + pen_im + 0.05 * pen_edge)

        loss = loss_degeneracy + penalty

        # Anti-EP2 plateau kick (only when stagnating)
        if loss < objective_function.best_loss - 1e-6:
            objective_function.best_loss = loss
            objective_function.stagnation_counter = 0
        else:
            objective_function.stagnation_counter += 1

        kick = 0.0
        if (
            objective_function.stagnation_counter > 400
            and d_pair < 1e-3
            and d_third > 0.1
        ):
            kick = 0.1 * np.exp(-d_pair / 1e-4)
            loss += kick

        if return_details:
            # Return full eigvals real/imag for printing (first 3 are most important)
            return (
                float(loss),
                eigvals,
                np.real(eigvals),
                np.imag(eigvals),
                G,
                G1,
                float(loss_degeneracy),
                (float(pen_re), float(pen_im)),
                float(d_pair),
                float(d_third),
                float(alpha_eff),
                float(kick),
            )

        return float(loss)

    except Exception:
        if return_details:
            return 1e12, None, None, None, None, None, None, None, None, None, None, None
        return 1e12


# ============================================================
# Optimization routine with detailed progress
# ============================================================
def optimize_exceptional_point(
    maxiter_de=1000,
    maxiter_nm=500,
    seed=812,
    threshold=5.0,
    penalty_weight=2.0
):
    np.random.seed(seed)

    output_dir = os.path.join(
        "results",
        f"EP3_robust_progress_s{seed}_i1-{maxiter_de}_i2-{maxiter_nm}_thr{threshold}_pw{penalty_weight}"
    )
    os.makedirs(output_dir, exist_ok=True)

    # Fixed material parameters
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    # Bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
    bounds = [
        (2.0, 10.0),
        (0.5, 10.0),
        (0.1, 50.0),
        (0.5, 3.0),
        (0.1, 50.0),
        (0.5, 3.0),
        (0.1, 50.0),
        (0.5, 3.0),
        (0.1, 50.0),
    ]

    # Log file
    log_path = os.path.join(output_dir, "optimization_log.txt")
    log_file = open(log_path, "w", encoding="utf-8")

    def log_print(msg: str):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Wrapper for DE/NM
    def obj_wrap(x):
        return objective_function(
            x,
            fixed_materials,
            bounds=bounds,
            threshold=threshold,
            penalty_weight=penalty_weight,
        )

    # Progress bar for DE
    de_iter_counter = [0]
    pbar_de = tqdm(
        total=maxiter_de,
        desc="DE Progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    def de_callback(xk, convergence=None):
        de_iter_counter[0] += 1
        pbar_de.update(1)

        it = de_iter_counter[0]

        # Every 10 generations: compact detailed loss line
        if it % 10 == 0:
            (
                loss, _, _, _, _, _, loss_degen, (pen_re, pen_im),
                d_pair, d_third, alpha_eff, kick
            ) = objective_function(
                xk,
                fixed_materials,
                bounds=bounds,
                return_details=True,
                threshold=threshold,
                penalty_weight=penalty_weight,
            )

            msg = (
                f"DE Gen {it:4d}/{maxiter_de} | "
                f"Loss={loss:.6e} | Degen={loss_degen:.6e} | "
                f"Pen_Re={pen_re:.3e} Pen_Im={pen_im:.3e} | "
                f"d_pair={d_pair:.3e} d_third={d_third:.3e} alpha={alpha_eff:.3e} kick={kick:.3e}"
            )
            pbar_de.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        # Every 100 generations: print eigenvalues (first 3)
        if it % 100 == 0 or it == 1:
            (
                loss, eigvals, re_all, im_all, _, _, loss_degen, (pen_re, pen_im),
                d_pair, d_third, alpha_eff, kick
            ) = objective_function(
                xk,
                fixed_materials,
                bounds=bounds,
                return_details=True,
                threshold=threshold,
                penalty_weight=penalty_weight,
            )

            pbar_de.write("")
            pbar_de.write(f"--- DE Generation {it}/{maxiter_de} ---")
            pbar_de.write(
                f"Loss={loss:.6e} | Degen={loss_degen:.6e} | Pen_Re={pen_re:.3e} Pen_Im={pen_im:.3e} | "
                f"d_pair={d_pair:.3e} d_third={d_third:.3e} alpha={alpha_eff:.3e} kick={kick:.3e}"
            )
            pbar_de.write("Eigenvalues (first 3):")
            for j in range(3):
                pbar_de.write(f"  lambda_{j+1} = {re_all[j]:+18.10f} {im_all[j]:+18.10f}i")

            log_file.write(f"\n--- DE Generation {it}/{maxiter_de} ---\n")
            log_file.write(
                f"Loss={loss:.6e} | Degen={loss_degen:.6e} | Pen_Re={pen_re:.3e} Pen_Im={pen_im:.3e} | "
                f"d_pair={d_pair:.3e} d_third={d_third:.3e} alpha={alpha_eff:.3e} kick={kick:.3e}\n"
            )
            log_file.write("Eigenvalues (first 3):\n")
            for j in range(3):
                log_file.write(f"  lambda_{j+1} = {re_all[j]:+18.10f} {im_all[j]:+18.10f}i\n")
            log_file.flush()

        return False

    log_print("=" * 70)
    log_print("Starting Differential Evolution (robust EP search)")
    log_print("=" * 70)

    result_de = differential_evolution(
        obj_wrap,
        bounds,
        maxiter=maxiter_de,
        popsize=20,
        seed=seed,
        polish=False,
        updating="immediate",
        workers=1,
        disp=False,              # disable SciPy default line
        callback=de_callback,    # our detailed progress
        tol=0,
        atol=0,
    )

    pbar_de.close()
    log_print(f"DE finished. Best loss = {result_de.fun:.6e}")

    log_print("=" * 70)
    log_print("Phase 2: Local refinement (Nelder-Mead)")
    log_print("=" * 70)

    # Local progress bar
    nm_iter_counter = [0]
    pbar_nm = tqdm(
        total=maxiter_nm,
        desc="Nelder-Mead Progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    def nm_callback(xk):
        nm_iter_counter[0] += 1
        pbar_nm.update(1)
        it = nm_iter_counter[0]

        if it % 50 == 0 or it == 1:
            (
                loss, _, re_all, im_all, _, _, loss_degen, (pen_re, pen_im),
                d_pair, d_third, alpha_eff, kick
            ) = objective_function(
                xk,
                fixed_materials,
                bounds=bounds,
                return_details=True,
                threshold=threshold,
                penalty_weight=penalty_weight,
            )
            msg = (
                f"NM Iter {it:4d}/{maxiter_nm} | "
                f"Loss={loss:.6e} | Degen={loss_degen:.6e} | "
                f"Pen_Re={pen_re:.3e} Pen_Im={pen_im:.3e} | "
                f"d_pair={d_pair:.3e} d_third={d_third:.3e} alpha={alpha_eff:.3e} kick={kick:.3e}"
            )
            pbar_nm.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if it % 100 == 0 or it == 1:
                pbar_nm.write("Eigenvalues (first 3):")
                for j in range(3):
                    pbar_nm.write(f"  lambda_{j+1} = {re_all[j]:+18.10f} {im_all[j]:+18.10f}i")

        return False

    result_nm = minimize(
        obj_wrap,
        result_de.x,
        method="Nelder-Mead",
        callback=nm_callback,
        options={"maxiter": maxiter_nm, "adaptive": True},
    )

    pbar_nm.close()

    best = result_nm if result_nm.fun < result_de.fun else result_de

    log_print("=" * 70)
    log_print("FINAL RESULT")
    log_print("=" * 70)

    (
        loss, eigvals, re_all, im_all, _, _, loss_degen, (pen_re, pen_im),
        d_pair, d_third, alpha_eff, kick
    ) = objective_function(
        best.x,
        fixed_materials,
        bounds=bounds,
        return_details=True,
        threshold=threshold,
        penalty_weight=penalty_weight,
    )

    theta0, Layers, C0 = build_layers(best.x, fixed_materials)

    log_print(f"Total Loss = {loss:.12e}")
    log_print(f"  Degeneracy Loss = {loss_degen:.12e}")
    log_print(f"  Penalty Re = {pen_re:.12e}")
    log_print(f"  Penalty Im = {pen_im:.12e}")
    log_print(f"  d_pair  = {d_pair:.12e}")
    log_print(f"  d_third = {d_third:.12e}")
    log_print(f"  alpha_eff = {alpha_eff:.12e}")
    log_print(f"  kick = {kick:.12e}")
    log_print("")
    log_print(f"theta0 = {theta0:.15f}")
    log_print(f"C0     = {C0:.15f} (fixed)")
    log_print("")
    log_print("Layer thicknesses (finite layers):")
    for idx, layer in enumerate(Layers[:-1]):
        log_print(f"  Layer {idx}: thickness = {layer[1]:.15f} nm, resonant={layer[2]}")

    log_print("")
    log_print("Eigenvalues (first 3):")
    for j in range(3):
        log_print(f"  lambda_{j+1} = {re_all[j]:+22.15f} {im_all[j]:+22.15f}i")

    log_print("")
    log_print(f"Log saved to: {log_path}")

    log_file.close()
    return best, output_dir


# ============================================================
# Entry point with CLI args (like your original)
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Robust EP3 Search with detailed progress")
    parser.add_argument("-i1", "--iterations-de", type=int, default=1000, help="DE generations")
    parser.add_argument("-i2", "--iterations-nm", type=int, default=500, help="Nelder-Mead iterations")
    parser.add_argument("-s", "--seed", type=int, default=812, help="Random seed")
    parser.add_argument("-t", "--threshold", type=float, default=5.0, help="Threshold for |Re| and |Im|")
    parser.add_argument("-pw", "--penalty-weight", type=float, default=2.0, help="Penalty weight")

    args = parser.parse_args()

    optimize_exceptional_point(
        maxiter_de=args.iterations_de,
        maxiter_nm=args.iterations_nm,
        seed=args.seed,
        threshold=args.threshold,
        penalty_weight=args.penalty_weight,
    )
