"""
Exceptional Point (EP3) Discrete Optimizer - Experimental Precision Version
Only uses Differential Evolution with 1 decimal place precision (0.1 step)
For EP3 (3 resonant layers) with Carbon spacer layer only

This simulates experimental constraints where parameters can only be set to 1 decimal place
"""

import numpy as np
from scipy.optimize import differential_evolution
import sys
import os
import importlib.util
from tqdm import tqdm

# Import shared configuration
from config import (
    FIXED_MATERIALS, C0_FIXED, DEFAULT_SEEDS,
    PARAM_NAMES, PARAM_LABELS, LAYER_NAMES,
    BOUNDS_EXTENDED, IMAG_MIN, IMAG_PENALTY,
    DEFAULT_DE_ITERATIONS
)

# Import common functions (EP3 version)
from common_functions import (
    build_layers, objective_function_control, objective_function_cached, clear_eval_cache,
    format_eigenvalues_string, save_eigenvalues_txt,
    save_params_npz, save_parameters_txt, format_final_result_string
)

# Import plotting utilities
from plotting_utils import plot_optimization_history, scan_parameters_around_optimum

# ============================================================
# Dynamic Import of Green Function
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
green_script_path = os.path.join(current_dir, "green_function.py")

if not os.path.exists(green_script_path):
    green_script_path = "green_function.py"

try:
    spec = importlib.util.spec_from_file_location("green_function", green_script_path)
    green_function_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(green_function_module)
    GreenFun = green_function_module.GreenFun
except Exception as e:
    print(f"Error importing GreenFun: {e}")
    print("Make sure 'green_function.py' is in the same directory.")
    sys.exit(1)

# ============================================================
# Discrete precision parameters (configurable)
# ============================================================
DISCRETE_PRECISION_ANGLE = 2      # Angle (theta0) precision in decimal places (e.g., 2 = 0.01 mrad step)
DISCRETE_PRECISION_THICKNESS = 3  # Thickness precision in decimal places (e.g., 2 = 0.01 nm step)

# ============================================================
# Module-level wrapper for multiprocessing (must be picklable)
# ============================================================

def _objective_wrapper_discrete(params):
    """
    Module-level wrapper with discrete precision rounding.
    Angle (params[0]) is rounded to DISCRETE_PRECISION_ANGLE decimal places.
    Other parameters (thickness) are rounded to DISCRETE_PRECISION_THICKNESS decimal places.
    Uses cached version to avoid redundant Green function calculations.
    """
    params_rounded = params.copy()
    params_rounded[0] = np.round(params[0], decimals=DISCRETE_PRECISION_ANGLE)  # Angle
    params_rounded[1:] = np.round(params[1:], decimals=DISCRETE_PRECISION_THICKNESS)  # Thickness
    return objective_function_cached(params_rounded, FIXED_MATERIALS, GreenFun, build_layers_func=build_layers, use_cache=True)


# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point_discrete(maxiter_de, seed, n_workers, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'ep3_discrete_a{DISCRETE_PRECISION_ANGLE}t{DISCRETE_PRECISION_THICKNESS}_w{n_workers}_s{seed}_DE{maxiter_de}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")
        print("=" * 70)
        print(f"DISCRETE OPTIMIZATION MODE")
        print(f"Precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad ({DISCRETE_PRECISION_ANGLE} decimals), "
              f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.1f} nm ({DISCRETE_PRECISION_THICKNESS} decimal)")
        print(f"Structure: EP3 (3 resonant layers) with Carbon spacer")
        print(f"Algorithm: Differential Evolution only (no L-BFGS-B)")
        print("=" * 70)

    # Clear evaluation cache at start of optimization
    clear_eval_cache()

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS_EXTENDED  # Use extended bounds for broader search space

    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    def log_print(msg, console=True):
        if console and verbose:
            print(f"[Seed {seed}] {msg}")
        log_file.write(msg + "\n")
        log_file.flush()

    history = {'iteration': [], 'loss': [], 'phase': [], 'eigvals_real': [], 'eigvals_imag': []}
    iteration_count = [0]

    # --- Differential Evolution with Discrete Precision ---
    log_print("=" * 70, console=False)
    log_print(f"Differential Evolution (Discrete) - Seed {seed}", console=False)
    log_print(f"Precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad ({DISCRETE_PRECISION_ANGLE} decimals), "
              f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.2f} nm ({DISCRETE_PRECISION_THICKNESS} decimals)", console=False)
    log_print("=" * 70, console=False)

    pbar_de = tqdm(total=maxiter_de, desc=f"Seed {seed} DE-Discrete", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    def callback_de(xk, convergence=None):
        nonlocal best_loss_so_far, best_solution_so_far

        iteration_count[0] += 1
        pbar_de.update(1)

        # Round parameters (angle: DISCRETE_PRECISION_ANGLE, thickness: DISCRETE_PRECISION_THICKNESS)
        xk_rounded = xk.copy()
        xk_rounded[0] = np.round(xk[0], decimals=DISCRETE_PRECISION_ANGLE)  # Angle
        xk_rounded[1:] = np.round(xk[1:], decimals=DISCRETE_PRECISION_THICKNESS)  # Thickness

        # Use cached version - will return cached value from DE's evaluation (no redundant calculation)
        loss_current = objective_function_cached(xk_rounded, fixed_materials, GreenFun, build_layers_func=build_layers, use_cache=True)

        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk_rounded.copy()

        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk_rounded, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers
            )
            # Format eigenvalues for logging
            eigvals_str = format_eigenvalues_string(real_parts, imag_parts)

            # Show rounded parameters with correct precision
            params_str = (f"{xk_rounded[0]:.{DISCRETE_PRECISION_ANGLE}f}, "
                         f"{xk_rounded[1]:.{DISCRETE_PRECISION_THICKNESS}f}, "
                         f"{xk_rounded[2]:.{DISCRETE_PRECISION_THICKNESS}f}")

            msg = f"DE Iter {iteration_count[0]:5d}: Loss={loss:.6e} (Spread={spread:.6e}, PenIm={pen_im:.2e})\n"
            msg += f"  Params[0:3]=[{params_str}, ...]\n"
            msg += f"  Eigenvalues: {eigvals_str}"

            if verbose:
                pbar_de.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('DE-Discrete')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    # DE settings optimized for discrete optimization
    log_print(f"Using workers={n_workers}, popsize=100 (optimized for discrete space)", console=verbose)

    result_de = differential_evolution(
        _objective_wrapper_discrete,  # Use discrete wrapper
        bounds,
        maxiter=maxiter_de,
        popsize=50,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0,
        atol=0,
        updating='deferred',
        workers=n_workers,
        polish=False,
        seed=seed,
        disp=False,
        callback=callback_de,
    )

    pbar_de.close()

    if best_solution_so_far is None:
        best_solution_so_far = result_de.x.copy()
        best_solution_so_far[0] = np.round(best_solution_so_far[0], decimals=DISCRETE_PRECISION_ANGLE)
        best_solution_so_far[1:] = np.round(best_solution_so_far[1:], decimals=DISCRETE_PRECISION_THICKNESS)

    # Ensure final result is rounded with correct precision
    final_x = best_solution_so_far.copy()
    final_x[0] = np.round(final_x[0], decimals=DISCRETE_PRECISION_ANGLE)  # Angle
    final_x[1:] = np.round(final_x[1:], decimals=DISCRETE_PRECISION_THICKNESS)  # Thickness

    # --- Discrete Local Search (Greedy Hill Climbing) ---
    if verbose:
        print(f"\n[Seed {seed}] Starting discrete local search (greedy hill climbing)...")

    log_print("\n" + "=" * 70, console=False)
    log_print("Discrete Local Search (Greedy Hill Climbing)", console=False)
    log_print("=" * 70, console=False)

    best_x_local = final_x.copy()
    best_loss_local = objective_function_cached(best_x_local, fixed_materials, GreenFun,
                                                 build_layers_func=build_layers, use_cache=True)

    improved = True
    local_iter = 0

    while improved:
        improved = False
        local_iter += 1

        for i in range(len(best_x_local)):
            step_size = 10**(-DISCRETE_PRECISION_ANGLE) if i == 0 else 10**(-DISCRETE_PRECISION_THICKNESS)
            precision = DISCRETE_PRECISION_ANGLE if i == 0 else DISCRETE_PRECISION_THICKNESS

            # Try +step
            test_x = best_x_local.copy()
            test_x[i] += step_size
            if bounds[i][0] <= test_x[i] <= bounds[i][1]:
                # Ensure proper rounding
                test_x[i] = np.round(test_x[i], decimals=precision)

                test_loss = objective_function_cached(test_x, fixed_materials, GreenFun,
                                                     build_layers_func=build_layers, use_cache=True)

                if test_loss < best_loss_local:
                    best_x_local = test_x.copy()
                    best_loss_local = test_loss
                    improved = True
                    log_print(f"  Local iter {local_iter}: param[{i}] +{step_size} -> loss={test_loss:.6e}", console=False)
                    continue

            # Try -step
            test_x = best_x_local.copy()
            test_x[i] -= step_size
            if bounds[i][0] <= test_x[i] <= bounds[i][1]:
                # Ensure proper rounding
                test_x[i] = np.round(test_x[i], decimals=precision)

                test_loss = objective_function_cached(test_x, fixed_materials, GreenFun,
                                                     build_layers_func=build_layers, use_cache=True)

                if test_loss < best_loss_local:
                    best_x_local = test_x.copy()
                    best_loss_local = test_loss
                    improved = True
                    log_print(f"  Local iter {local_iter}: param[{i}] -{step_size} -> loss={test_loss:.6e}", console=False)

    improvement = best_loss_so_far - best_loss_local
    log_print(f"\nLocal search completed after {local_iter} iterations", console=False)
    log_print(f"  Before: {best_loss_so_far:.6e}", console=False)
    log_print(f"  After:  {best_loss_local:.6e}", console=False)
    log_print(f"  Improvement: {improvement:.6e} ({improvement/best_loss_so_far*100:.2f}%)", console=False)

    if verbose:
        print(f"[Seed {seed}] Local search: {best_loss_so_far:.6e} -> {best_loss_local:.6e} (Δ={improvement:.6e})")

    # Use locally optimized result
    final_x = best_x_local
    final_loss = best_loss_local

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        final_x, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers
    )

    theta0, Layers, C0 = build_layers(final_x, fixed_materials)

    output = format_final_result_string(loss, spread, pen_im, real_parts, imag_parts, IMAG_MIN)
    output = "\n" + "="*70 + "\nFINAL RESULT (DISCRETE)\n" + "="*70 + "\n" + \
             f"Precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad ({DISCRETE_PRECISION_ANGLE} decimals), " \
             f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.2f} nm ({DISCRETE_PRECISION_THICKNESS} decimals)\n" + output

    log_print(output, console=verbose)
    log_file.close()

    # Save results using common functions
    save_params_npz(output_dir, final_x, loss, eigvals, theta0, C0)

    save_parameters_txt(
        output_dir=output_dir,
        seed=seed,
        ep_name=f'EP3-Discrete-a{DISCRETE_PRECISION_ANGLE}t{DISCRETE_PRECISION_THICKNESS}',
        final_x=final_x,
        final_loss=loss,
        spread=spread,
        pen_im=pen_im,
        theta0=theta0,
        Layers=Layers,
        bounds=bounds,
        layer_names=LAYER_NAMES
    )

    save_eigenvalues_txt(output_dir, seed, real_parts, imag_parts, ep_type=f'EP3-Discrete-a{DISCRETE_PRECISION_ANGLE}t{DISCRETE_PRECISION_THICKNESS}')

    plot_optimization_history(history, output_dir, seed, 'EP3 Discrete Optimizer: DE (0.1 step)')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, GreenFun, build_layers_func=build_layers, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-4,
        n_points=51,
        param_names=PARAM_NAMES,
        param_labels=PARAM_LABELS
    )

    return seed, loss, final_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f'EP3 Discrete Optimizer - Angle: {DISCRETE_PRECISION_ANGLE} decimals, Thickness: {DISCRETE_PRECISION_THICKNESS} decimals')
    parser.add_argument('-i', '--iter', type=int, default=DEFAULT_DE_ITERATIONS*2,
                       help='DE iterations (default: 2x normal, for discrete space)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-w', '--workers', type=int, default=20, help='Number of workers for DE')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EP3 DISCRETE Optimizer (workers={args.workers}, seed={args.seed})")
    print(f"Algorithm: Differential Evolution ONLY (no L-BFGS-B)")
    print(f"Precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad ({DISCRETE_PRECISION_ANGLE} decimals), "
          f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.2f} nm ({DISCRETE_PRECISION_THICKNESS} decimals)")
    print(f"Matrix: -G - 0.5j*I | Constraint: |Im(λ)| >= {IMAG_MIN}")
    print(f"Structure: Pt / C / Fe* / C / Fe* / C / Fe* / C / Pt(substrate)")
    print("=" * 70)

    optimize_exceptional_point_discrete(
        maxiter_de=args.iter,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True
    )
