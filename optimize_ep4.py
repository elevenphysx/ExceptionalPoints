"""
Exceptional Point (EP4) Optimizer
Two-stage optimization: Differential Evolution + L-BFGS-B
Uses variance-based loss function with -G-0.5j*I matrix
Optimizes for 4 resonant layers (EP4)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys
import os
import importlib.util
from tqdm import tqdm

# Import shared configuration (EP4 version)
from config import (
    FIXED_MATERIALS, C0_FIXED, DEFAULT_SEEDS,
    PARAM_NAMES_EP4, PARAM_LABELS_EP4, LAYER_NAMES_EP4,
    BOUNDS_EP4, IMAG_MIN, IMAG_PENALTY,
    DEFAULT_DE_ITERATIONS, DEFAULT_LBFGSB_ITERATIONS
)

# Import common functions (EP4 version)
from common_functions import (
    build_layers_ep4, objective_function_control,
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
# Module-level wrapper for multiprocessing (must be picklable)
# ============================================================

def _objective_wrapper_for_de(params):
    """
    Module-level wrapper that can be pickled for multiprocessing.
    Uses module-level GreenFun, FIXED_MATERIALS, and build_layers_ep4.
    """
    return objective_function_control(params, FIXED_MATERIALS, GreenFun, build_layers_func=build_layers_ep4)


# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_de, maxiter_lbfgsb, seed, n_workers, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'ep4_w{n_workers}_s{seed}_DE{maxiter_de}_LB{maxiter_lbfgsb}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS_EP4


    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    def log_print(msg, console=True):
        if console and verbose:
            print(f"[Seed {seed}] {msg}")
        log_file.write(msg + "\n")
        log_file.flush()

    history = {'iteration': [], 'loss': [], 'phase': [], 'eigvals_real': [], 'eigvals_imag': []}
    iteration_count = [0]

    # --- Phase 1: Differential Evolution (Control_N settings) ---
    log_print("=" * 70, console=False)
    log_print(f"Phase 1: Differential Evolution - Seed {seed}", console=False)
    log_print("=" * 70, console=False)

    pbar_de = tqdm(total=maxiter_de, desc=f"Seed {seed} DE", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    def callback_de(xk, convergence=None):
        nonlocal best_loss_so_far, best_solution_so_far

        iteration_count[0] += 1
        pbar_de.update(1)

        loss_current = objective_function_control(xk, fixed_materials, GreenFun, build_layers_func=build_layers_ep4)

        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk.copy()

        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers_ep4
            )
            # Format eigenvalues for logging
            eigvals_str = format_eigenvalues_string(real_parts, imag_parts)
            msg = f"DE Iter {iteration_count[0]:5d}: Loss={loss:.6e} (Spread={spread:.6e}, PenIm={pen_im:.2e}, convergence={convergence if convergence else 0:.3e})\n"
            msg += f"  Eigenvalues: {eigvals_str}"
            if verbose: pbar_de.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()  # Flush immediately for real-time logging

            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('DE')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    # Control_N DE settings with multiprocessing fix
    log_print(f"Using workers={n_workers}", console=verbose)

    result_de = differential_evolution(
        _objective_wrapper_for_de,  # Use module-level wrapper (can be pickled)
        bounds,
        maxiter=maxiter_de,
        popsize=25,
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
        best_solution_so_far = result_de.x

    result_de_x = best_solution_so_far

    # --- Phase 2: L-BFGS-B Refinement (Control_N settings) ---
    log_print("\n" + "=" * 70, console=False)
    log_print("Phase 2: L-BFGS-B Refinement", console=False)
    log_print("=" * 70, console=False)

    lbfgsb_iter = [0]
    pbar_lbfgsb = tqdm(total=maxiter_lbfgsb, desc=f"Seed {seed} L-BFGS-B", disable=not verbose)

    def lbfgsb_callback(xk):
        lbfgsb_iter[0] += 1
        pbar_lbfgsb.update(1)
        if lbfgsb_iter[0] % 100 == 0:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers_ep4
            )
            # Format eigenvalues for logging
            eigvals_str = format_eigenvalues_string(real_parts, imag_parts)
            msg = f"L-BFGS-B Iter {lbfgsb_iter[0]}: Loss={loss:.6e}\n"
            msg += f"  Eigenvalues: {eigvals_str}"
            if verbose: pbar_lbfgsb.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()  # Flush immediately for real-time logging

            # Record to history for plotting
            if eigvals is not None:
                history['iteration'].append(iteration_count[0] + lbfgsb_iter[0])
                history['loss'].append(loss)
                history['phase'].append('L-BFGS-B')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    try:
        res_lbfgsb = minimize(
            lambda p: objective_function_control(p, fixed_materials, GreenFun, build_layers_func=build_layers_ep4),
            result_de_x,
            method='L-BFGS-B',
            bounds=bounds,
            callback=lbfgsb_callback,
            options={'maxiter': maxiter_lbfgsb, 'ftol': 1e-20, 'gtol': 1e-14, 'disp': False}
        )
        final_x = res_lbfgsb.x
        final_loss = res_lbfgsb.fun
        pbar_lbfgsb.close()
    except Exception as e:
        print(f"L-BFGS-B Warning: {e}")
        final_x = result_de_x
        final_loss = best_loss_so_far
        pbar_lbfgsb.close()

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        final_x, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers_ep4
    )

    theta0, Layers, C0 = build_layers_ep4(final_x, fixed_materials)

    output = format_final_result_string(loss, spread, pen_im, real_parts, imag_parts, IMAG_MIN)

    log_print(output, console=verbose)
    log_file.close()

    # Save results
    save_params_npz(output_dir, final_x, final_loss, eigvals, theta0, C0)

    save_parameters_txt(
        output_dir=output_dir,
        seed=seed,
        ep_name='EP4',
        final_x=final_x,
        final_loss=final_loss,
        spread=spread,
        pen_im=pen_im,
        theta0=theta0,
        Layers=Layers,
        bounds=bounds,
        layer_names=LAYER_NAMES_EP4
    )

    save_eigenvalues_txt(output_dir, seed, real_parts, imag_parts, ep_type='EP4')

    plot_optimization_history(history, output_dir, seed, 'EP4 Optimizer: DE + L-BFGS-B')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, GreenFun, build_layers_func=build_layers_ep4, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-4,
        n_points=51,
        param_names=PARAM_NAMES_EP4,
        param_labels=PARAM_LABELS_EP4
    )

    return seed, final_loss, final_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point (EP4) Optimizer - 4 Resonant Layers')
    parser.add_argument('-i1', type=int, default=DEFAULT_DE_ITERATIONS, help='DE iterations')
    parser.add_argument('-i2', type=int, default=DEFAULT_LBFGSB_ITERATIONS, help='L-BFGS-B iterations')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-w', '--workers', type=int, default=20, help='Number of workers for DE')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EP4 Optimizer (workers={args.workers}, seed={args.seed})")
    print(f"Algorithm: DE ({args.i1} iter) -> L-BFGS-B ({args.i2} iter)")
    print(f"Matrix: -G - 0.5j*I | Constraint: |Im(λ)| >= {IMAG_MIN}")
    print(f"Structure: Pt / (C/Fe*)×4 / C / Pt(substrate) - 4 resonant layers")
    print("=" * 70)

    optimize_exceptional_point(
        maxiter_de=args.i1,
        maxiter_lbfgsb=args.i2,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True
    )
