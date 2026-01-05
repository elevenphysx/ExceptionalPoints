"""
Exceptional Point (EP3) Hybrid Optimizer
Three-stage optimization:
1. Differential Evolution (continuous)
2. L-BFGS-B Refinement (continuous)
3. Discrete Local Search (discretization + greedy hill climbing)

Uses variance-based loss function with -G-0.5j*I matrix
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys
import os
import importlib.util
from tqdm import tqdm

# Import shared configuration
from config import (
    FIXED_MATERIALS, C0_FIXED, DEFAULT_SEEDS,
    PARAM_NAMES, PARAM_LABELS, LAYER_NAMES,
    BOUNDS_EXTENDED, IMAG_MIN, IMAG_PENALTY,
    DEFAULT_DE_ITERATIONS, DEFAULT_LBFGSB_ITERATIONS
)

# Import common functions
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
# Discrete precision parameters (for final discretization stage)
# ============================================================
DISCRETE_PRECISION_ANGLE = 2      # Angle precision in decimal places (0.01 mrad)
DISCRETE_PRECISION_THICKNESS = 3  # Thickness precision in decimal places (0.001 nm)

# ============================================================
# Module-level wrapper for multiprocessing (must be picklable)
# ============================================================

def _objective_wrapper_for_de(params):
    """
    Module-level wrapper that can be pickled for multiprocessing.
    Uses module-level GreenFun and FIXED_MATERIALS.
    Uses cached version to avoid redundant Green function calculations.
    """
    return objective_function_cached(params, FIXED_MATERIALS, GreenFun, use_cache=True)


# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point_hybrid(maxiter_de, maxiter_lbfgsb, seed, n_workers, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'ep3_hybrid_w{n_workers}_s{seed}_DE{maxiter_de}_LB{maxiter_lbfgsb}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")
        print("=" * 70)
        print("HYBRID OPTIMIZATION (Continuous -> Discrete)")
        print(f"Stage 1-2: Continuous optimization (DE + L-BFGS-B)")
        print(f"Stage 3: Discretization + Local search")
        print(f"Discrete precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad, "
              f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.3f} nm")
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

        # Use cached version - will return cached value from DE's evaluation (no redundant calculation)
        loss_current = objective_function_cached(xk, fixed_materials, GreenFun, use_cache=True)

        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk.copy()

        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk, fixed_materials, GreenFun, return_details=True
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
                xk, fixed_materials, GreenFun, return_details=True
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
            lambda p: objective_function_control(p, fixed_materials, GreenFun),
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

    # Store continuous optimization result
    continuous_optimal_x = final_x.copy()
    continuous_optimal_loss = final_loss

    # --- Phase 3: Discrete Local Search (Greedy Hill Climbing) ---
    if verbose:
        print(f"\n[Seed {seed}] Phase 3: Discretization + Local Search")
        print(f"  Continuous optimal loss: {continuous_optimal_loss:.6e}")

    log_print("\n" + "=" * 70, console=False)
    log_print("Phase 3: Discrete Local Search (Greedy Hill Climbing)", console=False)
    log_print(f"Continuous optimal: {continuous_optimal_loss:.6e}", console=False)
    log_print("=" * 70, console=False)

    # Discretize continuous solution
    discrete_start = continuous_optimal_x.copy()
    discrete_start[0] = np.round(discrete_start[0], decimals=DISCRETE_PRECISION_ANGLE)
    discrete_start[1:] = np.round(discrete_start[1:], decimals=DISCRETE_PRECISION_THICKNESS)

    if verbose:
        print(f"  Discretized starting point")

    best_x_local = discrete_start.copy()
    best_loss_local = objective_function_cached(best_x_local, fixed_materials, GreenFun,
                                                 build_layers_func=build_layers, use_cache=True)

    log_print(f"Discrete starting loss: {best_loss_local:.6e}", console=False)

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
                test_x[i] = np.round(test_x[i], decimals=precision)

                test_loss = objective_function_cached(test_x, fixed_materials, GreenFun,
                                                     build_layers_func=build_layers, use_cache=True)

                if test_loss < best_loss_local:
                    best_x_local = test_x.copy()
                    best_loss_local = test_loss
                    improved = True
                    log_print(f"  Local iter {local_iter}: param[{i}] -{step_size} -> loss={test_loss:.6e}", console=False)

    improvement = continuous_optimal_loss - best_loss_local
    log_print(f"\nLocal search completed after {local_iter} iterations", console=False)
    log_print(f"  Continuous optimal: {continuous_optimal_loss:.6e}", console=False)
    log_print(f"  Discrete optimal:   {best_loss_local:.6e}", console=False)
    log_print(f"  Improvement: {improvement:.6e}", console=False)

    if verbose:
        print(f"[Seed {seed}] Discrete optimization complete")
        print(f"  Continuous: {continuous_optimal_loss:.6e}")
        print(f"  Discrete:   {best_loss_local:.6e}")
        print(f"  Improvement: {improvement:.6e}")

    # Use discrete optimal result
    final_x = best_x_local
    final_loss = best_loss_local

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        final_x, fixed_materials, GreenFun, return_details=True
    )

    theta0, Layers, C0 = build_layers(final_x, fixed_materials)

    output = format_final_result_string(loss, spread, pen_im, real_parts, imag_parts, IMAG_MIN)

    log_print(output, console=verbose)
    log_file.close()

    # Save results
    save_params_npz(output_dir, final_x, final_loss, eigvals, theta0, C0)

    save_parameters_txt(
        output_dir=output_dir,
        seed=seed,
        ep_name='EP3-Hybrid',
        final_x=final_x,
        final_loss=final_loss,
        spread=spread,
        pen_im=pen_im,
        theta0=theta0,
        Layers=Layers,
        bounds=bounds,
        layer_names=LAYER_NAMES
    )

    save_eigenvalues_txt(output_dir, seed, real_parts, imag_parts, ep_type='EP3-Hybrid')

    plot_optimization_history(history, output_dir, seed, 'EP3 Hybrid Optimizer: DE + L-BFGS-B + Discrete')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, GreenFun, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-4,
        n_points=51,
        param_names=PARAM_NAMES,
        param_labels=PARAM_LABELS
    )

    return seed, final_loss, final_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point (EP3) Hybrid Optimizer')
    parser.add_argument('-i1', type=int, default=DEFAULT_DE_ITERATIONS, help='DE iterations')
    parser.add_argument('-i2', type=int, default=DEFAULT_LBFGSB_ITERATIONS, help='L-BFGS-B iterations')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-w', '--workers', type=int, default=20, help='Number of workers for DE')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EP3 HYBRID Optimizer (workers={args.workers}, seed={args.seed})")
    print(f"Algorithm: DE ({args.i1} iter) -> L-BFGS-B ({args.i2} iter) -> Discrete Local Search")
    print(f"Bounds: BOUNDS_EXTENDED (broader search space)")
    print(f"Discrete precision: Angle={10**(-DISCRETE_PRECISION_ANGLE)} mrad, "
          f"Thickness={10**(-DISCRETE_PRECISION_THICKNESS):.3f} nm")
    print(f"Matrix: -G - 0.5j*I | Constraint: |Im(λ)| >= {IMAG_MIN}")
    print("=" * 70)

    optimize_exceptional_point_hybrid(
        maxiter_de=args.i1,
        maxiter_lbfgsb=args.i2,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True
    )