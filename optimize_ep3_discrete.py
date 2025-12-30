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

# Import shared configuration (EP3 version)
from config import (
    FIXED_MATERIALS, C0_FIXED,
    PARAM_NAMES, PARAM_LABELS, LAYER_NAMES,
    BOUNDS, IMAG_MIN, IMAG_PENALTY,
    DEFAULT_DE_ITERATIONS
)

# Import common functions (EP3 version)
from common_functions import build_layers, objective_function_control

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
# Discrete precision parameter (global)
# ============================================================
DISCRETE_PRECISION = 1  # Decimal places (0.1 step size)

# ============================================================
# Module-level wrapper for multiprocessing (must be picklable)
# ============================================================

def _objective_wrapper_discrete(params):
    """
    Module-level wrapper with discrete precision rounding.
    Rounds parameters to DISCRETE_PRECISION decimal places before evaluation.
    """
    params_rounded = np.round(params, decimals=DISCRETE_PRECISION)
    return objective_function_control(params_rounded, FIXED_MATERIALS, GreenFun, build_layers_func=build_layers)


# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point_discrete(maxiter_de, seed, n_workers, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'ep3_discrete_p{DISCRETE_PRECISION}_w{n_workers}_s{seed}_DE{maxiter_de}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")
        print("=" * 70)
        print(f"DISCRETE OPTIMIZATION MODE")
        print(f"Precision: {DISCRETE_PRECISION} decimal place(s) (step = 0.{'0'*(DISCRETE_PRECISION-1)}1)")
        print(f"Structure: EP3 (3 resonant layers) with Carbon spacer")
        print(f"Algorithm: Differential Evolution only (no L-BFGS-B)")
        print("=" * 70)

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS  # EP3 bounds

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
    log_print(f"Precision: {DISCRETE_PRECISION} decimal place(s)", console=False)
    log_print("=" * 70, console=False)

    pbar_de = tqdm(total=maxiter_de, desc=f"Seed {seed} DE-Discrete", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    def callback_de(xk, convergence=None):
        nonlocal best_loss_so_far, best_solution_so_far

        iteration_count[0] += 1
        pbar_de.update(1)

        # Round parameters before evaluation
        xk_rounded = np.round(xk, decimals=DISCRETE_PRECISION)
        loss_current = objective_function_control(xk_rounded, fixed_materials, GreenFun, build_layers_func=build_layers)

        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk_rounded.copy()

        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk_rounded, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers
            )
            # Format eigenvalues for logging
            eigvals_str = ", ".join([f"λ{i+1}={re:+.3f}{im:+.3f}i" for i, (re, im) in enumerate(zip(real_parts, imag_parts))])

            # Show rounded parameters
            params_str = ", ".join([f"{val:.1f}" for val in xk_rounded[:3]])  # Show first 3 params

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

    # DE settings (increased popsize for discrete optimization)
    log_print(f"Using workers={n_workers}, popsize=150 (increased for discrete space)", console=verbose)

    result_de = differential_evolution(
        _objective_wrapper_discrete,  # Use discrete wrapper
        bounds,
        maxiter=maxiter_de,
        popsize=150,  # Increased for discrete space
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
        best_solution_so_far = np.round(result_de.x, decimals=DISCRETE_PRECISION)

    # Ensure final result is rounded
    final_x = np.round(best_solution_so_far, decimals=DISCRETE_PRECISION)

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        final_x, fixed_materials, GreenFun, return_details=True, build_layers_func=build_layers
    )

    theta0, Layers, C0 = build_layers(final_x, fixed_materials)

    output = "\n" + "="*70 + "\nFINAL RESULT (DISCRETE)\n" + "="*70 + "\n"
    output += f"Precision: {DISCRETE_PRECISION} decimal place(s)\n"
    output += f"Total Loss: {loss:.6e}\n"
    output += f"Spread (Variance): {spread:.6e}\n"
    output += f"Penalty Im: {pen_im:.6e}\n\n"

    output += "Eigenvalues:\n"
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        output += f"  lambda_{i+1} = {re:+.10f} {im:+.10f}i\n"

    output += "\nDegeneracy Check:\n"
    output += f"  Std(Re) = {np.std(real_parts):.6e}\n"
    output += f"  Std(Im) = {np.std(imag_parts):.6e}\n"
    output += f"  Min |Im(λ)| = {np.min(np.abs(imag_parts)):.4f} (constraint: |Im(λ)| >= {IMAG_MIN})\n"

    log_print(output, console=verbose)
    log_file.close()

    # Save results
    np.savez(os.path.join(output_dir, 'params.npz'),
             params=final_x,
             loss=loss,
             eigenvalues=eigvals,
             theta0=theta0,
             C0=C0,
             precision=DISCRETE_PRECISION)

    params_txt_path = os.path.join(output_dir, 'parameters_discrete.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Exceptional Point (EP3) Discrete Parameters - Seed {seed}\n")
        f.write(f"Precision: {DISCRETE_PRECISION} decimal place(s)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {loss:.15e}\n")
        f.write(f"Spread:      {spread:.15e}\n")
        f.write(f"Penalty Im:  {pen_im:.15e}\n")
        f.write(f"theta0 = {theta0:.1f} mrad\n")
        f.write("-" * 50 + "\n")
        layer_names = LAYER_NAMES
        thicknesses = [layer[1] for layer in Layers[:-1]]
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.1f} nm{resonant}\n")

        f.write("\nParameter Values (1 decimal precision):\n")
        for i, (name, val) in enumerate(zip(PARAM_NAMES, final_x)):
            f.write(f"  {name:8s} = {val:10.1f}\n")

        f.write("\nBounds Check:\n")
        all_ok = True
        for i, (val, (low, high)) in enumerate(zip(final_x, bounds)):
            status = "OK" if low <= val <= high else "VIOLATION"
            if status == "VIOLATION": all_ok = False
            f.write(f"  Param {i}: {val:.1f} [{low}, {high}] -> {status}\n")
        f.write(f"\nOverall Status: {'PASSED' if all_ok else 'FAILED'}\n")

    eigvals_txt_path = os.path.join(output_dir, 'eigenvalues.txt')
    with open(eigvals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Eigenvalue Analysis (Discrete) - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")

    plot_optimization_history(history, output_dir, seed, 'EP3 Discrete Optimizer: DE (0.1 step)')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, GreenFun, build_layers_func=build_layers, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-4,
        n_points=21,
        param_names=PARAM_NAMES,
        param_labels=PARAM_LABELS
    )

    return seed, loss, final_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EP3 Discrete Optimizer - 1 Decimal Place Precision')
    parser.add_argument('-i', '--iter', type=int, default=DEFAULT_DE_ITERATIONS*2,
                       help='DE iterations (default: 2x normal, for discrete space)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-w', '--workers', type=int, default=20, help='Number of workers for DE')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EP3 DISCRETE Optimizer (workers={args.workers}, seed={args.seed})")
    print(f"Algorithm: Differential Evolution ONLY (no L-BFGS-B)")
    print(f"Precision: {DISCRETE_PRECISION} decimal place (0.1 step size)")
    print(f"Matrix: -G - 0.5j*I | Constraint: |Im(λ)| >= {IMAG_MIN}")
    print(f"Structure: Pt / C / Fe* / C / Fe* / C / Fe* / C / Pt(substrate)")
    print("=" * 70)

    optimize_exceptional_point_discrete(
        maxiter_de=args.iter,
        seed=args.seed,
        n_workers=args.workers,
        verbose=True
    )
