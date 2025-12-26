"""
Exceptional Point Finder - DE + L-BFGS-B (Control_N Algorithm, -G-0.5i Matrix)
Uses Control_N.py's objective function (variance-based) and constraints
With shifted matrix: -G - 0.5j*I and constraint |Im(λ)| >= 5
With multi-core parallel execution and plotting utilities
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import sys
import os
import importlib.util
from tqdm import tqdm
import concurrent.futures
import time
import argparse

# Import shared configuration
from config import FIXED_MATERIALS, C0_FIXED, DEFAULT_SEEDS, PARAM_NAMES, PARAM_LABELS, LAYER_NAMES

# Import common functions
from common_functions import build_layers

# Import plotting utilities
from plotting_utils import plot_optimization_history, scan_parameters_around_optimum

# ============================================================
# Control_N Configuration
# ============================================================

# Use Control_N bounds
BOUNDS = [
    (2.0, 8.0),    # theta0 (mrad)
    (0.5, 4.0),    # Pt
    (1.0, 40.0),   # C
    (0.8, 3.0),    # Fe*
    (1.0, 40.0),   # C
    (0.5, 3.0),    # Fe*
    (1.0, 40.0),   # C
    (0.5, 3.0),    # Fe*
    (1.0, 40.0),   # C
]

# BOUNDS = [
#         (2.0, 10.0),     # theta0 (mrad)
#         (3.0, 10.0),     # Pt thickness (nm)
#         (5.0, 50.0),     # C layer 1
#         (0.5, 3.0),      # Fe layer 1 (resonant)
#         (5.0, 50.0),     # C layer 2
#         (0.5, 3.0),      # Fe layer 2 (resonant)
#         (5.0, 50.0),     # C layer 3
#         (0.5, 3.0),      # Fe layer 3 (resonant)
#         (5.0, 50.0),     # C layer 4
#     ]


IMAG_MIN = 5.0
IMAG_PENALTY = 1e4

# DE settings matching Control_N
DEFAULT_DE_ITERATIONS = 100000
DEFAULT_LBFGSB_ITERATIONS = 5000

# ============================================================
# Dynamic Import of Green Function
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
green_script_path = os.path.join(current_dir, "green function-new.py")

if not os.path.exists(green_script_path):
    green_script_path = "green function-new.py"

try:
    spec = importlib.util.spec_from_file_location("green_function_new", green_script_path)
    green_function_new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(green_function_new)
    GreenFun = green_function_new.GreenFun
except Exception as e:
    print(f"Error importing GreenFun: {e}")
    print("Make sure 'green function-new.py' is in the same directory.")
    sys.exit(1)

# ============================================================
# Control_N Style Objective Function (Variance-based)
# ============================================================

def objective_function_control(params, fixed_materials, return_details=False):
    """
    Variance-based loss matching Control_N.py
    Loss = Σ|λ - mean(λ)|² where λ are eigenvalues of -G - 0.5j*I
    Constraint: |Im(λ)| >= IMAG_MIN (absolute value, since eigenvalues shifted to negative)
    """
    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)

        # Use -G - 0.5j*I (shifted matrix)
        I = np.eye(G.shape[0])
        G_shifted = -G - 0.5j * I
        eigvals = np.linalg.eigvals(G_shifted)

        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            if return_details:
                return 1e10, eigvals, np.zeros(3), np.zeros(3), G_shifted, G_shifted, 1e10, 0.0
            return 1e10

        # Variance-based loss (Control_N style)
        mean_eig = np.mean(eigvals)
        diff = eigvals - mean_eig
        spread = np.sum(diff.real**2 + diff.imag**2)

        # |Im(λ)| >= IMAG_MIN constraint (absolute value)
        imag_parts = np.imag(eigvals)
        min_abs_im = np.min(np.abs(imag_parts))

        penalty_imag = 0.0
        if min_abs_im < IMAG_MIN:
            penalty_imag = IMAG_PENALTY * (IMAG_MIN - min_abs_im) ** 2

        loss = spread + penalty_imag

        if return_details:
            real_parts = np.real(eigvals)
            return loss, eigvals, real_parts, imag_parts, G_shifted, G_shifted, spread, penalty_imag
        return loss

    except Exception as e:
        if return_details:
            return 1e10, None, None, None, None, None, None, None
        return 1e10


# ============================================================
# Global wrapper for multiprocessing
# ============================================================
_global_fixed_materials = None

def _objective_wrapper(p):
    """Global wrapper for DE multiprocessing"""
    return objective_function_control(p, _global_fixed_materials)


# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_de, maxiter_lbfgsb, seed, verbose=True):
    from functools import partial

    np.random.seed(seed)

    output_dir = os.path.join('results', f'DE_Control_Gshift_s{seed}_DE{maxiter_de}_LB{maxiter_lbfgsb}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS

    # Create a partial function that embeds fixed_materials (works with multiprocessing)
    objective_for_de = partial(objective_function_control, fixed_materials=fixed_materials)

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
    log_print(f"Phase 1: DE Optimization (Control_N Algorithm) - Seed {seed}", console=False)
    log_print("=" * 70, console=False)

    pbar_de = tqdm(total=maxiter_de, desc=f"Seed {seed} DE", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    def callback_de(xk, convergence=None):
        nonlocal best_loss_so_far, best_solution_so_far

        iteration_count[0] += 1
        pbar_de.update(1)

        loss_current = objective_function_control(xk, fixed_materials)

        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk.copy()

        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                xk, fixed_materials, return_details=True
            )
            msg = f"DE Iter {iteration_count[0]:5d}: Loss={loss:.6e} (Spread={spread:.6e}, PenIm={pen_im:.2e}, convergence={convergence if convergence else 0:.3e})"
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
    result_de = differential_evolution(
        objective_for_de,
        bounds,
        maxiter=maxiter_de,
        popsize=25,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        tol=0,
        atol=0,
        updating='deferred',
        workers=1,
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
                xk, fixed_materials, return_details=True
            )
            msg = f"L-BFGS-B Iter {lbfgsb_iter[0]}: Loss={loss:.6e}"
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
            lambda p: objective_function_control(p, fixed_materials),
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
    loss, eigvals, real_parts, imag_parts, G, _, spread, pen_im = objective_function_control(
        final_x, fixed_materials, return_details=True
    )

    theta0, Layers, C0 = build_layers(final_x, fixed_materials)

    output = "\n" + "="*70 + "\nFINAL RESULT\n" + "="*70 + "\n"
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
             loss=final_loss,
             eigenvalues=eigvals,
             theta0=theta0,
             C0=C0)

    params_txt_path = os.path.join(output_dir, 'parameters_high_precision.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Exceptional Point Parameters (Control_N Algorithm, -G-0.5i Matrix) - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {final_loss:.15e}\n")
        f.write(f"Spread:      {spread:.15e}\n")
        f.write(f"Penalty Im:  {pen_im:.15e}\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write("-" * 50 + "\n")
        layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']
        thicknesses = [layer[1] for layer in Layers[:-1]]
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n")

        f.write("\nBounds Check:\n")
        all_ok = True
        for i, (val, (low, high)) in enumerate(zip(final_x, bounds)):
            status = "OK" if low <= val <= high else "VIOLATION"
            if status == "VIOLATION": all_ok = False
            f.write(f"  Param {i}: {val:.4f} [{low}, {high}] -> {status}\n")
        f.write(f"\nOverall Status: {'PASSED' if all_ok else 'FAILED'}\n")

    eigvals_txt_path = os.path.join(output_dir, 'eigenvalues.txt')
    with open(eigvals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Eigenvalue Analysis - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")

    plot_optimization_history(history, output_dir, seed, 'DE (Control_N, -G-0.5i) + L-BFGS-B')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=0.5,
        n_points=21
    )

    return seed, final_loss, final_x


# ============================================================
# Parallel Execution Helpers
# ============================================================

def run_single_seed(args):
    seed, max_de, max_lbfgsb = args
    try:
        _, loss, final_x = optimize_exceptional_point(
            maxiter_de=max_de,
            maxiter_lbfgsb=max_lbfgsb,
            seed=seed,
            verbose=False
        )
        return seed, loss, final_x, "Success"
    except Exception as e:
        return seed, None, None, str(e)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Control_N Algorithm)')
    parser.add_argument('-i1', type=int, default=DEFAULT_DE_ITERATIONS, help='DE iterations')
    parser.add_argument('-i2', type=int, default=DEFAULT_LBFGSB_ITERATIONS, help='L-BFGS-B iterations')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Workers')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='Seeds')

    args = parser.parse_args()

    task_args = [(seed, args.i1, args.i2) for seed in args.seeds]

    print("=" * 70)
    print(f"Starting Parallel Optimization (Control_N Algorithm, -G-0.5i Matrix)")
    print(f"Algorithm: DE (Control_N Settings) -> L-BFGS-B")
    print(f"Matrix: -G - 0.5j*I (shifted eigenvalues)")
    print(f"DE Settings: maxiter={args.i1}, popsize=25, updating='deferred'")
    print(f"Parallelization: {len(args.seeds)} seeds across {args.workers if args.workers else 'all'} cores")
    print(f"Constraint: |Im(λ)| >= {IMAG_MIN}")
    print("=" * 70)

    start_time = time.time()
    fixed_materials = FIXED_MATERIALS

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, t): t[0] for t in task_args}

        for future in concurrent.futures.as_completed(futures):
            seed_val = futures[future]
            try:
                seed, loss, final_x, status = future.result()
                if status == "Success":
                    print(f"✅ Seed {seed} Finished | Final Loss: {loss:.6e}")
                    results.append((seed, loss, final_x))
                else:
                    print(f"❌ Seed {seed} Failed | Error: {status}")
            except Exception as exc:
                print(f"❌ Seed {seed_val} generated an exception: {exc}")

    print("\n" + "=" * 70)
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print("DETAILED SUMMARY (Sorted by Loss):")

    results.sort(key=lambda x: x[1])

    for s, l, x in results:
        theta0, Layers, C0 = build_layers(x, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        eigvals = np.linalg.eigvals(G)
        re = np.real(eigvals)
        im = np.imag(eigvals)
        idx = np.argsort(re)
        re = re[idx]
        im = im[idx]

        print("-" * 60)
        print(f"Seed {s} | Loss = {l:.6e}")
        for k in range(3):
            print(f"   lambda_{k+1} = {re[k]:8.5f} {im[k]:+8.5f}j")

    if results:
        print("\n" + "=" * 70)
        print(f"🏆 BEST SEED: {results[0][0]}")
