"""
Exceptional Point (EP4) Optimizer using CMA-ES - NO CONSTRAINT VERSION
CMA-ES without |Im(λ)| >= 5.0 constraint (only minimize variance)
For testing whether constraint is too restrictive
"""

import numpy as np
import cma
import sys
import os
import importlib.util
from tqdm import tqdm

# Import shared configuration (EP4 version)
from config import (
    FIXED_MATERIALS, C0_FIXED, DEFAULT_SEEDS,
    PARAM_NAMES_EP4, PARAM_LABELS_EP4, LAYER_NAMES_EP4,
    BOUNDS_EP4, IMAG_MIN, IMAG_PENALTY,
    DEFAULT_CMA_ITERATIONS
)

# Import common functions (EP4 version)
from common_functions import build_layers_ep4, objective_function_control

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
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_cma, seed, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'ep4_cma_noconstraint_s{seed}_iter{maxiter_cma}')
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

    # --- CMA-ES Optimization ---
    log_print("=" * 70, console=False)
    log_print(f"CMA-ES Optimization - Seed {seed}", console=False)
    log_print("=" * 70, console=False)

    # Initial point: center of bounds
    x0 = np.array([(low + high) / 2.0 for low, high in bounds])

    # Initial standard deviation: 1/4 of the range
    sigma0 = np.array([(high - low) / 4.0 for low, high in bounds])
    sigma0_mean = np.mean(sigma0)

    # Extract bounds for CMA-ES
    bounds_lower = np.array([b[0] for b in bounds])
    bounds_upper = np.array([b[1] for b in bounds])

    # CMA-ES options
    opts = {
        'bounds': [bounds_lower, bounds_upper],
        'maxiter': maxiter_cma,
        'popsize': 30,  # Population size (default is 4 + 3*log(N))
        'seed': seed,
        'verb_disp': 0,  # Disable CMA-ES output
        'verb_log': 0,
        'verbose': -9,
    }

    log_print(f"Initial point (center): {x0[:3]}... (11 params)", console=verbose)
    log_print(f"Initial sigma: {sigma0_mean:.3f}", console=verbose)
    log_print("=" * 70, console=verbose)
    log_print("WARNING: Constraint |Im(λ)| >= 5.0 is DISABLED", console=verbose)
    log_print("Penalty weight set to 0 (only minimize variance)", console=verbose)
    log_print("=" * 70, console=verbose)

    # Define objective function wrapper - NO CONSTRAINT VERSION
    def objective_cma(params):
        # Call with penalty_weight=0 to disable constraint
        return objective_function_control(params, fixed_materials, GreenFun,
                                         build_layers_func=build_layers_ep4,
                                         penalty_weight=0)

    # Progress bar
    pbar = tqdm(total=maxiter_cma, desc=f"Seed {seed} CMA-ES", disable=not verbose)

    best_loss = float('inf')
    best_solution = None

    # Callback for logging
    iteration_count[0] = 0
    current_solutions = None
    current_fitness = None

    def callback_cma(es):
        nonlocal best_loss, best_solution, current_solutions, current_fitness
        iteration_count[0] += 1
        pbar.update(1)

        # Get best solution from current generation
        if current_fitness is not None and len(current_fitness) > 0:
            best_idx = np.argmin(current_fitness)
            current_best = current_fitness[best_idx]
            current_x = current_solutions[best_idx]

            if current_best < best_loss:
                best_loss = current_best
                best_solution = current_x.copy()
        else:
            # Fallback to es.result
            current_best = es.result.fbest
            current_x = es.result.xbest
            if current_best < best_loss:
                best_loss = current_best
                best_solution = current_x.copy()

        # Log every 10 iterations
        if iteration_count[0] % 10 == 0 or iteration_count[0] == 1:
            # Use best solution found so far
            if best_solution is not None:
                x_to_evaluate = best_solution
            else:
                x_to_evaluate = current_x

            loss, eigvals, real_parts, imag_parts, _, _, spread, pen_im = objective_function_control(
                x_to_evaluate, fixed_materials, GreenFun, return_details=True,
                build_layers_func=build_layers_ep4, penalty_weight=0
            )

            # Format eigenvalues for logging
            eigvals_str = ", ".join([f"λ{i+1}={re:+.3f}{im:+.3f}i" for i, (re, im) in enumerate(zip(real_parts, imag_parts))])
            msg = f"CMA Iter {iteration_count[0]:5d}: Loss={loss:.6e} (Spread={spread:.6e}, PenIm={pen_im:.2e}, sigma={es.sigma:.3e})\n"
            msg += f"  Eigenvalues: {eigvals_str}"
            if verbose: pbar.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            # Record to history
            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('CMA-ES')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    # Run CMA-ES
    try:
        es = cma.CMAEvolutionStrategy(x0, sigma0_mean, opts)

        while not es.stop():
            solutions = es.ask()
            current_solutions = solutions  # Store for callback
            fitness = [objective_cma(x) for x in solutions]
            current_fitness = fitness  # Store for callback
            es.tell(solutions, fitness)
            callback_cma(es)

            if iteration_count[0] >= maxiter_cma:
                break

        es_result = es.result
        final_x = es_result.xbest
        final_loss = es_result.fbest

    except Exception as e:
        log_print(f"CMA-ES error: {e}", console=True)
        if best_solution is not None:
            final_x = best_solution
            final_loss = best_loss
        else:
            final_x = x0
            final_loss = objective_cma(x0)

    pbar.close()

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        final_x, fixed_materials, GreenFun, return_details=True,
        build_layers_func=build_layers_ep4, penalty_weight=0
    )

    theta0, Layers, C0 = build_layers_ep4(final_x, fixed_materials)

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
        f.write(f"Exceptional Point (EP4) Parameters - CMA-ES - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {final_loss:.15e}\n")
        f.write(f"Spread:      {spread:.15e}\n")
        f.write(f"Penalty Im:  {pen_im:.15e}\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write("-" * 50 + "\n")
        layer_names = LAYER_NAMES_EP4
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
        f.write(f"Eigenvalue Analysis - CMA-ES - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")

    plot_optimization_history(history, output_dir, seed, 'EP4 Optimizer: CMA-ES')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=lambda p, fm, **kw: objective_function_control(p, fm, GreenFun, build_layers_func=build_layers_ep4, penalty_weight=0, **kw),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=0.5,
        n_points=21,
        param_names=PARAM_NAMES_EP4,
        param_labels=PARAM_LABELS_EP4
    )

    return seed, final_loss, final_x


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EP4 Optimizer - CMA-ES - NO CONSTRAINT VERSION')
    parser.add_argument('-i', '--iter', type=int, default=10000, help='CMA-ES iterations (default: 10000)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EP4 Optimizer - CMA-ES NO CONSTRAINT (seed={args.seed})")
    print(f"Algorithm: CMA-ES ({args.iter} iterations)")
    print(f"Matrix: -G - 0.5j*I | Constraint: DISABLED (only minimize variance)")
    print(f"Structure: Pt / (C/Fe*)×4 / C / Pt(substrate) - 4 resonant layers")
    print("=" * 70)

    optimize_exceptional_point(
        maxiter_cma=args.iter,
        seed=args.seed,
        verbose=True
    )
