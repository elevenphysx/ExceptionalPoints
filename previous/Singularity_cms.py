"""
Exceptional Point Finder (CMA-ES + Nelder-Mead Strategy)
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find parameters where all eigenvalues degenerate (lambda1 = lambda2 = lambda3)
        AND Absolute Real/Imaginary parts are large (> threshold).

Strategy:
    Stage 1: Global Search using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
             Best for non-convex, rugged landscapes and finding deep minima.
    Stage 2: Local Refinement using Nelder-Mead to polish the result.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm
import cma

# Import shared configuration
from config import (FIXED_MATERIALS, BOUNDS, C0_FIXED, DEFAULT_THRESHOLD,
                    DEFAULT_PENALTY_WEIGHT, DEFAULT_CMA_ITERATIONS,
                    DEFAULT_NM_ITERATIONS, DEFAULT_SEEDS,
                    DEFAULT_SCAN_RANGE, PARAM_NAMES, PARAM_LABELS, LAYER_NAMES)

# Import common functions
from common_functions import build_layers, objective_function as _objective_function

# Import plotting utilities
from plotting_utils import plot_optimization_history, scan_parameters_around_optimum

# Import from green function-new.py (with space in filename)
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# ============================================================
# Wrapper for objective_function with GreenFun
# ============================================================

def objective_function(params, fixed_materials, return_details=False, threshold=5.0, penalty_weight=100.0, bounds=None):
    """Wrapper that injects GreenFun into the common objective_function"""
    return _objective_function(params, fixed_materials, GreenFun, return_details, threshold, penalty_weight, bounds)

# ============================================================
# Main Optimization Logic
# ============================================================

def optimize_exceptional_point(maxiter_cma=DEFAULT_CMA_ITERATIONS, maxiter_nm=DEFAULT_NM_ITERATIONS,
                              seed=DEFAULT_SEEDS[0], threshold=DEFAULT_THRESHOLD, penalty_weight=DEFAULT_PENALTY_WEIGHT):
    """
    Main optimization routine using CMA-ES followed by Nelder-Mead.
    """
    np.random.seed(seed)

    output_dir = os.path.join('results', f'CMA-ES_s{seed}_i1-{maxiter_cma}_i2-{maxiter_nm}_thr{threshold:.1f}_pw{penalty_weight:.1f}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    print(f"Random seed: {seed}")

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS

    # Initialize log file
    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # Force UTF-8 output
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    log_file.write("Exceptional Point Optimization Log (CMA-ES)\n")
    log_file.write("=" * 70 + "\n\n")

    def log_print(message, end='\n'):
        print(message, end=end)
        log_file.write(message + end)
        log_file.flush()

    # History tracking
    history = {'iteration': [], 'loss': [], 'phase': [], 'eigvals_real': [], 'eigvals_imag': []}
    iteration_count = [0]

    # -------------------------------------------------------------------------
    # Phase 1: Global Search with CMA-ES
    # -------------------------------------------------------------------------
    log_print("=" * 70)
    log_print(f"Phase 1: CMA-ES Optimization (Max Iterations: {maxiter_cma})")
    log_print("Target: Find EP with |Re| > threshold and |Im| > threshold")
    log_print("=" * 70)

    # 1. Prepare bounds for CMA (needs list of lows and list of highs)
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    
    # 2. Generate random initial guess within bounds
    x0 = np.array([np.random.uniform(l, u) for l, u in bounds])
    
    # 3. Initialize CMA-ES
    # sigma0=1.0 is a heuristic for normalized parameter space, adequate given bounds.
    es = cma.CMAEvolutionStrategy(x0, 1.0, {
        'bounds': [lower_bounds, upper_bounds],
        'seed': seed,
        'popsize': 40,  # Increased population size for better exploration
        'maxiter': maxiter_cma,
        'verbose': -9 # Suppress internal logging, we will do it manually
    })

    # Progress bar for CMA
    pbar_cma = tqdm(total=maxiter_cma, desc="CMA-ES Progress",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   position=0, leave=True)

    best_solution_so_far = None
    best_loss_so_far = float('inf')

    # CMA-ES Loop
    while not es.stop():
        iteration_count[0] += 1
        solutions = es.ask() # Ask for candidate solutions
        
        # Evaluate solutions using our objective function
        fitnesses = [objective_function(x, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight) for x in solutions]
        
        es.tell(solutions, fitnesses) # Tell CMA the results
        
        # Tracking best result in this generation
        current_best_idx = np.argmin(fitnesses)
        current_best_loss = fitnesses[current_best_idx]
        current_best_x = solutions[current_best_idx]

        if current_best_loss < best_loss_so_far:
            best_loss_so_far = current_best_loss
            best_solution_so_far = current_best_x

        pbar_cma.update(1)

        # Logging logic (every 10 iterations)
        if iteration_count[0] % 10 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
                current_best_x, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )
            
            msg = f"CMA-ES Gen {iteration_count[0]}: Loss={loss:.6e} (Degen={loss_degeneracy:.6e}, Pen_Re={penalty_real:.6e}, Pen_Im={penalty_imag:.6e})"
            pbar_cma.write(msg)
            log_file.write(msg + "\n")
            
            # Record history
            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('CMA-ES')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())
            
            # Detailed check every 50 gens
            if iteration_count[0] % 50 == 0:
                output = "  Current Best Eigenvalues:\n"
                for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                    output += f"    λ_{i+1} = {re:+10.5f} {im:+10.5f}i\n"
                pbar_cma.write(output)
                log_file.write(output)

    pbar_cma.close()
    
    # Get final best solution from CMA
    result_cma_x = es.result.xbest
    result_cma_loss = es.result.fbest
    
    log_print("\n" + "=" * 70)
    log_print(f"CMA-ES Finished. Best Loss = {result_cma_loss:.6e}")
    log_print("=" * 70)

    # -------------------------------------------------------------------------
    # Phase 2: Local Refinement (Nelder-Mead)
    # -------------------------------------------------------------------------
    log_print("\n" + "=" * 70)
    log_print("Phase 2: Local Refinement (Nelder-Mead)")
    log_print(f"Starting from CMA-ES result")
    log_print("=" * 70)

    iteration_offset = iteration_count[0]
    
    # Progress bar for NM
    pbar_local = tqdm(total=maxiter_nm, desc="Nelder-Mead",
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                     position=0, leave=True)
    
    nm_iter = [0]
    
    def local_callback(xk):
        nm_iter[0] += 1
        pbar_local.update(1)
        
        if nm_iter[0] % 50 == 0:
            loss, eigvals, real_parts, imag_parts, _, _, loss_deg, (_, p_im) = objective_function(
                xk, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )
            msg = f"NM Iter {nm_iter[0]}: Loss={loss:.6e} (Degen={loss_deg:.6e})"
            pbar_local.write(msg)
            log_file.write(msg + "\n")
            
            global_iter = iteration_offset + nm_iter[0]
            history['iteration'].append(global_iter)
            history['loss'].append(loss)
            history['phase'].append('Nelder-Mead')
            if eigvals is not None:
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    # Optimize wrapper
    def objective_wrapper(params):
        return objective_function(params, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight)

    try:
        result_nm = minimize(
            objective_wrapper,
            result_cma_x,
            method='Nelder-Mead',
            bounds=None, # Nelder-Mead in scipy doesn't strictly support bounds, but we check inside objective
            callback=local_callback,
            options={'maxiter': maxiter_nm, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
        )
        pbar_local.close()
        log_print(f"\nNelder-Mead finished. Final Loss: {result_nm.fun:.6e}")
        final_result_x = result_nm.x
        final_loss = result_nm.fun
        
    except Exception as e:
        log_print(f"Nelder-Mead failed: {e}")
        final_result_x = result_cma_x
        final_loss = result_cma_loss

    # -------------------------------------------------------------------------
    # Final Reporting
    # -------------------------------------------------------------------------
    output = "\n" + "=" * 70 + "\n"
    output += "FINAL RESULT:\n"
    output += "=" * 70 + "\n"

    loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
        final_result_x, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
    )

    theta0, Layers, C0 = build_layers(final_result_x, fixed_materials)

    output += f"\nTotal Loss = {loss:.6e}\n"
    output += f"  ├─ Degeneracy Loss (Pairwise) = {loss_degeneracy:.6e}\n"
    output += f"  ├─ Penalty (Re < {threshold}) = {penalty_real:.6e}\n"
    output += f"  └─ Penalty (Im < {threshold}) = {penalty_imag:.6e}\n\n"
    output += f"theta0 = {theta0:.15f} mrad\n"
    output += f"C0     = {C0:.15f} (fixed)\n\n"

    output += "Layer Structure (nm):\n"
    layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Pt(sub)']
    for i, (name, layer) in enumerate(zip(layer_names, Layers)):
        thickness = layer[1]
        resonant = ' (resonant)' if layer[2] == 1 else ''
        output += f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n"

    output += "\nEigenvalue Analysis:\n"
    output += "  λᵢ = Re + Im·i\n"
    output += "  " + "-" * 50 + "\n"
    for i, (eig, re, im) in enumerate(zip(eigvals, real_parts, imag_parts)):
        output += f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n"
    
    # Check bounds satisfaction
    min_re = np.min(np.abs(real_parts))
    min_im = np.min(np.abs(imag_parts))
    output += "\n  Constraint Check:\n"
    output += f"    Min|Re| = {min_re:.4f} > {threshold}? {'YES' if min_re > threshold else 'NO'}\n"
    output += f"    Min|Im| = {min_im:.4f} > {threshold}? {'YES' if min_im > threshold else 'NO'}\n"

    print(output, end='')
    log_file.write(output)

    # Save Matrix
    output_mat = "\nMatrix G1:\n" + str(G1)
    log_file.write(output_mat)
    log_file.close()

    # Plot
    plot_optimization_history(history, output_dir, seed, 'CMA-ES + Nelder-Mead')
    print(f"\nLog saved to: {log_file_path}")

    # Return result object-like structure for compatibility
    class Result:
        pass
    res = Result()
    res.x = final_result_x
    res.fun = final_loss

    return res, theta0, Layers, C0, output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (CMA-ES)')
    parser.add_argument('-i1', '--iterations-cma', type=int, default=DEFAULT_CMA_ITERATIONS,
                        help='Number of generations for CMA-ES')
    parser.add_argument('-i2', '--iterations-nm', type=int, default=DEFAULT_NM_ITERATIONS,
                        help='Number of iterations for Nelder-Mead')
    parser.add_argument('-s', '--seed', type=int, default=DEFAULT_SEEDS[0],
                        help='Random seed for reproducibility')
    parser.add_argument('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Constraint threshold for |Re| and |Im|')
    parser.add_argument('-pw', '--penalty-weight', type=float, default=DEFAULT_PENALTY_WEIGHT,
                        help='Penalty weight for constraint violations')
    args = parser.parse_args()

    result, theta0, Layers, C0, output_dir = optimize_exceptional_point(
        maxiter_cma=args.iterations_cma,
        maxiter_nm=args.iterations_nm,
        seed=args.seed,
        threshold=args.threshold,
        penalty_weight=args.penalty_weight
    )

    # Extract layer thicknesses for saving
    thicknesses = [layer[1] for layer in Layers[:-1]]

    params_path = os.path.join(output_dir, 'exceptional_point_params.npz')
    np.savez(params_path,
             params=result.x,
             theta0=theta0,
             C0=C0,
             thicknesses=thicknesses,
             loss=result.fun)
    print(f"\nParameters saved to: {params_path}")

    # Detailed text save
    params_txt_path = os.path.join(output_dir, 'exceptional_point_params.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Final Loss: {result.fun}\n")
        f.write(f"Params: {result.x}\n")

    # Run parameter sensitivity scan
    scan_parameters_around_optimum(
        params_optimal=result.x,
        objective_func=objective_function,
        fixed_materials=FIXED_MATERIALS,
        output_dir=output_dir,
        scan_range=DEFAULT_SCAN_RANGE,
        n_points=21,
        threshold=args.threshold,
        penalty_weight=args.penalty_weight
    )