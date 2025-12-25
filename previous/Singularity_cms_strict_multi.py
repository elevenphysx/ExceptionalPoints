import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm
import cma
import concurrent.futures
import time
import argparse

# Import shared configuration
from config import (FIXED_MATERIALS, BOUNDS, C0_FIXED, DEFAULT_THRESHOLD,
                    DEFAULT_PENALTY_WEIGHT, DEFAULT_CMA_ITERATIONS,
                    DEFAULT_NM_ITERATIONS, DEFAULT_SEEDS,
                    DEFAULT_SCAN_RANGE, PARAM_NAMES, PARAM_LABELS, LAYER_NAMES)

# Import common functions
from common_functions import build_layers, objective_function as _objective_function

# Import plotting utilities
from plotting_utils import plot_optimization_history, scan_parameters_around_optimum

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
# Wrapper for objective_function with GreenFun
# ============================================================

def objective_function(params, fixed_materials, bounds=None, return_details=False, threshold=5.0, penalty_weight=100.0):
    """Wrapper that injects GreenFun into the common objective_function"""
    return _objective_function(params, fixed_materials, GreenFun, return_details, threshold, penalty_weight, bounds)

# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_cma, maxiter_powell, seed, threshold, penalty_weight, verbose=True):
    np.random.seed(seed)
    
    # Unique output directory for this seed
    output_dir = os.path.join('results', f'StrictBounds_s{seed}_thr{threshold:.1f}_pw{penalty_weight:.1f}')
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS

    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    def log_print(msg, console=True):
        if console and verbose:
            print(f"[Seed {seed}] {msg}")
        log_file.write(msg + "\n")
        log_file.flush()

    history = {'iteration': [], 'loss': [], 'phase': [], 'eigvals_real': [], 'eigvals_imag': []}
    iteration_count = [0]

    # --- Phase 1: CMA-ES ---
    log_print("=" * 70, console=False)
    log_print(f"Phase 1: CMA-ES Optimization (Seed: {seed})", console=False)
    log_print("=" * 70, console=False)

    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]
    x0 = np.array([np.random.uniform(l, u) for l, u in bounds])

    # CMA-ES with bounds handling (using internal BoundTransform implicitly via 'bounds' arg)
    es = cma.CMAEvolutionStrategy(x0, 1.0, {
        'bounds': [lower_bounds, upper_bounds],
        'seed': seed,
        'popsize': 40,
        'maxiter': maxiter_cma,
        'verbose': -9,
        'tolfun': 1e-20,
        'tolx': 1e-20,
    })

    pbar_cma = tqdm(total=maxiter_cma, desc=f"Seed {seed} CMA-ES", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    while not es.stop():
        iteration_count[0] += 1
        solutions = es.ask()
        
        # Pass bounds to objective function for "Soft Barrier" calculation just in case
        fitnesses = [objective_function(x, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight) for x in solutions]
        es.tell(solutions, fitnesses)
        
        current_best_idx = np.argmin(fitnesses)
        if fitnesses[current_best_idx] < best_loss_so_far:
            best_loss_so_far = fitnesses[current_best_idx]
            best_solution_so_far = solutions[current_best_idx]

        pbar_cma.update(1)

        if iteration_count[0] % 10 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, loss_deg, (pen_re, pen_im) = objective_function(
                best_solution_so_far, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )
            msg = f"CMA Gen {iteration_count[0]}: Loss={loss:.6e} (Deg={loss_deg:.6e}, PenRe={pen_re:.2e}, PenIm={pen_im:.2e})"
            if verbose: pbar_cma.write(msg)
            log_file.write(msg + "\n")
            
            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('CMA-ES')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

    pbar_cma.close()
    result_cma_x = es.result.xbest

    # Ensure CMA result is within bounds before passing to Phase 2
    # (Sometimes CMA can be slightly out by 1e-10)
    result_cma_x = np.clip(result_cma_x, lower_bounds, upper_bounds)

    # --- Phase 2: Powell Method with STRICT Bounds ---
    log_print("\n" + "=" * 70, console=False)
    log_print("Phase 2: Powell Refinement (Strict Bounds)", console=False)
    log_print("=" * 70, console=False)

    def obj_wrapper(p):
        return objective_function(p, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight)

    powell_iter = [0]
    pbar_pw = tqdm(total=maxiter_powell, desc=f"Seed {seed} Powell", disable=not verbose)

    def pw_callback(xk):
        powell_iter[0] += 1
        pbar_pw.update(1)
        if powell_iter[0] % 50 == 0:
            loss = obj_wrapper(xk)
            msg = f"Powell Iter {powell_iter[0]}: Loss={loss:.6e}"
            if verbose: pbar_pw.write(msg)
            log_file.write(msg + "\n")

    try:
        # KEY CHANGE: Using 'Powell' with 'bounds'.
        # This is gradient-free (safe for physics) and supports bounds (safe for thickness).
        res_pw = minimize(obj_wrapper, result_cma_x, method='Powell', 
                         bounds=bounds,  # Strict bounds enforcement
                         callback=pw_callback, 
                         options={'maxiter': maxiter_powell, 'ftol': 1e-10, 'xtol': 1e-10})
        final_x = res_pw.x
        final_loss = res_pw.fun
        pbar_pw.close()
    except Exception as e:
        print(f"Optimizer Warning: {e}")
        final_x = result_cma_x
        final_loss = best_loss_so_far

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G1, loss_deg, (pen_re, pen_im) = objective_function(
        final_x, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
    )
    
    theta0, Layers, C0 = build_layers(final_x, fixed_materials)
    
    output = "\n" + "="*70 + "\nFINAL RESULT\n" + "="*70 + "\n"
    output += f"Total Loss: {loss:.6e}\n"
    output += f"Degeneracy Loss: {loss_deg:.6e}\n"
    output += f"Penalty Re: {pen_re:.6e}\n"
    output += f"Penalty Im: {pen_im:.6e}\n\n"
    
    output += "Eigenvalues:\n"
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        output += f"  lambda_{i+1} = {re:+.10f} {im:+.10f}i\n"
        
    output += "\nConstraint Check:\n"
    output += f"  Min|Re| = {np.min(np.abs(real_parts)):.4f} > {threshold}? {'YES' if pen_re==0 else 'NO'}\n"
    output += f"  Min|Im| = {np.min(np.abs(imag_parts)):.4f} > {threshold}? {'YES' if pen_im==0 else 'NO'}\n"

    log_print(output, console=verbose)
    log_file.close()

    # Save results
    np.savez(os.path.join(output_dir, 'params.npz'),
             params=final_x,
             loss=final_loss,
             eigenvalues=eigvals,
             theta0=theta0,
             C0=C0)

    # Save high-precision text
    params_txt_path = os.path.join(output_dir, 'parameters_high_precision.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Exceptional Point Parameters (Strict Bounds) - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {final_loss:.15e}\n")
        f.write(f"Degeneracy:  {loss_deg:.15e}\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write("-" * 50 + "\n")
        layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']
        thicknesses = [layer[1] for layer in Layers[:-1]] 
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n")
        
        # Verify bounds in log
        f.write("\nBounds Check:\n")
        all_ok = True
        for i, (val, (low, high)) in enumerate(zip(final_x, bounds)):
            status = "OK" if low <= val <= high else "VIOLATION"
            if status == "VIOLATION": all_ok = False
            f.write(f"  Param {i}: {val:.4f} [{low}, {high}] -> {status}\n")
        f.write(f"\nOverall Status: {'PASSED' if all_ok else 'FAILED'}\n")

    # Save Eigenvalues
    eigvals_txt_path = os.path.join(output_dir, 'eigenvalues.txt')
    with open(eigvals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Eigenvalue Analysis - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")

    plot_optimization_history(history, output_dir, seed, 'CMA-ES + Powell')

    scan_parameters_around_optimum(
        params_optimal=final_x,
        objective_func=objective_function,
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=DEFAULT_SCAN_RANGE,
        n_points=21,
        threshold=threshold,
        penalty_weight=penalty_weight
    )

    return seed, final_loss, final_x

# ============================================================
# Parallel Execution Helpers
# ============================================================

def run_single_seed(args):
    seed, max_cma, max_pw, threshold, penalty_weight = args
    try:
        _, loss, final_x = optimize_exceptional_point(
            maxiter_cma=max_cma,
            maxiter_powell=max_pw,
            seed=seed,
            threshold=threshold,
            penalty_weight=penalty_weight,
            verbose=False 
        )
        return seed, loss, final_x, "Success"
    except Exception as e:
        return seed, None, None, str(e)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Strict Bounds)')
    parser.add_argument('-i1', type=int, default=DEFAULT_CMA_ITERATIONS, help='CMA-ES iterations')
    parser.add_argument('-i2', type=int, default=DEFAULT_NM_ITERATIONS, help='Powell iterations')
    parser.add_argument('-t', type=float, default=DEFAULT_THRESHOLD, help='Threshold')
    parser.add_argument('-pw', type=float, default=DEFAULT_PENALTY_WEIGHT, help='Penalty Weight')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Workers')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=DEFAULT_SEEDS, help='Seeds')

    args = parser.parse_args()

    print("=" * 70)
    print(f"Starting Parallel Optimization (Strict Bounds Mode)")
    print(f"Algorithm: CMA-ES (Global) -> Powell (Local + Hard Bounds)")
    print("=" * 70)

    task_args = [(seed, args.i1, args.i2, args.t, args.pw) for seed in args.seeds]
    results = []

    fixed_materials = FIXED_MATERIALS

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, t): t[0] for t in task_args}
        for future in concurrent.futures.as_completed(futures):
            seed_val = futures[future]
            try:
                seed, loss, final_x, status = future.result()
                if status == "Success":
                    print(f"✅ Seed {seed} Finished | Loss: {loss:.6e}")
                    results.append((seed, loss, final_x))
                else:
                    print(f"❌ Seed {seed} Failed | {status}")
            except Exception as e:
                print(f"❌ Seed {seed_val} Exception: {e}")

    print("\n" + "=" * 70)
    results.sort(key=lambda x: x[1])
    
    for s, l, x in results:
        theta0, Layers, C0 = build_layers(x, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)
        re = np.real(eigvals)
        im = np.imag(eigvals)
        idx = np.argsort(re)
        
        print("-" * 60)
        print(f"Seed {s} | Loss = {l:.6e}")
        for k in range(3):
             print(f"   λ_{k+1} = {re[idx[k]]:8.5f} {im[idx[k]]:+8.5f}j")
             
    if results:
        print(f"\n🏆 BEST SEED: {results[0][0]}")