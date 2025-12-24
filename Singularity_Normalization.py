"""
Exceptional Point Finder (Parallel CMA-ES + Nelder-Mead Strategy)
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find parameters where all eigenvalues degenerate (lambda1 = lambda2 = lambda3)
        AND Absolute Real/Imaginary parts are large (> threshold).

Strategy:
    Stage 1: Global Search using CMA-ES.
    Stage 2: Local Refinement using Nelder-Mead.
    Parallel: Runs multiple seeds simultaneously on different CPU cores.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm
import cma  # Requires: pip install cma
import concurrent.futures # For parallel processing
import time

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
# Core Physics & Objective Functions
# ============================================================

def build_layers(params, fixed_materials):
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = 7.74 * 1.06 * 0.5  # Fixed value (CFe)
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

def objective_function(params, fixed_materials, bounds=None, return_details=False, threshold=5.0, penalty_weight=100.0):
    """
    Revised Objective function.
    """
    
    # Helper: Pairwise Squared Difference Sum
    def pairwise_diff_sq(values):
        diff_sum = 0.0
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                diff_sum += (values[i] - values[j]) ** 2
        return diff_sum

    try:
        # Check bounds
        if bounds is not None:
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                if not (low <= param <= high):
                    if return_details: return 1e10, None, None, None, None, None, None, None
                    else: return 1e10

        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)
        
        # Stability check
        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            if return_details: return 1e10, eigvals, np.zeros(3), np.zeros(3), G, G1, 1e10, (1e10, 1e10)
            return 1e10

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        
        loss_degeneracy = pairwise_diff_sq(real_parts) + pairwise_diff_sq(imag_parts)

        min_re = np.min(np.abs(real_parts))
        if min_re < threshold:
            gap_re = threshold - min_re
            penalty_real = (np.exp(gap_re) - 1.0)
        else:
            penalty_real = 0.0

        min_im = np.min(np.abs(imag_parts))
        if min_im < threshold:
            gap_im = threshold - min_im
            penalty_imag = (np.exp(gap_im) - 1.0)
        else:
            penalty_imag = 0.0
        
        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag)
        return loss

    except Exception as e:
        if return_details: return 1e10, None, None, None, None, None, None, None
        return 1e10

# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_cma=200, maxiter_nm=500, seed=812, threshold=5.0, penalty_weight=100.0, verbose=True):
    np.random.seed(seed)
    
    output_dir = os.path.join('results', f'CMA_Exact_s{seed}_thr{threshold:.1f}_pw{penalty_weight:.1f}')
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")
    
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    bounds = [
        (0.5, 10.0), (0.0, 10.0), (0.0, 50.0), (0.5, 10.0),
        (0.1, 50.0), (0.5, 10.0), (0.1, 50.0), (0.5, 10.0), (0.1, 50.0)
    ]

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
    
    es = cma.CMAEvolutionStrategy(x0, 1.0, {
        'bounds': [lower_bounds, upper_bounds],
        'seed': seed,
        'popsize': 40,
        'maxiter': maxiter_cma,
        'verbose': -9
    })

    # Disable tqdm if verbose is False (to keep parallel runs clean)
    pbar_cma = tqdm(total=maxiter_cma, desc=f"Seed {seed} CMA-ES", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    while not es.stop():
        iteration_count[0] += 1
        solutions = es.ask()
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

    # --- Phase 2: Nelder-Mead ---
    log_print("\n" + "=" * 70, console=False)
    log_print("Phase 2: Nelder-Mead Refinement", console=False)
    log_print("=" * 70, console=False)

    def obj_wrapper(p):
        return objective_function(p, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight)

    nm_iter = [0]
    pbar_nm = tqdm(total=maxiter_nm, desc=f"Seed {seed} Nelder-Mead", disable=not verbose)

    def nm_callback(xk):
        nm_iter[0] += 1
        pbar_nm.update(1)
        if nm_iter[0] % 50 == 0:
            loss = obj_wrapper(xk)
            msg = f"NM Iter {nm_iter[0]}: Loss={loss:.6e}"
            if verbose: pbar_nm.write(msg)
            log_file.write(msg + "\n")

    try:
        res_nm = minimize(obj_wrapper, result_cma_x, method='Nelder-Mead', callback=nm_callback, 
                         options={'maxiter': maxiter_nm, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        final_x = res_nm.x
        final_loss = res_nm.fun
        pbar_nm.close()
    except:
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
        output += f"  λ_{i+1} = {re:+.10f} {im:+.10f}i\n"
        
    output += "\nConstraint Check:\n"
    output += f"  Min|Re| = {np.min(np.abs(real_parts)):.4f} > {threshold}? {'YES' if pen_re==0 else 'NO'}\n"
    output += f"  Min|Im| = {np.min(np.abs(imag_parts)):.4f} > {threshold}? {'YES' if pen_im==0 else 'NO'}\n"

    log_print(output, console=verbose)
    log_file.close()
    
    np.savez(os.path.join(output_dir, 'params.npz'), params=final_x, loss=final_loss)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(history['iteration'], history['loss'], 'b.-')
        ax.set_title(f'Optimization History (Seed {seed})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True)
        plt.savefig(os.path.join(output_dir, 'history.png'))
        plt.close()
    except Exception: pass

    return seed, final_loss

# ============================================================
# Parallel Execution Helpers
# ============================================================

def run_single_seed(args):
    """Wrapper function to run a single seed optimization"""
    # Unpack including the new verbose flag
    seed, max_cma, max_nm, threshold, penalty_weight, verbose = args
    try:
        _, loss = optimize_exceptional_point(
            maxiter_cma=max_cma,
            maxiter_nm=max_nm,
            seed=seed,
            threshold=threshold,
            penalty_weight=penalty_weight,
            verbose=verbose  # Pass the flag down
        )
        return seed, loss, "Success"
    except Exception as e:
        return seed, None, str(e)

if __name__ == "__main__":
    # Prevent numpy from using multiple threads per process
    os.environ["OMP_NUM_THREADS"] = "1"

    import argparse
    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Parallel CMA-ES)')
    parser.add_argument('-i1', type=int, default=300, help='CMA-ES iterations')
    parser.add_argument('-i2', type=int, default=500, help='Nelder-Mead iterations')
    parser.add_argument('-t', type=float, default=5.0, help='Threshold')
    parser.add_argument('-pw', type=float, default=100.0, help='Penalty Weight')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=[812, 1001, 2023, 3030, 42], help='List of seeds to run')
    parser.add_argument('-v', '--verbose', action='store_true', help='Force verbose output even for multiple seeds')
    
    args = parser.parse_args()
    
    # 智能判断：如果只跑1个种子，或者用户强制加了-v，就开启详细输出
    is_single_seed = len(args.seeds) == 1
    should_be_verbose = is_single_seed or args.verbose

    # 把 verbose 标志打包进参数里
    task_args = [(seed, args.i1, args.i2, args.t, args.pw, should_be_verbose) for seed in args.seeds]
    
    print("=" * 60)
    print(f"Starting Parallel Optimization")
    print(f"Seeds to run: {args.seeds}")
    print(f"Target Threshold: > {args.t}")
    print(f"Workers: {args.workers if args.workers else 'Max Available'}")
    print(f"Verbose Mode: {'ON' if should_be_verbose else 'OFF (Logs only)'}")
    print("=" * 60)

    start_time = time.time()
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, t): t[0] for t in task_args}
        
        for future in concurrent.futures.as_completed(futures):
            seed_val = futures[future]
            try:
                seed, loss, status = future.result()
                if status == "Success":
                    # Single line summary upon completion
                    print(f"✅ Seed {seed} Finished | Final Loss: {loss:.6e}")
                    results.append((seed, loss))
                else:
                    print(f"❌ Seed {seed} Failed | Error: {status}")
            except Exception as exc:
                print(f"❌ Seed {seed_val} generated an exception: {exc}")

    print("\n" + "=" * 60)
    print("All Tasks Completed")
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print("Summary:")
    
    results.sort(key=lambda x: x[1])
    for s, l in results:
        print(f"  Seed {s}: Loss = {l:.6e}")
        
    if results:
        print(f"\nBest Seed: {results[0][0]} (Loss: {results[0][1]:.6e})")
    print("=" * 60)