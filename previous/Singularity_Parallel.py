"""
Parallel Exceptional Point Finder (Robust Version)
Method: Multi-Start Differential Evolution using Multiprocessing
Objective: Find parameters where lambda1 = lambda2 = lambda3 with |Re(lambda)| > threshold
"""

import numpy as np
from scipy.optimize import differential_evolution
import sys
import os
import importlib.util
from datetime import datetime
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# ============================================================
# 1. Dynamic Import of Green Function Module
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
# Note: Handling filenames with spaces requires specific loader logic
# Ensure 'green function-new.py' exists in the same directory
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# ============================================================
# 2. Core Logic (Layer Building & Robust Loss)
# ============================================================

def build_layers(params, fixed_materials):
    """
    Construct the physical layer structure from optimization parameters.
    """
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = 7.74 * 1.06 * 0.5  # Fixed constant

    Platinum, Carbon, Iron = fixed_materials

    # Structure: (Material_Tuple, Thickness, Type_Flag)
    Layers = [
        (Platinum, thicknesses[0], 0),
        (Carbon,   thicknesses[1], 0),
        (Iron,     thicknesses[2], 1),  # Resonant
        (Carbon,   thicknesses[3], 0),
        (Iron,     thicknesses[4], 1),  # Resonant
        (Carbon,   thicknesses[5], 0),
        (Iron,     thicknesses[6], 1),  # Resonant
        (Carbon,   thicknesses[7], 0),
        (Platinum, np.inf, 0),          # Substrate
    ]

    return theta0, Layers, C0

def objective_function(params, fixed_materials, threshold=5.0, penalty_weight=2.0):
    """
    Robust Loss Function:
    1. Degeneracy: log10(sum of pairwise distances) - Forces all 3 eigenvalues together.
    2. Constraints: Hard Penalty if < threshold.
    3. Guidance: Saturated Reward (capped) to pull optimizer out of low-magnitude traps.
    """
    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)

        # --- A. Degeneracy Term (Log Pairwise Distance) ---
        diffs = []
        for i in range(len(eigvals)):
            for j in range(i + 1, len(eigvals)):
                diffs.append(np.abs(eigvals[i] - eigvals[j]))
        
        sum_diffs = np.sum(diffs)
        # Using log10 provides strong gradients even at very small scales (e.g. 1e-10)
        loss_degeneracy = np.log10(sum_diffs + 1e-30)

        # --- B. Constraints & Guidance ---
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        min_abs_real = np.min(np.abs(real_parts))
        min_abs_imag = np.min(np.abs(imag_parts))

        # 1. Hard Penalty (Push): Quadratic penalty if below threshold
        pen_re = np.maximum(0, threshold - min_abs_real) ** 2
        pen_im = np.maximum(0, threshold - min_abs_imag) ** 2
        
        # 2. Saturated Guidance (Pull): Reward larger values up to (threshold + 1.0)
        # This prevents the optimizer from sacrificing degeneracy for infinite magnitude.
        safe_cap = threshold + 1.0
        guidance = -1.0 * (np.minimum(min_abs_real, safe_cap) + np.minimum(min_abs_imag, safe_cap))

        # --- Total Loss ---
        return loss_degeneracy + penalty_weight * (pen_re + pen_im) + guidance

    except Exception:
        # Return high loss if calculation fails (e.g., singular matrix)
        return 1e10

# ============================================================
# 3. Worker Function (Runs on separate CPU cores)
# ============================================================

def run_single_optimization(task_id, seed, maxiter, bounds, fixed_materials, threshold, penalty_weight):
    """
    Independent optimization task.
    """
    # Important: Set unique seed for this process
    np.random.seed(seed)
    
    # Wrapper for DE
    def wrapper(p):
        return objective_function(p, fixed_materials, threshold=threshold, penalty_weight=penalty_weight)

    print(f"[Worker {task_id}] Started (Seed: {seed})...")
    
    # Differential Evolution Configuration
    # using 'rand1bin' for better global exploration capability
    result = differential_evolution(
        wrapper,
        bounds,
        maxiter=maxiter,
        popsize=30,           # Good balance for exploration
        strategy='rand1bin',  # Robust against local optima
        mutation=(0.5, 1.0),  # High mutation range
        recombination=0.7,
        seed=seed,
        disp=False,
        polish=True,          # Local refinement at end of DE
        tol=0,
        atol=0
    )
    
    print(f"[Worker {task_id}] Finished. Loss: {result.fun:.6f}")
    
    return {
        'id': task_id,
        'seed': seed,
        'x': result.x,
        'fun': result.fun,
        'message': result.message
    }

# ============================================================
# 4. Main Controller
# ============================================================

def main(args):
    # --- Parse Arguments ---
    NUM_WORKERS = args.workers
    MAXITER = args.iterations
    THRESHOLD = args.threshold
    PENALTY_WEIGHT = args.penalty_weight
    BASE_SEED = args.seed
    
    # Material Parameters (refractive index, extinction coefficient)
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)
    
    # Bounds: theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4
    bounds = [
        (2.0, 10.0), (0.5, 10.0), 
        (0.1, 50.0), (0.5, 3.0), 
        (0.1, 50.0), (0.5, 3.0), 
        (0.1, 50.0), (0.5, 3.0), 
        (0.1, 50.0)
    ]

    print("=" * 70)
    print(f"PARALLEL OPTIMIZATION STARTED")
    print(f"Workers (CPUs): {NUM_WORKERS}")
    print(f"Iterations/Worker: {MAXITER}")
    print(f"Target |Re| > {THRESHOLD}")
    print(f"Penalty Weight: {PENALTY_WEIGHT}")
    print("=" * 70)

    results = []
    
    # --- Parallel Execution ---
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = []
        for i in range(NUM_WORKERS):
            # Generate deterministic but distinct seeds
            current_seed = BASE_SEED + i * 997
            futures.append(
                executor.submit(
                    run_single_optimization, 
                    i, current_seed, MAXITER, bounds, fixed_materials, THRESHOLD, PENALTY_WEIGHT
                )
            )
        
        # Collect results
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                print(f"Worker Error: {e}")

    if not results:
        print("No results collected. Exiting.")
        return

    # --- Aggregation ---
    print("\n" + "=" * 70)
    print("RANKING RESULTS")
    print("=" * 70)
    
    # Sort by Loss (Ascending)
    results.sort(key=lambda x: x['fun'])
    
    for res in results:
        print(f"Worker {res['id']} (Seed {res['seed']}): Loss = {res['fun']:.6e}")

    best = results[0]
    print("\n" + "=" * 70)
    print(f"WINNER: Worker {best['id']}")
    print(f"Loss: {best['fun']:.6e}")
    print("=" * 70)

    # --- Verification & Saving ---
    theta0, Layers, C0 = build_layers(best['x'], fixed_materials)
    G, _ = GreenFun(theta0, Layers, C0)
    G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
    eigvals = np.linalg.eigvals(G1)
    
    re = np.real(eigvals)
    im = np.imag(eigvals)
    
    print("\nEigenvalues of Best Solution:")
    for i in range(len(eigvals)):
        print(f"  lambda_{i+1} = {re[i]:.8f} {im[i]:+.8f}i  (|Re|={np.abs(re[i]):.4f})")
    
    min_re = np.min(np.abs(re))
    status = "PASS" if min_re > THRESHOLD else "FAIL"
    print(f"\nConstraint Check (|Re| > {THRESHOLD}): {status} (Min |Re|={min_re:.4f})")
    
    # Generate Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join('results', f'Parallel_Best_{timestamp}')
    os.makedirs(out_dir, exist_ok=True)
    
    # Save NPZ
    # FIX: Explicitly set dtype=object for Layers to avoid inhomogeneous shape error
    np.savez(os.path.join(out_dir, 'params_best.npz'), 
             params=best['x'], 
             loss=best['fun'], 
             eigvals=eigvals, 
             theta0=theta0, 
             layers=np.array(Layers, dtype=object))
    
    # Save Text Report
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(f"Best Loss: {best['fun']}\n")
        f.write(f"Constraint > {THRESHOLD}: {status}\n")
        f.write(f"Theta0: {theta0}\n")
        f.write("Eigenvalues:\n")
        for i in range(3):
            f.write(f"{re[i]} + {im[i]}j\n")
    
    print(f"\nResults saved to: {out_dir}")

if __name__ == '__main__':
    # Essential for Windows Multiprocessing
    multiprocessing.freeze_support()
    
    parser = argparse.ArgumentParser(description="Parallel EP Finder")
    
    # Arguments
    parser.add_argument('-i', '--iterations', type=int, default=500, 
                        help='DE Iterations per worker (default: 500)')
    
    # Default workers = CPU count - 2 (leave some for system)
    default_workers = max(1, multiprocessing.cpu_count() - 2)
    parser.add_argument('-w', '--workers', type=int, default=default_workers, 
                        help=f'Number of parallel workers (default: {default_workers})')
    
    parser.add_argument('-s', '--seed', type=int, default=1115, 
                        help='Base random seed (default: 1115)')
    
    parser.add_argument('-t', '--threshold', type=float, default=2.0, 
                        help='Target threshold for |Re| (default: 2.0)')
    
    parser.add_argument('-pw', '--penalty-weight', type=float, default=2.0, 
                        help='Penalty weight (default: 2.0)')
    
    args = parser.parse_args()
    
    main(args)