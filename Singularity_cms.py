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
import cma  # Requires: pip install cma

# Import from green function-new.py (with space in filename)
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# ============================================================
# Core Physics & Objective Functions
# ============================================================

def build_layers(params, fixed_materials):
    """
    Build layer structure from optimization parameters
    Structure: Pt-C-Fe-C-Fe-C-Fe-C-Pt(inf)
    """
    theta0 = params[0]
    thicknesses = params[1:9]  # 8 finite layers
    C0 = 7.74 * 1.06 * 0.5  # Fixed value (CFe)

    Platinum, Carbon, Iron = fixed_materials

    Layers = [
        (Platinum, thicknesses[0], 0),  # Pt
        (Carbon,   thicknesses[1], 0),  # C
        (Iron,     thicknesses[2], 1),  # Fe (resonant)
        (Carbon,   thicknesses[3], 0),  # C
        (Iron,     thicknesses[4], 1),  # Fe (resonant)
        (Carbon,   thicknesses[5], 0),  # C
        (Iron,     thicknesses[6], 1),  # Fe (resonant)
        (Carbon,   thicknesses[7], 0),  # C
        (Platinum, np.inf, 0),  # Pt substrate (infinite thickness)
    ]

    return theta0, Layers, C0

def objective_function(params, fixed_materials, bounds=None, return_details=False, threshold=5.0, penalty_weight=100.0):
    """
    Revised Objective function for finding Exceptional Points.
    
    Improvements:
    1. Uses Normalized Sum of Pairwise Differences (L1 Norm) instead of Variance.
       This prevents gradient vanishing when differences are very small.
    2. Uses Exponential Penalty for BOTH Real and Imaginary parts.
    3. Checks the minimum absolute value to ensure ALL eigenvalues satisfy the threshold.
    """
    
    # Helper: Normalized Pairwise Difference Sum
    # Calculates sum(|x_i - x_j|) / mean_abs for scale invariance.
    def normalized_pairwise_diff(values, mean_val):
        # Calculate sum of absolute differences between all unique pairs
        # For 3 eigenvalues: |v0-v1| + |v0-v2| + |v1-v2|
        diff_sum = 0.0
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                diff_sum += np.abs(values[i] - values[j])
        
        if mean_val < 1e-9: # Avoid division by zero
            return diff_sum
        return diff_sum / mean_val

    try:
        # Check bounds if provided (CMA-ES handles bounds internally, but double check is safe)
        if bounds is not None:
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                if not (low <= param <= high):
                    if return_details:
                        return 1e10, None, None, None, None, None, None, None
                    else:
                        return 1e10

        theta0, Layers, C0 = build_layers(params, fixed_materials)

        # Compute Green matrix
        G, _ = GreenFun(theta0, Layers, C0)
        # Transform matrix (Singularity condition)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

        # Compute eigenvalues
        eigvals = np.linalg.eigvals(G1)
        
        # Stability check: Return high loss if NaN or Inf is detected
        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            if return_details:
                 return 1e10, eigvals, np.zeros(3), np.zeros(3), G, G1, 1e10, (1e10, 1e10)
            return 1e10

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        
        # Calculate mean absolute magnitude (used only for normalization)
        mean_abs = np.mean(np.abs(eigvals)) + 1e-9

        # 1. Degeneracy Loss: Pairwise Differences
        loss_degeneracy = normalized_pairwise_diff(real_parts, mean_abs) + normalized_pairwise_diff(imag_parts, mean_abs)

        # 2. Threshold Penalty (Exponential) - Check BOTH Real and Imaginary
        # We need ALL eigenvalues to have |Re| > threshold AND |Im| > threshold.
        # So we check the minimum absolute values.
        
        # Check Real Parts
        min_re = np.min(np.abs(real_parts))
        if min_re < threshold:
            gap_re = threshold - min_re
            penalty_real = (np.exp(gap_re) - 1.0)
        else:
            penalty_real = 0.0

        # Check Imaginary Parts
        min_im = np.min(np.abs(imag_parts))
        if min_im < threshold:
            gap_im = threshold - min_im
            penalty_imag = (np.exp(gap_im) - 1.0)
        else:
            penalty_imag = 0.0
        
        # Total Loss
        # Combine degeneracy loss with both penalties
        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag)
        return loss

    except Exception as e:
        # print(f"Error in objective: {e}") 
        if return_details:
            return 1e10, None, None, None, None, None, None, None
        else:
            return 1e10

# ============================================================
# Main Optimization Logic
# ============================================================

def optimize_exceptional_point(maxiter_cma=200, maxiter_nm=500, seed=812, threshold=5.0, penalty_weight=100.0):
    """
    Main optimization routine using CMA-ES followed by Nelder-Mead.
    """
    # Random seed
    np.random.seed(seed)

    # Create output directory
    output_dir = os.path.join('results', f'CMA-ES_s{seed}_i1-{maxiter_cma}_i2-{maxiter_nm}_thr{threshold:.1f}_pw{penalty_weight:.1f}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    print(f"Random seed: {seed}")
    
    # Material Parameters
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    # Relaxed Bounds (Allowing 0 thickness and small angles)
    # [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
    bounds = [
        (0.5, 10.0),     # theta0 (mrad)
        (0.0, 10.0),     # Pt thickness (nm)
        (0.0, 50.0),     # C layer 1
        (0.5, 10.0),     # Fe layer 1
        (0.1, 50.0),     # C layer 2
        (0.5, 10.0),     # Fe layer 2
        (0.1, 50.0),     # C layer 3
        (0.5, 10.0),     # Fe layer 3
        (0.1, 50.0),     # C layer 4
    ]

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
    plot_results(history, output_dir)
    print(f"\nLog saved to: {log_file_path}")

    # Return result object-like structure for compatibility
    class Result:
        pass
    res = Result()
    res.x = final_result_x
    res.fun = final_loss
    
    return res, theta0, Layers, C0, output_dir


def plot_results(history, output_dir):
    """
    Visualize optimization history
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'DejaVu Sans'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = np.array(history['iteration'])
    losses = np.array(history['loss'])
    phases = np.array(history['phase'])
    eigvals_real = history['eigvals_real']
    eigvals_imag = history['eigvals_imag']

    # Loss plot
    phase_colors = {'CMA-ES': '#1f77b4', 'Nelder-Mead': '#ff7f0e'}
    for phase_name in phase_colors:
        mask = phases == phase_name
        if np.any(mask):
            ax1.semilogy(iterations[mask], losses[mask], 'o-',
                        color=phase_colors[phase_name],
                        label=phase_name, markersize=3, alpha=0.6)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Optimization History')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Eigenvalue Distances
    if len(eigvals_real) > 0:
        d12, d13, d23 = [], [], []
        for r, i in zip(eigvals_real, eigvals_imag):
            if len(r) >= 3:
                e = r + 1j * i
                d12.append(np.abs(e[0]-e[1]))
                d13.append(np.abs(e[0]-e[2]))
                d23.append(np.abs(e[1]-e[2]))
            else:
                d12.append(np.nan); d13.append(np.nan); d23.append(np.nan)
        
        ax2.semilogy(iterations, d12, label='|λ1-λ2|', alpha=0.6)
        ax2.semilogy(iterations, d13, label='|λ1-λ3|', alpha=0.6)
        ax2.semilogy(iterations, d23, label='|λ2-λ3|', alpha=0.6)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Diff')
        ax2.set_title('Eigenvalue Coalescence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_result.png'), dpi=300)
    plt.close()


def scan_parameters_around_optimum(params_optimal, fixed_materials, output_dir, scan_range=1e-6):
    """
    Parameter Sensitivity Scan
    """
    # (Existing scan logic kept as is...)
    import matplotlib.pyplot as plt
    param_names = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2', 't_C3', 't_Fe3', 't_C4']
    
    print(f"\nScanning parameters around optimum (range +/- {scan_range})...")
    
    for i, name in enumerate(param_names):
        vals = np.linspace(params_optimal[i] - scan_range, params_optimal[i] + scan_range, 21)
        re_list, im_list = [], []
        
        for v in vals:
            p = params_optimal.copy()
            p[i] = v
            try:
                _, _, re, im, _, _, _, _ = objective_function(p, fixed_materials, return_details=True, threshold=0, penalty_weight=0)
                re_list.append(re)
                im_list.append(im)
            except:
                re_list.append([np.nan]*3)
                im_list.append([np.nan]*3)
                
        re_arr = np.array(re_list)
        im_arr = np.array(im_list)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        for k in range(3):
            ax1.plot(vals, re_arr[:, k], label=f'Re(λ{k+1})')
            ax2.plot(vals, im_arr[:, k], label=f'Im(λ{k+1})')
            
        ax1.set_title(f'Real vs {name}')
        ax2.set_title(f'Imag vs {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scan_{name}.png'))
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (CMA-ES)')
    parser.add_argument('-i1', '--iterations-cma', type=int, default=200,
                        help='Number of generations for CMA-ES (default: 200)')
    parser.add_argument('-i2', '--iterations-nm', type=int, default=500,
                        help='Number of iterations for Nelder-Mead (default: 500)')
    parser.add_argument('-s', '--seed', type=int, default=812,
                        help='Random seed for reproducibility (default: 812)')
    parser.add_argument('-t', '--threshold', type=float, default=5.0,
                        help='Constraint threshold for |Re| and |Im| (default: 5.0)')
    parser.add_argument('-pw', '--penalty-weight', type=float, default=100.0,
                        help='Penalty weight for constraint violations (default: 100.0)')
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
        fixed_materials=( (1.713e-5, 2.518e-6), (2.257e-6, 1.230e-9), (7.298e-6, 3.33e-7) ),
        output_dir=output_dir,
        scan_range=1e-6
    )