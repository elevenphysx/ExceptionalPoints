"""
Exceptional Point Finder for Nuclear Resonance Cavity
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find parameters where all eigenvalues degenerate (λ₁ = λ₂ = λ₃)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm

# Import shared configuration
from config import (FIXED_MATERIALS, BOUNDS, C0_FIXED, DEFAULT_THRESHOLD,
                    DEFAULT_PENALTY_WEIGHT, DEFAULT_DE_ITERATIONS,
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

def optimize_exceptional_point(maxiter_de=DEFAULT_DE_ITERATIONS, maxiter_nm=DEFAULT_NM_ITERATIONS,
                              maxiter_powell=500, seed=DEFAULT_SEEDS[0], threshold=DEFAULT_THRESHOLD,
                              penalty_weight=DEFAULT_PENALTY_WEIGHT):
    """
    Main optimization routine using Differential Evolution

    Args:
        maxiter_de: Maximum iterations for DE
        maxiter_nm: Maximum iterations for Nelder-Mead
        maxiter_powell: Maximum iterations for Powell
        seed: Random seed for reproducibility
        threshold: Constraint threshold for |Re| and |Im|
        penalty_weight: Weight for constraint penalties
    """
    np.random.seed(seed)

    output_dir = os.path.join('results', f'DE_s{seed}_i1-{maxiter_de}_i2-{maxiter_nm}_thr{threshold:.1f}_pw{penalty_weight:.2f}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    print(f"Random seed: {seed}")
    print(f"Iterations: DE={maxiter_de}, Nelder-Mead={maxiter_nm}, Powell={maxiter_powell}\n")

    fixed_materials = FIXED_MATERIALS
    bounds = BOUNDS

    # Convergence threshold for monitoring (not for early stopping)
    convergence_threshold = 1e-10
    iteration_count = [0]  # Use list to allow modification in callback

    # History tracking for plotting
    history = {
        'iteration': [],
        'loss': [],
        'phase': [],
        'eigvals_real': [],  # Real parts of eigenvalues
        'eigvals_imag': []   # Imaginary parts of eigenvalues
    }

    # Open log file for writing iteration history
    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    # Force UTF-8 output for console (Windows compatibility)
    import sys
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

    log_file.write("Exceptional Point Optimization Log\n")
    log_file.write("=" * 70 + "\n\n")

    # Helper function to log to both console and file
    def log_print(message, end='\n'):
        print(message, end=end)
        log_file.write(message + end)
        log_file.flush()

    # Create progress bar for DE phase
    pbar_de = tqdm(total=maxiter_de, desc="DE Progress",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                   position=0, leave=True)

    # Callback function to display progress and check convergence
    def callback(xk, convergence=None):
        """
        Callback function called at each iteration
        xk: current best solution
        convergence: convergence metric from DE
        """
        iteration_count[0] += 1
        pbar_de.update(1)  # Update progress bar
        progress_pct = iteration_count[0] / maxiter_de * 100

        # Every 100 iterations: full output with eigenvalues
        if iteration_count[0] % 100 == 0 or iteration_count[0] == 1:
            # Compute eigenvalues for current best solution
            loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
                xk, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )

            # Display progress (to both console and file)
            pbar_de.write(f"\n--- Iteration {iteration_count[0]}/{maxiter_de} ({progress_pct:.1f}%) ---")
            output = f"Total Loss = {loss:.6e}\n"
            output += f"  ├─ Degeneracy Loss = {loss_degeneracy:.6e}\n"
            output += f"  ├─ Penalty (|Re|>{threshold}) = {penalty_real:.6e} (weighted: {penalty_weight * penalty_real:.6e})\n"
            output += f"  └─ Penalty (|Im|>{threshold}) = {penalty_imag:.6e} (weighted: {penalty_weight * penalty_imag:.6e})\n"
            output += "Eigenvalues:\n"
            for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                output += f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n"

            # Check degeneracy
            re_std = np.std(real_parts)
            im_std = np.std(imag_parts)
            output += f"Std(Re) = {re_std:.6e}, Std(Im) = {im_std:.6e}"

            # Write to both console and file
            pbar_de.write(output)
            log_file.write(f"\n--- Iteration {iteration_count[0]}/{maxiter_de} ({progress_pct:.1f}%) ---\n")
            log_file.write(output + "\n")
            log_file.flush()

            # Monitor convergence (info only, no early stopping)
            if loss < convergence_threshold:
                msg = f"[INFO] Loss < {convergence_threshold} (continuing to maxiter)"
                pbar_de.write(msg)
                log_file.write(msg + "\n")
                log_file.flush()

        # Every 10 iterations (but not 100): simple loss output and record history
        elif iteration_count[0] % 10 == 0:
            loss, eigvals, _, _, _, _, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(xk, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight)
            penalty_total = penalty_real + penalty_imag
            msg = f"differential_evolution step {iteration_count[0]}/{maxiter_de} ({progress_pct:.1f}%): f(x)= {loss:.15f} (degeneracy={loss_degeneracy:.6e}, penalty_re={penalty_real:.6e}, penalty_im={penalty_imag:.6e})"
            pbar_de.write(msg)
            log_file.write(msg + "\n")
            log_file.flush()

            # Record history for plotting
            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('DE')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

        return False  # Continue

    # Wrapper function for multiprocessing compatibility
    def objective_wrapper(params):
        return objective_function(params, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight)

    # Phase 1: Global search with Differential Evolution
    result_de = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=maxiter_de,
        popsize=20,        # Increase population size for better exploration
        strategy='best1bin',
        seed=seed,         # Use defined random seed
        disp=False,        # Disable default output, use callback instead
        workers=1,         # Use single process to avoid pickle issues
        updating='immediate',
        polish=False,      # Manual refinement later
        callback=callback, # Display eigenvalues at each iteration
        atol=0,            # Disable absolute tolerance
        tol=0              # Disable relative tolerance - run to maxiter
    )

    pbar_de.close()  # Close DE progress bar

    print("\n" + "=" * 70)
    print("Differential Evolution Result:")
    print(f"Loss = {result_de.fun:.6e}")
    print("=" * 70)

    # Phase 2: Local refinement (always execute for better convergence)
    log_print("\n" + "=" * 70)
    log_print("Phase 2: Local Refinement (Gradient-Free Optimizers)")
    log_print("=" * 70)

    # Try multiple gradient-free optimizers
    best_result = result_de
    iteration_offset = iteration_count[0]  # Track global iteration for plotting

    optimizers = [
        ('Nelder-Mead', {'maxiter': maxiter_nm, 'disp': True,
                        'xatol': 0, 'fatol': 0, 'adaptive': True}),  # Set tolerances to 0, use adaptive for better exploration
        # ('Powell', {'maxiter': maxiter_powell, 'disp': True,
        #            'ftol': 0, 'xtol': 0}),  # Temporarily disabled - early stopping issue
    ]

    for opt_name, opt_options in optimizers:
        log_print(f"\n{'=' * 70}")
        log_print(f"[Phase 2.{optimizers.index((opt_name, opt_options)) + 1}] {opt_name} Optimizer")
        log_print(f"Starting from loss = {best_result.fun:.6e}")
        log_print(f"Maxiter = {opt_options['maxiter']} (or optimizer's own convergence criteria)")
        log_print(f"{'=' * 70}\n")

        # Create progress bar for this optimizer
        pbar_local = tqdm(total=opt_options['maxiter'], desc=f"{opt_name} Progress",
                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                         position=0, leave=True)

        # Callback to track progress
        iteration_count_local = [0]

        def local_callback(xk):
            iteration_count_local[0] += 1
            pbar_local.update(1)  # Update progress bar
            progress_pct = iteration_count_local[0] / opt_options['maxiter'] * 100

            # Every 100 iterations: full output with eigenvalues
            if iteration_count_local[0] % 100 == 0 or iteration_count_local[0] == 1:
                loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
                    xk, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
                )

                # Display progress (to both console and file)
                pbar_local.write(f"\n--- {opt_name} Iteration {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%) ---")
                output = f"Total Loss = {loss:.6e}\n"
                output += f"  ├─ Degeneracy Loss = {loss_degeneracy:.6e}\n"
                output += f"  ├─ Penalty (|Re|>{threshold}) = {penalty_real:.6e} (weighted: {penalty_weight * penalty_real:.6e})\n"
                output += f"  └─ Penalty (|Im|>{threshold}) = {penalty_imag:.6e} (weighted: {penalty_weight * penalty_imag:.6e})\n"
                output += "Eigenvalues:\n"
                for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                    output += f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n"

                # Check degeneracy
                re_std = np.std(real_parts)
                im_std = np.std(imag_parts)
                output += f"Std(Re) = {re_std:.6e}, Std(Im) = {im_std:.6e}"

                pbar_local.write(output)
                log_file.write(f"\n--- {opt_name} Iteration {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%) ---\n")
                log_file.write(output + "\n")
                log_file.flush()

                # Monitor convergence (info only, no early stopping)
                if loss < convergence_threshold:
                    msg = f"[INFO] Loss < {convergence_threshold} (continuing to maxiter)"
                    pbar_local.write(msg)
                    log_file.write(msg + "\n")
                    log_file.flush()

            # Every 10 iterations (but not 100): simple loss output and record history
            elif iteration_count_local[0] % 10 == 0:
                loss, eigvals, _, _, _, _, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(xk, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight)
                penalty_total = penalty_real + penalty_imag
                msg = f"{opt_name} step {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%): f(x)= {loss:.15f} (degeneracy={loss_degeneracy:.6e}, penalty_re={penalty_real:.6e}, penalty_im={penalty_imag:.6e})"
                pbar_local.write(msg)
                log_file.write(msg + "\n")
                log_file.flush()

                # Record history for plotting
                if eigvals is not None:
                    global_iter = iteration_offset + iteration_count_local[0]
                    history['iteration'].append(global_iter)
                    history['loss'].append(loss)
                    history['phase'].append(opt_name)
                    history['eigvals_real'].append(np.real(eigvals).copy())
                    history['eigvals_imag'].append(np.imag(eigvals).copy())

            return False

        try:
            result_local = minimize(
                objective_wrapper,
                best_result.x,
                method=opt_name,
                bounds=None,  # Don't use native bounds for Powell - use penalty method instead
                callback=local_callback,
                options=opt_options
            )

            pbar_local.close()  # Close progress bar

            log_print(f"\n{opt_name} finished:")
            log_print(f"  Final loss = {result_local.fun:.6e}")
            log_print(f"  Iterations = {iteration_count_local[0]}")
            log_print(f"  Status: {result_local.message if hasattr(result_local, 'message') else 'Completed'}")

            # Update global iteration offset for next optimizer
            iteration_offset += iteration_count_local[0]

            if result_local.fun < best_result.fun:
                improvement = (best_result.fun - result_local.fun) / best_result.fun * 100
                log_print(f"✓ {opt_name} improved by {improvement:.2f}%")
                best_result = result_local
            else:
                log_print(f"✗ {opt_name} did not improve")

        except Exception as e:
            pbar_local.close()  # Close progress bar on error
            log_print(f"✗ {opt_name} failed: {e}")

    log_print(f"\n{'=' * 70}")
    log_print(f"Best optimizer: Final loss = {best_result.fun:.6e}")
    log_print(f"{'=' * 70}")

    final_result = best_result

    # Display final results
    output = "\n" + "=" * 70 + "\n"
    output += "FINAL RESULT:\n"
    output += "=" * 70 + "\n"

    loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
        final_result.x, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
    )

    theta0, Layers, C0 = build_layers(final_result.x, fixed_materials)

    output += f"\nTotal Loss = {loss:.6e}\n"
    output += f"  ├─ Degeneracy Loss = {loss_degeneracy:.6e}\n"
    output += f"  ├─ Penalty (|Re|>{threshold}) = {penalty_real:.6e} (weighted: {penalty_weight * penalty_real:.6e})\n"
    output += f"  └─ Penalty (|Im|>{threshold}) = {penalty_imag:.6e} (weighted: {penalty_weight * penalty_imag:.6e})\n\n"
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

    # Check degeneracy
    re_std = np.std(real_parts)
    im_std = np.std(imag_parts)
    output += "\n  Degeneracy Check:\n"
    output += f"    Std(Re) = {re_std:.6e}  {'✓' if re_std < 0.01 else '✗'}\n"
    output += f"    Std(Im) = {im_std:.6e}  {'✓' if im_std < 0.01 else '✗'}\n"

    # Display matrices
    output += "\n" + "=" * 70 + "\n"
    output += "Matrix Analysis:\n"
    output += "=" * 70 + "\n"

    output += f"\nOriginal Green Matrix G (shape: {G.shape}):\n"
    np.set_printoptions(precision=6, suppress=True, linewidth=100)
    output += str(G) + "\n"

    output += f"\nTransformed Matrix G1 = -G - 0.5i·I (shape: {G1.shape}):\n"
    output += str(G1) + "\n"

    # Write to both console and file
    print(output, end='')
    log_file.write(output)
    log_file.flush()

    # Visualization
    plot_optimization_history(history, output_dir, seed, 'DE + Nelder-Mead')

    # Close log file
    log_file.close()
    print(f"\nOptimization log saved to: {log_file_path}")

    return final_result, theta0, Layers, C0, output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Variance Method)')
    parser.add_argument('-i1', '--iterations-de', type=int, default=DEFAULT_DE_ITERATIONS,
                        help='Number of iterations for DE')
    parser.add_argument('-i2', '--iterations-nm', type=int, default=DEFAULT_NM_ITERATIONS,
                        help='Number of iterations for Nelder-Mead')
    parser.add_argument('-i3', '--iterations-powell', type=int, default=500,
                        help='Number of iterations for Powell')
    parser.add_argument('-s', '--seed', type=int, default=DEFAULT_SEEDS[0],
                        help='Random seed for reproducibility')
    parser.add_argument('-t', '--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='Constraint threshold for |Re| and |Im|')
    parser.add_argument('-pw', '--penalty-weight', type=float, default=DEFAULT_PENALTY_WEIGHT,
                        help='Penalty weight for constraint violations')
    args = parser.parse_args()

    result, theta0, Layers, C0, output_dir = optimize_exceptional_point(
        maxiter_de=args.iterations_de,
        maxiter_nm=args.iterations_nm,
        maxiter_powell=args.iterations_powell,
        seed=args.seed,
        threshold=args.threshold,
        penalty_weight=args.penalty_weight
    )

    # Save results (save parameter array instead of Layers structure)
    # Extract layer thicknesses for saving
    thicknesses = [layer[1] for layer in Layers[:-1]]  # Exclude substrate

    params_path = os.path.join(output_dir, 'exceptional_point_params.npz')
    np.savez(params_path,
             params=result.x,  # All optimization parameters
             theta0=theta0,
             C0=C0,
             thicknesses=thicknesses,  # Layer thicknesses
             loss=result.fun)
    print(f"\nParameters saved to: {params_path}")

    # Save detailed parameters as text file with high precision
    params_txt_path = os.path.join(output_dir, 'exceptional_point_params.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Exceptional Point Parameters (High Precision)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Final Loss:  {result.fun:.12e}\n\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write(f"C0     = {C0:.15f} (fixed)\n\n")
        f.write("Layer Thicknesses (nm):\n")
        f.write("-" * 50 + "\n")
        layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n")
        f.write(f"  Layer 8: Pt(sub)  = inf nm\n\n")
        f.write("All Optimization Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  params[0] (theta0) = {result.x[0]:.15f} mrad\n")
        for i in range(1, len(result.x)):
            f.write(f"  params[{i}] (Layer {i-1} thickness) = {result.x[i]:.15f} nm\n")
    print(f"Detailed parameters saved to: {params_txt_path}")

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
