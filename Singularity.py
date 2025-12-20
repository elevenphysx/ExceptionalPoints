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

# Import from green function-new.py (with space in filename)
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# ============================================================
# Objective: Find Exceptional Point where λ₁ = λ₂ = λ₃
# ============================================================

def build_layers(params, fixed_materials):
    """
    Build layer structure from optimization parameters

    Structure: Pt-C-Fe-C-Fe-C-Fe-C-Pt(inf)
    params = [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter (fixed)
    """
    theta0 = params[0]
    thicknesses = params[1:9]  # 8 finite layers
    C0 = 7.74 * 1.06 * 0.5  # Fixed value from original code (CFe)

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


def objective_function(params, fixed_materials, bounds=None, return_details=False):
    """
    Objective function: make all eigenvalues degenerate (coincide)

    For exceptional point: λ₁ = λ₂ = λ₃
    i.e., Re(λ₁) = Re(λ₂) = Re(λ₃) and Im(λ₁) = Im(λ₂) = Im(λ₃)

    Loss = variance(Re) + variance(Im)
    """
    try:
        # Check bounds if provided
        if bounds is not None:
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                if not (low <= param <= high):
                    if return_details:
                        return 1e10, None, None, None, None, None
                    else:
                        return 1e10

        theta0, Layers, C0 = build_layers(params, fixed_materials)

        # Compute Green matrix
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

        # Compute eigenvalues
        eigvals = np.linalg.eigvals(G1)

        # Loss: variance of real parts + variance of imaginary parts
        # This forces all eigenvalues to collapse to the same point
        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)
        loss = np.var(real_parts) + np.var(imag_parts)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1
        return loss

    except Exception as e:
        print(f"Error in objective: {e}")
        if return_details:
            return 1e10, None, None, None, None, None
        else:
            return 1e10


def optimize_exceptional_point(maxiter_de=100, maxiter_nm=500, maxiter_powell=500, seed=812):
    """
    Main optimization routine using Differential Evolution

    Args:
        maxiter_de: Maximum iterations for DE (default: 100)
        maxiter_nm: Maximum iterations for Nelder-Mead (default: 500)
        maxiter_powell: Maximum iterations for Powell (default: 500)
        seed: Random seed for reproducibility (default: 812)
    """
    # Random seed for reproducibility
    np.random.seed(seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join('results', f'variance_method_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")
    print(f"Random seed: {seed}")
    print(f"Iterations: DE={maxiter_de}, Nelder-Mead={maxiter_nm}, Powell={maxiter_powell}\n")

    # Fixed material parameters (physical constants)
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

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
            loss, eigvals, real_parts, imag_parts, G, G1 = objective_function(
                xk, fixed_materials, bounds=bounds, return_details=True
            )

            # Display progress (to both console and file)
            pbar_de.write(f"\n--- Iteration {iteration_count[0]}/{maxiter_de} ({progress_pct:.1f}%) ---")
            output = f"Loss (variance) = {loss:.6e}\n"
            output += "Eigenvalues:\n"
            for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                output += f"  λ_{i+1} = {re:+15.8f} {im:+15.8f}i\n"

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
            loss, eigvals, _, _, _, _ = objective_function(xk, fixed_materials, bounds=bounds, return_details=True)
            msg = f"differential_evolution step {iteration_count[0]}/{maxiter_de} ({progress_pct:.1f}%): f(x)= {loss:.15f}"
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

    # Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
    # C0 is fixed at 7.74 * 1.06 * 0.5 = 4.1022 (not optimized)
    bounds = [
        (2.0, 10.0),     # theta0 (mrad) - user specified range
        (0.5, 10.0),     # Pt thickness (nm)
        (1.0, 50.0),     # C layer 1
        (0.5, 3.0),      # Fe layer 1 (resonant)
        (1.0, 50.0),     # C layer 2
        (0.5, 3.0),      # Fe layer 2 (resonant)
        (1.0, 50.0),     # C layer 3
        (0.5, 3.0),      # Fe layer 3 (resonant)
        (1.0, 50.0),     # C layer 4
    ]

    print("=" * 70)
    print("Starting Global Search (Differential Evolution)...")
    print("Target: Find Exceptional Point where λ₁ = λ₂ = λ₃")
    print("=" * 70)

    # Wrapper function for multiprocessing compatibility
    def objective_wrapper(params):
        return objective_function(params, fixed_materials, bounds=bounds)

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
                        'xatol': 1e-20, 'fatol': 1e-20, 'adaptive': False}),  # Extremely small tolerances - run to maxiter
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
                loss, eigvals, real_parts, imag_parts, G, G1 = objective_function(
                    xk, fixed_materials, bounds=bounds, return_details=True
                )

                # Display progress (to both console and file)
                pbar_local.write(f"\n--- {opt_name} Iteration {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%) ---")
                output = f"Loss (variance) = {loss:.6e}\n"
                output += "Eigenvalues:\n"
                for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                    output += f"  λ_{i+1} = {re:+15.8f} {im:+15.8f}i\n"

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
                loss, eigvals, _, _, _, _ = objective_function(xk, fixed_materials, bounds=bounds, return_details=True)
                msg = f"{opt_name} step {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%): f(x)= {loss:.15f}"
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

    loss, eigvals, real_parts, imag_parts, G, G1 = objective_function(
        final_result.x, fixed_materials, bounds=bounds, return_details=True
    )

    theta0, Layers, C0 = build_layers(final_result.x, fixed_materials)

    output += f"\nLoss = {loss:.6e}\n\n"
    output += f"theta0 = {theta0:.8f} mrad\n"
    output += f"C0     = {C0:.8f} (fixed)\n\n"

    output += "Layer Structure (nm):\n"
    layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Pt(sub)']
    for i, (name, layer) in enumerate(zip(layer_names, Layers)):
        thickness = layer[1]
        resonant = ' (resonant)' if layer[2] == 1 else ''
        output += f"  Layer {i}: {name:8s} = {thickness:12.8f} nm{resonant}\n"

    output += "\nEigenvalue Analysis:\n"
    output += "  λᵢ = Re + Im·i\n"
    output += "  " + "-" * 50 + "\n"
    for i, (eig, re, im) in enumerate(zip(eigvals, real_parts, imag_parts)):
        output += f"  λ_{i+1} = {re:+15.8f} {im:+15.8f}i\n"

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
    plot_results(history, output_dir)

    # Close log file
    log_file.close()
    print(f"\nOptimization log saved to: {log_file_path}")

    return final_result, theta0, Layers, C0, output_dir


def plot_results(history, output_dir):
    """
    Visualize optimization history with publication-quality plots

    Args:
        history: Dictionary containing iteration, loss, phase, and eigenvalues
        output_dir: Directory to save the figure
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set publication-quality style
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'DejaVu Sans'  # Supports Unicode subscripts
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.major.size'] = 5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = np.array(history['iteration'])
    losses = np.array(history['loss'])
    phases = np.array(history['phase'])
    eigvals_real = history['eigvals_real']
    eigvals_imag = history['eigvals_imag']

    # ===== Left plot: Loss vs Iteration =====
    # Define colors for each phase
    phase_colors = {'DE': '#1f77b4', 'Nelder-Mead': '#ff7f0e', 'Powell': '#2ca02c'}
    phase_names = ['DE', 'Nelder-Mead', 'Powell']

    for phase_name in phase_names:
        mask = phases == phase_name
        if np.any(mask):
            ax1.semilogy(iterations[mask], losses[mask], 'o-',
                        color=phase_colors[phase_name],
                        label=phase_name, markersize=4, alpha=0.8)

    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss (Variance)', fontsize=14, fontweight='bold')
    ax1.set_title('Optimization Progress', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(fontsize=11, framealpha=0.9, loc='best')
    ax1.tick_params(labelsize=11)

    # ===== Right plot: Eigenvalue distances vs Iteration =====
    distances_12 = []
    distances_13 = []
    distances_23 = []

    for real_parts, imag_parts in zip(eigvals_real, eigvals_imag):
        if len(real_parts) >= 3:
            # Calculate complex distances
            eig1 = real_parts[0] + 1j * imag_parts[0]
            eig2 = real_parts[1] + 1j * imag_parts[1]
            eig3 = real_parts[2] + 1j * imag_parts[2]

            distances_12.append(np.abs(eig1 - eig2))
            distances_13.append(np.abs(eig1 - eig3))
            distances_23.append(np.abs(eig2 - eig3))
        else:
            distances_12.append(np.nan)
            distances_13.append(np.nan)
            distances_23.append(np.nan)

    distances_12 = np.array(distances_12)
    distances_13 = np.array(distances_13)
    distances_23 = np.array(distances_23)

    ax2.semilogy(iterations, distances_12, 'o-', color='#d62728',
                label='|λ₁ - λ₂|', markersize=3, alpha=0.8)
    ax2.semilogy(iterations, distances_13, 's-', color='#9467bd',
                label='|λ₁ - λ₃|', markersize=3, alpha=0.8)
    ax2.semilogy(iterations, distances_23, '^-', color='#8c564b',
                label='|λ₂ - λ₃|', markersize=3, alpha=0.8)

    ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Distance', fontsize=14, fontweight='bold')
    ax2.set_title('Eigenvalue Convergence', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.legend(fontsize=11, framealpha=0.9, loc='best')
    ax2.tick_params(labelsize=11)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'exceptional_point_result.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {fig_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Variance Method)')
    parser.add_argument('-i1', '--iterations-de', type=int, default=100,
                        help='Number of iterations for DE (default: 100)')
    parser.add_argument('-i2', '--iterations-nm', type=int, default=500,
                        help='Number of iterations for Nelder-Mead (default: 500)')
    parser.add_argument('-i3', '--iterations-powell', type=int, default=500,
                        help='Number of iterations for Powell (default: 500)')
    parser.add_argument('-s', '--seed', type=int, default=812,
                        help='Random seed for reproducibility (default: 812)')
    args = parser.parse_args()

    result, theta0, Layers, C0, output_dir = optimize_exceptional_point(
        maxiter_de=args.iterations_de,
        maxiter_nm=args.iterations_nm,
        maxiter_powell=args.iterations_powell,
        seed=args.seed
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
        f.write(f"theta0 = {theta0:.8f} mrad\n")
        f.write(f"C0     = {C0:.8f} (fixed)\n\n")
        f.write("Layer Thicknesses (nm):\n")
        f.write("-" * 50 + "\n")
        layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:12.8f} nm{resonant}\n")
        f.write(f"  Layer 8: Pt(sub)  = inf nm\n\n")
        f.write("All Optimization Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  params[0] (theta0) = {result.x[0]:.8f} mrad\n")
        for i in range(1, len(result.x)):
            f.write(f"  params[{i}] (Layer {i-1} thickness) = {result.x[i]:.8f} nm\n")
    print(f"Detailed parameters saved to: {params_txt_path}")
