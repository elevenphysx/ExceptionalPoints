"""
Exceptional Point Finder for Nuclear Resonance Cavity (L-BFGS-B Version)
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find parameters where all eigenvalues degenerate (λ₁ = λ₂ = λ₃)
Algorithm: L-BFGS-B with multiple random restarts
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

    Constraints: |Re(λᵢ)| > 5 and |Im(λᵢ)| > 5 for all i
    Loss = variance(Re) + variance(Im) + constraint penalties
    Uses smooth-min for better gradient properties
    """

    def smooth_min(x, beta=10.0):
        """Smooth minimum function for better optimization"""
        return -np.log(np.sum(np.exp(-beta * x))) / beta

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
        loss_degeneracy = np.var(real_parts) + np.var(imag_parts)

        # Constraint: all real and imaginary parts should be |Re(λ)| > 5 and |Im(λ)| > 5
        # Use smooth-min instead of hard min for better gradient
        abs_real_parts = np.abs(real_parts)
        abs_imag_parts = np.abs(imag_parts)
        min_abs_real = smooth_min(abs_real_parts, beta=10.0)
        min_abs_imag = smooth_min(abs_imag_parts, beta=10.0)
        penalty_real = np.maximum(0, 5.0 - min_abs_real)**2
        penalty_imag = np.maximum(0, 5.0 - min_abs_imag)**2

        # Total loss = degeneracy loss + penalty
        penalty_weight = 0.1  # Adjustable weight
        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag)
        return loss

    except Exception as e:
        print(f"Error in objective: {e}")
        if return_details:
            return 1e10, None, None, None, None, None, None, None
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

    # Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
    # C0 is fixed at 7.74 * 1.06 * 0.5 = 4.1022 (not optimized)
    bounds = [
        (2.0, 10.0),     # theta0 (mrad) - user specified range
        (0.5, 10.0),     # Pt thickness (nm)
        (0.1, 50.0),     # C layer 1
        (0.5, 3.0),      # Fe layer 1 (resonant)
        (0.1, 50.0),     # C layer 2
        (0.5, 3.0),      # Fe layer 2 (resonant)
        (0.1, 50.0),     # C layer 3
        (0.5, 3.0),      # Fe layer 3 (resonant)
        (0.1, 50.0),     # C layer 4
    ]

    print("=" * 70)
    print("Starting Global Search (L-BFGS-B with Random Restarts)...")
    print("Target: Find Exceptional Point where λ₁ = λ₂ = λ₃")
    print("=" * 70)

    # Wrapper function for multiprocessing compatibility
    def objective_wrapper(params):
        return objective_function(params, fixed_materials, bounds=bounds)

    # Phase 1: Multi-start L-BFGS-B
    n_restarts = maxiter_de  # Use maxiter_de as number of random restarts
    best_result = None
    best_loss = np.inf

    print(f"\nRunning L-BFGS-B with {n_restarts} random restarts...")

    for restart_idx in range(n_restarts):
        # Generate random initial point within bounds
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])

        try:
            # Run L-BFGS-B from this initial point
            result = minimize(
                objective_wrapper,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-20, 'gtol': 1e-20}
            )

            # Output every restart
            loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(
                result.x, fixed_materials, bounds=bounds, return_details=True
            )

            progress_pct = (restart_idx + 1) / n_restarts * 100
            print(f"\nRestart {restart_idx + 1}/{n_restarts} ({progress_pct:.1f}%)")
            print(f"  Current Loss = {result.fun:.6e}")
            print(f"  Best Loss So Far = {best_loss:.6e}")

            if eigvals is not None:
                print(f"  ├─ Degeneracy    = {loss_degeneracy:.6e}")
                print(f"  ├─ Penalty (Re)  = {penalty_real:.6e}")
                print(f"  └─ Penalty (Im)  = {penalty_imag:.6e}")
                print(f"  Eigenvalues:")
                for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                    print(f"    λ_{i+1} = {re:+.6f} {im:+.6f}i")
                print(f"  Min |Re(λ)| = {np.min(np.abs(real_parts)):.3f}")
                print(f"  Min |Im(λ)| = {np.min(np.abs(imag_parts)):.3f}")

            # Keep track of best result
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result

        except Exception as e:
            print(f"Restart {restart_idx + 1} failed: {e}")
            continue

    result_de = best_result  # Use same variable name for compatibility

    print("\n" + "=" * 70)
    print("L-BFGS-B Multi-Start Result:")
    print(f"Loss = {result_de.fun:.6e}")
    print(f"Total restarts: {n_restarts}")
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
                    xk, fixed_materials, bounds=bounds, return_details=True
                )

                # Display progress (to both console and file)
                pbar_local.write(f"\n--- {opt_name} Iteration {iteration_count_local[0]}/{opt_options['maxiter']} ({progress_pct:.1f}%) ---")
                output = f"Total Loss = {loss:.6e}\n"
                output += f"  ├─ Degeneracy Loss = {loss_degeneracy:.6e}\n"
                output += f"  ├─ Penalty (|Re|>5) = {penalty_real:.6e} (weighted: {0.1 * penalty_real:.6e})\n"
                output += f"  └─ Penalty (|Im|>5) = {penalty_imag:.6e} (weighted: {0.1 * penalty_imag:.6e})\n"
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
                loss, eigvals, _, _, _, _, loss_degeneracy, (penalty_real, penalty_imag) = objective_function(xk, fixed_materials, bounds=bounds, return_details=True)
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
        final_result.x, fixed_materials, bounds=bounds, return_details=True
    )

    theta0, Layers, C0 = build_layers(final_result.x, fixed_materials)

    output += f"\nTotal Loss = {loss:.6e}\n"
    output += f"  ├─ Degeneracy Loss = {loss_degeneracy:.6e}\n"
    output += f"  ├─ Penalty (|Re|>5) = {penalty_real:.6e} (weighted: {0.1 * penalty_real:.6e})\n"
    output += f"  └─ Penalty (|Im|>5) = {penalty_imag:.6e} (weighted: {0.1 * penalty_imag:.6e})\n\n"
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


def scan_parameters_around_optimum(params_optimal, fixed_materials, output_dir, scan_range=1e-6):
    """
    Scan parameters around the optimal solution to observe eigenvalue sensitivity

    Args:
        params_optimal: Optimal parameters [theta0, t_Pt, t_C1, t_Fe1, ...]
        fixed_materials: Material constants
        output_dir: Directory to save scan plots
        scan_range: Scan range around optimal value (default: 1e-6)
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    # Set publication-quality style
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5

    # Professional color scheme (Nature journal style)
    colors = ['#4472C4', '#ED7D31', '#70AD47']  # Blue, Orange, Green

    param_names = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2', 't_C3', 't_Fe3', 't_C4']
    param_labels = ['θ₀ (mrad)', 't_Pt (nm)', 't_C₁ (nm)', 't_Fe₁ (nm)',
                    't_C₂ (nm)', 't_Fe₂ (nm)', 't_C₃ (nm)', 't_Fe₃ (nm)', 't_C₄ (nm)']

    print(f"\n{'='*70}")
    print("Parameter Sensitivity Scan")
    print(f"Scan range: ±{scan_range:.2e} around optimal values")
    print(f"{'='*70}\n")

    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        # Scan range around optimal value
        n_points = 21  # Including center point
        param_values = np.linspace(
            params_optimal[i] - scan_range,
            params_optimal[i] + scan_range,
            n_points
        )

        eigvals_real_list = []
        eigvals_imag_list = []

        print(f"Scanning {param_name}...")
        for param_val in tqdm(param_values, desc=f"  {param_name}", ncols=80, leave=False):
            # Create modified params
            params_test = params_optimal.copy()
            params_test[i] = param_val

            # Compute eigenvalues
            try:
                loss, eigvals, re, im, G, G1, _, _ = objective_function(
                    params_test, fixed_materials, return_details=True
                )
                eigvals_real_list.append(re)
                eigvals_imag_list.append(im)
            except:
                eigvals_real_list.append([np.nan, np.nan, np.nan])
                eigvals_imag_list.append([np.nan, np.nan, np.nan])

        eigvals_real_array = np.array(eigvals_real_list)
        eigvals_imag_array = np.array(eigvals_imag_list)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot real parts
        for j in range(3):
            ax1.plot(param_values, eigvals_real_array[:, j], 'o-',
                    color=colors[j], label=f'Re(λ_{j+1})',
                    markersize=6, alpha=0.8, linewidth=2)

        ax1.axvline(params_optimal[i], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Optimal')
        ax1.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Re(λ)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Real Parts vs {param_label}', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.tick_params(labelsize=11)

        # Plot imaginary parts
        for j in range(3):
            ax2.plot(param_values, eigvals_imag_array[:, j], 's-',
                    color=colors[j], label=f'Im(λ_{j+1})',
                    markersize=6, alpha=0.8, linewidth=2)

        ax2.axvline(params_optimal[i], color='red', linestyle='--',
                   linewidth=2, alpha=0.7, label='Optimal')
        ax2.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Im(λ)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Imaginary Parts vs {param_label}', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, framealpha=0.9, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.tick_params(labelsize=11)

        plt.tight_layout()
        scan_path = os.path.join(output_dir, f'scan_{param_name}.png')
        plt.savefig(scan_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\nParameter scan plots saved to: {output_dir}/scan_*.png")


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
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    scan_parameters_around_optimum(
        params_optimal=result.x,
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-6
    )
