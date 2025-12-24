"""
Two-Stage Exceptional Point Finder
两阶段分离优化策略

Stage 1: Find parameters satisfying constraints (|Re| > 5 AND |Im| > 5)
Stage 2: Optimize degeneracy from Stage 1 results
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm

# Try to import cma
try:
    import cma
except ImportError:
    print("ERROR: CMA-ES requires the 'cma' package.")
    print("Please install it with: pip install cma")
    sys.exit(1)

# Import from green function-new.py
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun


def build_layers(params, fixed_materials):
    """Build layer structure from optimization parameters"""
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = 7.74 * 1.06 * 0.5

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
        (Platinum, np.inf, 0),  # Pt substrate
    ]

    return theta0, Layers, C0


def stage1_objective(params, fixed_materials, bounds=None, return_details=False, threshold=5.0):
    """
    Stage 1: Only optimize constraint satisfaction

    Loss = penalty(|Re| < threshold) + penalty(|Im| < threshold)
    Goal: Find ANY parameters where |Re| > threshold AND |Im| > threshold
    """
    def smooth_min(x, beta=10.0):
        return -np.log(np.sum(np.exp(-beta * x))) / beta

    try:
        if bounds is not None:
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                if not (low <= param <= high):
                    if return_details:
                        return 1e10, None, None, None
                    else:
                        return 1e10

        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        # Constraint penalties (want to MINIMIZE this)
        abs_real_parts = np.abs(real_parts)
        abs_imag_parts = np.abs(imag_parts)
        min_abs_real = smooth_min(abs_real_parts, beta=10.0)
        min_abs_imag = smooth_min(abs_imag_parts, beta=10.0)

        penalty_real = np.maximum(0, threshold - min_abs_real)**2
        penalty_imag = np.maximum(0, threshold - min_abs_imag)**2

        # Stage 1: ONLY care about constraints
        loss = penalty_real + penalty_imag

        if return_details:
            return loss, eigvals, real_parts, imag_parts
        return loss

    except Exception as e:
        if return_details:
            return 1e10, None, None, None
        return 1e10


def stage2_objective(params, fixed_materials, bounds=None, return_details=False, threshold=5.0, penalty_weight=0.1):
    """
    Stage 2 Modified: Force EP3 coalescence with outlier penalty

    Loss = pairwise_distances + outlier_penalty + constraint_penalties
    Goal: Make all three eigenvalues degenerate (λ₁ = λ₂ = λ₃)
    """
    def smooth_min(x, beta=10.0):
        return -np.log(np.sum(np.exp(-beta * x))) / beta

    try:
        # Check bounds violation
        bounds_violated = False
        if bounds is not None:
            for i, (param, (low, high)) in enumerate(zip(params, bounds)):
                if not (low <= param <= high):
                    bounds_violated = True
                    if return_details:
                        return 1e10, None, None, None, None, None, True
                    else:
                        return 1e10

        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        # ====================================================
        # Core modification: More aggressive degeneracy loss
        # ====================================================

        # 1. Compute pairwise distances
        # This ensures each eigenvalue must be close to all others
        # |λ1-λ2| + |λ1-λ3| + |λ2-λ3|
        diffs = []
        for i in range(len(eigvals)):
            for j in range(i + 1, len(eigvals)):
                # Euclidean distance in complex plane
                dist = np.abs(eigvals[i] - eigvals[j])
                diffs.append(dist)

        # Base degeneracy loss: sum of all pairwise distances
        loss_degeneracy = np.sum(diffs)

        # 2. Outlier penalty
        # If max_dist / min_dist ratio is large, two eigenvalues are close but one is far
        # Apply extra penalty to the maximum distance
        max_dist = np.max(diffs)
        loss_degeneracy += 2.0 * max_dist  # Extra weight on maximum gap

        # ====================================================

        # Constraint penalties
        abs_real_parts = np.abs(real_parts)
        abs_imag_parts = np.abs(imag_parts)
        min_abs_real = smooth_min(abs_real_parts, beta=10.0)
        min_abs_imag = smooth_min(abs_imag_parts, beta=10.0)
        penalty_real = np.maximum(0, threshold - min_abs_real)**2
        penalty_imag = np.maximum(0, threshold - min_abs_imag)**2

        # Stage 2: Heavy penalty to prevent leaving feasible region
        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, loss_degeneracy, (penalty_real, penalty_imag), False
        return loss

    except Exception as e:
        if return_details:
            return 1e10, None, None, None, None, None, False
        return 1e10


def optimize_two_stage(maxiter_stage1=5000, maxiter_stage2=10000, seed=812, threshold=5.0, penalty_weight=0.1):
    """
    Two-stage optimization

    Args:
        maxiter_stage1: CMA-ES iterations for Stage 1 (constraint satisfaction)
        maxiter_stage2: CMA-ES iterations for Stage 2 (degeneracy optimization)
        seed: Random seed
        threshold: Constraint threshold for |Re| and |Im| (default: 5.0)
        penalty_weight: Weight for constraint penalties in Stage 2 (default: 0.1)
    """
    np.random.seed(seed)

    # Create output directory with parameters in name
    output_dir = os.path.join('results', f'TwoStage_s{seed}_i1-{maxiter_stage1}_i2-{maxiter_stage2}_thr{threshold:.1f}_pw{penalty_weight:.2f}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Fixed material parameters
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    # Parameter bounds
    bounds = [
        (2.0, 10.0),     # theta0 (mrad)
        (0.5, 10.0),     # Pt thickness (nm)
        (0.1, 50.0),     # C layer 1
        (0.5, 3.0),      # Fe layer 1 (resonant)
        (0.1, 50.0),     # C layer 2
        (0.5, 3.0),      # Fe layer 2 (resonant)
        (0.1, 50.0),     # C layer 3
        (0.5, 3.0),      # Fe layer 3 (resonant)
        (0.1, 50.0),     # C layer 4
    ]

    # Open log file
    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    def log_print(message, end='\n'):
        print(message, end=end)
        log_file.write(message + end)
        log_file.flush()

    # ============================================================
    # STAGE 1: Find feasible region (|Re| > threshold AND |Im| > threshold)
    # ============================================================
    log_print("=" * 70)
    log_print("STAGE 1: Finding Constraint-Satisfying Parameters")
    log_print(f"Goal: |Re(λ)| > {threshold} AND |Im(λ)| > {threshold} for all eigenvalues")
    log_print("=" * 70)

    x0_stage1 = np.array([(low + high) / 2 for low, high in bounds])
    sigma0_stage1 = np.array([(high - low) * 0.3 for low, high in bounds])

    log_print(f"\nCMA-ES Configuration:")
    log_print(f"  Max iterations: {maxiter_stage1}")
    log_print(f"  Population size: {4 + int(3 * np.log(len(bounds)))}")
    log_print(f"  Initial sigma: {np.mean(sigma0_stage1):.3f}")
    log_print("")

    def stage1_wrapper(params):
        return stage1_objective(params, fixed_materials, bounds=bounds, threshold=threshold)

    es_stage1 = cma.CMAEvolutionStrategy(
        x0_stage1,
        np.mean(sigma0_stage1),
        {
            'bounds': [list(zip(*bounds))[0], list(zip(*bounds))[1]],
            'popsize': 4 + int(3 * np.log(len(bounds))),
            'seed': seed,
            'verbose': -1,
            'maxiter': maxiter_stage1,
        }
    )

    iteration_stage1 = 0
    best_stage1_loss = np.inf
    best_stage1_params = None

    # History tracking for plotting (both stages)
    history = {
        'iteration': [],
        'loss': [],
        'degeneracy': [],
        'phase': [],
        'eigvals_real': [],
        'eigvals_imag': []
    }

    log_print("Running Stage 1...")
    while not es_stage1.stop() and iteration_stage1 < maxiter_stage1:
        solutions = es_stage1.ask()
        fitness = [stage1_wrapper(x) for x in solutions]
        es_stage1.tell(solutions, fitness)
        iteration_stage1 += 1

        current_best_idx = np.argmin(fitness)
        current_best_loss = fitness[current_best_idx]

        if current_best_loss < best_stage1_loss:
            best_stage1_loss = current_best_loss
            best_stage1_params = solutions[current_best_idx]

        # Output every 100 iterations
        if iteration_stage1 % 100 == 0:
            loss, eigvals, real_parts, imag_parts = stage1_objective(
                solutions[current_best_idx], fixed_materials, bounds=bounds, return_details=True, threshold=threshold
            )

            if eigvals is not None:
                min_abs_re = np.min(np.abs(real_parts))
                min_abs_im = np.min(np.abs(imag_parts))
                msg = f"Stage1 iter {iteration_stage1}/{maxiter_stage1}: loss={loss:.6e}, min|Re|={min_abs_re:.3f}, min|Im|={min_abs_im:.3f}"
                log_print(msg)

                # Check if constraints are satisfied
                if min_abs_re > threshold and min_abs_im > threshold:
                    log_print(f"  ✓ CONSTRAINTS SATISFIED! |Re|={min_abs_re:.3f}, |Im|={min_abs_im:.3f}")

                # Record history for plotting
                history['iteration'].append(iteration_stage1)
                history['loss'].append(loss)
                history['degeneracy'].append(0.0)  # Stage 1 doesn't optimize degeneracy
                history['phase'].append('Stage1')
                history['eigvals_real'].append(real_parts.copy())
                history['eigvals_imag'].append(imag_parts.copy())

    # Stage 1 results
    log_print(f"\n{'=' * 70}")
    log_print("STAGE 1 RESULTS:")
    log_print(f"{'=' * 70}")
    log_print(f"Final loss: {best_stage1_loss:.6e}")
    log_print(f"Total iterations: {iteration_stage1}")

    loss_s1, eigvals_s1, real_s1, imag_s1 = stage1_objective(
        best_stage1_params, fixed_materials, bounds=bounds, return_details=True, threshold=threshold
    )

    if eigvals_s1 is not None:
        min_abs_re = np.min(np.abs(real_s1))
        min_abs_im = np.min(np.abs(imag_s1))
        log_print(f"\nFinal eigenvalues:")
        for i, (re, im) in enumerate(zip(real_s1, imag_s1)):
            log_print(f"  λ_{i+1} = {re:+.6f} {im:+.6f}i  (|Re|={np.abs(re):.3f}, |Im|={np.abs(imag_s1[i]):.3f})")

        constraints_met = min_abs_re > threshold and min_abs_im > threshold
        status = "✓ SATISFIED" if constraints_met else "✗ NOT SATISFIED"
        log_print(f"\nConstraint status: {status}")
        log_print(f"  min|Re| = {min_abs_re:.6f} (need > {threshold})")
        log_print(f"  min|Im| = {min_abs_im:.6f} (need > {threshold})")

        if not constraints_met:
            log_print("\n⚠ WARNING: Stage 1 did not find feasible solution!")
            log_print("   Proceeding to Stage 2 anyway with best found parameters.")

    # ============================================================
    # STAGE 2: Optimize degeneracy from Stage 1 result
    # ============================================================
    log_print(f"\n{'=' * 70}")
    log_print("STAGE 2: Optimizing Degeneracy")
    log_print("Goal: λ₁ = λ₂ = λ₃ while maintaining constraints")
    log_print(f"{'=' * 70}")

    # Start from Stage 1 best result with smaller sigma
    x0_stage2 = best_stage1_params
    sigma0_stage2 = np.array([(high - low) * 0.1 for low, high in bounds])  # Smaller exploration

    log_print(f"\nCMA-ES Configuration:")
    log_print(f"  Starting from Stage 1 best result")
    log_print(f"  Max iterations: {maxiter_stage2}")
    log_print(f"  Population size: {4 + int(3 * np.log(len(bounds)))}")
    log_print(f"  Initial sigma: {np.mean(sigma0_stage2):.3f}")
    log_print("")

    def stage2_wrapper(params):
        return stage2_objective(params, fixed_materials, bounds=bounds, threshold=threshold, penalty_weight=penalty_weight)

    es_stage2 = cma.CMAEvolutionStrategy(
        x0_stage2,
        np.mean(sigma0_stage2),
        {
            'bounds': [list(zip(*bounds))[0], list(zip(*bounds))[1]],
            'popsize': 4 + int(3 * np.log(len(bounds))),
            'seed': seed + 1,  # Different seed
            'verbose': -1,
            'maxiter': maxiter_stage2,
        }
    )

    iteration_stage2 = 0
    best_stage2_loss = np.inf
    best_stage2_params = None

    log_print("Running Stage 2...")
    while not es_stage2.stop() and iteration_stage2 < maxiter_stage2:
        solutions = es_stage2.ask()
        fitness = [stage2_wrapper(x) for x in solutions]
        es_stage2.tell(solutions, fitness)
        iteration_stage2 += 1

        current_best_idx = np.argmin(fitness)
        current_best_loss = fitness[current_best_idx]

        if current_best_loss < best_stage2_loss:
            best_stage2_loss = current_best_loss
            best_stage2_params = solutions[current_best_idx]

        # Output every 100 iterations
        if iteration_stage2 % 100 == 0:
            loss, eigvals, real_parts, imag_parts, loss_deg, (pen_re, pen_im), bounds_viol = stage2_objective(
                solutions[current_best_idx], fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )

            if eigvals is not None and not bounds_viol:
                re_std = np.std(real_parts)
                im_std = np.std(imag_parts)
                min_abs_re = np.min(np.abs(real_parts))
                min_abs_im = np.min(np.abs(imag_parts))
                msg = f"Stage2 iter {iteration_stage2}/{maxiter_stage2}: loss={loss:.6e}, degeneracy={loss_deg:.6e}, Std(Re)={re_std:.6e}, Std(Im)={im_std:.6e}"
                log_print(msg)
                log_print(f"  Constraints: min|Re|={min_abs_re:.3f}, min|Im|={min_abs_im:.3f}")
                # Display all three eigenvalues
                log_print("  Eigenvalues:")
                for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
                    log_print(f"    λ_{i+1} = {re:+.6f} {im:+.6f}i  (|Re|={np.abs(re):.3f}, |Im|={np.abs(im):.3f})")

                # Record history for plotting
                history['iteration'].append(iteration_stage2)
                history['loss'].append(loss)
                history['degeneracy'].append(loss_deg)
                history['phase'].append('Stage2')
                history['eigvals_real'].append(real_parts.copy())
                history['eigvals_imag'].append(imag_parts.copy())
            elif bounds_viol:
                log_print(f"Stage2 iter {iteration_stage2}/{maxiter_stage2}: ⚠ BOUNDS VIOLATION - loss={loss:.2e} (penalty)")


    # Visualization
    log_print("\n" + "=" * 70)
    log_print("Generating plots...")
    log_print("=" * 70)
    plot_results(history, output_dir)

    # Final results
    log_print(f"\n{'=' * 70}")
    log_print("FINAL RESULTS (STAGE 2):")
    log_print(f"{'=' * 70}")

    loss_final, eigvals_final, real_final, imag_final, loss_deg_final, (pen_re_final, pen_im_final), _ = stage2_objective(
        best_stage2_params, fixed_materials, bounds=bounds, return_details=True, threshold=threshold, penalty_weight=penalty_weight
    )

    theta0, Layers, C0 = build_layers(best_stage2_params, fixed_materials)

    log_print(f"\nTotal Loss = {loss_final:.6e}")
    log_print(f"  ├─ Degeneracy Loss = {loss_deg_final:.6e}")
    log_print(f"  ├─ Penalty (Re) = {pen_re_final:.6e}")
    log_print(f"  └─ Penalty (Im) = {pen_im_final:.6e}")

    log_print(f"\ntheta0 = {theta0:.15f} mrad")
    log_print(f"C0     = {C0:.15f} (fixed)\n")

    log_print("Layer Structure (nm):")
    layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Pt(sub)']
    for i, (name, layer) in enumerate(zip(layer_names, Layers)):
        thickness = layer[1]
        resonant = ' (resonant)' if layer[2] == 1 else ''
        log_print(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}")

    log_print("\nEigenvalue Analysis:")
    log_print("  λᵢ = Re + Im·i")
    log_print("  " + "-" * 50)
    for i, (re, im) in enumerate(zip(real_final, imag_final)):
        log_print(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i")

    re_std = np.std(real_final)
    im_std = np.std(imag_final)
    min_abs_re = np.min(np.abs(real_final))
    min_abs_im = np.min(np.abs(imag_final))

    log_print("\n  Degeneracy Check:")
    log_print(f"    Std(Re) = {re_std:.6e}  {'✓' if re_std < 0.01 else '✗'}")
    log_print(f"    Std(Im) = {im_std:.6e}  {'✓' if im_std < 0.01 else '✗'}")

    log_print("\n  Constraint Check:")
    log_print(f"    min|Re| = {min_abs_re:.6f}  {'✓' if min_abs_re > threshold else '✗'} (need > {threshold})")
    log_print(f"    min|Im| = {min_abs_im:.6f}  {'✓' if min_abs_im > threshold else '✗'} (need > {threshold})")

    # Save results
    params_path = os.path.join(output_dir, 'exceptional_point_params.npz')
    thicknesses = [layer[1] for layer in Layers[:-1]]
    np.savez(params_path,
             params=best_stage2_params,
             theta0=theta0,
             C0=C0,
             thicknesses=thicknesses,
             loss=loss_final,
             stage1_loss=best_stage1_loss,
             stage2_loss=best_stage2_loss)
    log_print(f"\nParameters saved to: {params_path}")

    log_file.close()
    print(f"Log saved to: {log_file_path}")

    return best_stage2_params, theta0, Layers, C0, output_dir


def plot_results(history, output_dir):
    """
    Visualize two-stage optimization history with publication-quality plots

    Args:
        history: Dictionary containing iteration, loss, degeneracy, phase, and eigenvalues
        output_dir: Directory to save the figure
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
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.major.size'] = 5

    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

    iterations = np.array(history['iteration'])
    losses = np.array(history['loss'])
    degeneracies = np.array(history['degeneracy'])
    phases = np.array(history['phase'])
    eigvals_real = history['eigvals_real']
    eigvals_imag = history['eigvals_imag']

    # ===== Left plot: Total Loss vs Iteration (Stage 2 only has degeneracy) =====
    phase_colors = {'Stage1': '#e74c3c', 'Stage2': '#3498db'}

    # Stage 1: Constraint penalties only
    mask_s1 = phases == 'Stage1'
    if np.any(mask_s1):
        ax1.semilogy(iterations[mask_s1], losses[mask_s1], 'o-',
                    color=phase_colors['Stage1'],
                    label='Stage 1 (Constraints)', markersize=4, alpha=0.8)

    # Stage 2: Degeneracy
    mask_s2 = phases == 'Stage2'
    if np.any(mask_s2):
        ax1.semilogy(iterations[mask_s2], degeneracies[mask_s2], 's-',
                    color=phase_colors['Stage2'],
                    label='Stage 2 (Degeneracy)', markersize=4, alpha=0.8)

    ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=14, fontweight='bold')
    ax1.set_title('Two-Stage Optimization Progress', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.legend(fontsize=11, framealpha=0.9, loc='best')
    ax1.tick_params(labelsize=11)

    # ===== Middle plot: Degeneracy Loss (Stage 2 only) =====
    if np.any(mask_s2):
        ax2.semilogy(iterations[mask_s2], degeneracies[mask_s2], 'o-',
                    color='#2ecc71', markersize=4, alpha=0.8)

    ax2.set_xlabel('Iteration (Stage 2)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Degeneracy Loss (Variance)', fontsize=14, fontweight='bold')
    ax2.set_title('Eigenvalue Degeneracy Progress', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax2.tick_params(labelsize=11)

    # ===== Right plot: Eigenvalue Distances =====
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

    ax3.semilogy(iterations, distances_12, 'o-', color='#d62728',
                label='|λ₁ - λ₂|', markersize=3, alpha=0.8)
    ax3.semilogy(iterations, distances_13, 's-', color='#9467bd',
                label='|λ₁ - λ₃|', markersize=3, alpha=0.8)
    ax3.semilogy(iterations, distances_23, '^-', color='#8c564b',
                label='|λ₂ - λ₃|', markersize=3, alpha=0.8)

    ax3.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Distance', fontsize=14, fontweight='bold')
    ax3.set_title('Eigenvalue Convergence', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax3.legend(fontsize=11, framealpha=0.9, loc='best')
    ax3.tick_params(labelsize=11)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'two_stage_result.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {fig_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Two-Stage Exceptional Point Finder')
    parser.add_argument('-i1', '--stage1-iter', type=int, default=5000,
                        help='Stage 1 iterations (constraint satisfaction, default: 5000)')
    parser.add_argument('-i2', '--stage2-iter', type=int, default=10000,
                        help='Stage 2 iterations (degeneracy optimization, default: 10000)')
    parser.add_argument('-s', '--seed', type=int, default=812,
                        help='Random seed (default: 812)')
    parser.add_argument('-t', '--threshold', type=float, default=5.0,
                        help='Constraint threshold for |Re| and |Im| (default: 5.0)')
    parser.add_argument('-pw', '--penalty-weight', type=float, default=0.1,
                        help='Penalty weight for constraint violations in Stage 2 (default: 0.1)')
    args = parser.parse_args()

    optimize_two_stage(
        maxiter_stage1=args.stage1_iter,
        maxiter_stage2=args.stage2_iter,
        seed=args.seed,
        threshold=args.threshold,
        penalty_weight=args.penalty_weight
    )
