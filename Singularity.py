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
    params = [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, C0]

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter
    """
    theta0 = params[0]
    thicknesses = params[1:9]  # 8 finite layers
    C0 = params[9]

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
        (Platinum, 10.0, 0),            # Pt substrate (thickness doesn't matter, will be treated as inf)
    ]

    return theta0, Layers, C0


def objective_function(params, fixed_materials, return_details=False):
    """
    Objective function: make all eigenvalues degenerate (coincide)

    For exceptional point: λ₁ = λ₂ = λ₃
    i.e., Re(λ₁) = Re(λ₂) = Re(λ₃) and Im(λ₁) = Im(λ₂) = Im(λ₃)

    Loss = variance(Re) + variance(Im)
    """
    try:
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
        return 1e10  # Large penalty for failed evaluation


def optimize_exceptional_point():
    """
    Main optimization routine using Differential Evolution
    """
    # Fixed material parameters (physical constants)
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    # Convergence threshold for early stopping
    convergence_threshold = 1e-6
    iteration_count = [0]  # Use list to allow modification in callback

    # Open log file for writing iteration history
    log_file = open('optimization_log.txt', 'w', encoding='utf-8')
    log_file.write("Exceptional Point Optimization Log\n")
    log_file.write("=" * 70 + "\n\n")

    # Callback function to display progress and check convergence
    def callback(xk, convergence=None):
        """
        Callback function called at each iteration
        xk: current best solution
        convergence: convergence metric from DE
        """
        iteration_count[0] += 1

        # Compute eigenvalues for current best solution
        loss, eigvals, real_parts, imag_parts, G, G1 = objective_function(
            xk, fixed_materials, return_details=True
        )

        # Display progress (to both console and file)
        output = f"\n--- Iteration {iteration_count[0]} ---\n"
        output += f"Loss (variance) = {loss:.6e}\n"
        output += "Eigenvalues:\n"
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            output += f"  λ_{i+1} = {re:+10.6f} {im:+10.6f}i\n"

        # Check degeneracy
        re_std = np.std(real_parts)
        im_std = np.std(imag_parts)
        output += f"Std(Re) = {re_std:.6e}, Std(Im) = {im_std:.6e}\n"

        # Write to both console and file
        print(output, end='')
        log_file.write(output)
        log_file.flush()

        # Early stopping check
        if loss < convergence_threshold:
            msg = f"\n✓ Converged! Loss < {convergence_threshold}\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            return True  # Stop optimization

        return False  # Continue

    # Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, C0]
    bounds = [
        (0.0, 10.0),     # theta0 (mrad)
        (0.5, 10.0),     # Pt thickness (nm)
        (1.0, 50.0),     # C layer 1
        (0.5, 3.0),      # Fe layer 1 (resonant)
        (1.0, 50.0),     # C layer 2
        (0.5, 3.0),      # Fe layer 2 (resonant)
        (1.0, 50.0),     # C layer 3
        (0.5, 3.0),      # Fe layer 3 (resonant)
        (1.0, 50.0),     # C layer 4
        (3.0, 5.0),      # C0
    ]

    print("=" * 70)
    print("Starting Global Search (Differential Evolution)...")
    print("Target: Find Exceptional Point where λ₁ = λ₂ = λ₃")
    print("=" * 70)

    # Wrapper function for multiprocessing compatibility
    def objective_wrapper(params):
        return objective_function(params, fixed_materials)

    # Phase 1: Global search with Differential Evolution
    result_de = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=1000,
        popsize=15,
        strategy='best1bin',
        seed=42,
        disp=True,
        workers=1,         # Use single process to avoid pickle issues
        updating='immediate',
        polish=False,      # Manual refinement later
        callback=callback, # Display eigenvalues at each iteration
        atol=convergence_threshold,  # Absolute tolerance for early stopping
        tol=0.01           # Relative tolerance
    )

    print("\n" + "=" * 70)
    print("Differential Evolution Result:")
    print(f"Loss = {result_de.fun:.6e}")
    print("=" * 70)

    # Phase 2: Local refinement (if DE found a good solution)
    if result_de.fun < 1e-2:
        print("\nStarting Local Refinement (L-BFGS-B)...")
        result_local = minimize(
            objective_wrapper,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'disp': True}
        )

        if result_local.fun < result_de.fun:
            print("\nLocal refinement improved the result!")
            final_result = result_local
        else:
            final_result = result_de
    else:
        final_result = result_de

    # Display final results
    output = "\n" + "=" * 70 + "\n"
    output += "FINAL RESULT:\n"
    output += "=" * 70 + "\n"

    loss, eigvals, real_parts, imag_parts, G, G1 = objective_function(
        final_result.x, fixed_materials, return_details=True
    )

    theta0, Layers, C0 = build_layers(final_result.x, fixed_materials)

    output += f"\nLoss = {loss:.6e}\n\n"
    output += f"theta0 = {theta0:.4f} mrad\n"
    output += f"C0     = {C0:.4f}\n\n"

    output += "Layer Structure (nm):\n"
    layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Pt(sub)']
    for i, (name, layer) in enumerate(zip(layer_names, Layers)):
        thickness = layer[1]
        resonant = ' (resonant)' if layer[2] == 1 else ''
        output += f"  Layer {i}: {name:8s} = {thickness:7.3f} nm{resonant}\n"

    output += "\nEigenvalue Analysis:\n"
    output += "  λᵢ = Re + Im·i\n"
    output += "  " + "-" * 50 + "\n"
    for i, (eig, re, im) in enumerate(zip(eigvals, real_parts, imag_parts)):
        output += f"  λ_{i+1} = {re:+10.6f} {im:+10.6f}i\n"

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
    plot_results(eigvals, real_parts, imag_parts)

    # Close log file
    log_file.close()
    print("\nOptimization log saved to: optimization_log.txt")

    return final_result, theta0, Layers, C0


def plot_results(eigvals, real_parts, imag_parts):
    """
    Visualize eigenvalues in complex plane
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Complex plane
    ax1 = axes[0]
    ax1.scatter(real_parts, imag_parts, s=200, c='red', marker='o',
                edgecolors='black', linewidths=2, zorder=3, label='Eigenvalues')

    # Mark the mean point (target EP)
    mean_re = np.mean(real_parts)
    mean_im = np.mean(imag_parts)
    ax1.scatter(mean_re, mean_im, s=300, c='green', marker='*',
                edgecolors='black', linewidths=2, zorder=4, label='Mean (EP Target)')

    ax1.set_xlabel('Re(λ)', fontsize=14)
    ax1.set_ylabel('Im(λ)', fontsize=14)
    ax1.set_title('Eigenvalues in Complex Plane', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='k', linewidth=0.8)
    ax1.axvline(0, color='k', linewidth=0.8)
    ax1.legend(fontsize=12)
    ax1.set_aspect('equal')

    # Right: Bar chart comparison
    ax2 = axes[1]
    indices = np.arange(len(real_parts))
    width = 0.35

    bars1 = ax2.bar(indices - width/2, real_parts, width, label='Re(λ)', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(indices + width/2, imag_parts, width, label='Im(λ)', alpha=0.8, color='coral')

    # Add mean lines
    ax2.axhline(np.mean(real_parts), color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Mean Re')
    ax2.axhline(np.mean(imag_parts), color='red', linestyle='--', linewidth=2, alpha=0.5, label='Mean Im')

    ax2.set_xlabel('Eigenvalue Index', fontsize=14)
    ax2.set_ylabel('Value', fontsize=14)
    ax2.set_title('Real vs Imaginary Parts', fontsize=16, fontweight='bold')
    ax2.set_xticks(indices)
    ax2.set_xticklabels([f'λ_{i+1}' for i in indices])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('exceptional_point_result.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: exceptional_point_result.png")
    plt.show()


if __name__ == "__main__":
    result, theta0, Layers, C0 = optimize_exceptional_point()

    # Save results
    np.savez('exceptional_point_params.npz',
             theta0=theta0,
             Layers=Layers,
             C0=C0,
             loss=result.fun)
    print("\nParameters saved to: exceptional_point_params.npz")
