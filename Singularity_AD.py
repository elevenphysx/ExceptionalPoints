"""
Exceptional Point Finder with Automatic Differentiation
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find EP3 where λ₁ = λ₂ = λ₃ using algebraic conditions

Key innovation: Use trace/det instead of eigvals for AD compatibility
EP3 condition: 3b - a² = 0 and 27c + 9ab + 2a³ = 0
where a = -tr(G1), b = [(tr G1)² - tr(G1²)]/2, c = -det(G1)
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
# Objective: Find EP3 using algebraic conditions (AD-friendly)
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
        (Platinum, 10.0, 0),            # Pt substrate (thickness doesn't matter)
    ]

    return theta0, Layers, C0


def objective_trace_det(params, fixed_materials, return_details=False):
    """
    Objective function using algebraic EP3 conditions

    For 3×3 matrix G1, EP3 occurs when characteristic polynomial has triple root:

    Characteristic polynomial: det(λI - G1) = λ³ + a₂λ² + a₁λ + a₀
    where:
        a₂ = -tr(G1)
        a₁ = [(tr G1)² - tr(G1²)] / 2
        a₀ = -det(G1)

    EP3 conditions (triple root):
        Discriminant = 0:  3a₁ - a₂² = 0
        Resultant = 0:     27a₀ + 9a₂a₁ + 2a₂³ = 0

    For complex matrices, we apply conditions to real and imaginary parts separately.

    Loss = |3b - a²| + |27c + 9ab + 2a³|
    """
    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)

        # Compute Green matrix (numpy version)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

        # Compute characteristic polynomial coefficients
        tr_G1 = np.trace(G1)
        tr_G1_sq = np.trace(G1 @ G1)
        det_G1 = np.linalg.det(G1)

        a = -tr_G1  # coefficient of λ²
        b = 0.5 * (tr_G1**2 - tr_G1_sq)  # coefficient of λ
        c = -det_G1  # constant term

        # EP3 conditions (for complex coefficients, check magnitude)
        condition1 = 3*b - a**2
        condition2 = 27*c + 9*a*b + 2*a**3

        # Loss: sum of magnitudes (works for complex values)
        loss = np.abs(condition1) + np.abs(condition2)

        if return_details:
            # Compute eigenvalues for verification only
            eigvals = np.linalg.eigvals(G1)
            return loss, eigvals, np.real(eigvals), np.imag(eigvals), a, b, c, G, G1

        return loss

    except Exception as e:
        print(f"Error in objective: {e}")
        return 1e10  # Large penalty for failed evaluation


def optimize_exceptional_point():
    """
    Main optimization routine: DE global search + gradient-based refinement
    """
    # Fixed material parameters (physical constants)
    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    # Convergence threshold for early stopping
    convergence_threshold = 1e-6
    iteration_count = [0]

    # Open log file
    log_file = open('optimization_log_AD.txt', 'w', encoding='utf-8')
    log_file.write("Exceptional Point Optimization Log (Algebraic Method)\n")
    log_file.write("=" * 70 + "\n\n")

    # Callback function to display progress
    def callback(xk, convergence=None):
        """Display eigenvalues at each iteration"""
        iteration_count[0] += 1

        # Compute eigenvalues for current best solution
        loss, eigvals, real_parts, imag_parts, a, b, c, G, G1 = objective_trace_det(
            xk, fixed_materials, return_details=True
        )

        # Display progress (to both console and file)
        output = f"\n--- Iteration {iteration_count[0]} ---\n"
        output += f"Loss = {loss:.6e}\n"
        output += "Eigenvalues:\n"
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            output += f"  λ_{i+1} = {re:+10.6f} {im:+10.6f}i\n"

        # Check degeneracy
        re_std = np.std(real_parts)
        im_std = np.std(imag_parts)
        output += f"Std(Re) = {re_std:.6e}, Std(Im) = {im_std:.6e}\n"

        # EP3 conditions
        cond1 = 3*b - a**2
        cond2 = 27*c + 9*a*b + 2*a**3
        output += f"Algebraic: |3b-a²| = {np.abs(cond1):.3e}, |27c+9ab+2a³| = {np.abs(cond2):.3e}\n"

        # Write to both console and file
        print(output, end='')
        log_file.write(output)
        log_file.flush()

        # Early stopping
        if loss < convergence_threshold:
            msg = f"\n✓ Converged! Loss < {convergence_threshold}\n"
            print(msg, end='')
            log_file.write(msg)
            log_file.flush()
            return True

        return False

    # Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, C0]
    bounds = [
        (0.1, 10.0),     # theta0 (mrad) - must be > 0 to avoid division by zero
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
    print("Exceptional Point Finder with Algebraic Conditions")
    print("Method: trace/det (no eigenvalue computation in optimization)")
    print("Strategy: DE global search → Gradient-based refinement")
    print("=" * 70)

    # Wrapper function for multiprocessing compatibility
    def objective_wrapper(params):
        return objective_trace_det(params, fixed_materials)

    # Phase 1: Global search with Differential Evolution
    print("\n[Phase 1] Differential Evolution (Global Search)...")
    result_de = differential_evolution(
        objective_wrapper,
        bounds,
        maxiter=500,      # Reduced for faster testing
        popsize=15,
        strategy='best1bin',
        seed=812,
        disp=True,
        workers=1,        # Use single process to avoid pickle issues
        updating='immediate',
        polish=False,
        callback=callback,  # Display eigenvalues at each iteration
        atol=convergence_threshold,  # Absolute tolerance
        tol=0.01          # Relative tolerance
    )

    print("\n" + "=" * 70)
    print("DE Result:")
    print(f"Loss = {result_de.fun:.6e}")
    print("=" * 70)

    # Phase 2: Gradient-based refinement
    # Note: scipy uses numerical gradients (finite differences) automatically
    # Since objective_trace_det uses trace/det instead of eigvals,
    # the numerical gradient should be much more stable

    if result_de.fun < 0.1:  # If DE found a reasonable solution
        print("\n[Phase 2] Gradient-based Refinement (L-BFGS-B with numerical gradients)...")
        print("Note: Using trace/det method makes numerical gradients stable")

        result_local = minimize(
            objective_wrapper,
            result_de.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': True}
        )

        if result_local.fun < result_de.fun:
            print("\n✓ Gradient refinement improved the result!")
            final_result = result_local
        else:
            print("\n✗ Gradient refinement did not improve (DE solution kept)")
            final_result = result_de
    else:
        print("\n[Phase 2] Skipped (DE loss too high)")
        final_result = result_de

    # Display final results
    output = "\n" + "=" * 70 + "\n"
    output += "FINAL RESULT:\n"
    output += "=" * 70 + "\n"

    loss, eigvals, real_parts, imag_parts, a, b, c, G, G1 = objective_trace_det(
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

    output += "\nAlgebraic Conditions Check:\n"
    output += f"  a = -tr(G1)            = {a:.6f}\n"
    output += f"  b = [tr²-tr(G²)]/2     = {b:.6f}\n"
    output += f"  c = -det(G1)           = {c:.6f}\n"

    cond1 = 3*b - a**2
    cond2 = 27*c + 9*a*b + 2*a**3
    output += f"\n  EP3 Condition 1: |3b - a²|              = {np.abs(cond1):.6e}  {'✓' if np.abs(cond1) < 1e-3 else '✗'}\n"
    output += f"  EP3 Condition 2: |27c + 9ab + 2a³|      = {np.abs(cond2):.6e}  {'✓' if np.abs(cond2) < 1e-3 else '✗'}\n"

    output += "\nEigenvalue Verification (computed for validation):\n"
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
    plot_results(eigvals, real_parts, imag_parts, loss)

    # Close log file
    log_file.close()
    print("\nOptimization log saved to: optimization_log_AD.txt")

    return final_result, theta0, Layers, C0


def plot_results(eigvals, real_parts, imag_parts, loss):
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
    ax1.set_title(f'Eigenvalues in Complex Plane (Loss={loss:.3e})', fontsize=16, fontweight='bold')
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
    plt.savefig('exceptional_point_AD.png', dpi=300, bbox_inches='tight')
    print("\nFigure saved as: exceptional_point_AD.png")
    plt.show()


if __name__ == "__main__":
    result, theta0, Layers, C0 = optimize_exceptional_point()

    # Save results (save parameter array instead of Layers structure)
    # Extract layer thicknesses for saving
    thicknesses = [layer[1] for layer in Layers[:-1]]  # Exclude substrate

    np.savez('exceptional_point_params_AD.npz',
             params=result.x,  # All optimization parameters
             theta0=theta0,
             C0=C0,
             thicknesses=thicknesses,  # Layer thicknesses
             loss=result.fun)
    print("\nParameters saved to: exceptional_point_params_AD.npz")
