"""
Scan eigenvalue range in parameter space
扫描参数空间中特征值的实部和虚部能达到的范围
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from tqdm import tqdm

# Import from green function-new.py
current_dir = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun


def build_layers(params, fixed_materials):
    """Build layer structure from parameters"""
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


def compute_eigenvalues(params, fixed_materials):
    """Compute eigenvalues for given parameters"""
    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)
        return eigvals
    except Exception as e:
        return None


def scan_parameter_space(n_samples=5000, seed=812):
    """
    Random sampling in parameter space

    Args:
        n_samples: Number of random samples (default: 5000)
        seed: Random seed
    """
    np.random.seed(seed)

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

    print("="*70)
    print("Scanning eigenvalue range in parameter space")
    print(f"Number of samples: {n_samples}")
    print(f"Random seed: {seed}")
    print("="*70)

    # Store all eigenvalues
    all_real_parts = []
    all_imag_parts = []
    all_abs_real_parts = []
    all_abs_imag_parts = []

    # Random sampling
    print("\nSampling parameter space...")
    for i in tqdm(range(n_samples), desc="Sampling"):
        # Generate random parameters within bounds
        params = np.array([np.random.uniform(low, high) for low, high in bounds])

        # Compute eigenvalues
        eigvals = compute_eigenvalues(params, fixed_materials)

        if eigvals is not None:
            real_parts = np.real(eigvals)
            imag_parts = np.imag(eigvals)

            all_real_parts.extend(real_parts)
            all_imag_parts.extend(imag_parts)
            all_abs_real_parts.extend(np.abs(real_parts))
            all_abs_imag_parts.extend(np.abs(imag_parts))

    all_real_parts = np.array(all_real_parts)
    all_imag_parts = np.array(all_imag_parts)
    all_abs_real_parts = np.array(all_abs_real_parts)
    all_abs_imag_parts = np.array(all_abs_imag_parts)

    # Statistics
    print("\n" + "="*70)
    print("STATISTICS:")
    print("="*70)
    print(f"\nTotal eigenvalues collected: {len(all_real_parts)}")
    print(f"\nReal parts (Re):")
    print(f"  Range: [{np.min(all_real_parts):.6f}, {np.max(all_real_parts):.6f}]")
    print(f"  Mean:  {np.mean(all_real_parts):.6f}")
    print(f"  Std:   {np.std(all_real_parts):.6f}")
    print(f"\nImaginary parts (Im):")
    print(f"  Range: [{np.min(all_imag_parts):.6f}, {np.max(all_imag_parts):.6f}]")
    print(f"  Mean:  {np.mean(all_imag_parts):.6f}")
    print(f"  Std:   {np.std(all_imag_parts):.6f}")
    print(f"\nAbsolute values:")
    print(f"  |Re| max: {np.max(all_abs_real_parts):.6f}")
    print(f"  |Im| max: {np.max(all_abs_imag_parts):.6f}")

    # Count how many satisfy constraints
    mask_re = all_abs_real_parts > 5.0
    mask_im = all_abs_imag_parts > 5.0
    mask_both = mask_re & mask_im

    print(f"\nConstraint satisfaction:")
    print(f"  |Re| > 5: {np.sum(mask_re)} / {len(all_real_parts)} ({np.sum(mask_re)/len(all_real_parts)*100:.2f}%)")
    print(f"  |Im| > 5: {np.sum(mask_im)} / {len(all_imag_parts)} ({np.sum(mask_im)/len(all_imag_parts)*100:.2f}%)")
    print(f"  Both:     {np.sum(mask_both)} / {len(all_real_parts)} ({np.sum(mask_both)/len(all_real_parts)*100:.2f}%)")

    # Find maximum |Re| and |Im| that can be achieved simultaneously
    print(f"\nSimultaneous maximum:")
    max_both_idx = np.argmax(all_abs_real_parts + all_abs_imag_parts)
    print(f"  Max (|Re| + |Im|) at same eigenvalue:")
    print(f"    Re = {all_real_parts[max_both_idx]:+.6f}")
    print(f"    Im = {all_imag_parts[max_both_idx]:+.6f}")
    print(f"    |Re| = {all_abs_real_parts[max_both_idx]:.6f}")
    print(f"    |Im| = {all_abs_imag_parts[max_both_idx]:.6f}")

    # Plotting
    print("\n" + "="*70)
    print("Generating scatter plots...")
    print("="*70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Re vs Im (all eigenvalues)
    ax = axes[0, 0]
    ax.scatter(all_real_parts, all_imag_parts, s=1, alpha=0.3, c='blue')
    ax.axhline(y=-5, color='red', linestyle='--', linewidth=1, alpha=0.7, label='|Im|=5')
    ax.axhline(y=5, color='red', linestyle='--', linewidth=1, alpha=0.7)
    ax.axvline(x=-5, color='green', linestyle='--', linewidth=1, alpha=0.7, label='|Re|=5')
    ax.axvline(x=5, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Re(λ)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Im(λ)', fontsize=12, fontweight='bold')
    ax.set_title(f'Eigenvalue Distribution (n={n_samples} samples)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 2: |Re| vs |Im| (absolute values)
    ax = axes[0, 1]
    ax.scatter(all_abs_real_parts, all_abs_imag_parts, s=1, alpha=0.3, c='purple')
    ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='|Im|=5')
    ax.axvline(x=5, color='green', linestyle='--', linewidth=2, label='|Re|=5')
    ax.set_xlabel('|Re(λ)|', fontsize=12, fontweight='bold')
    ax.set_ylabel('|Im(λ)|', fontsize=12, fontweight='bold')
    ax.set_title('Absolute Value Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Plot 3: Histogram of |Re|
    ax = axes[1, 0]
    ax.hist(all_abs_real_parts, bins=100, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='|Re|=5')
    ax.set_xlabel('|Re(λ)|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of |Re(λ)|', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    # Plot 4: Histogram of |Im|
    ax = axes[1, 1]
    ax.hist(all_abs_imag_parts, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=5, color='red', linestyle='--', linewidth=2, label='|Im|=5')
    ax.set_xlabel('|Im(λ)|', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of |Im(λ)|', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, 'eigenvalue_range_scan.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {fig_path}")

    plt.show()

    return all_real_parts, all_imag_parts, all_abs_real_parts, all_abs_imag_parts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Scan eigenvalue range in parameter space')
    parser.add_argument('-n', '--samples', type=int, default=5000,
                        help='Number of random samples (default: 5000)')
    parser.add_argument('-s', '--seed', type=int, default=812,
                        help='Random seed (default: 812)')
    args = parser.parse_args()

    scan_parameter_space(n_samples=args.samples, seed=args.seed)
