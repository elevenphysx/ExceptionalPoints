"""
Verify parameter precision impact on exceptional points
Simulates experimental precision limitations by rounding optimized parameters

Usage: python verify_precision.py <result_folder> [--precision N]
Example: python verify_precision.py ep3_w10_s810_DE100000_LB5000 --precision 1
"""

import numpy as np
import sys
import os
import importlib.util

# Import shared configuration
from config import (
    FIXED_MATERIALS, Platinum, Carbon, SiC, Iron,
    PARAM_NAMES, PARAM_LABELS, PARAM_NAMES_EP4, PARAM_LABELS_EP4,
    IMAG_MIN
)

# Import common functions
from common_functions import build_layers, build_layers_ep4, objective_function_control

# Import plotting utilities
from plotting_utils import scan_parameters_around_optimum

# ============================================================
# Dynamic Import of Green Function
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
green_script_path = os.path.join(current_dir, "green_function.py")

if not os.path.exists(green_script_path):
    green_script_path = "green_function.py"

try:
    spec = importlib.util.spec_from_file_location("green_function", green_script_path)
    green_function_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(green_function_module)
    GreenFun = green_function_module.GreenFun
except Exception as e:
    print(f"Error importing GreenFun: {e}")
    print("Make sure 'green_function.py' is in the same directory.")
    sys.exit(1)


def verify_precision(result_folder, precision=1, angle_precision=2, scan_range=1e-4, n_points=21):
    """
    Verify experimental precision impact by rounding parameters

    Args:
        result_folder: name or path of results folder
        precision: decimal places for thickness parameters (default: 1)
        angle_precision: decimal places for angle (theta0) (default: 2)
        scan_range: scan range around rounded value (default: 1e-4)
        n_points: number of scan points (default: 21)
    """
    # Handle both absolute path and relative path
    if os.path.isabs(result_folder):
        result_dir = result_folder
    else:
        result_dir = os.path.join('results', result_folder)

    if not os.path.exists(result_dir):
        print(f"Error: Result directory not found: {result_dir}")
        sys.exit(1)

    # Load saved parameters
    params_file = os.path.join(result_dir, 'params.npz')
    if not os.path.exists(params_file):
        print(f"Error: params.npz not found in {result_dir}")
        sys.exit(1)

    print("=" * 70)
    print(f"Loading parameters from: {params_file}")
    data = np.load(params_file)
    params_original = data['params']
    loss_original = data['loss']

    print(f"Original loss: {loss_original:.6e}")
    print(f"Number of parameters: {len(params_original)}")

    # Determine EP3 or EP4
    ep_type = None
    if len(params_original) == 9:
        print("Detected: EP3 (3 resonant layers)")
        param_names = PARAM_NAMES
        param_labels = PARAM_LABELS
        build_layers_func = build_layers
        ep_type = "EP3 (3 resonant layers)"
    elif len(params_original) == 11:
        print("Detected: EP4 (4 resonant layers)")
        param_names = PARAM_NAMES_EP4
        param_labels = PARAM_LABELS_EP4
        build_layers_func = build_layers_ep4
        ep_type = "EP4 (4 resonant layers)"
    else:
        print(f"Warning: Unexpected parameter count: {len(params_original)}")
        param_names = [f"param_{i}" for i in range(len(params_original))]
        param_labels = [f"Param {i}" for i in range(len(params_original))]
        build_layers_func = build_layers
        ep_type = f"Unknown ({len(params_original)} parameters)"

    # Detect SiC or C
    folder_name = os.path.basename(result_dir)
    spacer_type = None
    if 'sic' in folder_name.lower():
        print("Detected: SiC spacer layer")
        fixed_materials = (Platinum, SiC, Iron)
        spacer_type = "SiC (Silicon Carbide)"
    else:
        print("Detected: C (Carbon) spacer layer")
        fixed_materials = FIXED_MATERIALS
        spacer_type = "C (Carbon)"

    # Detect constraint
    constraint_type = None
    if 'noconstraint' in folder_name.lower():
        print("Detected: NO CONSTRAINT (penalty_weight=0)")
        penalty_weight = 0
        constraint_type = "No constraint (penalty_weight=0)"
    else:
        print("Using constraint: |Im(λ)| >= 5.0")
        penalty_weight = None
        constraint_type = f"|Im(λ)| >= {IMAG_MIN}"

    print("=" * 70)

    # Round parameters to specified precision (angle separate from thicknesses)
    params_rounded = params_original.copy()
    params_rounded[0] = np.round(params_original[0], decimals=angle_precision)  # theta0
    params_rounded[1:] = np.round(params_original[1:], decimals=precision)      # thicknesses

    print(f"\nParameter precision:")
    print(f"  Angle (theta0):  {angle_precision} decimal place(s)")
    print(f"  Thicknesses:     {precision} decimal place(s)")
    print("-" * 70)
    print(f"{'Parameter':<15} {'Original':<20} {'Rounded':<20} {'Diff':<15}")
    print("-" * 70)
    for i, (name, orig, rounded) in enumerate(zip(param_names, params_original, params_rounded)):
        diff = abs(orig - rounded)
        print(f"{name:<15} {orig:<20.10f} {rounded:<20.10f} {diff:<15.10f}")
    print("-" * 70)

    # Recalculate eigenvalues with rounded parameters
    print("\nRecalculating eigenvalues with rounded parameters...")
    loss_rounded, eigvals, real_parts, imag_parts, G, G_shifted, spread, pen_im = objective_function_control(
        params_rounded, fixed_materials, GreenFun,
        return_details=True,
        build_layers_func=build_layers_func,
        penalty_weight=penalty_weight
    )

    print("\n" + "=" * 70)
    print("ROUNDED PARAMETER RESULTS")
    print("=" * 70)
    print(f"Loss:              {loss_rounded:.6e}")
    print(f"Spread (Variance): {spread:.6e}")
    print(f"Penalty Im:        {pen_im:.6e}")
    print(f"\nLoss change: {loss_original:.6e} → {loss_rounded:.6e} (Δ={loss_rounded-loss_original:.6e})")

    print("\nEigenvalues:")
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        print(f"  λ_{i+1} = {re:+.10f} {im:+.10f}i")

    print("\nDegeneracy Check:")
    print(f"  Std(Re) = {np.std(real_parts):.6e}")
    print(f"  Std(Im) = {np.std(imag_parts):.6e}")
    print(f"  Min |Im(λ)| = {np.min(np.abs(imag_parts)):.4f}")

    # Create output directory
    output_dir = os.path.join(result_dir, f'precision_a{angle_precision}_t{precision}decimal')
    os.makedirs(output_dir, exist_ok=True)

    # Save rounded parameters
    np.savez(os.path.join(output_dir, 'params_rounded.npz'),
             params_original=params_original,
             params_rounded=params_rounded,
             loss_original=loss_original,
             loss_rounded=loss_rounded,
             eigenvalues=eigvals,
             angle_precision=angle_precision,
             precision=precision)

    # Save detailed text report
    report_path = os.path.join(output_dir, 'precision_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Precision Verification Report\n")
        f.write("=" * 70 + "\n\n")

        # Add detection information
        f.write("Configuration:\n")
        f.write(f"  Type:       {ep_type}\n")
        f.write(f"  Spacer:     {spacer_type}\n")
        f.write(f"  Constraint: {constraint_type}\n")
        f.write(f"  Angle Precision (theta0):  {angle_precision} decimal place(s)\n")
        f.write(f"  Thickness Precision:       {precision} decimal place(s)\n")
        f.write("\n" + "=" * 70 + "\n\n")

        f.write("Parameter Comparison:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Parameter':<15} {'Original':<20} {'Rounded':<20} {'Diff':<15}\n")
        f.write("-" * 70 + "\n")
        for i, (name, orig, rounded) in enumerate(zip(param_names, params_original, params_rounded)):
            diff = abs(orig - rounded)
            f.write(f"{name:<15} {orig:<20.15f} {rounded:<20.15f} {diff:<15.15f}\n")
        f.write("-" * 70 + "\n\n")

        f.write("Eigenvalue Comparison:\n")
        f.write(f"Original Loss:  {loss_original:.15e}\n")
        f.write(f"Rounded Loss:   {loss_rounded:.15e}\n")
        f.write(f"Loss Change:    {loss_rounded-loss_original:.15e}\n\n")

        f.write("Rounded Parameter Eigenvalues:\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")

        f.write(f"\nDegeneracy Check:\n")
        f.write(f"  Std(Re) = {np.std(real_parts):.15e}\n")
        f.write(f"  Std(Im) = {np.std(imag_parts):.15e}\n")
        f.write(f"  Min |Im(λ)| = {np.min(np.abs(imag_parts)):.15f}\n")

    print(f"\nReport saved: {report_path}")

    # Generate parameter scan plots around rounded values
    print("\n" + "=" * 70)
    print("Generating parameter scan plots...")
    print(f"  scan_range = {scan_range}")
    print(f"  n_points = {n_points}")
    print(f"  Step size = {2*scan_range/(n_points-1):.2e}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    scan_parameters_around_optimum(
        params_optimal=params_rounded,  # Use rounded parameters as center
        objective_func=lambda p, fm, **kw: objective_function_control(
            p, fm, GreenFun,
            build_layers_func=build_layers_func,
            penalty_weight=penalty_weight,
            **kw
        ),
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=scan_range,
        n_points=n_points,
        param_names=param_names,
        param_labels=param_labels
    )

    print("\n" + "=" * 70)
    print(f"Precision verification complete!")
    print(f"Results saved in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Verify experimental precision impact on exceptional points',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 1 decimal place precision
  python verify_precision.py ep3_w10_s810_DE100000_LB5000

  # Custom precision
  python verify_precision.py ep4_w20_s812_DE10000_LB5000 --precision 2

  # Custom scan settings
  python verify_precision.py ep3_sic_s0 --precision 1 --scan-range 1e-3 --n-points 41
        """
    )

    parser.add_argument('result_folder', type=str,
                       help='Name or path of results folder')
    parser.add_argument('--precision', type=int, default=1,
                       help='Decimal places for thickness parameters (default: 1)')
    parser.add_argument('--angle-precision', type=int, default=2,
                       help='Decimal places for angle (theta0) (default: 2)')
    parser.add_argument('--scan-range', type=float, default=1e-4,
                       help='Scan range around rounded value (default: 1e-4)')
    parser.add_argument('--n-points', type=int, default=21,
                       help='Number of scan points (default: 21)')

    args = parser.parse_args()

    verify_precision(
        result_folder=args.result_folder,
        precision=args.precision,
        angle_precision=args.angle_precision,
        scan_range=args.scan_range,
        n_points=args.n_points
    )
