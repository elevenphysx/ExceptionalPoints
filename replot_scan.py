"""
Replot parameter scan from saved optimization results
Usage: python replot_scan.py <result_folder_name>
Example: python replot_scan.py ep4_noconstraint_w20_s812_DE10000_LB5000
"""

import numpy as np
import sys
import os
import importlib.util

# Import shared configuration
from config import FIXED_MATERIALS, PARAM_NAMES_EP4, PARAM_LABELS_EP4

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


def replot_scan(result_folder, scan_range=1e-4, n_points=51, use_constraint=False):
    """
    Replot parameter scan from saved results

    Args:
        result_folder: name or path of results folder
        scan_range: scan range around optimal value (default: 1e-4)
        n_points: number of points to scan (default: 51)
        use_constraint: whether to use |Im(λ)| >= 5.0 constraint (default: False for no-constraint results)
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

    print(f"Loading parameters from: {params_file}")
    data = np.load(params_file)
    params_optimal = data['params']
    loss = data['loss']

    print(f"Loaded optimal parameters (loss = {loss:.6e})")
    print(f"Number of parameters: {len(params_optimal)}")

    # Determine if EP3 or EP4 based on parameter count
    if len(params_optimal) == 9:
        print("Detected: EP3 (3 resonant layers)")
        from config import PARAM_NAMES, PARAM_LABELS
        param_names = PARAM_NAMES
        param_labels = PARAM_LABELS
        build_layers_func = build_layers  # Use EP3 version
    elif len(params_optimal) == 11:
        print("Detected: EP4 (4 resonant layers)")
        param_names = PARAM_NAMES_EP4
        param_labels = PARAM_LABELS_EP4
        build_layers_func = build_layers_ep4  # Use EP4 version
    else:
        print(f"Warning: Unexpected parameter count: {len(params_optimal)}")
        param_names = [f"param_{i}" for i in range(len(params_optimal))]
        param_labels = [f"Param {i}" for i in range(len(params_optimal))]
        build_layers_func = build_layers  # Default to EP3

    # Check if this is a no-constraint result
    folder_name = os.path.basename(result_dir)
    if 'noconstraint' in folder_name.lower():
        print("Detected: NO CONSTRAINT result (penalty_weight=0)")
        penalty_weight = 0
    else:
        print(f"Using constraint: |Im(λ)| >= 5.0")
        penalty_weight = None  # Use default from config

    if use_constraint:
        penalty_weight = None
        print("Override: Using constraint (penalty_weight from config)")

    fixed_materials = FIXED_MATERIALS

    print(f"\nReplotting parameter scan...")
    print(f"  scan_range = {scan_range}")
    print(f"  n_points = {n_points}")
    print(f"  Step size = {2*scan_range/(n_points-1):.2e}")
    print(f"  Output directory: {result_dir}")

    # Replot using scan_parameters_around_optimum
    scan_parameters_around_optimum(
        params_optimal=params_optimal,
        objective_func=lambda p, fm, **kw: objective_function_control(
            p, fm, GreenFun,
            build_layers_func=build_layers_func,  # Use detected version (EP3 or EP4)
            penalty_weight=penalty_weight,
            **kw
        ),
        fixed_materials=fixed_materials,
        output_dir=result_dir,
        scan_range=scan_range,
        n_points=n_points,
        param_names=param_names,
        param_labels=param_labels
    )

    print(f"\nDone! Scan plots saved in: {result_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Replot parameter scan from saved optimization results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python replot_scan.py ep4_noconstraint_w20_s812_DE10000_LB5000
  python replot_scan.py ep4_w20_s0_DE1000_LB1000 --scan-range 1e-3
  python replot_scan.py results/ep4_cma_s0_iter10000 --n-points 41
        """
    )

    parser.add_argument('result_folder', type=str,
                       help='Name or path of results folder')
    parser.add_argument('--scan-range', type=float, default=1e-4,
                       help='Scan range around optimal (default: 1e-4)')
    parser.add_argument('--n-points', type=int, default=51,
                       help='Number of scan points (default: 51)')
    parser.add_argument('--use-constraint', action='store_true',
                       help='Force use of |Im(λ)| >= 5.0 constraint (ignore folder name)')

    args = parser.parse_args()

    replot_scan(
        result_folder=args.result_folder,
        scan_range=args.scan_range,
        n_points=args.n_points,
        use_constraint=args.use_constraint
    )
