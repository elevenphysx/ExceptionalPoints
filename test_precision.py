"""
Precision Test: Compare eigenvalue results using different decimal precisions
Tests 8, 10, 12, 14, and 15 decimal places
"""

import numpy as np
import sys
import os
import importlib.util

# Import GreenFun
current_dir = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# Load FULL PRECISION from .npz
results_dir = os.path.join(current_dir, 'results')
result_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith('variance_method_')], reverse=True)
latest_dir = os.path.join(results_dir, result_dirs[0])
npz_path = os.path.join(latest_dir, 'exceptional_point_params.npz')

data = np.load(npz_path)
params_full = data['params']      # Full precision (15+ digits)
theta0_full = data['theta0']
C0_full = data['C0']

# Material parameters
Iron = (7.298e-6, 3.33e-7)
Carbon = (2.257e-6, 1.230e-9)
Platinum = (1.713e-5, 2.518e-6)

print("=" * 100)
print("PRECISION TEST: Effect of Decimal Precision on Eigenvalue Calculation")
print("=" * 100)

print(f"\nFull precision parameters (from .npz):")
print(f"  theta0 = {theta0_full:.16f}")
for i in range(1, len(params_full)):
    print(f"  params[{i}] = {params_full[i]:.16f}")

print("\n" + "=" * 100)
print("Testing different precision levels...")
print("=" * 100)

# Test different precision levels
precisions = [8, 10, 12, 14, 15]
results = []

for precision in precisions:
    print(f"\n{'-' * 100}")
    print(f"PRECISION: {precision} decimal places")
    print(f"{'-' * 100}")

    # Round parameters to specified precision
    theta0 = np.round(theta0_full, precision)
    params = np.round(params_full, precision)
    C0 = np.round(C0_full, precision)

    print(f"  theta0 = {theta0:.{precision}f}")
    print(f"  Truncated params:")
    for i in range(1, min(4, len(params))):  # Show first 3 for brevity
        print(f"    params[{i}] = {params[i]:.{precision}f}")
    print(f"    ...")

    # Build layers
    Layers = [
        (Platinum, params[1], 0),
        (Carbon,   params[2], 0),
        (Iron,     params[3], 1),
        (Carbon,   params[4], 0),
        (Iron,     params[5], 1),
        (Carbon,   params[6], 0),
        (Iron,     params[7], 1),
        (Carbon,   params[8], 0),
        (Platinum, np.inf, 0),
    ]

    # Compute eigenvalues
    G, _ = GreenFun(theta0, Layers, C0)
    G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
    eigvals = np.linalg.eigvals(G1)
    real_parts = np.real(eigvals)
    imag_parts = np.imag(eigvals)
    loss = np.var(real_parts) + np.var(imag_parts)

    print(f"\n  Results:")
    print(f"    Loss = {loss:.15e}")
    print(f"    Eigenvalues:")
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        print(f"      lambda_{i+1} = {re:+.12f} {im:+.12f}j")
    print(f"    Std(Re) = {np.std(real_parts):.12e}")
    print(f"    Std(Im) = {np.std(imag_parts):.12e}")

    # Store results
    results.append({
        'precision': precision,
        'loss': loss,
        'eigvals': eigvals.copy(),
        'std_re': np.std(real_parts),
        'std_im': np.std(imag_parts)
    })

# Summary comparison
print("\n" + "=" * 100)
print("SUMMARY: Precision vs Accuracy")
print("=" * 100)

print(f"\n{'Precision':<12} {'Loss':<20} {'Std(Re)':<18} {'Std(Im)':<18} {'Status'}")
print("-" * 100)

target_loss = results[-1]['loss']  # 15-digit precision as reference

for r in results:
    loss_diff = abs(r['loss'] - target_loss)
    if r['loss'] < 1e-9:
        status = "[EXCELLENT]" if loss_diff < 1e-12 else "[GOOD]"
    elif r['loss'] < 1e-7:
        status = "[OK]"
    else:
        status = "[POOR]"

    print(f"{r['precision']:>3} digits   {r['loss']:<20.12e} {r['std_re']:<18.12e} {r['std_im']:<18.12e} {status}")

print("\n" + "=" * 100)
print("CONCLUSION:")
print("=" * 100)

# Find minimum precision needed
for i, r in enumerate(results):
    if r['loss'] < 1e-10:  # Threshold for acceptable loss
        print(f"\nMinimum precision needed: {r['precision']} decimal places")
        print(f"  Loss at {r['precision']} digits: {r['loss']:.6e}")
        print(f"  Loss at 15 digits: {results[-1]['loss']:.6e}")
        print(f"  Difference: {abs(r['loss'] - results[-1]['loss']):.6e}")
        break
else:
    print(f"\nNone of the tested precisions achieved Loss < 1e-10")

print("\nKey finding: Higher precision is critical near exceptional points!")
print("=" * 100)
