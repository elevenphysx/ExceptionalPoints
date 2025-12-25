"""
Verification script to check exceptional point parameters
Compute G, G1 matrices and eigenvalues for given parameters
Uses 15-digit precision parameters (hardcoded)
"""

import numpy as np
import sys
import os
import importlib.util

# ============================================================
# FULL PRECISION parameters (15 decimal places)
# ============================================================

# Fixed parameter
C0 = 4.102200000000000

# Optimization parameters
theta0 = 5.094015909616491  # mrad

# Layer thicknesses (nm) - 15 decimal precision
params = np.array([
    5.094015909616491,   # theta0 (mrad)
    8.862610010691775,   # Pt
    43.356799668547424,  # C1
    0.525859681105625,   # Fe1 (resonant)
    28.073182751441934,  # C2
    0.597994139103292,   # Fe2 (resonant)
    20.574401876939078,  # C3
    0.674835471839100,   # Fe3 (resonant)
    15.471969870765989,  # C4
])

print(f"Using 15-digit precision parameters (hardcoded)")
print(f"  theta0 = {theta0:.15f}")
print(f"  C0     = {C0:.15f}")
for i in range(1, len(params)):
    print(f"  params[{i}] = {params[i]:.15f}")

# Import GreenFun from green function-new.py
current_dir = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("green_function_new",
                                               os.path.join(current_dir, "green function-new.py"))
green_function_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(green_function_new)
GreenFun = green_function_new.GreenFun

# Material parameters (physical constants)
Iron = (7.298e-6, 3.33e-7)
Carbon = (2.257e-6, 1.230e-9)
Platinum = (1.713e-5, 2.518e-6)

# Build layer structure from FULL PRECISION parameters
Layers = [
    (Platinum, params[1], 0),   # Pt
    (Carbon,   params[2], 0),   # C
    (Iron,     params[3], 1),   # Fe (resonant)
    (Carbon,   params[4], 0),   # C
    (Iron,     params[5], 1),   # Fe (resonant)
    (Carbon,   params[6], 0),   # C
    (Iron,     params[7], 1),   # Fe (resonant)
    (Carbon,   params[8], 0),   # C
    (Platinum, np.inf, 0),       # Pt substrate (infinite thickness)
]

# ============================================================
# Compute Green matrix and eigenvalues
# ============================================================

print("=" * 80)
print("VERIFICATION: Exceptional Point Parameter Check (15-DIGIT PRECISION)")
print("=" * 80)

print(f"\nInput Parameters (hardcoded, 15 decimal places):")
print(f"  theta0 = {theta0:.15f} mrad")
print(f"  C0     = {C0:.15f} (fixed)")

print(f"\nLayer Structure:")
layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Pt(sub)']
for i, (name, layer) in enumerate(zip(layer_names, Layers)):
    thickness = layer[1]
    resonant = ' (resonant)' if layer[2] == 1 else ''
    if i < len(Layers) - 1:
        print(f"  Layer {i}: {name:8s} = {thickness:18.15f} nm{resonant}")
    else:
        print(f"  Layer {i}: {name:8s} = {'inf':>18s} nm{resonant}")

print("\n" + "=" * 80)
print("Computing Green Matrix...")
print("=" * 80)

# Compute Green matrix
G, O1 = GreenFun(theta0, Layers, C0)

print(f"\nGreen Matrix G (shape: {G.shape}):")
print("-" * 80)
np.set_printoptions(precision=8, suppress=False, linewidth=120)
for i in range(G.shape[0]):
    row_str = "  ["
    for j in range(G.shape[1]):
        val = G[i, j]
        row_str += f"{val.real:+12.8f}{val.imag:+12.8f}j"
        if j < G.shape[1] - 1:
            row_str += ",  "
    row_str += "]"
    print(row_str)

print(f"\nO1 vector (coupling vector, shape: {O1.shape}):")
print("-" * 80)
for i, val in enumerate(O1):
    print(f"  O1[{i}] = {val.real:+12.8f}{val.imag:+12.8f}j")

# Compute G1 = -G - 0.5i·I
G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)

print(f"\nTransformed Matrix G1 = -G - 0.5i·I (shape: {G1.shape}):")
print("-" * 80)
for i in range(G1.shape[0]):
    row_str = "  ["
    for j in range(G1.shape[1]):
        val = G1[i, j]
        row_str += f"{val.real:+12.8f}{val.imag:+12.8f}j"
        if j < G1.shape[1] - 1:
            row_str += ",  "
    row_str += "]"
    print(row_str)

# Compute eigenvalues
eigvals = np.linalg.eigvals(G1)
real_parts = np.real(eigvals)
imag_parts = np.imag(eigvals)

print("\n" + "=" * 80)
print("Eigenvalue Analysis")
print("=" * 80)

print("\nEigenvalues of G1:")
print("  lambda_i = Re + Im*j")
print("  " + "-" * 70)
for i, (eig, re, im) in enumerate(zip(eigvals, real_parts, imag_parts)):
    print(f"  lambda_{i+1} = {re:+15.8f} {im:+15.8f}j")

# Check degeneracy
re_std = np.std(real_parts)
im_std = np.std(imag_parts)
re_mean = np.mean(real_parts)
im_mean = np.mean(imag_parts)

print("\nDegeneracy Check:")
print("  " + "-" * 70)
print(f"  Mean(Re) = {re_mean:+15.8f}")
print(f"  Mean(Im) = {im_mean:+15.8f}")
print(f"  Std(Re)  = {re_std:.12e}  {'[PASS] Degenerate' if re_std < 1e-4 else '[FAIL] Not degenerate'}")
print(f"  Std(Im)  = {im_std:.12e}  {'[PASS] Degenerate' if im_std < 1e-4 else '[FAIL] Not degenerate'}")

# Calculate pairwise distances
dist_12 = np.abs(eigvals[0] - eigvals[1])
dist_13 = np.abs(eigvals[0] - eigvals[2])
dist_23 = np.abs(eigvals[1] - eigvals[2])

print("\nPairwise Eigenvalue Distances:")
print("  " + "-" * 70)
print(f"  |lambda_1 - lambda_2| = {dist_12:.12e}")
print(f"  |lambda_1 - lambda_3| = {dist_13:.12e}")
print(f"  |lambda_2 - lambda_3| = {dist_23:.12e}")

# Loss function value
loss = np.var(real_parts) + np.var(imag_parts)
print("\nLoss Function (Variance Method):")
print("  " + "-" * 70)
print(f"  Loss = Var(Re) + Var(Im) = {loss:.12e}")

print("\n" + "=" * 80)
print("Verification Complete")
print("=" * 80)
