"""
Export exceptional point parameters at different precision levels
Generates formatted parameter tables for sharing
"""

import numpy as np
import os

# Load full precision from .npz
results_dir = 'results'
result_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith('variance_method_')], reverse=True)
latest_dir = os.path.join(results_dir, result_dirs[0])
npz_path = os.path.join(latest_dir, 'exceptional_point_params.npz')

data = np.load(npz_path)
params_full = data['params']
theta0_full = data['theta0']
C0_full = data['C0']
loss_full = data['loss']

print("=" * 120)
print("EXCEPTIONAL POINT PARAMETERS - DIFFERENT PRECISION LEVELS")
print("=" * 120)
print(f"\nSource: {npz_path}")
print(f"Original Loss: {loss_full:.15e}\n")

# Define precision levels
precisions = [8, 10, 12, 14, 15]
param_names = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2', 't_C3', 't_Fe3', 't_C4']

# Generate tables for each precision
for precision in precisions:
    print("=" * 120)
    print(f"PRECISION: {precision} DECIMAL PLACES")
    print("=" * 120)

    # Round to specified precision
    theta0 = np.round(theta0_full, precision)
    params = np.round(params_full, precision)
    C0 = np.round(C0_full, precision)

    print(f"\nFixed parameter:")
    print(f"  C0 = {C0:.{precision}f}\n")

    print(f"Optimization parameters (copy-paste ready):")
    print(f"  theta0 = {theta0:.{precision}f}  # mrad\n")

    print(f"Layer thicknesses (nm):")
    layer_names = ['Pt', 'C1', 'Fe1 (resonant)', 'C2', 'Fe2 (resonant)', 'C3', 'Fe3 (resonant)', 'C4']
    for i, name in enumerate(layer_names):
        print(f"  {name:20s} = {params[i+1]:.{precision}f}")

    print(f"\n" + "-" * 120)
    print(f"Python code format:")
    print(f"-" * 120)
    print(f"theta0 = {theta0:.{precision}f}")
    print(f"C0 = {C0:.{precision}f}")
    print(f"layer_thicknesses = [")
    for i in range(1, 9):
        print(f"    {params[i]:.{precision}f},  # {layer_names[i-1]}")
    print(f"]")

    print(f"\n" + "-" * 120)
    print(f"Full parameter array format:")
    print(f"-" * 120)
    print(f"params = np.array([")
    print(f"    {theta0:.{precision}f},  # theta0 (mrad)")
    for i in range(1, len(params)):
        print(f"    {params[i]:.{precision}f},  # {layer_names[i-1]}")
    print(f"])")
    print()

# Create a comparison table
print("\n" + "=" * 120)
print("PARAMETER COMPARISON TABLE (All Precision Levels)")
print("=" * 120)

# Header
header = f"{'Parameter':<20}"
for p in precisions:
    header += f"{p:>18} digits"
print(header)
print("-" * 120)

# Each parameter row
all_params = [theta0_full] + list(params_full[1:9])
for i, (name, val_full) in enumerate(zip(['theta0'] + layer_names, all_params)):
    row = f"{name:<20}"
    for p in precisions:
        val = np.round(val_full, p)
        row += f"{val:>25.{p}f}"
    print(row)

print("\n" + "=" * 120)
print("USAGE NOTES:")
print("=" * 120)
print("""
1. **8 decimal places**: ❌ NOT RECOMMENDED (Loss ~6.3e-08, 762x worse)
   - Only for rough estimates or visualization

2. **10 decimal places**: ⚠️ MARGINAL (Loss ~6.8e-09, 83x worse)
   - Acceptable for preliminary studies

3. **12 decimal places**: ✅ GOOD (Loss ~1.4e-10, 1.7x worse)
   - Recommended for most applications
   - Sufficient for publication-quality results

4. **14 decimal places**: ⭐ EXCELLENT (Loss ~8.24e-11, 0.01% diff)
   - Recommended for high-precision work
   - Nearly identical to full precision

5. **15 decimal places**: ⭐ FULL PRECISION (Loss ~8.24e-11, reference)
   - Maximum precision from float64
   - Use for definitive calculations

RECOMMENDATION: Use 12 digits for sharing, 14 digits for critical work.
""")

# Save to file
output_file = os.path.join(latest_dir, 'parameters_all_precisions.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("EXCEPTIONAL POINT PARAMETERS - ALL PRECISION LEVELS\n")
    f.write("=" * 100 + "\n\n")

    for precision in precisions:
        f.write(f"\n{'=' * 100}\n")
        f.write(f"PRECISION: {precision} DECIMAL PLACES\n")
        f.write(f"{'=' * 100}\n\n")

        theta0 = np.round(theta0_full, precision)
        params = np.round(params_full, precision)
        C0 = np.round(C0_full, precision)

        f.write(f"C0 = {C0:.{precision}f}\n")
        f.write(f"theta0 = {theta0:.{precision}f}  # mrad\n\n")
        f.write(f"Layer thicknesses (nm):\n")
        for i, name in enumerate(layer_names):
            f.write(f"  {name:20s} = {params[i+1]:.{precision}f}\n")
        f.write("\n")

print(f"\nParameters saved to: {output_file}")
print("=" * 120)
