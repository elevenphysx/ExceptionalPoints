"""
Configuration file for Exceptional Point optimization
Contains material parameters, bounds, and other shared settings
"""

import numpy as np

# ============================================================
# Material Parameters (Physical Constants)
# ============================================================

# Iron (57Fe resonance)
Iron = (7.298e-6, 3.33e-7)

# Carbon
Carbon = (2.257e-6, 1.230e-9)

# Platinum
Platinum = (1.713e-5, 2.518e-6)

# SiC
SiC = (3.22628057e-06, 1.7340934e-08)

# Fixed materials tuple (order: Platinum, Carbon, Iron)
FIXED_MATERIALS = (Platinum, Carbon, Iron)

# Fixed constant C0 (CFe)
C0_FIXED = 7.74 * 1.06 * 0.5

# ============================================================
# Optimization Bounds (Control_N Verified Bounds)
# ============================================================

# Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
# These bounds are verified to work well with Control_N algorithm
BOUNDS = [
    (2.0, 8.0),    # theta0 (mrad)
    (0.5, 4.0),    # Pt thickness (nm)
    (1.0, 40.0),   # C layer 1
    (0.8, 3.0),    # Fe layer 1 (resonant) - first layer has tighter bound
    (1.0, 40.0),   # C layer 2
    (0.5, 3.0),    # Fe layer 2 (resonant)
    (1.0, 40.0),   # C layer 3
    (0.5, 3.0),    # Fe layer 3 (resonant)
    (1.0, 40.0),   # C layer 4
]

# ============================================================
# Constraint Parameters
# ============================================================

# Imaginary part minimum constraint: Im(λ) >= IMAG_MIN
IMAG_MIN = 5.0

# Penalty weight for imaginary part constraint violations
IMAG_PENALTY = 1e4

# ============================================================
# Default Optimization Parameters
# ============================================================

# Default iteration counts (Control_N verified values)
DEFAULT_DE_ITERATIONS = 100000      # Differential Evolution iterations
DEFAULT_LBFGSB_ITERATIONS = 5000    # L-BFGS-B refinement iterations

# Legacy iteration counts (kept for backward compatibility)
DEFAULT_CMA_ITERATIONS = 300
DEFAULT_NM_ITERATIONS = 500

# Default seeds for multi-seed optimization
DEFAULT_SEEDS = [812, 1001, 2023, 3030, 42]

# ============================================================
# Scanning Parameters
# ============================================================

# Scan range for parameter sensitivity analysis (±range)
DEFAULT_SCAN_RANGE = 1e-5

# Number of points in parameter scan
DEFAULT_SCAN_POINTS = 21

# ============================================================
# Layer Configuration
# ============================================================

# Layer names for output formatting
LAYER_NAMES = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']

# Parameter names for scanning
PARAM_NAMES = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2', 't_C3', 't_Fe3', 't_C4']

# Parameter labels for plotting (with subscripts)
PARAM_LABELS = ['θ₀ (mrad)', 't_Pt (nm)', 't_C₁ (nm)', 't_Fe₁ (nm)',
                't_C₂ (nm)', 't_Fe₂ (nm)', 't_C₃ (nm)', 't_Fe₃ (nm)', 't_C₄ (nm)']

# ============================================================
# EP4 Configuration (4 resonant layers)
# ============================================================

# EP4 Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, t_Fe4, t_C5]
BOUNDS_EP4 = [
    (2.0, 15.0),    # theta0 (mrad)
    (0.5, 10),    # Pt
    (0.5, 65.0),   # C
    (0.2, 3.5),    # Fe*
    (0.5, 65.0),   # C
    (0.2, 3.5),    # Fe*
    (0.5, 65.0),   # C
    (0.2, 3.5),    # Fe*
    (0.5, 65.0),   # C
    (0.2, 3.5),    # Fe*
    (0.5, 65.0),   # C
]

# EP4 Layer names (10 layers + substrate)
LAYER_NAMES_EP4 = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']

# EP4 Parameter names
PARAM_NAMES_EP4 = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2',
                   't_C3', 't_Fe3', 't_C4', 't_Fe4', 't_C5']

# EP4 Parameter labels for plotting
PARAM_LABELS_EP4 = ['θ₀ (mrad)', 't_Pt (nm)', 't_C₁ (nm)', 't_Fe₁ (nm)',
                    't_C₂ (nm)', 't_Fe₂ (nm)', 't_C₃ (nm)', 't_Fe₃ (nm)',
                    't_C₄ (nm)', 't_Fe₄ (nm)', 't_C₅ (nm)']
