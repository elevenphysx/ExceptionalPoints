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

# Fixed materials tuple (order: Platinum, Carbon, Iron)
FIXED_MATERIALS = (Platinum, Carbon, Iron)

# Fixed constant C0 (CFe)
C0_FIXED = 7.74 * 1.06 * 0.5

# ============================================================
# Optimization Bounds
# ============================================================

# Parameter bounds: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
BOUNDS = [
    (2.0, 10.0),   # theta0 (mrad)
    (3.0, 10.0),   # Pt thickness (nm)
    (1.0, 50.0),   # C layer 1
    (0.5, 3.0),    # Fe layer 1 (resonant)
    (1.0, 50.0),   # C layer 2
    (0.5, 3.0),    # Fe layer 2 (resonant)
    (1.0, 50.0),   # C layer 3
    (0.5, 3.0),    # Fe layer 3 (resonant)
    (1.0, 50.0),   # C layer 4
]

# ============================================================
# Default Optimization Parameters
# ============================================================

# Threshold for eigenvalue magnitude constraint
DEFAULT_THRESHOLD = 5.0

# Penalty weight for constraint violations
DEFAULT_PENALTY_WEIGHT = 100.0

# Default iteration counts
DEFAULT_CMA_ITERATIONS = 300
DEFAULT_DE_ITERATIONS = 100
DEFAULT_NM_ITERATIONS = 500
DEFAULT_LBFGSB_ITERATIONS = 500

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
