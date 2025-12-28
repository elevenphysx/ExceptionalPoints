"""
Common functions shared across all Exceptional Point optimization algorithms
"""

import numpy as np
from config import C0_FIXED, IMAG_MIN, IMAG_PENALTY


def build_layers(params, fixed_materials):
    """
    Build layer structure from optimization parameters

    Args:
        params: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4]
        fixed_materials: (Platinum, Carbon, Iron) material tuples

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter (fixed)
    """
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = C0_FIXED
    Platinum, Carbon, Iron = fixed_materials
    Layers = [
        (Platinum, thicknesses[0], 0),
        (Carbon,   thicknesses[1], 0),
        (Iron,     thicknesses[2], 1),
        (Carbon,   thicknesses[3], 0),
        (Iron,     thicknesses[4], 1),
        (Carbon,   thicknesses[5], 0),
        (Iron,     thicknesses[6], 1),
        (Carbon,   thicknesses[7], 0),
        (Platinum, np.inf, 0),
    ]
    return theta0, Layers, C0


def build_layers_ep4(params, fixed_materials):
    """
    Build layer structure for EP4 (4 resonant layers) from optimization parameters

    Args:
        params: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, t_Fe4, t_C5]
        fixed_materials: (Platinum, Carbon, Iron) material tuples

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter (fixed)
    """
    theta0 = params[0]
    thicknesses = params[1:11]  # 10 thickness parameters
    C0 = C0_FIXED
    Platinum, Carbon, Iron = fixed_materials
    Layers = [
        (Platinum, thicknesses[0], 0),  # Pt
        (Carbon,   thicknesses[1], 0),  # C1
        (Iron,     thicknesses[2], 1),  # Fe1*
        (Carbon,   thicknesses[3], 0),  # C2
        (Iron,     thicknesses[4], 1),  # Fe2*
        (Carbon,   thicknesses[5], 0),  # C3
        (Iron,     thicknesses[6], 1),  # Fe3*
        (Carbon,   thicknesses[7], 0),  # C4
        (Iron,     thicknesses[8], 1),  # Fe4*
        (Carbon,   thicknesses[9], 0),  # C5
        (Platinum, np.inf, 0),           # Substrate
    ]
    return theta0, Layers, C0


def objective_function_control(params, fixed_materials, GreenFun, return_details=False, build_layers_func=None, penalty_weight=None):
    """
    Variance-based loss matching Control_N.py (Verified Working Implementation)

    Loss = Σ|λ - mean(λ)|² where λ are eigenvalues of -G - 0.5j*I
    Constraint: |Im(λ)| >= IMAG_MIN (absolute value, since eigenvalues shifted to negative)

    This is the verified implementation from Singularity_de_control_multi_Gshift.py

    Args:
        params: [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4] (EP3)
                or [theta0, t_Pt, t_C1, t_Fe1, t_C2, t_Fe2, t_C3, t_Fe3, t_C4, t_Fe4, t_C5] (EP4)
        fixed_materials: (Platinum, Carbon, Iron) material tuples
        GreenFun: Green function calculator
        return_details: if True, return detailed breakdown
        build_layers_func: custom layer building function (default: build_layers for EP3)
        penalty_weight: weight for imaginary constraint penalty (default: IMAG_PENALTY from config)
                        Set to 0 to disable constraint

    Returns:
        loss: total loss value
        (if return_details=True) additional: eigvals, real_parts, imag_parts, G, G_shifted, spread, penalty_imag
    """
    try:
        # Use default penalty weight if not specified
        if penalty_weight is None:
            penalty_weight = IMAG_PENALTY

        # Use custom build_layers function if provided, otherwise use default (EP3)
        if build_layers_func is None:
            build_layers_func = build_layers

        theta0, Layers, C0 = build_layers_func(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)

        # Use -G - 0.5j*I (shifted matrix)
        I = np.eye(G.shape[0])
        G_shifted = -G - 0.5j * I
        eigvals = np.linalg.eigvals(G_shifted)

        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            if return_details:
                n_eigvals = len(eigvals)  # Auto-detect: EP3=3, EP4=4, etc.
                return 1e10, eigvals, np.zeros(n_eigvals), np.zeros(n_eigvals), G, G_shifted, 1e10, 0.0
            return 1e10

        # Variance-based loss (Control_N style)
        mean_eig = np.mean(eigvals)
        diff = eigvals - mean_eig
        spread = np.sum(diff.real**2 + diff.imag**2)

        # |Im(λ)| >= IMAG_MIN constraint (absolute value)
        imag_parts = np.imag(eigvals)
        min_abs_im = np.min(np.abs(imag_parts))

        penalty_imag = 0.0
        if min_abs_im < IMAG_MIN:
            penalty_imag = penalty_weight * (IMAG_MIN - min_abs_im) ** 2  # Use parameter instead of constant

        loss = spread + penalty_imag

        if return_details:
            real_parts = np.real(eigvals)
            return loss, eigvals, real_parts, imag_parts, G, G_shifted, spread, penalty_imag
        return loss

    except Exception as e:
        if return_details:
            return 1e10, None, None, None, None, None, None, None
        return 1e10

