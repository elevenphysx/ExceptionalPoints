"""
Common functions shared across all Exceptional Point optimization algorithms
"""

import numpy as np
from config import C0_FIXED


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


def objective_function(params, fixed_materials, GreenFun, return_details=False, threshold=5.0, penalty_weight=100.0, bounds=None):
    """
    Objective function for Exceptional Point optimization

    Minimizes:
        1. Degeneracy loss: sum of squared pairwise differences in eigenvalues
        2. Magnitude constraint: exponential penalty if |Re| or |Im| < threshold
        3. (Optional) Soft bounds penalty if bounds is provided

    Args:
        params: optimization parameters
        fixed_materials: material constants
        GreenFun: Green function calculator
        return_details: if True, return detailed breakdown
        threshold: minimum magnitude constraint
        penalty_weight: weight for constraint penalties
        bounds: optional bounds for soft penalty

    Returns:
        loss: total loss value
        (if return_details=True) additional: eigvals, real_parts, imag_parts, G, G1, loss_deg, penalties
    """

    def pairwise_diff_sq(values):
        """Calculate sum of squared pairwise differences"""
        diff_sum = 0.0
        n = len(values)
        for i in range(n):
            for j in range(i + 1, n):
                diff_sum += (values[i] - values[j]) ** 2
        return diff_sum

    try:
        theta0, Layers, C0 = build_layers(params, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)

        if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            if return_details:
                return 1e10, eigvals, np.zeros(3), np.zeros(3), G, G1, 1e10, (1e10, 1e10)
            return 1e10

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        loss_degeneracy = pairwise_diff_sq(real_parts) + pairwise_diff_sq(imag_parts)

        # Smooth threshold penalty using softplus (always differentiable)
        # softplus(x) = log(1 + exp(x)) is smooth approximation of max(0, x)
        min_re = np.min(np.abs(real_parts))
        gap_re = threshold - min_re
        # Use scaled softplus for smooth penalty
        penalty_real = np.log(1.0 + np.exp(np.clip(gap_re, -10, 10)))  # Clip to avoid overflow

        min_im = np.min(np.abs(imag_parts))
        gap_im = threshold - min_im
        penalty_imag = np.log(1.0 + np.exp(np.clip(gap_im, -10, 10)))

        # Optional soft bounds penalty (exponential with clipping to prevent overflow)
        penalty_bounds = 0.0
        if bounds is not None:
            for p, (low, high) in zip(params, bounds):
                if p < low:
                    gap = low - p
                    # Clip gap to max 20: exp(20) ≈ 4.8e8 is already huge enough
                    penalty_bounds += (np.exp(np.clip(gap, 0, 20)) - 1.0)
                elif p > high:
                    gap = p - high
                    # Clip gap to max 20 to prevent overflow
                    penalty_bounds += (np.exp(np.clip(gap, 0, 20)) - 1.0)

        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag + penalty_bounds)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag)
        return loss

    except Exception as e:
        if return_details:
            return 1e10, None, None, None, None, None, None, None
        return 1e10
