"""
Common functions shared across all Exceptional Point optimization algorithms
"""

import numpy as np
import os
from config import C0_FIXED, IMAG_MIN, IMAG_PENALTY


def build_layers(params, fixed_materials):
    """
    Build layer structure from optimization parameters

    Args:
        params: [theta0, t_Pt, t_Spacer1, t_Fe1, t_Spacer2, t_Fe2, t_Spacer3, t_Fe3, t_Spacer4]
        fixed_materials: (Platinum, Spacer, Iron) material tuples
                        Spacer can be Carbon, SiC, or other materials

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter (fixed)
    """
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = C0_FIXED
    Platinum, Spacer, Iron = fixed_materials  # Generic naming: Spacer = Carbon or SiC
    Layers = [
        (Platinum, thicknesses[0], 0),
        (Spacer,   thicknesses[1], 0),
        (Iron,     thicknesses[2], 1),
        (Spacer,   thicknesses[3], 0),
        (Iron,     thicknesses[4], 1),
        (Spacer,   thicknesses[5], 0),
        (Iron,     thicknesses[6], 1),
        (Spacer,   thicknesses[7], 0),
        (Platinum, np.inf, 0),
    ]
    return theta0, Layers, C0


def build_layers_ep4(params, fixed_materials):
    """
    Build layer structure for EP4 (4 resonant layers) from optimization parameters

    Args:
        params: [theta0, t_Pt, t_Spacer1, t_Fe1, t_Spacer2, t_Fe2, t_Spacer3, t_Fe3, t_Spacer4, t_Fe4, t_Spacer5]
        fixed_materials: (Platinum, Spacer, Iron) material tuples
                        Spacer can be Carbon, SiC, or other materials

    Returns:
        theta0: incident angle (mrad)
        Layers: list of (material, thickness, is_resonant)
        C0: constant parameter (fixed)
    """
    theta0 = params[0]
    thicknesses = params[1:11]  # 10 thickness parameters
    C0 = C0_FIXED
    Platinum, Spacer, Iron = fixed_materials  # Generic naming: Spacer = Carbon or SiC
    Layers = [
        (Platinum, thicknesses[0], 0),  # Pt
        (Spacer,   thicknesses[1], 0),  # Spacer1
        (Iron,     thicknesses[2], 1),  # Fe1*
        (Spacer,   thicknesses[3], 0),  # Spacer2
        (Iron,     thicknesses[4], 1),  # Fe2*
        (Spacer,   thicknesses[5], 0),  # Spacer3
        (Iron,     thicknesses[6], 1),  # Fe3*
        (Spacer,   thicknesses[7], 0),  # Spacer4
        (Iron,     thicknesses[8], 1),  # Fe4*
        (Spacer,   thicknesses[9], 0),  # Spacer5
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
        params: [theta0, t_Pt, t_Spacer1, t_Fe1, ...] (EP3: 9 params, EP4: 11 params)
        fixed_materials: (Platinum, Spacer, Iron) material tuples
                        Spacer can be Carbon, SiC, or other materials
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


# ============================================================
# Output Formatting and File I/O Functions
# ============================================================

def format_eigenvalues_string(real_parts, imag_parts, precision=3):
    """
    Format eigenvalues as string for logging

    Args:
        real_parts: array of real parts of eigenvalues
        imag_parts: array of imaginary parts of eigenvalues
        precision: number of decimal places (default: 3)

    Returns:
        Formatted string like "λ1=+0.123+0.456i, λ2=+0.124+0.457i, ..."
    """
    return ", ".join([f"λ{i+1}={re:+.{precision}f}{im:+.{precision}f}i"
                     for i, (re, im) in enumerate(zip(real_parts, imag_parts))])


def save_eigenvalues_txt(output_dir, seed, real_parts, imag_parts, ep_type='EP3'):
    """
    Save eigenvalues to txt file with 15-digit precision

    Args:
        output_dir: directory to save the file
        seed: random seed used for optimization
        real_parts: array of real parts of eigenvalues
        imag_parts: array of imaginary parts of eigenvalues
        ep_type: type of exceptional point (e.g., 'EP3', 'EP4', 'EP3-SiC')
    """
    eigvals_txt_path = os.path.join(output_dir, 'eigenvalues.txt')
    with open(eigvals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Eigenvalue Analysis ({ep_type}) - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")


def save_params_npz(output_dir, final_x, final_loss, eigvals, theta0, C0):
    """
    Save optimization results to npz file

    Args:
        output_dir: directory to save the file
        final_x: final optimized parameters
        final_loss: final loss value
        eigvals: final eigenvalues
        theta0: incident angle (mrad)
        C0: constant parameter
    """
    np.savez(os.path.join(output_dir, 'params.npz'),
             params=final_x,
             loss=final_loss,
             eigenvalues=eigvals,
             theta0=theta0,
             C0=C0)


def save_parameters_txt(output_dir, seed, ep_name, final_x, final_loss, spread, pen_im,
                       theta0, Layers, bounds, layer_names):
    """
    Save high-precision parameters to txt file

    Args:
        output_dir: directory to save the file
        seed: random seed used for optimization
        ep_name: name of exceptional point (e.g., 'EP3', 'EP4', 'EP3-SiC')
        final_x: final optimized parameters
        final_loss: final loss value
        spread: variance component of loss
        pen_im: imaginary penalty component of loss
        theta0: incident angle (mrad)
        Layers: layer structure list
        bounds: parameter bounds
        layer_names: list of layer names for output
    """
    params_txt_path = os.path.join(output_dir, 'parameters_high_precision.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Exceptional Point ({ep_name}) Parameters - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {final_loss:.15e}\n")
        f.write(f"Spread:      {spread:.15e}\n")
        f.write(f"Penalty Im:  {pen_im:.15e}\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write("-" * 50 + "\n")
        thicknesses = [layer[1] for layer in Layers[:-1]]
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n")

        f.write("\nBounds Check:\n")
        all_ok = True
        for i, (val, (low, high)) in enumerate(zip(final_x, bounds)):
            status = "OK" if low <= val <= high else "VIOLATION"
            if status == "VIOLATION": all_ok = False
            f.write(f"  Param {i}: {val:.4f} [{low}, {high}] -> {status}\n")
        f.write(f"\nOverall Status: {'PASSED' if all_ok else 'FAILED'}\n")


def format_final_result_string(loss, spread, pen_im, real_parts, imag_parts, imag_min):
    """
    Format final result summary string

    Args:
        loss: total loss value
        spread: variance component of loss
        pen_im: imaginary penalty component
        real_parts: array of real parts of eigenvalues
        imag_parts: array of imaginary parts of eigenvalues
        imag_min: minimum imaginary part constraint value

    Returns:
        Formatted string with final results
    """
    output = "\n" + "="*70 + "\nFINAL RESULT\n" + "="*70 + "\n"
    output += f"Total Loss: {loss:.6e}\n"
    output += f"Spread (Variance): {spread:.6e}\n"
    output += f"Penalty Im: {pen_im:.6e}\n\n"

    output += "Eigenvalues:\n"
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        output += f"  lambda_{i+1} = {re:+.10f} {im:+.10f}i\n"

    output += "\nDegeneracy Check:\n"
    output += f"  Std(Re) = {np.std(real_parts):.6e}\n"
    output += f"  Std(Im) = {np.std(imag_parts):.6e}\n"
    output += f"  Min |Im(λ)| = {np.min(np.abs(imag_parts)):.4f} (constraint: |Im(λ)| >= {imag_min})\n"

    return output

