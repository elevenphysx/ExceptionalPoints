"""
Plotting utilities for Exceptional Point optimization
Provides consistent visualization across all algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from config import PARAM_NAMES, PARAM_LABELS


# ============================================================
# Style Configuration
# ============================================================

# Color schemes
EIGENVALUE_COLORS = ['#4472C4', '#ED7D31', '#70AD47']
DISTANCE_COLORS = ['#d62728', '#9467bd', '#8c564b']


def setup_matplotlib_style():
    """Set publication-quality matplotlib style"""
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5
    mpl.rcParams['xtick.major.size'] = 5
    mpl.rcParams['ytick.major.size'] = 5


# ============================================================
# Optimization History Plot
# ============================================================

def plot_optimization_history(history, output_dir, seed, algorithm_name='Optimization'):
    """
    Plot optimization progress: loss evolution and eigenvalue convergence

    Args:
        history: dict with keys:
            - 'iteration': list of iteration numbers
            - 'loss': list of loss values
            - 'eigvals_real': list of real parts arrays
            - 'eigvals_imag': list of imaginary parts arrays
        output_dir: directory to save plot
        seed: random seed number
        algorithm_name: algorithm description for legend

    Saves:
        optimization_result.png in output_dir
    """
    try:
        setup_matplotlib_style()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = np.array(history['iteration'])
        losses = np.array(history['loss'])
        eigvals_real = history['eigvals_real']
        eigvals_imag = history['eigvals_imag']

        # Left panel: Loss vs Iteration
        ax1.semilogy(iterations, losses, 'o-', color='#1f77b4',
                    label=algorithm_name, markersize=4, alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss (Variance)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Optimization Progress (Seed {seed})', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.legend(fontsize=11, framealpha=0.9, loc='best')
        ax1.tick_params(labelsize=11)

        # Right panel: Eigenvalue distances
        # Dynamically compute all pairwise distances
        n_eigvals = len(eigvals_real[0]) if eigvals_real else 3
        distances = {f'{i+1}-{j+1}': [] for i in range(n_eigvals) for j in range(i+1, n_eigvals)}

        for real_parts_iter, imag_parts_iter in zip(eigvals_real, eigvals_imag):
            if len(real_parts_iter) >= n_eigvals:
                eigs = [real_parts_iter[i] + 1j * imag_parts_iter[i] for i in range(n_eigvals)]

                for i in range(n_eigvals):
                    for j in range(i+1, n_eigvals):
                        distances[f'{i+1}-{j+1}'].append(np.abs(eigs[i] - eigs[j]))
            else:
                for key in distances:
                    distances[key].append(np.nan)

        # Plot up to 6 distance pairs (for visibility)
        colors = DISTANCE_COLORS + ['#ff7f0e', '#2ca02c', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        plot_count = 0
        for idx, (key, dist_values) in enumerate(distances.items()):
            if plot_count >= 6:
                break
            dist_array = np.array(dist_values)
            ax2.semilogy(iterations, dist_array, f'{markers[idx]}-',
                        color=colors[idx % len(colors)],
                        label=f'|λ_{key}|', markersize=3, alpha=0.8)
            plot_count += 1

        ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Distance', fontsize=14, fontweight='bold')
        ax2.set_title('Eigenvalue Convergence', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(fontsize=11, framealpha=0.9, loc='best')
        ax2.tick_params(labelsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_result.png'), dpi=300, bbox_inches='tight')
        plt.close()

        return True
    except Exception as e:
        print(f"Warning: Could not generate optimization plot: {e}")
        return False


# ============================================================
# Parameter Sensitivity Scan
# ============================================================

def scan_parameters_around_optimum(params_optimal, objective_func, fixed_materials,
                                   output_dir, scan_range, n_points=21,
                                   param_names=None, param_labels=None, **obj_kwargs):
    """
    Scan parameter sensitivity around optimal solution

    Args:
        params_optimal: optimal parameter vector
        objective_func: objective function that takes (params, fixed_materials, return_details=True, **kwargs)
        fixed_materials: material constants tuple
        output_dir: directory to save plots
        scan_range: scan range (±range around optimal)
        n_points: number of scan points (default: 21)
        param_names: list of parameter names (default: PARAM_NAMES from config)
        param_labels: list of parameter labels (default: PARAM_LABELS from config)
        **obj_kwargs: additional keyword arguments for objective_func (threshold, penalty_weight, etc.)

    Saves:
        scan_theta0.png, scan_t_Pt.png, ..., scan_t_C4.png in output_dir
    """
    try:
        setup_matplotlib_style()

        # Use default values if not provided
        if param_names is None:
            param_names = PARAM_NAMES
        if param_labels is None:
            param_labels = PARAM_LABELS

        for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            param_values = np.linspace(params_optimal[i] - scan_range,
                                      params_optimal[i] + scan_range, n_points)

            eigvals_real_list = []
            eigvals_imag_list = []
            n_eigvals_expected = len(params_optimal) - 8  # Estimate: EP3=3, EP4=4, etc.
            if n_eigvals_expected < 3:
                n_eigvals_expected = 3

            for param_val in param_values:
                params_test = params_optimal.copy()
                params_test[i] = param_val
                try:
                    loss, eigvals, re, im, G, G1, _, _ = objective_func(
                        params_test, fixed_materials, return_details=True, **obj_kwargs
                    )
                    eigvals_real_list.append(re)
                    eigvals_imag_list.append(im)
                except:
                    eigvals_real_list.append([np.nan] * n_eigvals_expected)
                    eigvals_imag_list.append([np.nan] * n_eigvals_expected)

            eigvals_real_array = np.array(eigvals_real_list)
            eigvals_imag_array = np.array(eigvals_imag_list)

            # Determine number of eigenvalues
            n_eigvals = eigvals_real_array.shape[1] if eigvals_real_array.size > 0 else 3
            colors_extended = EIGENVALUE_COLORS + ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

            # Create two-panel plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Left panel: Real parts
            for j in range(n_eigvals):
                ax1.plot(param_values, eigvals_real_array[:, j], 'o-',
                        color=colors_extended[j % len(colors_extended)], label=f'Re(λ_{j+1})',
                        markersize=6, alpha=0.8, linewidth=2)

            ax1.axvline(params_optimal[i], color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label='Optimal')
            ax1.set_xlabel(param_label, fontsize=14, fontweight='bold')
            ax1.set_ylabel('Re(λ)', fontsize=14, fontweight='bold')
            ax1.set_title(f'Real Parts vs {param_label}', fontsize=16, fontweight='bold')
            ax1.legend(fontsize=11, framealpha=0.9, loc='best')
            ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax1.tick_params(labelsize=11)
            # Format x-axis to avoid overlapping labels
            ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax1.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))

            # Right panel: Imaginary parts
            for j in range(n_eigvals):
                ax2.plot(param_values, eigvals_imag_array[:, j], 's-',
                        color=colors_extended[j % len(colors_extended)], label=f'Im(λ_{j+1})',
                        markersize=6, alpha=0.8, linewidth=2)

            ax2.axvline(params_optimal[i], color='red', linestyle='--',
                       linewidth=2, alpha=0.7, label='Optimal')
            ax2.set_xlabel(param_label, fontsize=14, fontweight='bold')
            ax2.set_ylabel('Im(λ)', fontsize=14, fontweight='bold')
            ax2.set_title(f'Imaginary Parts vs {param_label}', fontsize=16, fontweight='bold')
            ax2.legend(fontsize=11, framealpha=0.9, loc='best')
            ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
            ax2.tick_params(labelsize=11)
            # Format x-axis to avoid overlapping labels
            ax2.xaxis.set_major_locator(plt.MaxNLocator(6))
            ax2.ticklabel_format(style='sci', axis='x', scilimits=(-3,3))

            plt.tight_layout()
            scan_path = os.path.join(output_dir, f'scan_{param_name}.png')
            plt.savefig(scan_path, dpi=300, bbox_inches='tight')
            plt.close()

        return True
    except Exception as e:
        print(f"Warning: Could not generate parameter scan plots: {e}")
        return False
