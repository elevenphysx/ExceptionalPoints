"""
Exceptional Point Finder (Parallel DE + Nelder-Mead Strategy)
Structure: Pt-C-Iron-C-Iron-C-Iron-C-Pt(substrate, inf)
Target: Find parameters where all eigenvalues degenerate (lambda1 = lambda2 = lambda3)
        AND Absolute Real/Imaginary parts are large (> threshold).

Strategy:
    Stage 1: Global Search using Differential Evolution.
    Stage 2: Local Refinement using Nelder-Mead.
    Parallel: Runs multiple seeds simultaneously on different CPU cores.
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import sys
import os
import importlib.util
from datetime import datetime
from tqdm import tqdm
import concurrent.futures
import time
import argparse

# ============================================================
# Dynamic Import of Green Function
# ============================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
green_script_path = os.path.join(current_dir, "green function-new.py")

if not os.path.exists(green_script_path):
    green_script_path = "green function-new.py"

try:
    spec = importlib.util.spec_from_file_location("green_function_new", green_script_path)
    green_function_new = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(green_function_new)
    GreenFun = green_function_new.GreenFun
except Exception as e:
    print(f"Error importing GreenFun: {e}")
    print("Make sure 'green function-new.py' is in the same directory.")
    sys.exit(1)

# ============================================================
# Core Physics & Objective Functions
# ============================================================

def build_layers(params, fixed_materials):
    theta0 = params[0]
    thicknesses = params[1:9]
    C0 = 7.74 * 1.06 * 0.5
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

def objective_function(params, fixed_materials, return_details=False, threshold=5.0, penalty_weight=100.0):
    def pairwise_diff_sq(values):
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
            if return_details: return 1e10, eigvals, np.zeros(3), np.zeros(3), G, G1, 1e10, (1e10, 1e10)
            return 1e10

        real_parts = np.real(eigvals)
        imag_parts = np.imag(eigvals)

        loss_degeneracy = pairwise_diff_sq(real_parts) + pairwise_diff_sq(imag_parts)

        min_re = np.min(np.abs(real_parts))
        if min_re < threshold:
            gap_re = threshold - min_re
            penalty_real = (np.exp(gap_re) - 1.0)
        else:
            penalty_real = 0.0

        min_im = np.min(np.abs(imag_parts))
        if min_im < threshold:
            gap_im = threshold - min_im
            penalty_imag = (np.exp(gap_im) - 1.0)
        else:
            penalty_imag = 0.0

        loss = loss_degeneracy + penalty_weight * (penalty_real + penalty_imag)

        if return_details:
            return loss, eigvals, real_parts, imag_parts, G, G1, loss_degeneracy, (penalty_real, penalty_imag)
        return loss

    except Exception as e:
        if return_details: return 1e10, None, None, None, None, None, None, None
        return 1e10

# ============================================================
# Main Optimization Logic (Single Seed)
# ============================================================

def optimize_exceptional_point(maxiter_de, maxiter_nm, seed, threshold, penalty_weight, verbose=True):
    np.random.seed(seed)

    output_dir = os.path.join('results', f'DE_Exact_s{seed}_thr{threshold:.1f}_pw{penalty_weight:.1f}')
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print(f"--> [Seed {seed}] Output directory: {output_dir}")

    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    bounds = [
        (2.0, 10.0),
        (3.0, 10.0),
        (5.0, 50.0),
        (0.5, 3.0),
        (5.0, 50.0),
        (0.5, 3.0),
        (5.0, 50.0),
        (0.5, 3.0),
        (5.0, 50.0),
    ]

    log_file_path = os.path.join(output_dir, 'optimization_log.txt')
    log_file = open(log_file_path, 'w', encoding='utf-8')

    def log_print(msg, console=True):
        if console and verbose:
            print(f"[Seed {seed}] {msg}")
        log_file.write(msg + "\n")
        log_file.flush()

    history = {'iteration': [], 'loss': [], 'phase': [], 'eigvals_real': [], 'eigvals_imag': []}
    iteration_count = [0]

    # --- Phase 1: Differential Evolution ---
    log_print("=" * 70, console=False)
    log_print(f"Phase 1: DE Optimization (Seed: {seed})", console=False)
    log_print("=" * 70, console=False)

    pbar_de = tqdm(total=maxiter_de, desc=f"Seed {seed} DE", disable=not verbose)

    best_loss_so_far = float('inf')
    best_solution_so_far = None

    def callback_de(xk, convergence=None):
        iteration_count[0] += 1
        pbar_de.update(1)

        nonlocal best_loss_so_far, best_solution_so_far
        loss_current = objective_function(xk, fixed_materials, threshold=threshold, penalty_weight=penalty_weight)
        if loss_current < best_loss_so_far:
            best_loss_so_far = loss_current
            best_solution_so_far = xk.copy()

        if iteration_count[0] % 10 == 0 or iteration_count[0] == 1:
            loss, eigvals, real_parts, imag_parts, _, _, loss_deg, (pen_re, pen_im) = objective_function(
                best_solution_so_far, fixed_materials, return_details=True, threshold=threshold, penalty_weight=penalty_weight
            )
            msg = f"DE Iter {iteration_count[0]}: Loss={loss:.6e} (Deg={loss_deg:.6e}, PenRe={pen_re:.2e}, PenIm={pen_im:.2e})"
            if verbose: pbar_de.write(msg)
            log_file.write(msg + "\n")

            if eigvals is not None:
                history['iteration'].append(iteration_count[0])
                history['loss'].append(loss)
                history['phase'].append('DE')
                history['eigvals_real'].append(np.real(eigvals).copy())
                history['eigvals_imag'].append(np.imag(eigvals).copy())

        return False

    def obj_wrapper(p):
        return objective_function(p, fixed_materials, threshold=threshold, penalty_weight=penalty_weight)

    result_de = differential_evolution(
        obj_wrapper,
        bounds,
        maxiter=maxiter_de,
        popsize=20,
        strategy='best1bin',
        seed=seed,
        disp=False,
        workers=1,
        updating='immediate',
        polish=False,
        callback=callback_de,
        atol=0,
        tol=0
    )

    pbar_de.close()
    result_de_x = result_de.x

    # --- Phase 2: Nelder-Mead ---
    log_print("\n" + "=" * 70, console=False)
    log_print("Phase 2: Nelder-Mead Refinement", console=False)
    log_print("=" * 70, console=False)

    nm_iter = [0]
    pbar_nm = tqdm(total=maxiter_nm, desc=f"Seed {seed} Nelder-Mead", disable=not verbose)

    def nm_callback(xk):
        nm_iter[0] += 1
        pbar_nm.update(1)
        if nm_iter[0] % 50 == 0:
            loss = obj_wrapper(xk)
            msg = f"NM Iter {nm_iter[0]}: Loss={loss:.6e}"
            if verbose: pbar_nm.write(msg)
            log_file.write(msg + "\n")

    try:
        res_nm = minimize(obj_wrapper, result_de_x, method='Nelder-Mead', callback=nm_callback,
                         options={'maxiter': maxiter_nm, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
        final_x = res_nm.x
        final_loss = res_nm.fun
        pbar_nm.close()
    except:
        final_x = result_de_x
        final_loss = best_loss_so_far

    # --- Final Output ---
    loss, eigvals, real_parts, imag_parts, G, G1, loss_deg, (pen_re, pen_im) = objective_function(
        final_x, fixed_materials, return_details=True, threshold=threshold, penalty_weight=penalty_weight
    )

    theta0, Layers, C0 = build_layers(final_x, fixed_materials)

    output = "\n" + "="*70 + "\nFINAL RESULT\n" + "="*70 + "\n"
    output += f"Total Loss: {loss:.6e}\n"
    output += f"Degeneracy Loss: {loss_deg:.6e}\n"
    output += f"Penalty Re: {pen_re:.6e}\n"
    output += f"Penalty Im: {pen_im:.6e}\n\n"

    output += "Eigenvalues:\n"
    for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
        output += f"  lambda_{i+1} = {re:+.10f} {im:+.10f}i\n"

    output += "\nConstraint Check:\n"
    output += f"  Min|Re| = {np.min(np.abs(real_parts)):.4f} > {threshold}? {'YES' if pen_re==0 else 'NO'}\n"
    output += f"  Min|Im| = {np.min(np.abs(imag_parts)):.4f} > {threshold}? {'YES' if pen_im==0 else 'NO'}\n"

    log_print(output, console=verbose)
    log_file.close()

    # Save results
    np.savez(os.path.join(output_dir, 'params.npz'),
             params=final_x,
             loss=final_loss,
             eigenvalues=eigvals,
             theta0=theta0,
             C0=C0)

    params_txt_path = os.path.join(output_dir, 'parameters_high_precision.txt')
    with open(params_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Exceptional Point Parameters (15-digit Precision) - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Final Loss:  {final_loss:.15e}\n")
        f.write(f"Degeneracy:  {loss_deg:.15e}\n")
        f.write(f"Penalty Re:  {pen_re:.15e}\n")
        f.write(f"Penalty Im:  {pen_im:.15e}\n\n")
        f.write(f"theta0 = {theta0:.15f} mrad\n")
        f.write(f"C0     = {C0:.15f} (fixed)\n\n")
        f.write("Layer Thicknesses (nm):\n")
        f.write("-" * 50 + "\n")
        layer_names = ['Pt', 'C', 'Fe*', 'C', 'Fe*', 'C', 'Fe*', 'C']
        thicknesses = [layer[1] for layer in Layers[:-1]]
        for i, (name, thickness) in enumerate(zip(layer_names, thicknesses)):
            resonant = ' (resonant)' if Layers[i][2] == 1 else ''
            f.write(f"  Layer {i}: {name:8s} = {thickness:20.15f} nm{resonant}\n")
        f.write(f"  Layer 8: Pt(sub)  = inf nm\n\n")
        f.write("All Optimization Parameters:\n")
        f.write("-" * 50 + "\n")
        f.write(f"  params[0] (theta0) = {final_x[0]:.15f} mrad\n")
        for i in range(1, len(final_x)):
            f.write(f"  params[{i}] (Layer {i-1} thickness) = {final_x[i]:.15f} nm\n")

    eigvals_txt_path = os.path.join(output_dir, 'eigenvalues.txt')
    with open(eigvals_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(f"Eigenvalue Analysis - Seed {seed}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Eigenvalues (15-digit precision):\n")
        f.write("-" * 50 + "\n")
        for i, (re, im) in enumerate(zip(real_parts, imag_parts)):
            f.write(f"  λ_{i+1} = {re:+22.15f} {im:+22.15f}i\n")
        f.write("\n")
        f.write(f"Real parts: [{', '.join([f'{re:.15f}' for re in real_parts])}]\n")
        f.write(f"Imag parts: [{', '.join([f'{im:.15f}' for im in imag_parts])}]\n")
        f.write("\n")
        f.write("Degeneracy Check:\n")
        f.write(f"  Std(Re) = {np.std(real_parts):.15e}\n")
        f.write(f"  Std(Im) = {np.std(imag_parts):.15e}\n")
        f.write(f"  Min|Re| = {np.min(np.abs(real_parts)):.15f}\n")
        f.write(f"  Min|Im| = {np.min(np.abs(imag_parts)):.15f}\n")

    try:
        import matplotlib as mpl
        mpl.rcParams['font.size'] = 12
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams['lines.linewidth'] = 2.0
        mpl.rcParams['xtick.major.width'] = 1.5
        mpl.rcParams['ytick.major.width'] = 1.5
        mpl.rcParams['xtick.major.size'] = 5
        mpl.rcParams['ytick.major.size'] = 5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        iterations = np.array(history['iteration'])
        losses = np.array(history['loss'])
        eigvals_real = history['eigvals_real']
        eigvals_imag = history['eigvals_imag']

        ax1.semilogy(iterations, losses, 'o-', color='#1f77b4',
                    label='DE + NM', markersize=4, alpha=0.8)
        ax1.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Loss (Variance)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Optimization Progress (Seed {seed})', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.legend(fontsize=11, framealpha=0.9, loc='best')
        ax1.tick_params(labelsize=11)

        distances_12 = []
        distances_13 = []
        distances_23 = []

        for real_parts_iter, imag_parts_iter in zip(eigvals_real, eigvals_imag):
            if len(real_parts_iter) >= 3:
                eig1 = real_parts_iter[0] + 1j * imag_parts_iter[0]
                eig2 = real_parts_iter[1] + 1j * imag_parts_iter[1]
                eig3 = real_parts_iter[2] + 1j * imag_parts_iter[2]

                distances_12.append(np.abs(eig1 - eig2))
                distances_13.append(np.abs(eig1 - eig3))
                distances_23.append(np.abs(eig2 - eig3))
            else:
                distances_12.append(np.nan)
                distances_13.append(np.nan)
                distances_23.append(np.nan)

        distances_12 = np.array(distances_12)
        distances_13 = np.array(distances_13)
        distances_23 = np.array(distances_23)

        ax2.semilogy(iterations, distances_12, 'o-', color='#d62728',
                    label='|λ₁ - λ₂|', markersize=3, alpha=0.8)
        ax2.semilogy(iterations, distances_13, 's-', color='#9467bd',
                    label='|λ₁ - λ₃|', markersize=3, alpha=0.8)
        ax2.semilogy(iterations, distances_23, '^-', color='#8c564b',
                    label='|λ₂ - λ₃|', markersize=3, alpha=0.8)

        ax2.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Distance', fontsize=14, fontweight='bold')
        ax2.set_title('Eigenvalue Convergence', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.legend(fontsize=11, framealpha=0.9, loc='best')
        ax2.tick_params(labelsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'optimization_result.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        if verbose:
            print(f"Warning: Could not generate plots: {e}")

    scan_parameters_around_optimum(
        params_optimal=final_x,
        fixed_materials=fixed_materials,
        output_dir=output_dir,
        scan_range=1e-5,
        threshold=threshold,
        penalty_weight=penalty_weight
    )

    return seed, final_loss, final_x


def scan_parameters_around_optimum(params_optimal, fixed_materials, output_dir, scan_range, threshold, penalty_weight):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['lines.linewidth'] = 2.0
    mpl.rcParams['xtick.major.width'] = 1.5
    mpl.rcParams['ytick.major.width'] = 1.5

    colors = ['#4472C4', '#ED7D31', '#70AD47']
    param_names = ['theta0', 't_Pt', 't_C1', 't_Fe1', 't_C2', 't_Fe2', 't_C3', 't_Fe3', 't_C4']
    param_labels = ['θ₀ (mrad)', 't_Pt (nm)', 't_C₁ (nm)', 't_Fe₁ (nm)',
                    't_C₂ (nm)', 't_Fe₂ (nm)', 't_C₃ (nm)', 't_Fe₃ (nm)', 't_C₄ (nm)']

    for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
        n_points = 21
        param_values = np.linspace(params_optimal[i] - scan_range, params_optimal[i] + scan_range, n_points)

        eigvals_real_list = []
        eigvals_imag_list = []

        for param_val in param_values:
            params_test = params_optimal.copy()
            params_test[i] = param_val
            try:
                loss, eigvals, re, im, G, G1, _, _ = objective_function(
                    params_test, fixed_materials, return_details=True, threshold=threshold, penalty_weight=penalty_weight
                )
                eigvals_real_list.append(re)
                eigvals_imag_list.append(im)
            except:
                eigvals_real_list.append([np.nan, np.nan, np.nan])
                eigvals_imag_list.append([np.nan, np.nan, np.nan])

        eigvals_real_array = np.array(eigvals_real_list)
        eigvals_imag_array = np.array(eigvals_imag_list)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for j in range(3):
            ax1.plot(param_values, eigvals_real_array[:, j], 'o-',
                    color=colors[j], label=f'Re(λ_{j+1})', markersize=6, alpha=0.8, linewidth=2)

        ax1.axvline(params_optimal[i], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
        ax1.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Re(λ)', fontsize=14, fontweight='bold')
        ax1.set_title(f'Real Parts vs {param_label}', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=11, framealpha=0.9, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax1.tick_params(labelsize=11)

        for j in range(3):
            ax2.plot(param_values, eigvals_imag_array[:, j], 's-',
                    color=colors[j], label=f'Im(λ_{j+1})', markersize=6, alpha=0.8, linewidth=2)

        ax2.axvline(params_optimal[i], color='red', linestyle='--', linewidth=2, alpha=0.7, label='Optimal')
        ax2.set_xlabel(param_label, fontsize=14, fontweight='bold')
        ax2.set_ylabel('Im(λ)', fontsize=14, fontweight='bold')
        ax2.set_title(f'Imaginary Parts vs {param_label}', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11, framealpha=0.9, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        ax2.tick_params(labelsize=11)

        plt.tight_layout()
        scan_path = os.path.join(output_dir, f'scan_{param_name}.png')
        plt.savefig(scan_path, dpi=300, bbox_inches='tight')
        plt.close()


# ============================================================
# Parallel Execution Helpers
# ============================================================

def run_single_seed(args):
    seed, max_de, max_nm, threshold, penalty_weight = args
    try:
        _, loss, final_x = optimize_exceptional_point(
            maxiter_de=max_de,
            maxiter_nm=max_nm,
            seed=seed,
            threshold=threshold,
            penalty_weight=penalty_weight,
            verbose=False
        )
        return seed, loss, final_x, "Success"
    except Exception as e:
        return seed, None, None, str(e)

if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "1"

    import argparse
    parser = argparse.ArgumentParser(description='Exceptional Point Finder (Parallel DE)')
    parser.add_argument('-i1', type=int, default=100, help='DE iterations')
    parser.add_argument('-i2', type=int, default=500, help='Nelder-Mead iterations')
    parser.add_argument('-t', type=float, default=5.0, help='Threshold')
    parser.add_argument('-pw', type=float, default=100.0, help='Penalty Weight')
    parser.add_argument('-w', '--workers', type=int, default=None, help='Number of parallel processes')
    parser.add_argument('-s', '--seeds', type=int, nargs='+', default=[812, 1001, 2023, 3030, 42], help='List of seeds to run')

    args = parser.parse_args()

    task_args = [(seed, args.i1, args.i2, args.t, args.pw) for seed in args.seeds]

    print("=" * 70)
    print(f"Starting Parallel Optimization (DE)")
    print(f"Seeds to run: {args.seeds}")
    print(f"Target Threshold: > {args.t}")
    print(f"Workers: {args.workers if args.workers else 'Max Available'}")
    print("=" * 70)

    start_time = time.time()

    Iron = (7.298e-6, 3.33e-7)
    Carbon = (2.257e-6, 1.230e-9)
    Platinum = (1.713e-5, 2.518e-6)
    fixed_materials = (Platinum, Carbon, Iron)

    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_single_seed, t): t[0] for t in task_args}

        for future in concurrent.futures.as_completed(futures):
            seed_val = futures[future]
            try:
                seed, loss, final_x, status = future.result()
                if status == "Success":
                    print(f"✅ Seed {seed} Finished | Final Loss: {loss:.6e}")
                    results.append((seed, loss, final_x))
                else:
                    print(f"❌ Seed {seed} Failed | Error: {status}")
            except Exception as exc:
                print(f"❌ Seed {seed_val} generated an exception: {exc}")

    print("\n" + "=" * 70)
    print("All Tasks Completed")
    print(f"Total Time: {time.time() - start_time:.2f} seconds")
    print("=" * 70)
    print("DETAILED SUMMARY (Sorted by Loss):")

    results.sort(key=lambda x: x[1])

    for s, l, x in results:
        theta0, Layers, C0 = build_layers(x, fixed_materials)
        G, _ = GreenFun(theta0, Layers, C0)
        G1 = -G - 1j * 0.5 * np.eye(G.shape[0], dtype=complex)
        eigvals = np.linalg.eigvals(G1)
        re = np.real(eigvals)
        im = np.imag(eigvals)

        idx = np.argsort(re)
        re = re[idx]
        im = im[idx]

        print("-" * 60)
        print(f"Seed {s} | Loss = {l:.6e}")
        for k in range(3):
             print(f"   lambda_{k+1} = {re[k]:8.5f} {im[k]:+8.5f}j  (|Re|={np.abs(re[k]):.4f})")

        if np.std(re) < 1e-2 and np.std(im) < 1e-2:
             print("   [STATUS]: EXCELLENT DEGENERACY")
        elif np.std(re) < 0.1:
             print("   [STATUS]: Good Match")
        else:
             print("   [STATUS]: Split / Local Optima")

    if results:
        print("\n" + "=" * 70)
        print(f"🏆 BEST SEED: {results[0][0]} (Loss: {results[0][1]:.6e})")
        print("=" * 70)
