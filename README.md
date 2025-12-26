# EP3 Optimizer

Finding third-order exceptional points (EP3) in nuclear resonance cavity structures using two-stage optimization.

## Physical System

Multi-layer thin film structure:
```
Pt / (C/Fe*)×3 / C / Pt(substrate)
     └─────────────┘
    3 resonant layers
```

**Goal**: Find parameters where Green function matrix has three degenerate eigenvalues (triple exceptional point).

## Quick Start

```bash
# Run with default settings (20 workers, seed=0)
python optimize_ep3.py

# Custom configuration
python optimize_ep3.py -i1 10000 -i2 5000 -w 25 -s 812
```

**Command-line arguments**:
- `-i1`: DE iterations (default: 100000)
- `-i2`: L-BFGS-B iterations (default: 5000)
- `-w`: Number of parallel workers (default: 20)
- `-s`: Random seed (default: 0)

## Project Structure

```
optimize_ep3.py          # Main optimization script
config.py                # Configuration (bounds, material parameters)
common_functions.py      # Shared functions (build_layers, objective_function_control)
green_function.py        # Green's function calculation
plotting_utils.py        # Plotting and visualization
previous/                # Old versions (archived)
results/                 # Optimization outputs
```

## Algorithm

**Two-stage optimization**:
1. **Phase 1**: Differential Evolution (global search)
   - Population-based stochastic search
   - Highly parallelized (workers > 1)
   - Explores large parameter space

2. **Phase 2**: L-BFGS-B (local refinement)
   - Gradient-based optimization
   - Bounded constraints
   - High-precision convergence

**Loss function** (variance-based):
```python
Loss = Σ|λᵢ - mean(λ)|² + penalty_imag
```
where λᵢ are eigenvalues of matrix: **G_shifted = -G - 0.5j·I**

**Constraint**: `|Im(λ)| ≥ 5.0` (ensures physical significance)

## Optimization Parameters

**9 optimization variables**:
- `theta0`: Incident angle (2.0-8.0 mrad)
- `t_Pt`: Platinum layer thickness (0.5-4.0 nm)
- `t_C1, t_C2, t_C3, t_C4`: Carbon layer thicknesses (1.0-40.0 nm)
- `t_Fe1`: First Fe* layer (0.8-3.0 nm, resonant)
- `t_Fe2, t_Fe3`: Other Fe* layers (0.5-3.0 nm, resonant)

**Fixed material constants** (see `config.py`):
- Iron (⁵⁷Fe): δ = 7.298e-6, β = 3.33e-7
- Carbon: δ = 2.257e-6, β = 1.230e-9
- Platinum: δ = 1.713e-5, β = 2.518e-6

## Output Files

Results are saved to `results/ep3_w{workers}_s{seed}_DE{iter1}_LB{iter2}/`:

- `optimization_log.txt`: Iteration history
- `parameters_high_precision.txt`: Final parameters (15-digit precision)
- `eigenvalues.txt`: Final eigenvalues
- `optimization_result.png`: Loss and eigenvalue convergence plots
- `scan_*.png`: Parameter sensitivity scans
- `params.npz`: NumPy archive with results

## Key Features

✅ **Multi-core parallelization**: Utilizes all CPU cores for DE phase
✅ **Robust multiprocessing**: Handles Windows spawn mode correctly
✅ **Real-time logging**: Progress updates written immediately
✅ **Automatic visualization**: Generates convergence and sensitivity plots
✅ **High precision**: 15-digit floating point output
✅ **Bounds checking**: Verifies all parameters within valid ranges

## Requirements

```bash
numpy
scipy
matplotlib
tqdm
```

Install with:
```bash
conda install numpy scipy matplotlib tqdm
# or
pip install numpy scipy matplotlib tqdm
```

## Technical Notes

### Multiprocessing
- Uses module-level wrapper function for pickle compatibility
- Spawns child processes that re-import the module
- Verified working with workers up to 32 on Windows

### Convergence Settings
- `tol=0, atol=0`: Prevents premature convergence
- `updating='deferred'`: Batch updates for parallel efficiency
- `strategy='best1bin'`: DE strategy optimized for continuous problems

### Verified Configuration
All settings in `config.py` have been verified to find good solutions:
- Bounds: Based on Control_N reference implementation
- Constraint penalty: `1e4 × (5.0 - min|Im(λ)|)²`
- Default iterations: 100k DE + 5k L-BFGS-B

## Theory

**Exceptional Points (EPs)** are non-Hermitian degeneracies where:
- Eigenvalues coalesce: λ₁ = λ₂ = λ₃
- Eigenvectors also coalesce (non-diagonalizable matrix)

For nuclear resonance cavities, the effective non-Hermitian Hamiltonian is constructed from the Green's matrix of the layered structure. EP3 represents a third-order degeneracy with unique topological properties.

## Documentation

Detailed technical documentation and lessons learned:
- `.claude/optimization_lessons_log.md`: Troubleshooting guide, algorithm insights, known issues

## References

- Nuclear resonance optics and cavity QED
- Non-Hermitian quantum mechanics
- Differential evolution (Storn & Price, 1997)
- L-BFGS-B optimization (Byrd et al., 1995)
