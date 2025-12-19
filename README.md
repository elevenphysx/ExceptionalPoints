# Exceptional Point Finder

Finding exceptional points (EPs) in nuclear resonance cavity structures using optimization methods.

## Structure

```
Pt - C - Fe - C - Fe - C - Fe - C - Pt(substrate)
     |       |       |       |
     └───────┴───────┴───────┘
         3 resonant layers
```

## Files

- `green function-new.py`: Core physics calculations (Green's function, Parratt reflectivity)
- `Singularity.py`: EP finder using eigenvalue variance minimization
- `Singularity_AD.py`: EP finder using algebraic conditions (trace/det method)

## Methods

### Method 1: Eigenvalue Variance (Singularity.py)

Minimize the variance of eigenvalues to find degeneracy:
```
Loss = Var(Re(λ)) + Var(Im(λ))
```

### Method 2: Algebraic Conditions (Singularity_AD.py) ⭐ Recommended

Use characteristic polynomial coefficients to avoid eigenvalue computation:

For EP3 (triple degeneracy), the conditions are:
```
a = -tr(G1)
b = [(tr G1)² - tr(G1²)] / 2
c = -det(G1)

Condition 1: 3b - a² = 0
Condition 2: 27c + 9ab + 2a³ = 0

Loss = |3b - a²| + |27c + 9ab + 2a³|
```

**Advantages:**
- ✅ More stable (avoids eigenvalue computation in optimization)
- ✅ Better gradient quality for numerical differentiation
- ✅ Faster convergence

## Optimization Strategy

1. **Phase 1**: Differential Evolution (DE) for global search
2. **Phase 2**: L-BFGS-B with numerical gradients for local refinement

## Parameters

**Optimization variables (10 parameters):**
- `theta0`: Incident angle (0-10 mrad)
- Layer thicknesses:
  - Pt: 0.5-10 nm
  - C: 1-50 nm
  - Fe: 0.5-3 nm (resonant layers)
- `C0`: Constant parameter (3-5)

**Fixed material constants:**
- Iron: δ = 7.298e-6, β = 3.33e-7
- Carbon: δ = 2.257e-6, β = 1.230e-9
- Platinum: δ = 1.713e-5, β = 2.518e-6

## Usage

```bash
# Method 1: Eigenvalue variance
python Singularity.py

# Method 2: Algebraic conditions (recommended)
python Singularity_AD.py
```

## Output

- Console output: Optimization progress and final parameters
- `exceptional_point_result.png` / `exceptional_point_AD.png`: Eigenvalue visualization
- `exceptional_point_params.npz` / `exceptional_point_params_AD.npz`: Saved parameters

## Requirements

```
numpy
scipy
matplotlib
```

## Theory

**Exceptional Points (EPs)** are degeneracies in non-Hermitian systems where:
- Eigenvalues coalesce: λ₁ = λ₂ = λ₃
- Eigenvectors also coalesce

For nuclear resonance cavities, G1 matrix is computed from:
```
G1 = -G - i/2·I
```
where G is the Green's matrix from layer structure.

## References

- Nuclear resonance optics
- Non-Hermitian quantum mechanics
- Differential evolution optimization
