# Cahn-Hilliard Phase-Field Simulation

A 3-D numerical solver for the Poisson equaiton on a cubic lattice, with support for 
three iterative algorithms and two physical charge configurations.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
- SciPy
 
Install dependencies with:
 
```bash
pip install numpy matplotlib scipy
```

## poisson.py
This script has all the functions and classes to either run an animation or measurements of the simulation. 
The user needs to put in different arguements and customise the animation or measurement conditions.

### Arguments

- 'l', Lattice side length. The simulation grid is `l × l`, Default = 100
- 'omega', SOR relaxation parameter (only used with --alg sor), Default = 1.87
- 'tol', Convergence tolerance/accuracy, Default = 1e-6
- 'type', Experiment type: electric ('e'), magnetic ('m'), or SOR sweeps ('s')
- 'alg', Algorithm type: Jacobi ('j'), Gauss-Seidel ('gs'), or SOR ('sor')
- 'mp', Set True to use multiprocessing for SOR experiment, Default = False

## Command line examples

**Electric field** — from a point charge in a 100x100 lattice, solved with the Jacobi algorithm with a 1e-6 accuracy.

```
python3 poisson.py --type e --alg j --tol 1e-6 --l 100
```

**Magnetic field** — from an infinite wire in a 100x100 lattice, solved with the Jacobi algorithm with a 1e-6 accuracy.
```
python3 poisson.py --type m --alg j --tol 1e-6 --l 100
```

**SOR** — using all CPU cores with a 1e-3 accuracy in a 50x50 lattice.

```
python3 poisson.py --type s --alg sor --tol 1e-3 --l 50 --mp True
```

## Output
All outputs are saved relative to the scipt's directory:

```
outputs/
├── datafiles/     # Raw measurement data (.txt)
└── plots/         # Saved figures (.png, 300 dpi)
```
