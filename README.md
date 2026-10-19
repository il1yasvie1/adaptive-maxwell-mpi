# Adaptive Maxwell MPI

MPI-enabled DOLFINx experiments for regularized adaptive finite element method of quasilinear Maxwell problems.

## Repository structure

```text
.
├── main.py                 # Command-line driver
├── config.py               # YAML configuration and validation
├── problem.py              # Problem data and exact solutions
├── solver.py               # Linear, nonlinear, and mixed solvers
├── estimate.py             # Residual error estimator
├── mark.py                 # Distributed Dörfler marking
├── meshing.py              # Gmsh/Netgen meshes and refinement
├── gamma.py                # Gamma parameter studies
├── utils.py                # Checkpoints, errors, and CSV output
├── configs/                # YAML experiment configurations
├── data/                   # CSV convergence data
├── tests/                  # Configuration and MPI smoke tests
├── environment.yml         # Conda environment specification
└── work/                   # Generated checkpoints and visualization data
```

## Requirements

Tested environment:

```text
Python       3.14.6
DOLFINx     0.11.0
UFL         2026.1.0
Basix       0.11.0
PETSc       compatible with petsc4py 3.25.5
mpi4py      4.1.2
ADIOS4DOLFINx 0.11.0
ngsPETSc    0.2.1
Gmsh        4.15.2
NumPy       2.5.2
pandas      3.0.5
PyYAML      6.0.3
matplotlib  3.11.1
MPICH       5.0.1
```

The DOLFINx, UFL, Basix, PETSc, MPI, ADIOS4DOLFINx, and ngsPETSc versions must be mutually compatible. The environment specification is in [environment.yml](environment.yml).

## Running from the terminal

Activate the environment:

```bash
conda env create -f environment.yml
conda activate adaptive-maxwell-mpi
```

Validate a configuration:

```bash
python main.py validate configs/n1-example1.yaml
```

Run all experiments in a configuration:

```bash
python main.py run configs/n1-example1.yaml
```

Run with MPI:

```bash
mpiexec -n 4 python main.py run configs/n1-example1.yaml
```

Run only one configured experiment:

```bash
mpiexec -n 4 python main.py run configs/n1-example1.yaml --only n1-adaptive
```

Post-process stored checkpoints:

```bash
mpiexec -n 4 python main.py errors configs/n1-example1.yaml
```

Run the MPI smoke test:

```bash
mpiexec -n 4 python tests/test_mpi_smoke.py
```

Results are written to `data/example*/` and `work/`.
