"""Brute-force and exact-error gamma experiments on a fixed mesh."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
from petsc4py import PETSc

from main import solve_level
from meshing import create_mesh, refine_mesh
from problem import exact_curl, exact_solution
from utils import calc_hcurl_error


def _fixed_mesh(experiment, level, work_root):
    """Create the same uniformly refined mesh used by a normal experiment."""
    state = create_mesh(
        eg=experiment.geometry_example,
        backend=experiment.mesh_backend,
        comm=MPI.COMM_WORLD,
        gmsh_path=Path(work_root) / "mesh" / "gamma-initial.msh",
    )
    for _ in range(experiment.initial_refinements + level):
        state = refine_mesh(state)
    return state.domain


def _scan(experiment, run, level, gamma_values, work_root):
    """Solve every gamma on one fixed mesh and return root-rank rows."""
    domain = _fixed_mesh(experiment, level, work_root)
    rows = []
    u_exact = exact_solution(domain, experiment.geometry_example)

    for gamma in gamma_values:
        result = solve_level(domain, experiment, run, gamma_override=gamma)
        owned = domain.topology.index_map(domain.topology.dim).size_local

        def global_norm(field):
            local = np.sum(field.x.array[:owned] ** 2)
            return float(np.sqrt(domain.comm.allreduce(local, op=MPI.SUM)))

        rows.append(
            {
                "gamma": float(gamma),
                "eta1": global_norm(result.eta1),
                "eta2": global_norm(result.eta2),
                "eta3": global_norm(result.eta3),
                "estimator": result.estim_error,
                "hcurl_error": (
                    None
                    if u_exact is None
                    else calc_hcurl_error(
                        u_exact,
                        result.uh,
                        is_ref=False,
                        curl_u=exact_curl(
                            domain, experiment.geometry_example
                        ),
                    )
                ),
                "dofs": result.uh.function_space.dofmap.index_map.size_global,
                "iterations": result.num_iterations,
            }
        )
        PETSc.Sys.Print(
            f"gamma={gamma:.6g}: eta={result.estim_error:.6e}, "
            f"hcurl={rows[-1]['hcurl_error']}"
        )
    return rows


def _write_results(rows, output, use_exact):
    """Write gamma-scan data and available log-log diagnostic plots."""
    comm = MPI.COMM_WORLD
    output = Path(output)
    if comm.rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        data = pd.DataFrame(rows)
        data.to_csv(output / "gamma-scan.csv", index=False)
        plot_data = data.sort_values("gamma")

        figure, axis = plt.subplots()
        axis.loglog(
            plot_data["gamma"],
            plot_data["estimator"],
            "o-",
            label=r"$\eta$",
        )
        axis.set_xlabel(r"$\gamma$")
        axis.set_ylabel(r"estimator $\eta$")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / "eta-vs-gamma.png", dpi=180)
        plt.close(figure)

        if use_exact:
            best = data.loc[data["hcurl_error"].idxmin()]
            figure, axis = plt.subplots()
            axis.loglog(
                plot_data["gamma"],
                plot_data["hcurl_error"],
                "o-",
                label=r"$\|u-u_h\|_{H(\mathrm{curl})}$",
            )
            axis.loglog(
                best["gamma"],
                best["hcurl_error"],
                "r*",
                markersize=12,
                label=rf"best $\gamma={best['gamma']:.3g}$",
            )
            axis.set_xlabel(r"$\gamma$")
            axis.set_ylabel(r"$H(\mathrm{curl})$ error")
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                output / "hcurl-error-vs-gamma.png",
                dpi=180,
            )
            plt.close(figure)
            PETSc.Sys.Print(
                f"Optimal gamma by exact H(curl) error: {best['gamma']:.6g}"
            )
        else:
            PETSc.Sys.Print(
                "No exact solution: gamma-scan uses estimator only."
            )
    comm.Barrier()


def gamma_scan(experiment, run, level, gamma_values, output, work_root):
    """Run a brute-force scan and plot estimator versus gamma."""
    if run.formulation != "regularized":
        raise ValueError("Gamma scans require a regularized run.")
    rows = _scan(experiment, run, level, gamma_values, work_root)
    if MPI.COMM_WORLD.rank == 0:
        _write_results(rows, output, rows[0]["hcurl_error"] is not None)
    else:
        MPI.COMM_WORLD.Barrier()
    return rows


def find_optimal_gamma(experiment, run, level, gamma_values, output, work_root):
    """Select the gamma with minimum exact H(curl) error."""
    domain = _fixed_mesh(experiment, level, work_root)
    if exact_solution(domain, experiment.geometry_example) is None:
        raise ValueError(
            "No exact solution is available for this example; "
            "use the gamma-scan command instead."
        )
    rows = _scan(experiment, run, level, gamma_values, work_root)
    _write_results(rows, output, True)
    return min(rows, key=lambda row: row["hcurl_error"])
