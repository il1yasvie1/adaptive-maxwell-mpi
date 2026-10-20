"""Command-line driver for adaptive quasilinear Maxwell experiments.

The numerical work is intentionally kept in small modules.  This file only
coordinates one refinement run, writes its results, and optionally compares
stored solutions with a reference checkpoint.
"""

from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path

import adios4dolfinx
import dolfinx.fem as fem
from mpi4py import MPI
from petsc4py import PETSc

from config import load_config, write_resolved_config
from estimate import estimate
from mark import mark
from meshing import MeshState, create_mesh, refine_mesh
from problem import exact_curl, exact_solution, set_problem_data
from solver import set_LinearProblem, set_MixedProblem, set_NonlinearProblem
from utils import (
    RunPaths,
    append_csv_log,
    calc_hcurl_error,
    init_csv_log,
    update_data_hcurl_error,
    update_csv_log,
    vtx_write,
    write_data_csv,
)


@dataclass
class LevelResult:
    uh: object
    ph: object
    eta1: object
    eta2: object
    eta3: object
    gamma: float
    num_iterations: int
    estim_error: float
    hcurl_error: float | None


def _checkpoint_metadata(experiment, run, level, num_cells, gamma):
    """Describe all configuration values that affect a stored level."""
    metadata = {
        "version": 1,
        "example": experiment.example,
        "degree": experiment.degree,
        "mesh_backend": experiment.mesh_backend,
        "initial_refinements": experiment.initial_refinements,
        "run_id": run.id,
        "strategy": run.strategy,
        "formulation": run.formulation,
        "theta12": run.theta12,
        "theta3": run.theta3,
        "level": level,
        "num_cells": num_cells,
        "gamma": gamma,
    }
    if run.is_regularized:
        metadata["regularization"] = {
            "gamma0": experiment.gamma0,
            "n0": experiment.n0,
            "mu": experiment.gamma_exponent,
        }
    if experiment.version == "new":
        metadata["algorithm_version"] = "new"
    return metadata


def _checkpoint_ready(paths, experiment, run, level):
    """Return whether a level is complete and configuration-compatible.

    The pressure field is checked separately because adaptive marking uses
    both ``u_h`` and ``p_h``.  This also makes old checkpoints, which only
    stored ``u_h`` or have no metadata, safely recompute.  For regularized
    runs, changed gamma settings invalidate the checkpoint.
    """
    metadata_path = paths.checkpoint_metadata(run.id, level)
    if not (
        paths.checkpoint(run.id, level).exists()
        and paths.pressure_checkpoint(run.id, level).exists()
        and metadata_path.exists()
    ):
        return False

    try:
        with metadata_path.open() as stream:
            stored = json.load(stream)
        num_cells = int(stored["num_cells"])
        gamma = float(stored["gamma"])
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False

    expected = _checkpoint_metadata(
        experiment,
        run,
        level,
        num_cells,
        experiment.gamma(num_cells, run.formulation),
    )
    stored.pop("gamma", None)
    expected_gamma = expected.pop("gamma")
    return stored == expected and math.isclose(
        gamma,
        expected_gamma,
        rel_tol=1e-12,
        abs_tol=1e-14,
    )


def _read_checkpoint(checkpoint, pressure_checkpoint, degree):
    """Read a stored mesh and both solution fields."""
    domain = adios4dolfinx.read_mesh(str(checkpoint), MPI.COMM_WORLD)
    V = fem.functionspace(domain, (f"N{degree}E", 1))
    uh = fem.Function(V)
    adios4dolfinx.read_function(str(checkpoint), uh)

    S = fem.functionspace(domain, ("P", degree))
    ph = fem.Function(S)
    adios4dolfinx.read_function(str(pressure_checkpoint), ph)
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    return domain, uh, ph


def _problem_data(domain, experiment):
    problem_kwargs = {}
    if experiment.a1 is not None:
        problem_kwargs["a1"] = experiment.a1
        problem_kwargs["degree"] = experiment.degree
    return set_problem_data(
        domain,
        experiment.geometry_example,
        **problem_kwargs,
    )


def _result(domain, experiment, uh, ph, gamma, iterations, problem_data):
    alpha, f, g = problem_data
    eta1, eta2, eta3, estim_error = estimate(
        domain,
        (uh, ph),
        (alpha, f, g),
        gamma,
        experiment.version,
    )
    u_exact = exact_solution(domain, experiment.geometry_example)
    hcurl_error = None
    if u_exact is not None:
        hcurl_error = calc_hcurl_error(
            u_exact,
            uh,
            curl_u=exact_curl(domain, experiment.geometry_example),
        )
    return LevelResult(
        uh, ph, eta1, eta2, eta3, gamma, iterations, estim_error, hcurl_error
    )


def _result_from_solution(domain, experiment, run, uh, ph):
    num_cells = domain.topology.index_map(domain.topology.dim).size_global
    gamma = experiment.gamma(num_cells, run.formulation)
    return _result(
        domain,
        experiment,
        uh,
        ph,
        gamma,
        0,
        _problem_data(domain, experiment),
    )


def solve_level(domain, experiment, run, gamma_override=None):
    """Solve one mesh level and compute its estimator and optional exact error."""
    num_cells = domain.topology.index_map(
        domain.topology.dim
    ).size_global
    gamma = (
        experiment.gamma(num_cells, run.formulation)
        if gamma_override is None
        else float(gamma_override)
    )
    problem_data = _problem_data(domain, experiment)
    alpha, f, g = problem_data

    if run.is_regularized:
        linear_problem = set_LinearProblem(domain, g, experiment.degree)
        ph = linear_problem.solve()
        ph.x.scatter_forward()
        nonlinear_problem = set_NonlinearProblem(
            domain,
            ph,
            alpha,
            f,
            gamma,
            experiment.degree,
        )
        uh = nonlinear_problem.solve()
        uh.x.scatter_forward()
        num_iterations = (
            nonlinear_problem.solver.getLinearSolveIterations()
        )
        del nonlinear_problem
        del linear_problem
    else:
        uh, ph, num_iterations = set_MixedProblem(
            domain,
            alpha,
            f,
            g,
            experiment.degree,
            return_iterations=True,
        )

    return _result(
        domain,
        experiment,
        uh,
        ph,
        gamma,
        num_iterations,
        problem_data,
    )


def _initial_mesh(experiment, paths):
    state = create_mesh(
        eg=experiment.geometry_example,
        backend=experiment.mesh_backend,
        comm=MPI.COMM_WORLD,
        gmsh_path=paths.gmsh_file,
    )
    for _ in range(experiment.initial_refinements):
        state = refine_mesh(state)
    return state


def _data_row(domain, result):
    return {
        "Num of Cells": domain.topology.index_map(
            domain.topology.dim
        ).size_global,
        "DoFs": result.uh.function_space.dofmap.index_map.size_global,
        "Num of Iters": result.num_iterations,
        "Estim Errors": result.estim_error,
        "Hcurl Errors": result.hcurl_error,
    }


def _level_identity(experiment, run, level):
    return {
        "name": experiment.name,
        "eg": experiment.example,
        "degree": experiment.degree,
        "run_id": run.id,
        "mode": run.strategy,
        "formulation": run.formulation,
        "mesh_backend": experiment.mesh_backend,
        "is_regularized": run.is_regularized,
        "initial_level": experiment.initial_refinements,
        "current_level": level,
    }


def _write_level(paths, experiment, run, level, domain, result):
    checkpoint = paths.checkpoint(run.id, level)
    adios4dolfinx.write_mesh(str(checkpoint), domain)
    adios4dolfinx.write_function(str(checkpoint), result.uh)
    adios4dolfinx.write_mesh(
        str(paths.pressure_checkpoint(run.id, level)), domain
    )
    adios4dolfinx.write_function(
        str(paths.pressure_checkpoint(run.id, level)), result.ph
    )
    vtx_write(result.uh, str(paths.view(run.id, level)))

    if domain.comm.rank == 0:
        num_cells = domain.topology.index_map(
            domain.topology.dim
        ).size_global
        metadata = _checkpoint_metadata(
            experiment,
            run,
            level,
            num_cells,
            result.gamma,
        )
        with paths.checkpoint_metadata(run.id, level).open("w") as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)

        row = _level_identity(experiment, run, level)
        row.update(
            {
                "gamma": result.gamma,
                "th12": run.theta12,
                "th3": run.theta3,
                "ncell": num_cells,
                "dof": result.uh.function_space.dofmap.index_map.size_global,
                "estim_error": result.estim_error,
                "hcurl_error": result.hcurl_error,
            }
        )
        append_csv_log(paths.csv, row)
    domain.comm.Barrier()


def run_refinement(experiment, run, paths):
    """Execute the solve-estimate-mark-refine loop for one configured run."""
    PETSc.Sys.Print(
        f"Running {run.id}: strategy={run.strategy}, "
        f"formulation={run.formulation}, levels={run.levels}"
    )
    state = _initial_mesh(experiment, paths)
    initial_solution = None
    if _checkpoint_ready(paths, experiment, run, 0):
        domain, uh, ph = _read_checkpoint(
            paths.checkpoint(run.id, 0),
            paths.pressure_checkpoint(run.id, 0),
            experiment.degree,
        )
        state = MeshState(
            domain,
            experiment.mesh_backend,
            state.geo_model,
        )
        initial_solution = uh, ph
        del domain, uh, ph
        PETSc.Sys.Print(f"{run.id} level 0: checkpoint found, skipped solve")

    data_rows = []
    for level in range(run.levels + 1):
        if level == 0 and initial_solution is not None:
            result = _result_from_solution(
                state.domain, experiment, run, *initial_solution
            )
            initial_solution = None
        elif _checkpoint_ready(paths, experiment, run, level):
            domain, uh, ph = _read_checkpoint(
                paths.checkpoint(run.id, level),
                paths.pressure_checkpoint(run.id, level),
                experiment.degree,
            )
            state = MeshState(
                domain,
                experiment.mesh_backend,
                state.geo_model,
            )
            result = _result_from_solution(
                state.domain, experiment, run, uh, ph
            )
            del domain, uh, ph
            PETSc.Sys.Print(
                f"{run.id} level {level}: checkpoint found, skipped solve"
            )
        else:
            result = solve_level(state.domain, experiment, run)
            _write_level(paths, experiment, run, level, state.domain, result)
        data_rows.append(_data_row(state.domain, result))
        if state.domain.comm.rank == 0:
            write_data_csv(paths.data_csv(run.id), data_rows)

        PETSc.Sys.Print(
            f"{run.id} level {level}: "
            f"cells={data_rows[-1]['Num of Cells']}, "
            f"dofs={data_rows[-1]['DoFs']}, "
            f"estimator={result.estim_error:.6e}"
        )
        if level == run.levels:
            del result
            gc.collect()
            break

        cells = None
        if run.strategy == "adaptive":
            cells = mark(
                result.eta1,
                result.eta2,
                result.eta3,
                run.theta12,
                run.theta3,
                experiment.version,
            )
        del result
        gc.collect()

        old_state = state
        state = refine_mesh(old_state, cells)
        del old_state, cells
        gc.collect()


def compute_reference_errors(
    experiment,
    run,
    paths,
    reference_checkpoint,
    levels="all",
):
    """Interpolate a reference solution and update target CSV error columns."""
    comm = MPI.COMM_WORLD
    if levels == "all":
        levels = range(run.levels + 1)

    ref_domain = adios4dolfinx.read_mesh(
        str(reference_checkpoint),
        comm,
    )
    V_ref = fem.functionspace(
        ref_domain,
        (f"N{experiment.degree}E", 1),
    )
    u_ref = fem.Function(V_ref)
    adios4dolfinx.read_function(
        str(reference_checkpoint),
        u_ref,
    )

    for level in levels:
        checkpoint = paths.checkpoint(run.id, level)
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Missing target checkpoint: {checkpoint}"
            )
        current_domain = adios4dolfinx.read_mesh(
            str(checkpoint),
            comm,
        )
        V_cur = fem.functionspace(
            current_domain,
            (f"N{experiment.degree}E", 1),
        )
        u_cur = fem.Function(V_cur)
        adios4dolfinx.read_function(str(checkpoint), u_cur)

        hcurl_error = calc_hcurl_error(u_ref, u_cur, is_ref=True)
        PETSc.Sys.Print(
            f"{run.id} level {level}: Hcurl error = {hcurl_error}"
        )
        if comm.rank == 0:
            update_csv_log(
                paths.csv,
                _level_identity(experiment, run, level),
                {"hcurl_error": hcurl_error},
            )
            update_data_hcurl_error(
                paths.data_csv(run.id),
                level,
                hcurl_error,
            )


def run_hcurl_error_tasks(experiment, paths, only=None):
    only = set(only or [])
    for task in experiment.hcurl_errors:
        reference_run = experiment.run(task.reference_run)
        reference_level = (
            reference_run.levels
            if task.reference_level == "final"
            else task.reference_level
        )
        reference_checkpoint = paths.checkpoint(
            reference_run.id,
            reference_level,
        )
        if not reference_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing reference checkpoint: {reference_checkpoint}"
            )
        for target in task.targets:
            if only and target.run not in only:
                continue
            run = experiment.run(target.run)
            if not paths.data_csv(run.id).exists():
                raise FileNotFoundError(
                    f"Missing data file for target run {run.id}."
                )
            compute_reference_errors(
                experiment,
                run,
                paths,
                reference_checkpoint,
                target.levels,
            )


def prepare(experiment, paths):
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        paths.ensure(run.id for run in experiment.runs)
        init_csv_log(paths.csv)
        write_resolved_config(
            experiment,
            paths.resolved_config,
        )
    comm.Barrier()


def print_config(experiment, source):
    PETSc.Sys.Print(f"Configuration: {Path(source).resolve()}")
    PETSc.Sys.Print(f"Experiment: {experiment.name}")
    PETSc.Sys.Print(f"Version: {experiment.version}")
    PETSc.Sys.Print(f"Mode: {experiment.mode}")
    PETSc.Sys.Print(
        f"Example={experiment.example}, degree={experiment.degree}, "
        f"backend={experiment.mesh_backend}, "
        f"initial refinements={experiment.initial_refinements}"
    )
    PETSc.Sys.Print("Runs:")
    for run in experiment.runs:
        PETSc.Sys.Print(
            f"  {run.id}: {run.strategy}, {run.formulation}, "
            f"levels={run.levels}"
        )
    PETSc.Sys.Print(
        f"Hcurl error tasks: {len(experiment.hcurl_errors)}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive quasilinear Maxwell experiments"
    )
    parser.add_argument("command", choices=("validate", "run", "errors"))
    parser.add_argument("config", help="YAML experiment configuration")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="Run or postprocess only this run id; may be repeated.",
    )
    args = parser.parse_args()

    experiment = load_config(args.config)
    print_config(experiment, args.config)
    if args.command == "validate":
        PETSc.Sys.Print("Configuration is valid.")
        return

    paths = RunPaths(
        experiment=experiment.name,
        example=experiment.example,
    )
    prepare(experiment, paths)

    if args.command == "run" and experiment.mode == "gamma":
        from gamma import adjoint_gamma_levels, find_optimal_gamma, gamma_scan

        for test in experiment.gamma_tests:
            run = experiment.run(test.run)
            output = paths.root / "gamma" / test.id
            if test.method == "adjoint":
                adjoint_gamma_levels(
                    experiment, run, test.levels, test.values[0], test.values[-1],
                    test.tolerance, output, paths.root
                )
            elif experiment.geometry_example in (1, 2):
                find_optimal_gamma(
                    experiment, run, test.level, test.values, output, paths.root
                )
            else:
                gamma_scan(
                    experiment, run, test.level, test.values, output, paths.root
                )
        PETSc.Sys.Print(f"Gamma results: {paths.root / 'gamma'}")
        return

    requested = set(args.only)
    known = {run.id for run in experiment.runs}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"Unknown --only run ids: {', '.join(sorted(unknown))}"
        )

    if args.command == "run":
        selected = [
            run
            for run in experiment.runs
            if not requested or run.id in requested
        ]
        for run in selected:
            run_refinement(experiment, run, paths)
        if requested:
            PETSc.Sys.Print(
                "Partial run completed; reference errors were not run. "
                "Use the errors command after required checkpoints exist."
            )
        else:
            run_hcurl_error_tasks(experiment, paths)
        PETSc.Sys.Print(f"Results: {paths.root}")
        return

    run_hcurl_error_tasks(experiment, paths, only=requested)
    PETSc.Sys.Print("Hcurl error postprocessing completed.")


if __name__ == "__main__":
    main()
