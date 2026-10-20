import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("petsc4py")

import dolfinx.fem as fem
import dolfinx.mesh as mesh
from mpi4py import MPI
from petsc4py import PETSc
import ufl

from estimate import estimate
from mark import mark
from meshing import MeshState, refine_mesh
from solver import set_LinearProblem, set_NonlinearProblem


pytestmark = [pytest.mark.numerical, pytest.mark.mpi]


def _unit_mesh():
    return mesh.create_unit_cube(
        MPI.COMM_WORLD,
        2,
        2,
        2,
        ghost_mode=mesh.GhostMode.shared_facet,
    )


def test_zero_problem_solve_and_estimator():
    domain = _unit_mesh()
    zero_scalar = fem.Constant(domain, PETSc.ScalarType(0.0))
    zero_vector = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 0.0)))

    def alpha(_):
        return 1.0

    linear_problem = set_LinearProblem(domain, zero_scalar, degree=1)
    ph = linear_problem.solve()
    ph.x.scatter_forward()
    assert linear_problem.solver.getConvergedReason() > 0

    nonlinear_problem = set_NonlinearProblem(
        domain,
        ph,
        alpha,
        zero_vector,
        gamma=1.0,
        degree=1,
    )
    uh = nonlinear_problem.solve()
    uh.x.scatter_forward()
    assert nonlinear_problem.solver.getConvergedReason() > 0
    assert np.all(np.isfinite(uh.x.array))
    assert np.linalg.norm(uh.x.array) == pytest.approx(0.0, abs=1e-12)

    eta1, eta2, eta3, total = estimate(
        domain,
        (uh, ph),
        (alpha, zero_vector, zero_scalar),
        gam=1.0,
        version="old",
    )
    assert total == pytest.approx(0.0, abs=1e-12)
    for indicator in (eta1, eta2, eta3):
        assert np.all(np.isfinite(indicator.x.array))
        assert np.all(indicator.x.array >= 0.0)


def test_nonzero_nonlinear_problem_converges():
    domain = _unit_mesh()
    scalar_space = fem.functionspace(domain, ("P", 1))
    ph = fem.Function(scalar_space)
    forcing = fem.Constant(domain, PETSc.ScalarType((0.0, 0.0, 1.0)))

    def alpha(field):
        curl_norm_sq = ufl.inner(ufl.curl(field), ufl.curl(field))
        return 1.0 + 0.1 * curl_norm_sq

    problem = set_NonlinearProblem(
        domain,
        ph,
        alpha,
        forcing,
        gamma=1.0,
        degree=1,
    )
    uh = problem.solve()
    uh.x.scatter_forward()

    assert problem.solver.getConvergedReason() > 0
    assert problem.solver.getIterationNumber() > 0
    assert np.all(np.isfinite(uh.x.array))
    assert np.linalg.norm(uh.x.array) > 0.0


def test_global_marking_and_refinement_increase_cell_count():
    domain = _unit_mesh()
    dg0 = fem.functionspace(domain, ("DP", 0))
    indicators = [fem.Function(dg0) for _ in range(3)]
    owned = domain.topology.index_map(domain.topology.dim).size_local
    local_cells = np.arange(owned, dtype=np.int32)
    global_ids = domain.topology.index_map(
        domain.topology.dim
    ).local_to_global(local_cells)
    indicators[0].x.array[:owned] = global_ids + 1.0
    indicators[0].x.scatter_forward()

    cells = mark(
        *indicators,
        th12=0.5,
        th3=0.5,
        version="new",
    )
    global_marked = domain.comm.allreduce(len(cells), op=MPI.SUM)
    assert global_marked > 0

    before = domain.topology.index_map(domain.topology.dim).size_global
    refined = refine_mesh(MeshState(domain, "gmsh"), cells).domain
    after = refined.topology.index_map(refined.topology.dim).size_global
    assert after > before
