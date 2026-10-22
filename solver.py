"""Finite-element solvers for the two Maxwell formulations.

The module deliberately keeps the solver construction separate from the
experiment loop in ``main.py``.  This makes it possible to compare solvers
without changing mesh generation, marking, or output code.
"""

import basix.ufl
import dolfinx.fem as fem
import dolfinx.fem.petsc as petsc
import dolfinx.mesh as mesh
import ufl
import numpy as np


def _boundary_facets(domain):
    """Return exterior facets, creating the required mesh connectivity."""
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim - 1, tdim)
    return mesh.exterior_facet_indices(domain.topology)


def _zero_bc(space, domain, boundary_facets):
    """Create a homogeneous Dirichlet condition on a finite-element space."""
    dofs = fem.locate_dofs_topological(
        space,
        domain.topology.dim - 1,
        boundary_facets,
    )
    value = fem.Function(space)
    value.x.array[:] = 0.0
    return fem.dirichletbc(value, dofs)


def set_LinearProblem(domain, g, degree):
    """Build the scalar Poisson problem used by regularization."""
    S = fem.functionspace(domain, ("P", degree))
    p, q = ufl.TrialFunction(S), ufl.TestFunction(S)
    apq = ufl.inner(ufl.grad(p), ufl.grad(q))*ufl.dx
    lq = -g*q*ufl.dx
    boundary_facets = _boundary_facets(domain)
    bcs_S = [_zero_bc(S, domain, boundary_facets)]
    linear_problem_parameters = {
        "ksp_type": "cg",
        "pc_type": "gamg",
        "ksp_rtol": 1e-8,
        "ksp_view": None,
        }
    LinearProblem = petsc.LinearProblem(
                a = apq,
                L = lq,
                bcs = bcs_S,
                petsc_options_prefix="linear_",
                petsc_options=linear_problem_parameters,
                )
    return LinearProblem


def set_NonlinearProblem(domain, ph, alpha, f, gamma, degree):
    """Build the nonlinear H(curl) problem with mass regularization."""
    V = fem.functionspace(domain, (f"N{degree}E", 1))
    uh, v = fem.Function(V), ufl.TestFunction(V)
    F = (ufl.inner(alpha(uh) * ufl.curl(uh), ufl.curl(v)) \
         + ufl.inner(gamma * uh, v) \
         - ufl.inner(f, v) \
         - ufl.inner(gamma * ufl.grad(ph), v))*ufl.dx
    boundary_facets = _boundary_facets(domain)
    bcs_V = [_zero_bc(V, domain, boundary_facets)]
    nonlinear_problem_parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        "ksp_type": "cg",
        "pc_type": "gamg",
        "snes_view": None,
        "snes_monitor": None,
        "ksp_monitor": None,
        }
    NonlinearProblem = petsc.NonlinearProblem(
                F,
                uh,
                bcs = bcs_V,
                petsc_options_prefix="nonlinear_",
                petsc_options=nonlinear_problem_parameters,
                )
    return NonlinearProblem


def set_MixedProblem(
    domain,
    alpha,
    f,
    g,
    degree,
    return_iterations=False,
):
    """Solve the coupled nonlinear Maxwell saddle-point formulation.

    The Newton Jacobian is preconditioned by the block diagonal operator

        diag(A_k + M_edge, L),

    where ``A_k`` is the current nonlinear curl-curl Jacobian, ``M_edge``
    is the edge-element mass matrix, and ``L`` is the scalar Laplacian.
    The edge block is treated by hypre AMS and the scalar block by GAMG.
    This is the augmentation-free Maxwell preconditioner of
    Greif--Schotzau, with the H(curl) inverse supplied by the
    Hiptmair--Xu auxiliary-space method.

    A nested formulation is used so that AMS is attached to the edge
    block rather than to the full saddle-point matrix.  Both the discrete
    gradient and Cartesian constant edge fields are supplied to AMS.

    Hypre AMS is used here only for the lowest-order first Nedelec family.
    Although hypre has a high-order/second-family interface, the required
    interpolation layout is not provided safely by the current DOLFINx
    ``N2E`` path (it causes a hypre segmentation fault rather than a
    recoverable error), so reject that case explicitly.
    """
    if degree != 1:
        raise NotImplementedError(
            "The AMS mixed solver currently supports only degree=1 "
            "(lowest-order N1E). The N2E high-order AMS interface needs "
            "a separately validated interpolation operator."
        )

    cell = domain.basix_cell()
    curl_element = basix.ufl.element(f"N{degree}E", cell, 1)
    scalar_element = basix.ufl.element("Lagrange", cell, degree)
    V = fem.functionspace(domain, curl_element)
    Q = fem.functionspace(domain, scalar_element)

    uh, ph = fem.Function(V), fem.Function(Q)
    du, dp = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    v, q = ufl.TestFunction(V), ufl.TestFunction(Q)

    F_u = (
        ufl.inner(alpha(uh) * ufl.curl(uh), ufl.curl(v))
        + ufl.inner(ufl.grad(ph), v)
        - ufl.inner(f, v)
    ) * ufl.dx
    F_p = (
        ufl.inner(uh, ufl.grad(q))
        + g * q
    ) * ufl.dx
    F = [F_u, F_p]

    J_uu = ufl.derivative(F_u, uh, du)
    J_up = ufl.derivative(F_u, ph, dp)
    J_pu = ufl.derivative(F_p, uh, du)
    J_pp = ufl.derivative(F_p, ph, dp)
    J = [[J_uu, J_up], [J_pu, J_pp]]

    # Greif--Schotzau block preconditioner.  The added edge mass removes
    # the gradient kernel of the curl-curl block, while the scalar
    # Laplacian is spectrally equivalent to the Schur complement.
    P_uu = J_uu + ufl.inner(du, v) * ufl.dx
    P_pp = ufl.inner(ufl.grad(dp), ufl.grad(q)) * ufl.dx
    P = [[P_uu, None], [None, P_pp]]

    boundary_facets = _boundary_facets(domain)
    bcs = [
        _zero_bc(V, domain, boundary_facets),
        _zero_bc(Q, domain, boundary_facets),
    ]

    nonlinear_problem_parameters = {
        # -------- Nonlinear solver: Inexact Newton --------
        "snes_type": "newtonls",
        "snes_linesearch_type": "basic",
        # "snes_rtol": 1e-8,
        # "snes_atol": 1e-7,
        # "snes_stol": 1e-12,
        # "snes_max_it": 50,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_error_if_not_converged": None,
        "snes_view": None,

        # -------- Outer linear solver: FGMRES --------
        "ksp_type": "gmres",
        "ksp_rtol": 1e-2,
        "ksp_atol": 1e-7,
        "ksp_max_it": 200,
        "ksp_norm_type": "unpreconditioned",
        "ksp_error_if_not_converged": None,
        "ksp_monitor": None,
        "ksp_converged_reason": None,
        "ksp_view": None,

        # -------- Block diagonal preconditioner --------
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "additive",

        # -------- Edge block: HYPRE AMS --------
        # AMS is configured programmatically after
        # the FieldSplit sub-KSP objects are created.
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "none",

        # -------- Scalar block: GAMG --------
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "hypre",
        "fieldsplit_1_pc_hypre_type": "boomeramg",
    }

    nonlinear_problem = petsc.NonlinearProblem(
        F,
        [uh, ph],
        J=J,
        P=P,
        bcs=bcs,
        kind="nest",
        petsc_options_prefix="mixed_snes_",
        petsc_options=nonlinear_problem_parameters,
    )

    # Assemble once so PCFIELDSPLIT can create its sub-solvers before the
    # first Newton step.  NonlinearProblem reassembles J and P thereafter.
    snes = nonlinear_problem.solver
    snes.computeJacobian(
        nonlinear_problem.x,
        nonlinear_problem.A,
        nonlinear_problem.P_mat,
    )
    ksp = snes.getKSP()
    ksp.setOperators(nonlinear_problem.A, nonlinear_problem.P_mat)
    pc = ksp.getPC()
    pc.setUp()

    sub_ksp = pc.getFieldSplitSubKSP()
    if len(sub_ksp) != 2:
        raise RuntimeError(
            f"Expected two Maxwell field splits, got {len(sub_ksp)}."
        )

    edge_pc = sub_ksp[0].getPC()
    edge_pc.setType("hypre")
    edge_pc.setHYPREType("ams")

    # G maps the full scalar space into the Nedelec space.  Do not apply
    # boundary conditions to G: hypre AMS requires the full mesh complex.
    G = petsc.discrete_gradient(Q, V)
    G.assemble()
    edge_pc.setHYPREDiscreteGradient(G)

    constant_fields = []
    # Hypre recommends the low-order interface for lowest-order N1E.
    # Supplying edge representations of the Cartesian constants lets AMS
    # construct its Nedelec interpolation internally.
    for component in range(domain.geometry.dim):
        constant = fem.Function(V)
        constant.interpolate(
            lambda x, i=component: np.vstack(
                [
                    np.ones_like(x[0]) if j == i
                    else np.zeros_like(x[0])
                    for j in range(domain.geometry.dim)
                ]
            )
        )
        constant.x.scatter_forward()
        constant_fields.append(constant)
    edge_pc.setHYPRESetEdgeConstantVectors(
        constant_fields[0].x.petsc_vec,
        constant_fields[1].x.petsc_vec,
        constant_fields[2].x.petsc_vec,
    )

    P_00 = nonlinear_problem.P_mat.getNestSubMatrix(0, 0)
    P_11 = nonlinear_problem.P_mat.getNestSubMatrix(1, 1)
    P_00.setOption(P_00.Option.SPD, True)
    P_11.setOption(P_11.Option.SPD, True)

    nonlinear_problem.solve()

    uh.x.scatter_forward()
    ph.x.scatter_forward()
    num_iterations = snes.getLinearSolveIterations()

    # PETSc keeps its own references to the AMS auxiliary operators.
    G.destroy()
    del nonlinear_problem
    if return_iterations:
        return uh, ph, num_iterations
    return uh, ph
