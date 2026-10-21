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
    """Solve the coupled nonlinear (u, p) saddle-point formulation."""
    cell = domain.basix_cell()
    curl_element = basix.ufl.element(f"N{degree}E", cell, 1)
    scalar_element = basix.ufl.element("Lagrange", cell, degree)
    W = fem.functionspace(
        domain,
        basix.ufl.mixed_element([curl_element, scalar_element]),
    )

    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)

    F = (
        ufl.inner(alpha(u) * ufl.curl(u), ufl.curl(v))
        + ufl.inner(ufl.grad(p), v)
        - ufl.inner(f, v)
        + ufl.inner(u, ufl.grad(q))
        + g * q
    ) * ufl.dx

    du = ufl.TrialFunction(W)
    J = ufl.derivative(F, w, du)

    boundary_facets = _boundary_facets(domain)

    V, _ = W.sub(0).collapse()
    boundary_dofs_V = fem.locate_dofs_topological(
        (W.sub(0), V), domain.topology.dim - 1, boundary_facets
    )
    u_bc = fem.Function(V)
    u_bc.x.array[:] = 0.0
    bcs = [fem.dirichletbc(u_bc, boundary_dofs_V, W.sub(0))]

    Q, _ = W.sub(1).collapse()
    boundary_dofs_Q = fem.locate_dofs_topological(
        (W.sub(1), Q), domain.topology.dim - 1, boundary_facets
    )
    p_bc = fem.Function(Q)
    p_bc.x.array[:] = 0.0
    bcs.append(fem.dirichletbc(p_bc, boundary_dofs_Q, W.sub(1)))

    nonlinear_problem_parameters = {
        "snes_type": "newtonls",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-8,
        "snes_atol": 1e-10,
        "snes_stol": 1e-12,
        "snes_max_it": 50,
        "snes_monitor": None,
        "snes_converged_reason": None,
        "snes_error_if_not_converged": None,

        "ksp_type": "preonly",
        "ksp_error_if_not_converged": None,
        "ksp_monitor": None,

        "pc_type": "fieldsplit",
        "pc_fieldsplit_detect_saddle_point": None,
        "pc_fieldsplit_type": "gkb",
        "pc_fieldsplit_gkb_nu": 1.0,
        "pc_fieldsplit_gkb_tol": 1e-8,
        "pc_fieldsplit_gkb_delay": 5,
        "pc_fieldsplit_gkb_maxit": 200,

        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",
        "fieldsplit_0_pc_factor_mat_solver_type": "mumps",
    }

    nonlinear_problem = petsc.NonlinearProblem(
        F,
        w,
        J=J,
        bcs=bcs,
        petsc_options_prefix="mixed_snes_",
        petsc_options=nonlinear_problem_parameters,
    )

    w = nonlinear_problem.solve()

    uh = w.sub(0).collapse()
    ph = w.sub(1).collapse()
    uh.x.scatter_forward()
    ph.x.scatter_forward()
    num_iterations = nonlinear_problem.solver.getLinearSolveIterations()
    del nonlinear_problem
    del w
    if return_iterations:
        return (
            uh,
            ph,
            num_iterations,
        )
    return uh, ph
