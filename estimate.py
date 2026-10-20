"""Residual indicators used by the adaptive loop.

All returned indicators are DG0 functions: one value per cell.  Their
owned-cell values are used for MPI marking; ghost values are only needed when
assembling facet contributions.
"""

from petsc4py import PETSc
import numpy as np
import ufl
from mpi4py import MPI

import dolfinx.fem as fem
import dolfinx.fem.petsc as petsc


def assemble_indicator(DG0, form):
    """Assemble a non-negative cellwise squared indicator and take its root."""
    eta_sq = fem.Function(DG0)
    vec = eta_sq.x.petsc_vec

    vec.set(0.0)
    petsc.assemble_vector(vec, fem.form(form))

    vec.ghostUpdate(
        addv=PETSc.InsertMode.ADD_VALUES,
        mode=PETSc.ScatterMode.REVERSE,
    )
    eta_sq.x.scatter_forward()
    eta = fem.Function(DG0)
    eta.x.array[:] = np.sqrt(eta_sq.x.array)

    return eta


def estimate(domain, sol, dat, gam, version="old"):
    """Return (eta1, eta2, eta3, global_estimator).

    ``eta1`` measures the scalar auxiliary problem, ``eta2`` the divergence
    constraint, and ``eta3`` the Maxwell equation.  The global value is
    reduced over MPI, while the three DG0 functions remain distributed.
    """
    alpha, f, g = dat
    uh, ph = sol

    uh.x.scatter_forward()
    ph.x.scatter_forward()

    DG0 = fem.functionspace(domain, ("DP", 0))
    w = ufl.TestFunction(DG0)

    n = ufl.FacetNormal(domain)
    if version == "new":
        h = ufl.CellVolume(domain) ** (1.0 / 3.0)
        facet_weight = h("+") * w("+") + h("-") * w("-")
    else:
        h = ufl.CellDiameter(domain)
        h_avg = (h("+") + h("-")) / 2
        facet_weight = h_avg * w("+")

    r1 = ufl.div(ufl.grad(ph)) - g
    j1 = ufl.jump(ufl.grad(ph), n)
    G1 = (
        h**2 * r1**2 * w * ufl.dx
        + j1**2 * facet_weight * ufl.dS
    )

    r2 = ufl.div(ufl.grad(ph) - uh)
    j2 = ufl.jump(ufl.grad(ph) - uh, n)
    G2 = (
        h**2 * r2**2 * w * ufl.dx
        + j2**2 * facet_weight * ufl.dS
    )

    r3 = (
        f
        - ufl.curl(
            alpha(uh) * ufl.curl(uh)
        )
        + gam * (ufl.grad(ph) - uh)
    )

    flux = ufl.cross(
        alpha(uh) * ufl.curl(uh),
        n,
    )

    j3 = flux("+") + flux("-")

    G3 = (
        h**2 * ufl.inner(r3, r3) * w * ufl.dx
        + ufl.inner(j3, j3) * facet_weight * ufl.dS
    )

    eta1 = assemble_indicator(DG0, G1)
    eta2 = assemble_indicator(DG0, G2)
    eta3 = assemble_indicator(DG0, G3)

    num_owned = (
        DG0.dofmap.index_map.size_local
        * DG0.dofmap.index_map_bs
    )

    local_estimator_sq = (
        np.sum(eta1.x.array[:num_owned]**2)
        + np.sum(eta2.x.array[:num_owned]**2)
        + np.sum(eta3.x.array[:num_owned]**2)
    )

    global_estimator_sq = domain.comm.allreduce(
        local_estimator_sq,
        op=MPI.SUM,
    )

    estim_error = np.sqrt(global_estimator_sq)

    return eta1, eta2, eta3, estim_error
