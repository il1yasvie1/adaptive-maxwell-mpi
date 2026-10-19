"""Problem data for the manufactured and singular Maxwell examples."""

import ufl
import dolfinx.fem as fem
from petsc4py import PETSc


def _example5_fields(domain):
    """Nonlinear manufactured field with a reentrant-edge singularity.

    The scalar potential contains the classical r^(2/3) L-shaped-domain
    singularity.  Its gradient is curl-free and singular, while the smooth
    ``rotational`` field makes curl(u) nonzero so that alpha(u) is genuinely
    nonlinear.  Both components satisfy the homogeneous tangential boundary
    condition on the L-shaped prism.
    """
    x, y, z = ufl.SpatialCoordinate(domain)
    sqrt_two = ufl.sqrt(2.0)

    rotated_x = -(x + y) / sqrt_two
    rotated_y = (x - y) / sqrt_two
    theta = ufl.atan2(rotated_y, rotated_x) + 3 * ufl.pi / 4

    radius = ufl.sqrt(x**2 + y**2)
    exponent = 2.0 / 3.0
    angle = exponent * theta
    singular = radius**exponent * ufl.sin(angle)
    singular_x = exponent * radius ** (exponent - 2) * (
        x * ufl.sin(angle) - y * ufl.cos(angle)
    )
    singular_y = exponent * radius ** (exponent - 2) * (
        y * ufl.sin(angle) + x * ufl.cos(angle)
    )

    boundary_cutoff = (1 - x**2) * (1 - y**2)
    cutoff_x = -2 * x * (1 - y**2)
    cutoff_y = -2 * y * (1 - x**2)
    cutoff_laplacian = -4 + 2 * (x**2 + y**2)
    z_cutoff = z * (1 - z)
    z_derivative = 1 - 2 * z

    singular_gradient = ufl.as_vector(
        (
            z_cutoff * (cutoff_x * singular + boundary_cutoff * singular_x),
            z_cutoff * (cutoff_y * singular + boundary_cutoff * singular_y),
            z_derivative * boundary_cutoff * singular,
        )
    )
    smooth_rotational = ufl.as_vector((0, 0, boundary_cutoff))
    exact = singular_gradient + smooth_rotational

    exact_curl = ufl.as_vector((cutoff_y, -cutoff_x, 0))
    exact_divergence = (
        z_cutoff
        * (
            cutoff_laplacian * singular
            + 2 * (cutoff_x * singular_x + cutoff_y * singular_y)
        )
        - 2 * boundary_cutoff * singular
    )
    return exact, exact_curl, exact_divergence


def set_problem_data(domain, eg, **kwargs):
    """Return the nonlinear coefficient, Maxwell load, and divergence data."""
    x, y, z = ufl.SpatialCoordinate(domain)

    if eg == 1:
        alpha = lambda u: 1-.5/(1 + ufl.dot(ufl.curl(u), ufl.curl(u)))
        f = fem.Constant(domain, PETSc.ScalarType((0, 0, 0)))
        g = -3*ufl.pi**2*ufl.sin(ufl.pi*x)*ufl.sin(ufl.pi*y)*ufl.sin(ufl.pi*z)

    elif eg == 2:
        def alpha(u):
            a0 = 1.0
            a1 = kwargs["a1"]
            a2 = 1.0
            return a0 - a1*ufl.exp(-a2*ufl.dot(ufl.curl(u), ufl.curl(u)))
        import numpy as np

        def uexpr(points):
            values = np.zeros((3, points.shape[1]))
            values[2] = (
                np.cos(0.5 * np.pi * points[0])
                * np.cos(0.5 * np.pi * points[1])
                * np.cos(0.5 * np.pi * points[2])
            )
            return values

        degree = int(kwargs["degree"])
        Vex = fem.functionspace(domain, (f"N{degree}E", 3))
        u_interp = fem.Function(Vex)
        u_interp.interpolate(uexpr)
        f = ufl.curl(alpha(u_interp) * ufl.curl(u_interp))
        g = -.5*ufl.pi*ufl.cos(.5*ufl.pi*x)*ufl.cos(.5*ufl.pi*y)*ufl.sin(.5*ufl.pi*z)

    elif eg == 3:
        def alpha(u):
            ncurlu = ufl.dot(ufl.curl(u), ufl.curl(u))
            return 1 - ufl.conditional(ufl.gt(x, 0), 1, 0) / (4*(1+ncurlu)) - ufl.conditional(ufl.gt(y, 0), 1, 0) / (4*(1+ncurlu))
        r = 1/8
        f = ufl.as_vector([0, 0, 100*ufl.conditional(ufl.lt(abs(x), r),
                                                     ufl.conditional(ufl.lt(abs(y), r), 1, 0),
                                                     0)])
        g = fem.Constant(domain, 0.0)

    elif eg == 4:
        def alpha(u):
            ncurlu = ufl.dot(ufl.curl(u), ufl.curl(u))
            return 1 - ufl.conditional(ufl.gt(x, 0), 1, 0) / (4*(1+ncurlu)) - ufl.conditional(ufl.gt(y, 0), 1, 0) / (4*(1+ncurlu))
        r = 1/8
        one = fem.Constant(domain, PETSc.ScalarType(1))
        f = ufl.as_vector([one, one, one])
        g = fem.Constant(domain, 0.0)

    elif eg == 5:
        def alpha(u):
            curl_norm_sq = ufl.dot(ufl.curl(u), ufl.curl(u))
            return 1 - 0.5 / (1 + curl_norm_sq)

        _, curl_exact, g = _example5_fields(domain)
        alpha_exact = 1 - 0.5 / (
            1 + ufl.dot(curl_exact, curl_exact)
        )
        f = ufl.curl(alpha_exact * curl_exact)

    return (alpha, f, g)


def exact_solution(domain, eg):
    """Return a symbolic exact solution when the selected example has one."""
    x, y, z = ufl.SpatialCoordinate(domain)
    zero = fem.Constant(domain, 0.0)
    uex = None
    if eg == 1:
        uex = ufl.as_vector([ufl.pi*ufl.cos(ufl.pi*x)*ufl.sin(ufl.pi*y)*ufl.sin(ufl.pi*z),
                          ufl.pi*ufl.cos(ufl.pi*y)*ufl.sin(ufl.pi*z)*ufl.sin(ufl.pi*x),
                          ufl.pi*ufl.cos(ufl.pi*z)*ufl.sin(ufl.pi*y)*ufl.sin(ufl.pi*x)])
    elif eg == 2:
        uex = ufl.as_vector([zero, zero, 
                             ufl.cos(.5*ufl.pi*x)*ufl.cos(.5*ufl.pi*y)*ufl.cos(.5*ufl.pi*z)])
    elif eg == 5:
        uex, _, _ = _example5_fields(domain)

    return uex


def exact_curl(domain, eg):
    """Return a pre-simplified exact curl when one is available."""
    if eg == 5:
        _, curl_value, _ = _example5_fields(domain)
        return curl_value
    return None
