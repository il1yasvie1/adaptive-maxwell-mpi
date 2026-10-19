"""Initial mesh generation and MPI-compatible refinement backends."""

from dataclasses import dataclass
from pathlib import Path

import dolfinx
import dolfinx.mesh as mesh
try:
    from dolfinx.io import gmshio
except ImportError:
    from dolfinx.io import gmsh as gmshio
import numpy as np
from mpi4py import MPI


@dataclass
class MeshState:
    domain: object
    backend: str
    geo_model: object | None = None


def create_mesh(
    eg,
    backend="gmsh",
    comm=MPI.COMM_WORLD,
    gmsh_path=None,
    hmax=2.0,
):
    """Create the initial distributed mesh using Gmsh or Netgen."""
    if backend == "gmsh":
        if gmsh_path is None:
            raise ValueError("gmsh_path is required for the Gmsh backend.")
        path = Path(gmsh_path)
        if comm.rank == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            _write_gmsh_mesh(eg, path)
        comm.Barrier()
        domain = gmshio.read_from_msh(str(path), comm).mesh
        return MeshState(domain=domain, backend=backend)

    if backend == "netgen":
        import ngsPETSc.utils.fenicsx as ngfx

        geo_model = ngfx.GeometricModel(_netgen_geometry(eg), comm)
        domain, _, _ = geo_model.model_to_mesh(gdim=3, hmax=hmax)
        domain = _ensure_facet_ghosts(geo_model, domain)
        return MeshState(
            domain=domain,
            backend=backend,
            geo_model=geo_model,
        )

    raise ValueError(f"Unknown mesh backend: {backend}")


def refine_mesh(state, cells=None):
    """Refine ``cells``; with ``cells=None`` perform uniform refinement."""
    domain = state.domain

    if state.backend == "gmsh":
        domain.topology.create_entities(1)
        if cells is None:
            edges = None
        else:
            edges = mesh.compute_incident_entities(
                domain.topology,
                np.asarray(cells, dtype=np.int32),
                domain.topology.dim,
                1,
            )
        refined_domain, _, _ = mesh.refine(
            domain,
            edges=edges,
            partitioner=mesh.create_cell_partitioner(
                mesh.GhostMode.shared_facet,
                2,
            ),
            option=mesh.RefinementOption.none,
        )
        return MeshState(
            domain=refined_domain,
            backend=state.backend,
        )

    if state.backend == "netgen":
        if cells is None:
            num_owned = domain.topology.index_map(
                domain.topology.dim
            ).size_local
            cells = np.arange(num_owned, dtype=np.int32)
        else:
            cells = np.asarray(cells, dtype=np.int32)
            state.geo_model.ngmesh.Elements2D().NumPy()["refine"] = 0

        refined_domain, _ = state.geo_model.refineMarkedElements(
            domain.topology.dim,
            cells,
            netgen_flags={"refine_faces": True},
        )
        refined_domain = _ensure_facet_ghosts(
            state.geo_model,
            refined_domain,
        )
        return MeshState(
            domain=refined_domain,
            backend=state.backend,
            geo_model=state.geo_model,
        )

    raise ValueError(f"Unknown mesh backend: {state.backend}")


def _write_gmsh_mesh(eg, path):
    import gmsh

    if not gmsh.isInitialized():
        gmsh.initialize()
    gmsh.clear()
    gdim = 3

    if eg in (1, 3, 5):
        cube1 = gmsh.model.occ.addBox(-1, -1, 0, 1, 1, 1)
        cube2 = gmsh.model.occ.addBox(0, -1, 0, 1, 1, 1)
        cube3 = gmsh.model.occ.addBox(-1, 0, 0, 1, 1, 1)
        gmsh.model.occ.fuse(
            [(gdim, cube1)],
            [(gdim, cube2), (gdim, cube3)],
        )
    elif eg == 2:
        gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 2)
    else:
        gmsh.finalize()
        raise ValueError(f"Gmsh geometry is not defined for example {eg}.")

    gmsh.model.occ.synchronize()
    volumes = [
        tag for _, tag in gmsh.model.getEntities(dim=gdim)
    ]
    gmsh.model.addPhysicalGroup(gdim, volumes, 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2)
    gmsh.model.mesh.generate(gdim)
    gmsh.write(str(path))
    gmsh.finalize()


def _netgen_geometry(eg):
    import netgen.csg as csg

    if eg in (1, 3, 5):
        domain = (
            csg.OrthoBrick(csg.Pnt(-1, -1, 0), csg.Pnt(0, 0, 1))
            + csg.OrthoBrick(csg.Pnt(0, -1, 0), csg.Pnt(1, 0, 1))
            + csg.OrthoBrick(csg.Pnt(-1, 0, 0), csg.Pnt(0, 1, 1))
        )
    elif eg == 2:
        domain = csg.OrthoBrick(
            csg.Pnt(-1, -1, -1),
            csg.Pnt(1, 1, 1),
        )
    elif eg == 4:
        cube = csg.OrthoBrick(
            csg.Pnt(-1, -1, -1),
            csg.Pnt(1, 1, 1),
        )
        cylinder = csg.Cylinder(
            csg.Pnt(0, 0, 0),
            csg.Pnt(0, 0, 1),
            1,
        )
        left_half = csg.Plane(
            csg.Pnt(0, 0, 0),
            csg.Vec(-1, 0, 0),
        )
        right_half = csg.Plane(
            csg.Pnt(0, 0, 0),
            csg.Vec(1, 0, 0),
        )
        lower_half = csg.Plane(
            csg.Pnt(0, 0, 0),
            csg.Vec(0, -1, 0),
        )
        domain = (
            cube * cylinder * left_half
            + cube * cylinder * right_half * lower_half
        )
    else:
        raise ValueError(f"Netgen geometry is not defined for example {eg}.")

    geometry = csg.CSGeometry()
    geometry.Add(domain)
    return geometry


def _ensure_facet_ghosts(geo_model, domain):
    comm = domain.comm
    tdim = domain.topology.dim
    if comm.size == 1 or domain.topology.index_map(tdim).num_ghosts:
        return domain

    raw = geo_model.ngmesh.Elements3D().NumPy()
    if not np.all(raw["np"] == 4):
        raise RuntimeError(
            "MPI ghost reconstruction for Netgen currently requires tetrahedra."
        )

    order = np.argsort(raw["np"])
    geo_model._sorted_mapping = order
    if comm.rank == geo_model.comm_rank:
        cells = np.asarray(raw["nodes"][order, :4], dtype=np.int64) - 1
        permutation = dolfinx.cpp.io.perm_vtk(
            mesh.CellType.tetrahedron,
            4,
        )
        cells = np.ascontiguousarray(cells[:, permutation])
        coordinates = np.ascontiguousarray(
            geo_model.ngmesh.Coordinates()[:, :3]
        )
    else:
        cells = np.zeros((0, 4), dtype=np.int64)
        coordinates = np.zeros((0, 3), dtype=np.float64)

    domain = mesh.create_mesh(
        comm,
        cells,
        gmshio.ufl_mesh(4, 3, dolfinx.default_real_type),
        coordinates,
        partitioner=mesh.create_cell_partitioner(
            mesh.GhostMode.shared_facet,
            2,
        ),
    )
    geo_model._mesh = domain
    return domain
