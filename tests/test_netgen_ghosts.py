import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("netgen")
pytest.importorskip("ngsPETSc")

from mpi4py import MPI
import ngsPETSc.utils.fenicsx as ngfx

from meshing import (
    _ensure_facet_ghosts,
    _netgen_geometry,
    _tetrahedral_nodes_and_order,
)


pytestmark = [pytest.mark.numerical, pytest.mark.mpi]


def test_tetrahedral_nodes_support_old_and_new_netgen_layouts():
    old = np.zeros(
        2,
        dtype=[("nodes", np.int32, (20,)), ("np", np.int8)],
    )
    old["nodes"][:, :4] = [[1, 2, 3, 4], [2, 3, 4, 5]]
    old["np"] = 4

    new = np.zeros(
        2,
        dtype=[("nodes", np.int32, (4,)), ("type", np.uint8)],
    )
    new["nodes"] = [[1, 2, 3, 4], [2, 3, 4, 5]]
    new["type"] = 20

    for raw in (old, new):
        nodes, order = _tetrahedral_nodes_and_order(raw)
        assert np.array_equal(nodes[order, :4], new["nodes"])


def test_tetrahedral_nodes_reject_non_tetrahedra():
    raw = np.zeros(
        1,
        dtype=[("nodes", np.int32, (5,)), ("type", np.uint8)],
    )
    raw["nodes"] = [[1, 2, 3, 4, 5]]

    with pytest.raises(RuntimeError, match=r"detected nodes per cell: \[5\]"):
        _tetrahedral_nodes_and_order(raw)


@pytest.mark.skipif(
    MPI.COMM_WORLD.size < 2,
    reason="Facet-ghost reconstruction requires at least two MPI ranks.",
)
def test_ensure_facet_ghosts_preserves_cells_and_provides_ghosts():
    comm = MPI.COMM_WORLD
    geo_model = ngfx.GeometricModel(_netgen_geometry(3), comm)
    domain, _, _ = geo_model.model_to_mesh(gdim=3, hmax=2.0)

    tdim = domain.topology.dim
    before_map = domain.topology.index_map(tdim)
    before_owned = comm.allreduce(before_map.size_local, op=MPI.SUM)
    before_ghosts = comm.allreduce(before_map.num_ghosts, op=MPI.SUM)

    rebuilt = _ensure_facet_ghosts(geo_model, domain)

    after_map = rebuilt.topology.index_map(tdim)
    after_owned = comm.allreduce(after_map.size_local, op=MPI.SUM)
    after_ghosts = comm.allreduce(after_map.num_ghosts, op=MPI.SUM)

    assert after_ghosts > 0
    assert after_owned == before_owned
    assert geo_model._mesh is rebuilt
    if before_ghosts == 0:
        assert rebuilt is not domain
    else:
        assert rebuilt is domain
