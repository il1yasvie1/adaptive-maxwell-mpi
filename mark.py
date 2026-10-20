"""MPI-safe Dörfler marking for distributed DG0 indicators."""

import numpy as np
import dolfinx.mesh as mesh
from mpi4py import MPI


def _distributed_mark(sq, theta, comm):
    """Select local cell indices carrying a global fraction of ``sq``.

    The pivot iterations avoid gathering all indicator values on rank 0.
    They return indices in the local owned-cell array, which is exactly the
    format expected by DOLFINx refinement.
    """
    sq = np.asarray(sq, dtype=np.float64)

    global_total = comm.allreduce(
        np.sum(sq),
        op=MPI.SUM,
    )

    if global_total <= 0 or theta <= 0:
        return np.empty(0, dtype=np.int32)

    target = theta**2 * global_total
    remaining = target

    active = np.ones(len(sq), dtype=bool)
    marked = np.zeros(len(sq), dtype=bool)

    while True:
        local_values = sq[active]

        if len(local_values):
            local_values = np.sort(local_values)
            local_pivot = local_values[len(local_values) // 2]
        else:
            local_pivot = np.nan

        pivots = comm.allgather(local_pivot)
        pivots = np.asarray(
            [p for p in pivots if not np.isnan(p)]
        )

        if len(pivots) == 0:
            break

        pivots.sort()
        pivot = pivots[len(pivots) // 2]

        greater = active & (sq > pivot)
        equal = active & (sq == pivot)

        mass_greater = comm.allreduce(
            np.sum(sq[greater]),
            op=MPI.SUM,
        )
        mass_equal = comm.allreduce(
            np.sum(sq[equal]),
            op=MPI.SUM,
        )

        if mass_greater >= remaining:
            active = greater

        elif mass_greater + mass_equal >= remaining:
            marked |= greater | equal
            break

        else:
            marked |= greater | equal
            remaining -= mass_greater + mass_equal
            active &= sq < pivot

    return np.flatnonzero(marked).astype(np.int32)


def _global_cell_keys(domain):
    """Return local global IDs and partition-independent cell coordinates."""
    index_map = domain.topology.index_map(domain.topology.dim)
    local_cells = np.arange(index_map.size_local, dtype=np.int32)
    local_ids = index_map.local_to_global(local_cells)
    midpoints = mesh.compute_midpoints(
        domain,
        domain.topology.dim,
        local_cells,
    )
    return local_ids, midpoints


def _global_sort_mark(sq, theta, domain, cell_keys):
    """Reproduce serial global sorting with partition-independent ties."""
    comm = domain.comm
    local_ids, midpoints = cell_keys
    gathered_sq = comm.gather(np.asarray(sq, dtype=np.float64), root=0)
    gathered_ids = comm.gather(local_ids, root=0)
    gathered_midpoints = comm.gather(midpoints, root=0)

    selected = None
    if comm.rank == 0:
        global_sq = np.concatenate(gathered_sq)
        global_ids = np.concatenate(gathered_ids)
        global_midpoints = np.concatenate(gathered_midpoints)
        total = np.sum(global_sq)
        if total <= 0 or theta <= 0:
            selected = np.empty(0, dtype=np.int64)
        else:
            order = np.lexsort((
                global_ids,
                global_midpoints[:, 2],
                global_midpoints[:, 1],
                global_midpoints[:, 0],
                -global_sq,
            ))
            cumulative = np.cumsum(global_sq[order])
            count = np.searchsorted(
                cumulative,
                theta**2 * total,
                side="left",
            ) + 1
            selected = global_ids[order[:count]]

    selected = comm.bcast(selected, root=0)
    return np.flatnonzero(np.isin(local_ids, selected)).astype(np.int32)


def mark(eta1, eta2, eta3, th12, th3, version="old"):
    """Mark the union of the eta1/eta2 and eta3 Dörfler sets."""
    domain = eta1.function_space.mesh
    comm = domain.comm
    tdim = domain.topology.dim

    num_owned_cells = (
        domain.topology.index_map(tdim).size_local
    )

    sq12 = (
        eta1.x.array[:num_owned_cells]**2
        + eta2.x.array[:num_owned_cells]**2
    )

    sq3 = eta3.x.array[:num_owned_cells]**2

    if version == "new":
        cell_keys = _global_cell_keys(domain)
        cells12 = _global_sort_mark(sq12, th12, domain, cell_keys)
        cells3 = _global_sort_mark(sq3, th3, domain, cell_keys)
    else:
        cells12 = _distributed_mark(sq12, th12, comm)
        cells3 = _distributed_mark(sq3, th3, comm)

    return np.ascontiguousarray(np.union1d(
        cells12,
        cells3,
    ), dtype=np.int32).copy()
