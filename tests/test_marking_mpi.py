import numpy as np
import pytest

pytest.importorskip("dolfinx")

from mpi4py import MPI

from mark import _distributed_mark


pytestmark = [pytest.mark.numerical, pytest.mark.mpi]


def test_distributed_mark_reaches_global_dorfler_target():
    comm = MPI.COMM_WORLD
    values = np.asarray(
        [2 * comm.rank + 1, 2 * comm.rank + 2],
        dtype=np.float64,
    )
    theta = 0.6

    cells = _distributed_mark(values, theta, comm)

    assert cells.dtype == np.int32
    assert np.all((0 <= cells) & (cells < len(values)))
    total = comm.allreduce(float(np.sum(values)), op=MPI.SUM)
    marked = comm.allreduce(float(np.sum(values[cells])), op=MPI.SUM)
    assert marked >= theta**2 * total


def test_distributed_mark_handles_empty_and_full_targets():
    comm = MPI.COMM_WORLD
    values = np.asarray([comm.rank + 1.0, comm.rank + 2.0])

    assert _distributed_mark(values, 0.0, comm).size == 0
    assert np.array_equal(
        _distributed_mark(values, 1.0, comm),
        np.arange(len(values), dtype=np.int32),
    )
