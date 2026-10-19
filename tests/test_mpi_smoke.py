from mpi4py import MPI


def test_mpi_communicator():
    comm = MPI.COMM_WORLD
    total = comm.allreduce(1, op=MPI.SUM)
    assert total == comm.size


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    print(f"rank={comm.rank}/{comm.size}", flush=True)
