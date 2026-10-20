from pathlib import Path
import sys


# The ``pytest`` console script does not necessarily add the current working
# directory to sys.path.  Keep repository modules importable for both
# ``pytest tests`` and ``python -m pytest tests``.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numerical: requires the DOLFINx/PETSc runtime",
    )
    config.addinivalue_line(
        "markers",
        "mpi: safe to run under mpiexec with pytest on every rank",
    )
