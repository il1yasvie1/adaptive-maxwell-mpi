"""Output paths, field writers, error norms, and CSV bookkeeping."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx.io import VTXWriter
import dolfinx.fem as fem
import pandas as pd


@dataclass(frozen=True)
class RunPaths:
    experiment: str
    example: str = ""
    work_root: Path = Path(__file__).resolve().parent / "work"
    data_root: Path = Path(__file__).resolve().parent / "data"

    @property
    def root(self):
        return self.work_root / self.experiment

    @property
    def csv(self):
        return self.root / "run-log.csv"

    @property
    def resolved_config(self):
        return self.root / "resolved-config.yaml"

    @property
    def gmsh_file(self):
        return self.root / "mesh" / "initial.msh"

    def checkpoint(self, run_id, level):
        return self.root / "checkpoints" / run_id / f"{level}.bp"

    def pressure_checkpoint(self, run_id, level):
        return self.root / "checkpoints" / run_id / f"{level}-p.bp"

    def checkpoint_metadata(self, run_id, level):
        return self.root / "checkpoints" / run_id / f"{level}.json"

    def view(self, run_id, level):
        return self.root / "views" / run_id / f"{level}.bp"

    def data_csv(self, run_id):
        return self.data_root / f"example{self.example}" / f"{run_id}.csv"

    def ensure(self, run_ids):
        self.gmsh_file.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.data_root / f"example{self.example}").mkdir(
            parents=True,
            exist_ok=True,
        )
        for run_id in run_ids:
            for path in (self.checkpoint(run_id, 0), self.view(run_id, 0)):
                path.parent.mkdir(parents=True, exist_ok=True)


def vtx_write(uh, filename, uex=None):
    """Write a vector field in CG1 form for lightweight visualization."""
    CG1 = fem.functionspace(uh.function_space.mesh, ("P", 1, (3,)))
    ucg = fem.Function(CG1)
    if uex is not None:
        error_expr = fem.Expression(uh - uex, CG1.element.interpolation_points)
        ucg.interpolate(error_expr)
    else:
        ucg.interpolate(uh)
    with VTXWriter(
        uh.function_space.mesh.comm, filename, [ucg], engine="BP4") as vtx:
        vtx.write(0.0)


def calc_hcurl_error(u, uh, is_ref=False, curl_u=None):
    """Compute the global L2-plus-curl error, optionally across two meshes."""
    if is_ref:
        domain_fine = u.function_space.mesh
        u.x.scatter_forward()
        uh.x.scatter_forward()
        cell_map = domain_fine.topology.index_map(domain_fine.topology.dim)
        cells = np.arange(cell_map.size_local + cell_map.num_ghosts, dtype=np.int32)
        interpolation_data = fem.create_interpolation_data(
            u.function_space,
            uh.function_space,
            cells,
        )
        uh_interp = fem.Function(u.function_space)
        uh_interp.interpolate_nonmatching(uh, cells, interpolation_data)
        uh_interp.x.scatter_forward()
        uh = uh_interp

    curl_error = (
        ufl.curl(u - uh)
        if curl_u is None
        else curl_u - ufl.curl(uh)
    )
    error_form = fem.form(
        (
            ufl.inner(curl_error, curl_error)
            + ufl.inner(u - uh, u - uh)
        )
        * ufl.dx
    )

    local_error = fem.assemble_scalar(error_form)
    global_error_sq = uh.function_space.mesh.comm.allreduce(local_error, op=MPI.SUM)
    return np.sqrt(global_error_sq)


def calc_hcurl_norm(u, curl_u=None):
    """Compute the global H(curl) norm of a field."""
    curl_value = ufl.curl(u) if curl_u is None else curl_u
    norm_form = fem.form(
        (ufl.inner(curl_value, curl_value) + ufl.inner(u, u)) * ufl.dx
    )
    local_norm_sq = fem.assemble_scalar(norm_form)
    global_norm_sq = u.function_space.mesh.comm.allreduce(local_norm_sq, op=MPI.SUM)
    return np.sqrt(global_norm_sq)


LOG_COLUMNS = [
    "name",
    "eg",
    "degree",
    "run_id",
    "mode",
    "formulation",
    "mesh_backend",
    "is_regularized",
    "initial_level",
    "current_level",
    "gamma",
    "th12",
    "th3",
    "ncell",
    "dof",
    "estim_error",
    "hcurl_error",
    "rel_hcurl_error",
]


def init_csv_log(csv_path):
    if Path(csv_path).exists():
        data = pd.read_csv(csv_path)
        changed = False
        defaults = {
            "run_id": data.get("mode", "unknown"),
            "formulation": np.where(
                data.get("is_regularized", False),
                "regularized",
                "mixed",
            ),
            "mesh_backend": "gmsh",
        }
        for column, default in defaults.items():
            if column not in data.columns:
                data[column] = default
                changed = True
        if changed:
            data = data.reindex(columns=LOG_COLUMNS)
            data.to_csv(csv_path, index=False)
        return

    df = pd.DataFrame(columns=LOG_COLUMNS)
    df.to_csv(csv_path, index=False)


def append_csv_log(csv_path, row):
    df = pd.DataFrame([row], columns=LOG_COLUMNS)
    df.to_csv(
        csv_path,
        mode="a",
        header=False,
        index=False,
    )


def update_csv_log(csv_path, key, values):
    data = pd.read_csv(csv_path)
    row_mask = pd.Series(True, index=data.index)
    for column, value in key.items():
        row_mask &= data[column] == value
    for column, value in values.items():
        data.loc[row_mask, column] = value
    data.to_csv(csv_path, index=False)


DATA_COLUMNS = [
    "Num of Cells",
    "DoFs",
    "Num of Iters",
    "Estim Errors",
    "Hcurl Errors",
    "rel Hcurl Errors",
]


def write_data_csv(csv_path, rows):
    pd.DataFrame(rows, columns=DATA_COLUMNS).to_csv(
        csv_path,
        index=False,
    )


def update_data_hcurl_error(csv_path, level, value):
    data = pd.read_csv(csv_path)
    if level >= len(data.index):
        raise IndexError(
            f"Level {level} is missing from {csv_path}."
        )
    data.loc[level, "Hcurl Errors"] = value
    data.to_csv(csv_path, index=False)


def update_data_relative_hcurl_error(csv_path, level, value):
    data = pd.read_csv(csv_path)
    if level >= len(data.index):
        raise IndexError(f"Level {level} is missing from {csv_path}.")
    if "rel Hcurl Errors" not in data.columns:
        data["rel Hcurl Errors"] = np.nan
    data.loc[level, "rel Hcurl Errors"] = value
    data.to_csv(csv_path, index=False)
