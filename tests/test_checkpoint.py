import json

import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("adios4dolfinx")

from config import load_config
from main import _checkpoint_metadata, _checkpoint_ready
from utils import RunPaths


def test_checkpoint_requires_both_fields_and_matching_metadata(tmp_path):
    root = tmp_path / "repository"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "checkpoint.yaml"
    config_path.write_text(
        """
name: checkpoint-test
version: new
basic:
  eg: 1
  degree: 1
runs:
  - id: adaptive
    strategy: adaptive
    levels: 1
    formulation: regularized
"""
    )
    experiment = load_config(config_path)
    run = experiment.run("adaptive")
    paths = RunPaths(
        experiment=experiment.name,
        example=experiment.example,
        work_root=tmp_path / "work",
        data_root=tmp_path / "data",
    )
    paths.ensure([run.id])

    num_cells = 24
    gamma = experiment.gamma(num_cells, run.formulation)
    metadata = _checkpoint_metadata(
        experiment,
        run,
        level=0,
        num_cells=num_cells,
        gamma=gamma,
    )
    paths.checkpoint(run.id, 0).touch()
    paths.checkpoint_metadata(run.id, 0).write_text(json.dumps(metadata))

    assert not _checkpoint_ready(paths, experiment, run, 0)

    paths.pressure_checkpoint(run.id, 0).touch()
    assert _checkpoint_ready(paths, experiment, run, 0)

    metadata["theta12"] = 0.25
    paths.checkpoint_metadata(run.id, 0).write_text(json.dumps(metadata))
    assert not _checkpoint_ready(paths, experiment, run, 0)
