from pathlib import Path

import pytest
import pandas as pd

pytest.importorskip("ufl")

from utils import (
    DATA_COLUMNS,
    RunPaths,
    update_data_relative_hcurl_error,
)


def test_legacy_data_layout():
    root = Path(__file__).parents[1]
    paths = RunPaths(experiment="n1-example1", example="1")
    assert paths.data_csv("n1-adaptive") == (
        root / "data" / "example1" / "n1-adaptive.csv"
    )


def test_data_schema_contains_relative_hcurl_error():
    assert DATA_COLUMNS[-1] == "rel Hcurl Errors"


def test_relative_hcurl_error_updates_legacy_csv(tmp_path):
    csv_path = tmp_path / "example3.csv"
    pd.DataFrame(
        {
            "Num of Cells": [10],
            "DoFs": [20],
            "Num of Iters": [3],
            "Estim Errors": [1.5],
            "Hcurl Errors": [0.5],
        }
    ).to_csv(csv_path, index=False)

    update_data_relative_hcurl_error(csv_path, 0, 0.25)

    data = pd.read_csv(csv_path)
    assert "rel Hcurl Errors" in data.columns
    assert data.loc[0, "rel Hcurl Errors"] == pytest.approx(0.25)
