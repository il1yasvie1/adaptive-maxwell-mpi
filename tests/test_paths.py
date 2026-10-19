from pathlib import Path

from utils import RunPaths


def test_legacy_data_layout():
    root = Path(__file__).parents[1]
    paths = RunPaths(experiment="n1-example1", example="1")
    assert paths.data_csv("n1-adaptive") == (
        root / "data" / "example1" / "n1-adaptive.csv"
    )
