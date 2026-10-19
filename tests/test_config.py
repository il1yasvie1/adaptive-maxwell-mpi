from pathlib import Path

from config import load_config


def test_all_configurations_load():
    root = Path(__file__).parents[1]
    configs = sorted((root / "configs").glob("*.yaml"))
    assert configs
    loaded = [load_config(path) for path in configs]
    assert all(item.name for item in loaded)


def test_smoke_configuration_has_expected_runs():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "smoke-test.yaml")
    assert config.example == "3"
    assert {run.id for run in config.runs} == {
        "uniform-regularized",
        "adaptive-regularized",
        "uniform-mixed",
        "adaptive-mixed",
    }


def test_reference_configurations_match_legacy_parameters():
    root = Path(__file__).parents[1]
    expected = {
        "n1-example1.yaml": ("1", 1, 1024.0, 144, 1 / 3, 5, 13),
        "n2-example1.yaml": ("1", 2, 1024.0, 144, 1 / 3, 4, 13),
        "n1-example2a.yaml": ("2a", 1, 1.0407, 384, 0.3604, 2, 15),
        "n2-example2a.yaml": ("2a", 2, 1.2349, 384, 0.3939, 2, 8),
        "n1-example2b.yaml": ("2b", 1, 1.1680, 384, 0.4697, 2, 15),
        "n2-example2b.yaml": ("2b", 2, 2.0238, 384, 0.5191, 2, 8),
    }
    for filename, values in expected.items():
        config = load_config(root / "configs" / filename)
        uniform = config.run(f"n{config.degree}-uniform")
        adaptive = config.run(f"n{config.degree}-adaptive")
        actual = (
            config.example,
            config.degree,
            config.gamma0,
            config.n0,
            config.gamma_exponent,
            uniform.levels,
            adaptive.levels,
        )
        assert actual == values


def test_example3_reference_configuration():
    root = Path(__file__).parents[1]
    config = load_config(root / "configs" / "n1-example3-reference.yaml")
    assert config.example == "3"
    assert config.degree == 1
    assert config.gamma0 == 512
    assert config.n0 == 144
    assert config.gamma_exponent == 1 / 3
    assert config.run("n1-reference").levels == 4
    assert config.run("n1-uniform").levels == 3
    adaptive = config.run("n1-adaptive")
    assert adaptive.levels == 15
    assert adaptive.theta12 == 0.3
    assert adaptive.theta3 == 0.3
