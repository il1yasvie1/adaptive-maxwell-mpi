from pathlib import Path

import pytest

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
    assert config.version == "new"
    assert config.example == "3"
    assert {run.id for run in config.runs} == {
        "uniform-regularized",
        "adaptive-regularized",
    }


def test_reference_configurations_match_legacy_parameters():
    root = Path(__file__).parents[1]
    expected = {
        "n2-example1.yaml": ("1", 2, 1024.0, 192, 1 / 3, 5, 15),
        "n1-example2a.yaml": ("2a", 1, 1.0407, 384, 0.3604, 2, 15),
        "n2-example2a.yaml": ("2a", 2, 1.2349, 384, 0.3939, 2, 8),
        "n1-example2b.yaml": ("2b", 1, 1.1680, 384, 0.4697, 2, 15),
        "n2-example2b.yaml": ("2b", 2, 2.0238, 384, 0.5191, 2, 8),
    }
    new_version = {"n2-example1.yaml"}
    for filename, values in expected.items():
        config = load_config(root / "configs" / filename)
        assert config.version == ("new" if filename in new_version else "old")
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
    config = load_config(root / "configs" / "n1-example3.yaml")
    assert config.example == "3"
    assert config.degree == 1
    assert config.gamma0 == 256
    assert config.n0 == 1152
    assert config.gamma_exponent == 1 / 3
    assert config.run("n1-uniform-self").levels == 5
    adaptive = config.run("n1-adaptive-self")
    assert adaptive.levels == 15
    assert adaptive.theta12 == 0.3
    assert adaptive.theta3 == 0.3


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("strategy: invalid", "strategy must be uniform or adaptive"),
        ("levels: -1", "levels must be non-negative"),
        ("theta12: 0", "theta12 and theta3 must lie in"),
    ],
)
def test_invalid_run_configuration_is_rejected(tmp_path, replacement, message):
    source = """
name: invalid-run
basic:
  eg: 1
  degree: 1
runs:
  - id: test-run
    strategy: uniform
    levels: 0
    formulation: regularized
    marking:
      theta12: 0.5
      theta3: 0.5
"""
    if replacement.startswith("theta12"):
        source = source.replace("theta12: 0.5", replacement)
    elif replacement.startswith("strategy"):
        source = source.replace("strategy: uniform", replacement)
    else:
        source = source.replace("levels: 0", replacement)

    path = tmp_path / "invalid.yaml"
    path.write_text(source)
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_gamma_logspace_is_expanded(tmp_path):
    path = tmp_path / "gamma.yaml"
    path.write_text(
        """
name: gamma-test
mode: gamma
basic:
  eg: 1
  degree: 1
runs:
  - id: regularized
    strategy: uniform
    levels: 0
    formulation: regularized
gamma_tests:
  - id: scan
    run: regularized
    level: 0
    values:
      logspace: [-1, 1, 3]
"""
    )

    config = load_config(path)
    assert config.gamma_tests[0].values == pytest.approx((0.1, 1.0, 10.0))
