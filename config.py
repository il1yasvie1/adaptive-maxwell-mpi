"""Typed YAML configuration and defaults for reproducible experiments."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml


@dataclass(frozen=True)
class RunConfig:
    id: str
    strategy: str
    levels: int
    formulation: str
    theta12: float
    theta3: float

    @property
    def is_regularized(self):
        return self.formulation == "regularized"


@dataclass(frozen=True)
class ErrorTarget:
    run: str
    levels: str | tuple[int, ...]


@dataclass(frozen=True)
class GammaTest:
    id: str
    run: str
    level: int
    values: tuple[float, ...]
    method: str = "scan"
    levels: tuple[int, ...] = ()
    tolerance: float = 1e-3


@dataclass(frozen=True)
class HcurlErrorConfig:
    reference_run: str
    reference_level: str | int
    targets: tuple[ErrorTarget, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    version: str
    mode: str
    example: str
    degree: int
    mesh_backend: str
    initial_refinements: int
    gamma0: float
    n0: int
    gamma_exponent: float
    runs: tuple[RunConfig, ...]
    hcurl_errors: tuple[HcurlErrorConfig, ...]
    gamma_tests: tuple[GammaTest, ...]

    @property
    def geometry_example(self):
        return int(self.example[0])

    @property
    def a1(self):
        if self.example == "2a":
            return 0.5
        if self.example == "2b":
            return 0.999
        return None

    def gamma(self, num_cells, formulation):
        if formulation == "mixed":
            return 0.0
        return self.gamma0 * (
            num_cells / self.n0
        ) ** (-self.gamma_exponent)

    def run(self, run_id):
        for run in self.runs:
            if run.id == run_id:
                return run
        raise KeyError(f"Unknown run id: {run_id}")

    def resolved_dict(self):
        payload = {
            "name": self.name,
            "version": self.version,
            "mode": self.mode,
            "basic": {
                "eg": self.example,
                "degree": self.degree,
                "mesh_backend": self.mesh_backend,
            },
            "initialization": {
                "uniform_refinements": self.initial_refinements,
            },
            "regularization": {
                "gamma0": self.gamma0,
                "n0": self.n0,
                "mu": self.gamma_exponent,
            },
            "runs": [],
            "hcurl_errors": [],
            "gamma_tests": [],
        }
        for run in self.runs:
            item = {
                "id": run.id,
                "strategy": run.strategy,
                "levels": run.levels,
                "formulation": run.formulation,
            }
            if run.strategy == "adaptive":
                item["marking"] = {
                    "theta12": run.theta12,
                    "theta3": run.theta3,
                }
            payload["runs"].append(item)

        for task in self.hcurl_errors:
            payload["hcurl_errors"].append(
                {
                    "reference": {
                        "run": task.reference_run,
                        "level": task.reference_level,
                    },
                    "targets": [
                        {
                            "run": target.run,
                            "levels": (
                                target.levels
                                if target.levels == "all"
                                else list(target.levels)
                            ),
                        }
                        for target in task.targets
                    ],
                }
            )
        for test in self.gamma_tests:
            payload["gamma_tests"].append(
                {
                    "id": test.id,
                    "run": test.run,
                    "level": test.level,
                    "values": list(test.values),
                }
            )
        return payload


def _standard_defaults(example, degree):
    if example == "1":
        return {
            "initial_refinements": 1,
            "gamma0": 1024.0,
            "n0": 144,
            "mu": 1 / 3,
            "theta": 0.5,
            "uniform_levels": 5 if degree == 1 else 4,
            "adaptive_levels": 13,
        }
    if example in ("2a", "2b"):
        parameters = {
            ("2a", 1): (1.0407, 0.3604),
            ("2a", 2): (1.2349, 0.3939),
            ("2b", 1): (1.1680, 0.4697),
            ("2b", 2): (2.0238, 0.5191),
        }
        gamma0, mu = parameters[(example, degree)]
        return {
            "initial_refinements": 2,
            "gamma0": gamma0,
            "n0": 384,
            "mu": mu,
            "theta": 0.3,
            "uniform_levels": 2,
            "adaptive_levels": 15 if degree == 1 else 8,
        }
    if example in ("3", "4", "5"):
        return {
            "initial_refinements": 1,
            "gamma0": 512.0,
            "n0": 144,
            "mu": 1 / 3,
            "theta": 0.3,
            "uniform_levels": 4,
            "adaptive_levels": 15,
        }
    raise ValueError(f"Unsupported example: {example}")


def _as_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _parse_runs(raw_runs, defaults):
    if raw_runs is None:
        raw_runs = [
            {
                "id": "uniform-regularized",
                "strategy": "uniform",
                "levels": defaults["uniform_levels"],
                "formulation": "regularized",
            },
            {
                "id": "adaptive-regularized",
                "strategy": "adaptive",
                "levels": defaults["adaptive_levels"],
                "formulation": "regularized",
            },
        ]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("runs must be a non-empty YAML list.")

    runs = []
    seen = set()
    for index, raw in enumerate(raw_runs):
        raw = _as_mapping(raw, f"runs[{index}]")
        run_id = str(raw.get("id", "")).strip()
        if not run_id or "/" in run_id or "\\" in run_id:
            raise ValueError(f"runs[{index}].id is invalid.")
        if run_id in seen:
            raise ValueError(f"Duplicate run id: {run_id}")
        seen.add(run_id)

        strategy = str(raw.get("strategy", "")).lower()
        if strategy not in ("uniform", "adaptive"):
            raise ValueError(
                f"Run {run_id}: strategy must be uniform or adaptive."
            )
        levels_value = raw.get(
            "levels",
            raw.get("uniform_levels", raw.get("level")),
        )
        if levels_value is None:
            levels_value = defaults[f"{strategy}_levels"]
        levels = int(levels_value)
        if levels < 0:
            raise ValueError(f"Run {run_id}: levels must be non-negative.")

        if "formulation" in raw:
            formulation = str(raw["formulation"]).lower()
        elif "is_regularized" in raw:
            formulation = (
                "regularized" if bool(raw["is_regularized"]) else "mixed"
            )
        else:
            formulation = "regularized"
        if formulation not in ("mixed", "regularized"):
            raise ValueError(
                f"Run {run_id}: formulation must be mixed or regularized."
            )

        marking = raw.get("marking", {})
        marking = _as_mapping(marking, f"Run {run_id}.marking")
        theta12 = float(marking.get("theta12", defaults["theta"]))
        theta3 = float(marking.get("theta3", defaults["theta"]))
        if not 0 < theta12 <= 1 or not 0 < theta3 <= 1:
            raise ValueError(
                f"Run {run_id}: theta12 and theta3 must lie in (0, 1]."
            )
        runs.append(
            RunConfig(
                id=run_id,
                strategy=strategy,
                levels=levels,
                formulation=formulation,
                theta12=theta12,
                theta3=theta3,
            )
        )
    return tuple(runs)


def _parse_error_tasks(raw_tasks, runs, example):
    run_map = {run.id: run for run in runs}
    if raw_tasks is None:
        if example in ("3", "4") and "uniform-regularized" in run_map:
            raw_tasks = [
                {
                    "reference": {
                        "run": "uniform-regularized",
                        "level": "final",
                    },
                    "targets": [
                        {"run": run.id, "levels": "all"}
                        for run in runs
                    ],
                }
            ]
        else:
            return ()
    if not isinstance(raw_tasks, list):
        raise ValueError("hcurl_errors must be a YAML list.")

    tasks = []
    for index, raw in enumerate(raw_tasks):
        raw = _as_mapping(raw, f"hcurl_errors[{index}]")
        reference = _as_mapping(
            raw.get("reference"),
            f"hcurl_errors[{index}].reference",
        )
        reference_run = str(reference.get("run", ""))
        if reference_run not in run_map:
            raise ValueError(
                f"Unknown reference run: {reference_run}"
            )
        reference_level = reference.get("level", "final")
        if reference_level != "final":
            reference_level = int(reference_level)
            if not 0 <= reference_level <= run_map[reference_run].levels:
                raise ValueError(
                    f"Reference level is outside run {reference_run}."
                )

        raw_targets = raw.get("targets", [])
        if not isinstance(raw_targets, list) or not raw_targets:
            raise ValueError(
                f"hcurl_errors[{index}].targets must be a non-empty list."
            )
        targets = []
        for target_raw in raw_targets:
            target_raw = _as_mapping(target_raw, "error target")
            target_run = str(target_raw.get("run", ""))
            if target_run not in run_map:
                raise ValueError(f"Unknown target run: {target_run}")
            levels = target_raw.get("levels", "all")
            if levels != "all":
                if not isinstance(levels, list):
                    raise ValueError(
                        f"Target {target_run}: levels must be all or a list."
                    )
                levels = tuple(int(level) for level in levels)
                if any(
                    level < 0 or level > run_map[target_run].levels
                    for level in levels
                ):
                    raise ValueError(
                        f"Target levels are outside run {target_run}."
                    )
            targets.append(ErrorTarget(target_run, levels))
        tasks.append(
            HcurlErrorConfig(
                reference_run=reference_run,
                reference_level=reference_level,
                targets=tuple(targets),
            )
        )
    return tuple(tasks)


def _parse_gamma_tests(raw_tests, runs, mode):
    """Parse fixed-mesh gamma scans configured in YAML."""
    if raw_tests is None:
        if mode == "gamma":
            raise ValueError("mode=gamma requires a non-empty gamma_tests list.")
        return ()
    if not isinstance(raw_tests, list) or not raw_tests:
        raise ValueError("gamma_tests must be a non-empty YAML list.")

    run_map = {run.id: run for run in runs}
    tests = []
    seen = set()
    for index, raw in enumerate(raw_tests):
        raw = _as_mapping(raw, f"gamma_tests[{index}]")
        test_id = str(raw.get("id", f"gamma-{index}")).strip()
        if not test_id or test_id in seen or "/" in test_id or "\\" in test_id:
            raise ValueError(f"gamma_tests[{index}].id is invalid.")
        seen.add(test_id)
        run_id = str(raw.get("run", ""))
        if run_id not in run_map:
            raise ValueError(f"Unknown gamma test run: {run_id}")
        if run_map[run_id].formulation != "regularized":
            raise ValueError("Gamma tests require a regularized run.")
        level = int(raw.get("level", 0))
        if level < 0 or level > run_map[run_id].levels:
            raise ValueError(
                f"Gamma test level is outside run {run_id}."
            )
        raw_values = raw.get("values", [])
        if isinstance(raw_values, dict) and "logspace" in raw_values:
            logspace = raw_values["logspace"]
            if not isinstance(logspace, list) or len(logspace) not in (2, 3):
                raise ValueError(
                    "values.logspace must be [start, stop] or "
                    "[start, stop, num]."
                )
            start, stop = float(logspace[0]), float(logspace[1])
            num = int(logspace[2]) if len(logspace) == 3 else 50
            raw_values = np.logspace(start, stop, num).tolist()
        values = tuple(float(value) for value in raw_values)
        if not values or any(value <= 0 for value in values):
            raise ValueError("Gamma test values must be positive and non-empty.")
        method = str(raw.get("method", "scan")).lower()
        if method not in ("scan", "adjoint"):
            raise ValueError("gamma test method must be scan or adjoint.")
        levels = raw.get("levels", [level])
        levels = tuple(int(item) for item in levels)
        if any(item < 0 or item > run_map[run_id].levels for item in levels):
            raise ValueError("Gamma test levels are outside the selected run.")
        tolerance = float(raw.get("tolerance", 1e-3))
        if tolerance <= 0:
            raise ValueError("Gamma test tolerance must be positive.")
        tests.append(GammaTest(test_id, run_id, level, values, method, levels, tolerance))
    return tuple(tests)


def load_config(path):
    path = Path(path)
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    raw = _as_mapping(raw, "configuration")

    name = str(raw.get("name", path.stem)).strip()
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError("name must be a safe directory name.")

    version = str(raw.get("version", "old")).lower()
    if version not in ("old", "new"):
        raise ValueError("version must be old or new.")

    mode = str(raw.get("mode", "solve")).lower()
    if mode not in ("solve", "gamma"):
        raise ValueError("mode must be solve or gamma.")

    basic = _as_mapping(raw.get("basic"), "basic")
    example = str(basic.get("eg", "")).lower()
    if example not in ("1", "2a", "2b", "3", "4", "5"):
        raise ValueError("basic.eg must be one of 1, 2a, 2b, 3, 4, 5.")
    degree = int(basic.get("degree", 0))
    if degree not in (1, 2):
        raise ValueError("basic.degree must be 1 or 2.")
    backend = str(basic.get("mesh_backend", "netgen")).lower()
    if backend not in ("gmsh", "netgen"):
        raise ValueError("basic.mesh_backend must be gmsh or netgen.")
    if example == "4" and backend == "gmsh":
        raise ValueError("Example 4 currently requires the Netgen backend.")

    defaults = _standard_defaults(example, degree)
    initialization = _as_mapping(
        raw.get("initialization", {}),
        "initialization",
    )
    initial_refinements = int(
        initialization.get(
            "uniform_refinements",
            defaults["initial_refinements"],
        )
    )
    if initial_refinements < 0:
        raise ValueError(
            "initialization.uniform_refinements must be non-negative."
        )

    regularization = _as_mapping(
        raw.get("regularization", {}),
        "regularization",
    )
    gamma0 = float(regularization.get("gamma0", defaults["gamma0"]))
    n0 = int(regularization.get("n0", defaults["n0"]))
    mu = float(regularization.get("mu", defaults["mu"]))
    if gamma0 <= 0 or n0 <= 0 or mu <= 0:
        raise ValueError("gamma0, n0 and mu must be positive.")

    runs = _parse_runs(raw.get("runs"), defaults)
    hcurl_errors = _parse_error_tasks(
        raw.get("hcurl_errors"),
        runs,
        example,
    )
    gamma_tests = _parse_gamma_tests(
        raw.get("gamma_tests"),
        runs,
        mode,
    )
    return ExperimentConfig(
        name=name,
        version=version,
        mode=mode,
        example=example,
        degree=degree,
        mesh_backend=backend,
        initial_refinements=initial_refinements,
        gamma0=gamma0,
        n0=n0,
        gamma_exponent=mu,
        runs=runs,
        hcurl_errors=hcurl_errors,
        gamma_tests=gamma_tests,
    )


def write_resolved_config(config, path):
    path = Path(path)
    with path.open("w") as stream:
        yaml.safe_dump(
            config.resolved_dict(),
            stream,
            sort_keys=False,
        )
