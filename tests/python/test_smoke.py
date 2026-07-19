from pathlib import Path

import lsv
import lsv_cpp


def test_python_and_cpp_packages_import() -> None:
    assert lsv.__version__ == "0.1.0"
    assert lsv_cpp.__version__ == lsv.__version__


def test_configuration_is_deterministic_and_immutable() -> None:
    config_path = Path(__file__).parents[2] / "configs" / "calibration.yaml"
    first = lsv.load_config(config_path)
    second = lsv.load_config(config_path)
    assert first == second
    assert tuple(first) == tuple(sorted(first))
    assert "experiment" in first
    assert "exotic_calibration" in first
