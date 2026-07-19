import numpy as np
import pytest
from lsv.barrier_pricer import price_barrier, price_dnt


def _gbm_paths(paths: int = 12_000, steps: int = 252) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(123)
    dt = 1.0 / steps
    volatility = 0.2
    normals = generator.standard_normal((paths, steps))
    log_returns = (-0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * normals
    log_paths = np.cumsum(log_returns, axis=1)
    spots = 100.0 * np.exp(np.column_stack((np.zeros(paths), log_paths)))
    variances = np.full_like(spots, volatility**2)
    return spots, variances


def test_dnt_bounds_monotonicity_and_touch_parity() -> None:
    fine_spots, fine_variances = _gbm_paths()
    spots = fine_spots[:, ::21]
    variances = fine_variances[:, ::21]
    wide = price_dnt(spots, variances, 1.0, 0.02, 1.0, 70.0, 130.0, seed=123)
    medium = price_dnt(spots, variances, 1.0, 0.02, 1.0, 80.0, 120.0, seed=123)
    narrow = price_dnt(spots, variances, 1.0, 0.02, 1.0, 90.0, 110.0, seed=123)
    assert 0.0 <= narrow.estimate < medium.estimate < wide.estimate <= np.exp(-0.02)
    touch = price_barrier(
        spots,
        variances,
        1.0,
        0.02,
        1.0,
        "double",
        "touch",
        "continuous",
        lower_barrier=80.0,
        upper_barrier=120.0,
        seed=123,
    )
    assert touch.estimate + medium.estimate == pytest.approx(np.exp(-0.02), abs=1e-12)
    assert (
        medium.confidence_interval[0]
        <= medium.estimate
        <= medium.confidence_interval[1]
    )
    assert medium.paths == spots.shape[0]
    assert medium.steps == spots.shape[1] - 1


def test_bridge_correction_reduces_monitoring_bias() -> None:
    fine_spots, fine_variances = _gbm_paths()
    fine_discrete = price_dnt(
        fine_spots,
        fine_variances,
        1.0,
        0.0,
        1.0,
        80.0,
        120.0,
        monitoring="discrete",
    )
    coarse_spots = fine_spots[:, ::21]
    coarse_variances = fine_variances[:, ::21]
    coarse_discrete = price_dnt(
        coarse_spots,
        coarse_variances,
        1.0,
        0.0,
        1.0,
        80.0,
        120.0,
        monitoring="discrete",
    )
    coarse_continuous = price_dnt(
        coarse_spots,
        coarse_variances,
        1.0,
        0.0,
        1.0,
        80.0,
        120.0,
        monitoring="continuous",
    )
    discrete_bias = abs(coarse_discrete.estimate - fine_discrete.estimate)
    corrected_bias = abs(coarse_continuous.estimate - fine_discrete.estimate)
    assert coarse_continuous.estimate < coarse_discrete.estimate
    assert corrected_bias < discrete_bias


def test_invalid_barriers_fail_clearly() -> None:
    spots = np.full((10, 3), 100.0)
    variances = np.full_like(spots, 0.04)
    with pytest.raises(ValueError, match="bracket initial spot"):
        price_dnt(spots, variances, 1.0, 0.0, 1.0, 105.0, 120.0)
