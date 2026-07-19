import numpy as np
import pytest
from lsv.dupire import DupireConfig, DupireExtractor
from lsv.ssvi_surface import SSVISurface


def test_flat_implied_volatility_returns_flat_local_volatility() -> None:
    volatility = 0.24
    maturities = np.array([0.25, 0.5, 1.0, 2.0])
    surface = SSVISurface(
        maturities,
        volatility**2 * maturities,
        rho=0.0,
        eta=1e-7,
        gamma=0.5,
    )
    extractor = DupireExtractor(surface, spot=100.0, rate=0.03, dividend_yield=0.01)
    grid = extractor.extract_grid(
        np.array([0.35, 0.75, 1.5]), np.array([80.0, 100.0, 120.0])
    )
    assert np.max(np.abs(grid.local_volatilities - volatility)) < 2e-5
    assert grid.diagnostics.unstable_denominator_count == 0
    assert grid.diagnostics.negative_variance_count == 0
    assert grid.diagnostics.floor_count == 0
    assert grid.diagnostics.cap_count == 0
    assert np.all(np.isfinite(grid.local_volatilities))


def test_term_structure_recovers_instantaneous_variance() -> None:
    surface = SSVISurface(
        np.array([0.5, 1.5, 2.0]),
        np.array([0.02, 0.065, 0.0875]),
        rho=0.0,
        eta=1e-7,
        gamma=0.5,
    )
    extractor = DupireExtractor(
        surface,
        spot=100.0,
        rate=0.0,
        dividend_yield=0.0,
        config=DupireConfig(time_step=1e-4, relative_strike_step=5e-4),
    )
    point = extractor.local_volatility(1.0, 100.0)
    assert point.local_volatility == pytest.approx(np.sqrt(0.045), abs=2e-5)
    assert not point.floor_applied


def test_unstable_denominator_is_flagged_and_floored() -> None:
    surface = SSVISurface(
        np.array([0.5, 1.0]),
        np.array([0.02, 0.04]),
        rho=0.0,
        eta=1e-7,
        gamma=0.5,
    )
    config = DupireConfig(denominator_floor=1e9, local_variance_floor=1e-6)
    point = DupireExtractor(surface, 100.0, 0.0, 0.0, config).local_volatility(
        0.75, 100.0
    )
    assert point.denominator_unstable
    assert point.floor_applied
    assert np.isnan(point.raw_local_variance)
    assert point.local_volatility == pytest.approx(0.001)


def test_invalid_grid_fails_clearly() -> None:
    surface = SSVISurface(
        np.array([0.5, 1.0]),
        np.array([0.02, 0.04]),
        rho=0.0,
        eta=1e-7,
        gamma=0.5,
    )
    extractor = DupireExtractor(surface, 100.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="strikes"):
        extractor.extract_grid([0.5], [-100.0])
