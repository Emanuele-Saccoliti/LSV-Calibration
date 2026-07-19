import numpy as np
import pytest
from lsv.black_scholes import black_scholes_price
from lsv.diagnostics import diagnose_static_arbitrage
from lsv.ssvi_surface import SSVISurface, fit_ssvi_surface


def test_ssvi_calendar_and_butterfly_diagnostics() -> None:
    surface = SSVISurface(
        np.array([0.5, 1.0, 2.0]),
        np.array([0.02, 0.04, 0.08]),
        rho=-0.35,
        eta=0.45,
        gamma=0.5,
    )
    maturities = np.array([0.5, 1.0, 2.0])
    strikes = np.linspace(60.0, 140.0, 161)
    time_grid, strike_grid = np.meshgrid(maturities, strikes, indexing="ij")
    log_moneyness = np.log(strike_grid / 100.0)
    total_variance = surface.total_variance(time_grid, log_moneyness)
    vol = np.sqrt(total_variance / time_grid)
    calls = black_scholes_price(100.0, strike_grid, time_grid, vol)
    report = diagnose_static_arbitrage(strikes, calls, total_variance, tolerance=2e-8)
    assert report.is_arbitrage_free
    assert np.all(surface.sufficient_butterfly_violations() == 0.0)


def test_ssvi_fit_recovers_synthetic_surface() -> None:
    expected = SSVISurface(
        np.array([0.5, 1.0, 2.0]),
        np.array([0.025, 0.045, 0.075]),
        rho=-0.25,
        eta=0.35,
        gamma=0.45,
    )
    times = np.repeat(expected.maturities, 9)
    k = np.tile(np.linspace(-0.3, 0.3, 9), 3)
    vols = expected.implied_volatility(times, k)
    fitted = fit_ssvi_surface(times, k, vols)
    fitted_w = fitted.total_variance(times, k)
    expected_w = expected.total_variance(times, k)
    assert np.max(np.abs(fitted_w - expected_w)) < 2e-5


def test_ssvi_rejects_non_monotone_atm_variance() -> None:
    with pytest.raises(ValueError, match="non-decreasing"):
        SSVISurface(
            np.array([1.0, 2.0]),
            np.array([0.04, 0.03]),
            rho=0.0,
            eta=0.2,
            gamma=0.5,
        )
