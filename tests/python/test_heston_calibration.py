import numpy as np
import pytest
from lsv.black_scholes import black_scholes_price
from lsv.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationQuote,
    VarianceSwapQuote,
    calibrate_heston,
)
from lsv.heston_pricer import (
    HestonParameters,
    expected_average_variance,
    heston_characteristic_function,
    heston_price,
)


def test_heston_price_matches_published_reference_value() -> None:
    parameters = HestonParameters(2.0, 0.04, 0.3, -0.7, 0.04)
    price = heston_price(100.0, 100.0, 1.0, 0.05, 0.0, parameters)
    assert price == pytest.approx(10.394218565, abs=2e-8)


def test_characteristic_function_has_martingale_first_moment() -> None:
    parameters = HestonParameters(1.5, 0.05, 0.45, -0.6, 0.035)
    first_moment = heston_characteristic_function(
        -1j, 2.0, 100.0, 0.03, 0.01, parameters
    )
    assert first_moment.real == pytest.approx(100.0 * np.exp(0.04), rel=1e-11)
    assert first_moment.imag == pytest.approx(0.0, abs=1e-11)


def test_near_deterministic_variance_agrees_with_black_scholes() -> None:
    parameters = HestonParameters(3.0, 0.04, 1e-6, 0.0, 0.04)
    heston = heston_price(100.0, 105.0, 1.4, 0.02, 0.01, parameters)
    forward = 100.0 * np.exp(0.01 * 1.4)
    black = float(black_scholes_price(forward, 105.0, 1.4, 0.2, np.exp(-0.02 * 1.4)))
    assert heston == pytest.approx(black, abs=1e-12)


def test_calibration_report_is_reproducible_and_complete() -> None:
    expected = HestonParameters(1.7, 0.045, 0.35, -0.55, 0.04)
    specs = [(0.5, 90.0), (0.5, 100.0), (1.0, 90.0), (1.0, 110.0)]
    quotes = [
        HestonCalibrationQuote(
            maturity,
            strike,
            heston_price(100.0, strike, maturity, 0.02, 0.01, expected),
        )
        for maturity, strike in specs
    ]
    variance_quotes = [
        VarianceSwapQuote(1.0, expected_average_variance(1.0, expected), weight=2.0)
    ]
    arguments = (100.0, 0.02, 0.01, quotes, expected)
    first = calibrate_heston(*arguments, variance_swap_quotes=variance_quotes)
    second = calibrate_heston(*arguments, variance_swap_quotes=variance_quotes)
    assert first == second
    assert first.success
    assert first.objective_value < 1e-20
    assert max(map(abs, first.price_residuals)) < 1e-10
    assert max(map(abs, first.variance_swap_residuals)) < 1e-12
    assert first.iterations >= 1
    assert first.feller_satisfied == expected.feller_satisfied


def test_optimizer_failure_is_surfaced() -> None:
    target = HestonParameters(1.7, 0.045, 0.35, -0.55, 0.04)
    quote = HestonCalibrationQuote(
        1.0, 100.0, heston_price(100.0, 100.0, 1.0, 0.02, 0.0, target)
    )
    initial = HestonParameters(0.5, 0.12, 1.0, 0.2, 0.1)
    with pytest.raises(RuntimeError, match="calibration failed"):
        calibrate_heston(
            100.0,
            0.02,
            0.0,
            [quote],
            initial,
            config=HestonCalibrationConfig(maximum_evaluations=1),
        )
