import numpy as np
from lsv.black_scholes import black_scholes_price
from lsv.exotic_calibration import (
    DNTCalibrationQuote,
    ExoticCalibrationConfig,
    QuoteSource,
    VanillaRepricingTarget,
    calibrate_exotics,
)
from lsv.heston_pricer import HestonParameters
from lsv.leverage_calibration import LeverageCalibrationConfig


def test_synthetic_dnt_outer_calibration_improves_residuals() -> None:
    times = np.linspace(1.0 / 6.0, 1.0, 6)
    log_moneyness = np.linspace(-0.3, 0.3, 7)
    local_volatility = np.full((6, 7), 0.2)
    leverage_config = LeverageCalibrationConfig(
        paths=4_000,
        iterations=6,
        bandwidth=0.11,
        damping=0.7,
        smoothing_sigma=0.5,
        convergence_tolerance=0.015,
    )
    synthetic_quote = DNTCalibrationQuote(
        maturity=1.0,
        lower_barrier=80.0,
        upper_barrier=120.0,
        payout=1.0,
        price=0.40993707909989363,
        source=QuoteSource.SYNTHETIC,
    )
    vanilla_price = float(
        black_scholes_price(100.0 * np.exp(0.01), 100.0, 1.0, 0.2, np.exp(-0.01))
    )
    report = calibrate_exotics(
        100.0,
        0.01,
        0.0,
        HestonParameters(1.5, 0.04, 0.5, -0.2, 0.04),
        times,
        log_moneyness,
        local_volatility,
        [synthetic_quote],
        vanilla_targets=[VanillaRepricingTarget(1.0, 100.0, vanilla_price)],
        leverage_config=leverage_config,
        config=ExoticCalibrationConfig(
            calibrated_parameters=("rho",),
            lower_bounds=(-0.9,),
            upper_bounds=(-0.05,),
            maximum_evaluations=20,
            tolerance=1e-5,
            finite_difference_step=0.08,
            pricing_paths=10_000,
            pricing_seed=555,
        ),
    )
    assert report.success
    assert report.quote_sources == (QuoteSource.SYNTHETIC,)
    np.testing.assert_allclose(report.final_parameters.rho, -0.75, atol=0.02)
    assert report.objective_after < 1e-4 * report.objective_before
    assert abs(report.final_dnt_residuals[0].price_residual) < 1e-3
    assert abs(report.final_vanilla_residuals[0]) < 0.05
    assert report.cache_hits > 0
    assert report.final_leverage.converged
    assert report.pricing_seed == 555


def test_quote_source_is_mandatory_and_validated() -> None:
    try:
        DNTCalibrationQuote(1.0, 80.0, 120.0, 1.0, 0.4, "unknown")  # type: ignore[arg-type]
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("invalid quote source was accepted")
