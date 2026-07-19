import lsv_cpp
import numpy as np
from lsv.black_scholes import black_scholes_price
from lsv.heston_pricer import HestonParameters
from lsv.leverage_calibration import (
    LeverageCalibrationConfig,
    calibrate_leverage,
)


def test_kernel_estimator_reproduces_constant_and_flags_tail() -> None:
    coordinates = np.linspace(-0.3, 0.3, 1_001)
    variances = np.full_like(coordinates, 0.04)
    result = lsv_cpp.estimate_conditional_variance(
        coordinates, variances, np.array([0.0, 3.0]), 0.05, 1e-12, 20.0
    )
    estimates = np.asarray(result["conditional_variances"])
    flags = np.asarray(result["low_density"], dtype=bool)
    assert np.isclose(estimates[0], 0.04, atol=1e-14)
    assert not flags[0]
    assert flags[1]


def test_fixed_point_converges_and_reprices_flat_local_volatility() -> None:
    parameters = HestonParameters(1.5, 0.04, 0.3, -0.6, 0.04)
    times = np.linspace(0.125, 1.0, 8)
    log_moneyness = np.linspace(-0.3, 0.3, 7)
    local_volatility = np.full((8, 7), 0.2)
    config = LeverageCalibrationConfig(
        paths=8_000,
        iterations=7,
        bandwidth=0.1,
        damping=0.65,
        smoothing_sigma=0.5,
        convergence_tolerance=0.008,
    )
    result = calibrate_leverage(
        100.0,
        0.02,
        0.01,
        parameters,
        times,
        log_moneyness,
        local_volatility,
        config=config,
    )
    assert result.converged
    assert result.history[-1].relative_update_norm < config.convergence_tolerance
    assert result.history[-1].low_density_count == 0
    projection = result.leverage * np.sqrt(result.conditional_variance)
    assert np.max(np.abs(projection - local_volatility)) < 0.005

    simulation = lsv_cpp.simulate_lsv(
        100.0,
        1.0,
        0.02,
        0.01,
        parameters.kappa,
        parameters.theta,
        parameters.eta,
        parameters.rho,
        parameters.v0,
        50_000,
        8,
        999,
        True,
        times,
        log_moneyness,
        result.leverage,
    )
    terminal_spots = np.asarray(simulation["spots"])[:, -1]
    payoff = np.exp(-0.02) * np.maximum(terminal_spots - 100.0, 0.0)
    estimate = float(np.mean(payoff))
    standard_error = float(np.std(payoff, ddof=1) / np.sqrt(payoff.size))
    target = float(
        black_scholes_price(100.0 * np.exp(0.01), 100.0, 1.0, 0.2, np.exp(-0.02))
    )
    assert abs(estimate - target) < 3.0 * standard_error
