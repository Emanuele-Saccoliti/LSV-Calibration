import lsv_cpp
import numpy as np
from lsv.heston_pricer import HestonParameters, heston_price


def _simulate(paths: int, steps: int, seed: int) -> dict[str, object]:
    parameters = HestonParameters(1.7, 0.045, 0.5, -0.65, 0.04)
    return lsv_cpp.simulate_heston_qem(
        100.0,
        1.0,
        0.03,
        0.01,
        parameters.kappa,
        parameters.theta,
        parameters.eta,
        parameters.rho,
        parameters.v0,
        paths,
        steps,
        seed,
        True,
    )


def test_qem_binding_is_deterministic_and_non_negative() -> None:
    first = _simulate(2_000, 32, 42)
    second = _simulate(2_000, 32, 42)
    first_spots = np.asarray(first["spots"])
    first_variances = np.asarray(first["variances"])
    assert np.array_equal(first_spots, np.asarray(second["spots"]))
    assert np.array_equal(first_variances, np.asarray(second["variances"]))
    assert first_spots.shape == (2_000, 33)
    assert first_variances.shape == (2_000, 33)
    assert np.min(first_variances) >= 0.0


def test_qem_vanilla_price_agrees_with_fourier_within_mc_error() -> None:
    parameters = HestonParameters(1.7, 0.045, 0.5, -0.65, 0.04)
    result = _simulate(30_000, 64, 9_876)
    terminal_spots = np.asarray(result["spots"])[:, -1]
    payoffs = np.exp(-0.03) * np.maximum(terminal_spots - 100.0, 0.0)
    estimate = float(np.mean(payoffs))
    standard_error = float(np.std(payoffs, ddof=1) / np.sqrt(payoffs.size))
    benchmark = heston_price(100.0, 100.0, 1.0, 0.03, 0.01, parameters)
    assert abs(estimate - benchmark) < 4.0 * standard_error + 0.02


def test_qem_coarse_bias_decreases_with_step_refinement() -> None:
    parameters = HestonParameters(1.0, 0.04, 1.0, -0.8, 0.04)
    benchmark = heston_price(100.0, 100.0, 2.0, 0.01, 0.0, parameters)

    def estimate(steps: int) -> float:
        result = lsv_cpp.simulate_heston_qem(
            100.0,
            2.0,
            0.01,
            0.0,
            parameters.kappa,
            parameters.theta,
            parameters.eta,
            parameters.rho,
            parameters.v0,
            100_000,
            steps,
            12_345,
            True,
        )
        terminal = np.asarray(result["spots"])[:, -1]
        return float(np.mean(np.exp(-0.02) * np.maximum(terminal - 100.0, 0.0)))

    coarse_error = abs(estimate(2) - benchmark)
    refined_error = abs(estimate(64) - benchmark)
    assert refined_error < 0.25 * coarse_error
