"""Particle fixed-point calibration of the LSV leverage function."""

from __future__ import annotations

from dataclasses import dataclass

import lsv_cpp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

from lsv.heston_pricer import HestonParameters, expected_average_variance


@dataclass(frozen=True)
class LeverageCalibrationConfig:
    """Reproducible particle calibration controls."""

    paths: int = 20_000
    iterations: int = 8
    seed: int = 1729
    bandwidth: float = 0.08
    minimum_effective_sample_size: float = 30.0
    denominator_floor: float = 1e-12
    conditional_variance_floor: float = 1e-8
    damping: float = 0.5
    smoothing_sigma: float = 0.75
    leverage_minimum: float = 0.05
    leverage_maximum: float = 5.0
    convergence_tolerance: float = 2e-3
    antithetic: bool = True

    def __post_init__(self) -> None:
        if self.paths <= 0 or self.iterations <= 0:
            raise ValueError("paths and iterations must be positive")
        positive = (
            self.bandwidth,
            self.minimum_effective_sample_size,
            self.denominator_floor,
            self.conditional_variance_floor,
            self.leverage_minimum,
            self.leverage_maximum,
            self.convergence_tolerance,
        )
        if not all(np.isfinite(positive)) or any(value <= 0.0 for value in positive):
            raise ValueError("leverage calibration thresholds must be positive")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        if self.smoothing_sigma < 0.0 or not np.isfinite(self.smoothing_sigma):
            raise ValueError("smoothing_sigma must be finite and non-negative")
        if self.leverage_minimum >= self.leverage_maximum:
            raise ValueError("leverage minimum must be below maximum")


@dataclass(frozen=True)
class LeverageIteration:
    """Diagnostics for one global fixed-point update."""

    iteration: int
    relative_update_norm: float
    low_density_count: int
    lower_clip_count: int
    upper_clip_count: int
    minimum_effective_sample_size: float


@dataclass(frozen=True)
class LeverageCalibrationResult:
    """Calibrated leverage grid and complete convergence history."""

    times: NDArray[np.float64]
    log_moneyness: NDArray[np.float64]
    leverage: NDArray[np.float64]
    conditional_variance: NDArray[np.float64]
    history: tuple[LeverageIteration, ...]
    converged: bool
    seed: int
    paths: int
    steps: int


def calibrate_leverage(
    spot: float,
    rate: float,
    dividend_yield: float,
    heston_parameters: HestonParameters,
    times: ArrayLike,
    log_moneyness: ArrayLike,
    local_volatilities: ArrayLike,
    *,
    config: LeverageCalibrationConfig | None = None,
    initial_leverage: ArrayLike | None = None,
) -> LeverageCalibrationResult:
    """Calibrate ``L=local_vol/sqrt(E[V|S])`` by kernel particle regression.

    Time nodes must be a uniform simulation grid ending at maturity. Kernel
    regression is performed in forward log-moneyness. Low-density estimates are
    flagged but retained using the C++ estimator's nearest-particle boundary
    fallback. Updates are smoothed across moneyness, damped, and clipped with
    intervention counts recorded at every iteration.
    """
    settings = config or LeverageCalibrationConfig()
    time_grid = np.asarray(times, dtype=float)
    strike_grid = np.asarray(log_moneyness, dtype=float)
    local_vol = np.asarray(local_volatilities, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2 or np.any(time_grid <= 0.0):
        raise ValueError("times must be a positive vector with at least two nodes")
    if strike_grid.ndim != 1 or strike_grid.size < 2:
        raise ValueError("log_moneyness must contain at least two nodes")
    if np.any(np.diff(time_grid) <= 0.0) or np.any(np.diff(strike_grid) <= 0.0):
        raise ValueError("calibration grids must be strictly increasing")
    expected_times = np.arange(1, time_grid.size + 1) * time_grid[-1] / time_grid.size
    if not np.allclose(time_grid, expected_times, rtol=1e-10, atol=1e-12):
        raise ValueError("times must form a uniform grid ending at maturity")
    if local_vol.shape != (time_grid.size, strike_grid.size):
        raise ValueError("local_volatilities must have shape (times, log_moneyness)")
    if np.any(~np.isfinite(local_vol)) or np.any(local_vol <= 0.0):
        raise ValueError("local volatilities must be finite and positive")
    if not np.isfinite(spot + rate + dividend_yield) or spot <= 0.0:
        raise ValueError("spot must be positive and market inputs finite")
    if initial_leverage is None:
        average_variance = np.array(
            [
                expected_average_variance(float(time), heston_parameters)
                for time in time_grid
            ]
        )
        leverage = local_vol / np.sqrt(average_variance[:, None])
    else:
        leverage = np.asarray(initial_leverage, dtype=float).copy()
        if leverage.shape != local_vol.shape or np.any(leverage <= 0.0):
            raise ValueError("initial_leverage must be positive and match the grid")
    leverage = np.clip(leverage, settings.leverage_minimum, settings.leverage_maximum)
    history: list[LeverageIteration] = []
    conditional_variance = np.empty_like(leverage)
    converged = False
    for iteration in range(1, settings.iterations + 1):
        simulation = lsv_cpp.simulate_lsv(
            spot,
            float(time_grid[-1]),
            rate,
            dividend_yield,
            heston_parameters.kappa,
            heston_parameters.theta,
            heston_parameters.eta,
            heston_parameters.rho,
            heston_parameters.v0,
            settings.paths,
            time_grid.size,
            settings.seed,
            settings.antithetic,
            time_grid,
            strike_grid,
            leverage,
        )
        spots = np.asarray(simulation["spots"])
        variances = np.asarray(simulation["variances"])
        low_density_count = 0
        minimum_ess = float("inf")
        for time_index, time in enumerate(time_grid):
            coordinates = (
                np.log(spots[:, time_index + 1] / spot) - (rate - dividend_yield) * time
            )
            estimate = lsv_cpp.estimate_conditional_variance(
                coordinates,
                variances[:, time_index + 1],
                strike_grid,
                settings.bandwidth,
                settings.denominator_floor,
                settings.minimum_effective_sample_size,
            )
            conditional_variance[time_index] = np.asarray(
                estimate["conditional_variances"]
            )
            effective_samples = np.asarray(estimate["effective_sample_sizes"])
            low_density = np.asarray(estimate["low_density"], dtype=bool)
            low_density_count += int(np.count_nonzero(low_density))
            minimum_ess = min(minimum_ess, float(np.min(effective_samples)))
        raw = local_vol / np.sqrt(
            np.maximum(conditional_variance, settings.conditional_variance_floor)
        )
        if settings.smoothing_sigma > 0.0:
            raw = gaussian_filter1d(
                raw, settings.smoothing_sigma, axis=1, mode="nearest"
            )
        damped = (1.0 - settings.damping) * leverage + settings.damping * raw
        lower_count = int(np.count_nonzero(damped < settings.leverage_minimum))
        upper_count = int(np.count_nonzero(damped > settings.leverage_maximum))
        updated = np.clip(damped, settings.leverage_minimum, settings.leverage_maximum)
        relative_norm = float(
            np.linalg.norm(updated - leverage) / max(np.linalg.norm(leverage), 1e-12)
        )
        leverage = updated
        history.append(
            LeverageIteration(
                iteration,
                relative_norm,
                low_density_count,
                lower_count,
                upper_count,
                minimum_ess,
            )
        )
        if relative_norm <= settings.convergence_tolerance:
            converged = True
            break
    return LeverageCalibrationResult(
        time_grid.copy(),
        strike_grid.copy(),
        leverage.copy(),
        conditional_variance.copy(),
        tuple(history),
        converged,
        settings.seed,
        settings.paths,
        time_grid.size,
    )
