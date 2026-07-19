"""Nested calibration of stochastic-volatility parameters to DNT quotes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from time import perf_counter

import lsv_cpp
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from lsv.barrier_pricer import BarrierPrice, Monitoring, price_dnt
from lsv.heston_pricer import HestonParameters
from lsv.leverage_calibration import (
    LeverageCalibrationConfig,
    LeverageCalibrationResult,
    calibrate_leverage,
)


class QuoteSource(StrEnum):
    """Provenance of an exotic quote."""

    MARKET = "market"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class DNTCalibrationQuote:
    """Maturity-paid Double No-Touch calibration quote."""

    maturity: float
    lower_barrier: float
    upper_barrier: float
    payout: float
    price: float
    source: QuoteSource
    weight: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.maturity,
            self.lower_barrier,
            self.upper_barrier,
            self.payout,
            self.price,
            self.weight,
        )
        if not all(np.isfinite(values)):
            raise ValueError("DNT quote values must be finite")
        if self.maturity <= 0.0 or self.lower_barrier <= 0.0:
            raise ValueError("DNT maturity and lower barrier must be positive")
        if self.upper_barrier <= self.lower_barrier:
            raise ValueError("DNT upper barrier must exceed lower barrier")
        if self.payout <= 0.0 or not 0.0 <= self.price <= self.payout:
            raise ValueError("DNT price must lie between zero and payout")
        if self.weight <= 0.0:
            raise ValueError("DNT weight must be positive")
        object.__setattr__(self, "source", QuoteSource(self.source))


@dataclass(frozen=True)
class VanillaRepricingTarget:
    """Discounted vanilla call used to verify marginal preservation."""

    maturity: float
    strike: float
    price: float

    def __post_init__(self) -> None:
        if not all(np.isfinite((self.maturity, self.strike, self.price))):
            raise ValueError("vanilla target values must be finite")
        if self.maturity <= 0.0 or self.strike <= 0.0 or self.price < 0.0:
            raise ValueError("invalid vanilla repricing target")


@dataclass(frozen=True)
class ExoticCalibrationConfig:
    """Outer optimizer and common-random-number configuration."""

    calibrated_parameters: tuple[str, ...] = ("eta", "rho")
    lower_bounds: tuple[float, ...] = (0.05, -0.95)
    upper_bounds: tuple[float, ...] = (2.0, 0.95)
    maximum_evaluations: int = 30
    tolerance: float = 2e-3
    finite_difference_step: float = 0.02
    pricing_paths: int = 30_000
    pricing_seed: int = 9173
    monitoring: Monitoring = Monitoring.CONTINUOUS
    require_leverage_convergence: bool = True
    cache_rounding_digits: int = 10

    def __post_init__(self) -> None:
        allowed = {"kappa", "theta", "eta", "rho", "v0"}
        if not self.calibrated_parameters or any(
            parameter not in allowed for parameter in self.calibrated_parameters
        ):
            raise ValueError("unsupported or empty calibrated parameter set")
        if len(set(self.calibrated_parameters)) != len(self.calibrated_parameters):
            raise ValueError("calibrated parameters must be unique")
        if not (
            len(self.lower_bounds)
            == len(self.upper_bounds)
            == len(self.calibrated_parameters)
        ):
            raise ValueError("outer bounds must match calibrated parameters")
        if np.any(np.asarray(self.lower_bounds) >= np.asarray(self.upper_bounds)):
            raise ValueError("outer calibration bounds must be ordered")
        if self.maximum_evaluations <= 0 or self.pricing_paths <= 1:
            raise ValueError("outer evaluations and pricing paths must be positive")
        if self.tolerance <= 0.0 or self.finite_difference_step <= 0.0:
            raise ValueError("outer optimizer tolerances must be positive")
        object.__setattr__(self, "monitoring", Monitoring(self.monitoring))


@dataclass(frozen=True)
class ExoticResidual:
    """Price/probability residual and Monte Carlo noise for one DNT quote."""

    quote: DNTCalibrationQuote
    model_price: float
    price_residual: float
    probability_residual: float
    standard_error: float
    residual_standard_errors: float


@dataclass(frozen=True)
class ExoticModelEvaluation:
    """One cached nested LSV evaluation."""

    parameters: HestonParameters
    dnt_prices: tuple[BarrierPrice, ...]
    vanilla_prices: tuple[float, ...]
    leverage: LeverageCalibrationResult


@dataclass(frozen=True)
class ExoticCalibrationReport:
    """Complete outer-calibration result with before/after diagnostics."""

    initial_parameters: HestonParameters
    final_parameters: HestonParameters
    initial_dnt_residuals: tuple[ExoticResidual, ...]
    final_dnt_residuals: tuple[ExoticResidual, ...]
    initial_vanilla_residuals: tuple[float, ...]
    final_vanilla_residuals: tuple[float, ...]
    objective_before: float
    objective_after: float
    iterations: int
    success: bool
    message: str
    cache_hits: int
    cache_misses: int
    elapsed_seconds: float
    quote_sources: tuple[QuoteSource, ...]
    final_leverage: LeverageCalibrationResult
    pricing_seed: int


def _replace_parameters(
    base: HestonParameters, names: tuple[str, ...], values: NDArray[np.float64]
) -> HestonParameters:
    data = {
        "kappa": base.kappa,
        "theta": base.theta,
        "eta": base.eta,
        "rho": base.rho,
        "v0": base.v0,
    }
    data.update(zip(names, map(float, values), strict=True))
    return HestonParameters(**data)


def calibrate_exotics(
    spot: float,
    rate: float,
    dividend_yield: float,
    initial_parameters: HestonParameters,
    times: ArrayLike,
    log_moneyness: ArrayLike,
    local_volatilities: ArrayLike,
    dnt_quotes: list[DNTCalibrationQuote],
    *,
    vanilla_targets: list[VanillaRepricingTarget] | None = None,
    leverage_config: LeverageCalibrationConfig | None = None,
    config: ExoticCalibrationConfig | None = None,
) -> ExoticCalibrationReport:
    """Run nested LSV/DNT calibration with cached leverage and common RNGs.

    Every candidate stochastic-volatility parameter vector receives its own
    leverage fixed point. Both leverage and pricing simulations reuse fixed seeds
    across candidates, providing common random numbers. Failures are raised and
    never converted into successful reports.
    """
    if not dnt_quotes:
        raise ValueError("at least one DNT calibration quote is required")
    settings = config or ExoticCalibrationConfig()
    leverage_settings = leverage_config or LeverageCalibrationConfig()
    vanilla = vanilla_targets or []
    time_grid = np.asarray(times, dtype=float)
    strike_grid = np.asarray(log_moneyness, dtype=float)
    local_vol = np.asarray(local_volatilities, dtype=float)
    if time_grid.ndim != 1 or time_grid.size < 2:
        raise ValueError("outer calibration requires a time grid")
    dt = float(time_grid[-1] / time_grid.size)
    for maturity in [quote.maturity for quote in dnt_quotes] + [
        target.maturity for target in vanilla
    ]:
        index = round(maturity / dt)
        if index < 1 or index > time_grid.size or not np.isclose(index * dt, maturity):
            raise ValueError("all quote maturities must align with the simulation grid")
    leverage_cache: dict[tuple[float, ...], LeverageCalibrationResult] = {}
    evaluation_cache: dict[tuple[float, ...], ExoticModelEvaluation] = {}
    cache_hits = 0
    cache_misses = 0

    def cache_key(parameters: HestonParameters) -> tuple[float, ...]:
        return tuple(
            np.round(parameters.as_array(), settings.cache_rounding_digits).tolist()
        )

    def evaluate(parameters: HestonParameters) -> ExoticModelEvaluation:
        nonlocal cache_hits, cache_misses
        key = cache_key(parameters)
        cached = evaluation_cache.get(key)
        if cached is not None:
            cache_hits += 1
            return cached
        cache_misses += 1
        leverage = leverage_cache.get(key)
        if leverage is None:
            leverage = calibrate_leverage(
                spot,
                rate,
                dividend_yield,
                parameters,
                time_grid,
                strike_grid,
                local_vol,
                config=leverage_settings,
            )
            leverage_cache[key] = leverage
        if settings.require_leverage_convergence and not leverage.converged:
            raise RuntimeError(
                f"leverage calibration did not converge for parameters {parameters}"
            )
        simulation = lsv_cpp.simulate_lsv(
            spot,
            float(time_grid[-1]),
            rate,
            dividend_yield,
            parameters.kappa,
            parameters.theta,
            parameters.eta,
            parameters.rho,
            parameters.v0,
            settings.pricing_paths,
            time_grid.size,
            settings.pricing_seed,
            True,
            time_grid,
            strike_grid,
            leverage.leverage,
        )
        spots = np.asarray(simulation["spots"])
        variances = np.asarray(simulation["variances"])
        local_variance_paths = np.empty_like(variances)
        carry = rate - dividend_yield
        for time_index in range(time_grid.size + 1):
            time = time_index * dt
            row = max(time_index - 1, 0)
            coordinates = np.log(spots[:, time_index] / spot) - carry * time
            path_leverage = np.interp(
                coordinates,
                strike_grid,
                leverage.leverage[row],
                left=leverage.leverage[row, 0],
                right=leverage.leverage[row, -1],
            )
            local_variance_paths[:, time_index] = (
                path_leverage**2 * variances[:, time_index]
            )
        dnt_prices: list[BarrierPrice] = []
        for quote in dnt_quotes:
            end = round(quote.maturity / dt)
            dnt_prices.append(
                price_dnt(
                    spots[:, : end + 1],
                    variances[:, : end + 1],
                    quote.maturity,
                    rate,
                    quote.payout,
                    quote.lower_barrier,
                    quote.upper_barrier,
                    monitoring=settings.monitoring,
                    local_variances=local_variance_paths[:, : end + 1],
                    seed=settings.pricing_seed,
                )
            )
        vanilla_prices: list[float] = []
        for target in vanilla:
            end = round(target.maturity / dt)
            discounted_payoff = np.exp(-rate * target.maturity) * np.maximum(
                spots[:, end] - target.strike, 0.0
            )
            vanilla_prices.append(float(np.mean(discounted_payoff)))
        result = ExoticModelEvaluation(
            parameters, tuple(dnt_prices), tuple(vanilla_prices), leverage
        )
        evaluation_cache[key] = result
        return result

    def residual_vector(evaluation: ExoticModelEvaluation) -> NDArray[np.float64]:
        return np.asarray(
            [
                np.sqrt(quote.weight) * (model.estimate - quote.price) / quote.payout
                for quote, model in zip(dnt_quotes, evaluation.dnt_prices, strict=True)
            ]
        )

    def residual_report(
        evaluation: ExoticModelEvaluation,
    ) -> tuple[ExoticResidual, ...]:
        output: list[ExoticResidual] = []
        for quote, model in zip(dnt_quotes, evaluation.dnt_prices, strict=True):
            price_residual = model.estimate - quote.price
            discounted_payout = quote.payout * np.exp(-rate * quote.maturity)
            probability_residual = price_residual / discounted_payout
            noise_units = (
                price_residual / model.standard_error
                if model.standard_error > 0.0
                else float("inf")
            )
            output.append(
                ExoticResidual(
                    quote,
                    model.estimate,
                    price_residual,
                    probability_residual,
                    model.standard_error,
                    noise_units,
                )
            )
        return tuple(output)

    start = perf_counter()
    initial_evaluation = evaluate(initial_parameters)
    initial_values = np.array(
        [getattr(initial_parameters, name) for name in settings.calibrated_parameters]
    )

    def objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
        return residual_vector(
            evaluate(
                _replace_parameters(
                    initial_parameters, settings.calibrated_parameters, values
                )
            )
        )

    optimization = least_squares(
        objective,
        initial_values,
        bounds=(np.asarray(settings.lower_bounds), np.asarray(settings.upper_bounds)),
        max_nfev=settings.maximum_evaluations,
        xtol=settings.tolerance,
        ftol=settings.tolerance,
        gtol=settings.tolerance,
        diff_step=settings.finite_difference_step,
    )
    if not optimization.success:
        raise RuntimeError(f"exotic calibration failed: {optimization.message}")
    final_parameters = _replace_parameters(
        initial_parameters, settings.calibrated_parameters, optimization.x
    )
    final_evaluation = evaluate(final_parameters)
    initial_residuals = residual_report(initial_evaluation)
    final_residuals = residual_report(final_evaluation)
    elapsed = perf_counter() - start
    return ExoticCalibrationReport(
        initial_parameters,
        final_parameters,
        initial_residuals,
        final_residuals,
        tuple(
            model - target.price
            for model, target in zip(
                initial_evaluation.vanilla_prices, vanilla, strict=True
            )
        ),
        tuple(
            model - target.price
            for model, target in zip(
                final_evaluation.vanilla_prices, vanilla, strict=True
            )
        ),
        float(
            np.dot(
                residual_vector(initial_evaluation), residual_vector(initial_evaluation)
            )
        ),
        float(
            np.dot(residual_vector(final_evaluation), residual_vector(final_evaluation))
        ),
        int(optimization.nfev),
        True,
        str(optimization.message),
        cache_hits,
        cache_misses,
        elapsed,
        tuple(quote.source for quote in dnt_quotes),
        final_evaluation.leverage,
        settings.pricing_seed,
    )
