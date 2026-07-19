"""Weighted calibration of risk-neutral Heston parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from lsv.black_scholes import OptionType
from lsv.heston_pricer import (
    HestonIntegrationConfig,
    HestonParameters,
    expected_average_variance,
    heston_price,
)


@dataclass(frozen=True)
class HestonCalibrationQuote:
    """Observed discounted European option price and objective weight."""

    maturity: float
    strike: float
    price: float
    option_type: OptionType = OptionType.CALL
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not all(np.isfinite((self.maturity, self.strike, self.price, self.weight))):
            raise ValueError("calibration quote values must be finite")
        if self.maturity <= 0.0 or self.strike <= 0.0 or self.price < 0.0:
            raise ValueError("maturity/strike must be positive and price non-negative")
        if self.weight <= 0.0:
            raise ValueError("calibration weight must be positive")
        object.__setattr__(self, "option_type", OptionType(self.option_type))


@dataclass(frozen=True)
class VarianceSwapQuote:
    """Expected average variance quote in variance units."""

    maturity: float
    variance: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not all(np.isfinite((self.maturity, self.variance, self.weight))):
            raise ValueError("variance-swap quote values must be finite")
        if self.maturity <= 0.0 or self.variance <= 0.0 or self.weight <= 0.0:
            raise ValueError(
                "variance-swap maturity, variance, and weight must be positive"
            )


@dataclass(frozen=True)
class HestonCalibrationConfig:
    """Optimizer bounds and convergence controls."""

    lower_bounds: tuple[float, float, float, float, float] = (
        0.05,
        0.0025,
        0.01,
        -0.999,
        0.0025,
    )
    upper_bounds: tuple[float, float, float, float, float] = (
        15.0,
        1.0,
        5.0,
        0.999,
        1.0,
    )
    maximum_evaluations: int = 500
    tolerance: float = 1e-8
    enforce_feller: bool = False

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_bounds)
        upper = np.asarray(self.upper_bounds)
        if lower.shape != (5,) or upper.shape != (5,) or np.any(lower >= upper):
            raise ValueError("Heston bounds must contain five ordered intervals")
        if self.maximum_evaluations <= 0 or self.tolerance <= 0.0:
            raise ValueError("optimizer evaluations and tolerance must be positive")


@dataclass(frozen=True)
class HestonCalibrationReport:
    """Complete deterministic optimizer result."""

    parameters: HestonParameters
    objective_value: float
    iterations: int
    price_residuals: tuple[float, ...]
    variance_swap_residuals: tuple[float, ...]
    feller_satisfied: bool
    success: bool
    message: str


def calibrate_heston(
    spot: float,
    rate: float,
    dividend_yield: float,
    quotes: list[HestonCalibrationQuote],
    initial: HestonParameters,
    *,
    variance_swap_quotes: list[VarianceSwapQuote] | None = None,
    config: HestonCalibrationConfig | None = None,
    integration: HestonIntegrationConfig | None = None,
) -> HestonCalibrationReport:
    """Calibrate Heston by weighted least squares in normalized price space.

    Vanilla residuals are ``sqrt(weight)*(model-market)/spot``. Variance-swap
    residuals are in variance units. The deterministic least-squares optimizer
    uses no random initialization, so repeated calls are reproducible.
    """
    if not np.isfinite(spot) or spot <= 0.0 or not np.isfinite(rate + dividend_yield):
        raise ValueError("spot must be positive and rates finite")
    if not quotes:
        raise ValueError("at least one vanilla calibration quote is required")
    settings = config or HestonCalibrationConfig()
    variance_quotes = variance_swap_quotes or []
    lower = np.asarray(settings.lower_bounds)
    upper = np.asarray(settings.upper_bounds)
    if np.any(initial.as_array() <= lower) or np.any(initial.as_array() >= upper):
        raise ValueError("initial Heston parameters must lie strictly inside bounds")

    def parameters_from(
        values: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> HestonParameters:
        return HestonParameters(*map(float, values))

    def raw_residuals(parameters: HestonParameters) -> tuple[list[float], list[float]]:
        price_errors = [
            heston_price(
                spot,
                quote.strike,
                quote.maturity,
                rate,
                dividend_yield,
                parameters,
                quote.option_type,
                integration,
            )
            - quote.price
            for quote in quotes
        ]
        variance_errors = [
            expected_average_variance(quote.maturity, parameters) - quote.variance
            for quote in variance_quotes
        ]
        return price_errors, variance_errors

    def objective(
        values: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        parameters = parameters_from(values)
        price_errors, variance_errors = raw_residuals(parameters)
        residuals = [
            np.sqrt(quote.weight) * error / spot
            for quote, error in zip(quotes, price_errors, strict=True)
        ]
        residuals.extend(
            np.sqrt(quote.weight) * error
            for quote, error in zip(variance_quotes, variance_errors, strict=True)
        )
        if settings.enforce_feller:
            residuals.append(
                10.0
                * max(
                    parameters.eta**2 - 2.0 * parameters.kappa * parameters.theta, 0.0
                )
            )
        return np.asarray(residuals, dtype=float)

    result = least_squares(
        objective,
        initial.as_array(),
        bounds=(lower, upper),
        xtol=settings.tolerance,
        ftol=settings.tolerance,
        gtol=settings.tolerance,
        max_nfev=settings.maximum_evaluations,
    )
    parameters = parameters_from(result.x)
    price_errors, variance_errors = raw_residuals(parameters)
    success = bool(result.success)
    message = str(result.message)
    if settings.enforce_feller and not parameters.feller_satisfied:
        success = False
        message = "optimizer result violates configured Feller constraint"
    report = HestonCalibrationReport(
        parameters,
        float(2.0 * result.cost),
        int(result.nfev),
        tuple(map(float, price_errors)),
        tuple(map(float, variance_errors)),
        parameters.feller_satisfied,
        success,
        message,
    )
    if not report.success:
        raise RuntimeError(
            "Heston calibration failed after "
            f"{report.iterations} evaluations: {report.message}"
        )
    return report
