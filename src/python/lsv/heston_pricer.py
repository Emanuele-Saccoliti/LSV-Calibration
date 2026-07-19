"""Semi-analytic Heston European option pricing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from lsv.black_scholes import OptionType, black_scholes_price


@dataclass(frozen=True)
class HestonParameters:
    """Risk-neutral Heston parameters in annualized units."""

    kappa: float
    theta: float
    eta: float
    rho: float
    v0: float

    def __post_init__(self) -> None:
        values = (self.kappa, self.theta, self.eta, self.rho, self.v0)
        if not all(np.isfinite(values)):
            raise ValueError("Heston parameters must be finite")
        if self.kappa <= 0.0 or self.theta <= 0.0 or self.eta <= 0.0 or self.v0 <= 0.0:
            raise ValueError("kappa, theta, eta, and v0 must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")

    @property
    def feller_satisfied(self) -> bool:
        """Whether ``2*kappa*theta >= eta**2``."""
        return 2.0 * self.kappa * self.theta >= self.eta**2

    def as_array(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return parameters ordered as kappa, theta, eta, rho, v0."""
        return np.array([self.kappa, self.theta, self.eta, self.rho, self.v0])


@dataclass(frozen=True)
class HestonIntegrationConfig:
    """Fourier quadrature controls."""

    upper_limit: float = 150.0
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-7
    maximum_subintervals: int = 250

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.upper_limit)
            or self.upper_limit <= 0.0
            or self.absolute_tolerance <= 0.0
            or self.relative_tolerance <= 0.0
            or self.maximum_subintervals < 50
        ):
            raise ValueError("invalid Heston quadrature configuration")


def heston_characteristic_function(
    argument: complex,
    maturity: float,
    spot: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
) -> complex:
    """Return ``E[exp(i*u*log(S_T))]`` under the risk-neutral measure.

    The implementation uses the stable ``exp(-dT)`` representation and selects
    the square-root branch with non-negative real part.
    """
    if not np.isfinite(maturity) or maturity < 0.0:
        raise ValueError("maturity must be finite and non-negative")
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    if not np.isfinite(rate) or not np.isfinite(dividend_yield):
        raise ValueError("rates must be finite")
    if maturity == 0.0:
        return complex(np.exp(1j * argument * np.log(spot)))
    kappa, theta, eta, rho, v0 = parameters.as_array()
    iu = 1j * argument
    beta = kappa - rho * eta * iu
    discriminant = beta**2 + eta**2 * (argument**2 + iu)
    d = np.sqrt(discriminant)
    if np.real(d) < 0.0:
        d = -d
    g = (beta - d) / (beta + d)
    exponential = np.exp(-d * maturity)
    log_term = np.log((1.0 - g * exponential) / (1.0 - g))
    c = iu * (
        np.log(spot) + (rate - dividend_yield) * maturity
    ) + kappa * theta / eta**2 * ((beta - d) * maturity - 2.0 * log_term)
    d_coefficient = (beta - d) / eta**2 * (1.0 - exponential) / (1.0 - g * exponential)
    return complex(np.exp(c + d_coefficient * v0))


def heston_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    parameters: HestonParameters,
    option_type: OptionType | str = OptionType.CALL,
    integration: HestonIntegrationConfig | None = None,
) -> float:
    """Price a discounted European option using Heston ``P1/P2`` integrals."""
    kind = OptionType(option_type)
    if not all(np.isfinite((spot, strike, maturity, rate, dividend_yield))):
        raise ValueError("Heston pricing inputs must be finite")
    if spot <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("spot, strike, and maturity must be positive")
    config = integration or HestonIntegrationConfig()
    if parameters.eta < 1e-5:
        integrated_variance = (
            parameters.theta * maturity
            + (parameters.v0 - parameters.theta)
            * (1.0 - np.exp(-parameters.kappa * maturity))
            / parameters.kappa
        )
        volatility = np.sqrt(integrated_variance / maturity)
        forward = spot * np.exp((rate - dividend_yield) * maturity)
        call = float(
            black_scholes_price(
                forward,
                strike,
                maturity,
                volatility,
                np.exp(-rate * maturity),
            )
        )
    else:
        phi_minus_i = heston_characteristic_function(
            -1j, maturity, spot, rate, dividend_yield, parameters
        )

        def integrand(argument: float, probability: int) -> float:
            if probability == 1:
                characteristic = (
                    heston_characteristic_function(
                        argument - 1j,
                        maturity,
                        spot,
                        rate,
                        dividend_yield,
                        parameters,
                    )
                    / phi_minus_i
                )
            else:
                characteristic = heston_characteristic_function(
                    argument,
                    maturity,
                    spot,
                    rate,
                    dividend_yield,
                    parameters,
                )
            value = np.exp(-1j * argument * np.log(strike)) * characteristic
            return float(np.real(value / (1j * argument)))

        probabilities: list[float] = []
        for probability in (1, 2):
            integral, _ = quad(
                integrand,
                1e-10,
                config.upper_limit,
                args=(probability,),
                epsabs=config.absolute_tolerance,
                epsrel=config.relative_tolerance,
                limit=config.maximum_subintervals,
            )
            probabilities.append(0.5 + integral / np.pi)
        call = (
            spot * np.exp(-dividend_yield * maturity) * probabilities[0]
            - strike * np.exp(-rate * maturity) * probabilities[1]
        )
    lower = max(
        spot * np.exp(-dividend_yield * maturity) - strike * np.exp(-rate * maturity),
        0.0,
    )
    upper = spot * np.exp(-dividend_yield * maturity)
    tolerance = 1e-7 * max(1.0, upper)
    if call < lower - tolerance or call > upper + tolerance or not np.isfinite(call):
        raise RuntimeError(
            f"Heston quadrature produced invalid call price {call}; "
            f"bounds [{lower}, {upper}]"
        )
    call = min(max(call, lower), upper)
    if kind is OptionType.CALL:
        return float(call)
    return float(
        call
        - spot * np.exp(-dividend_yield * maturity)
        + strike * np.exp(-rate * maturity)
    )


def expected_average_variance(maturity: float, parameters: HestonParameters) -> float:
    """Return ``E[1/T integral_0^T V_t dt]`` under risk-neutral Heston."""
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    return float(
        parameters.theta
        + (parameters.v0 - parameters.theta)
        * (1.0 - np.exp(-parameters.kappa * maturity))
        / (parameters.kappa * maturity)
    )
