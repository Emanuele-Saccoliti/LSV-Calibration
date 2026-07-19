"""Dupire local-volatility extraction from discounted call prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from lsv.black_scholes import black_scholes_price


class ImpliedVolatilitySurface(Protocol):
    """Structural interface required by the Dupire extractor."""

    def implied_volatility(
        self, maturity: ArrayLike, log_moneyness: ArrayLike
    ) -> NDArray[np.float64]:
        """Return annualized implied volatility."""
        ...


@dataclass(frozen=True)
class DupireConfig:
    """Finite-difference, regularization, and clipping parameters."""

    time_step: float = 1e-3
    relative_strike_step: float = 1e-3
    denominator_floor: float = 1e-12
    local_variance_floor: float = 1e-8
    local_variance_cap: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.time_step,
            self.relative_strike_step,
            self.denominator_floor,
            self.local_variance_floor,
            self.local_variance_cap,
        )
        if not all(np.isfinite(values)) or any(value <= 0.0 for value in values):
            raise ValueError(
                "all Dupire configuration values must be finite and positive"
            )
        if self.local_variance_floor >= self.local_variance_cap:
            raise ValueError("local_variance_floor must be below local_variance_cap")


@dataclass(frozen=True)
class LocalVolPoint:
    """One local-volatility value and its unhidden regularization state."""

    local_volatility: float
    raw_local_variance: float
    numerator: float
    denominator: float
    denominator_unstable: bool
    negative_variance: bool
    floor_applied: bool
    cap_applied: bool


@dataclass(frozen=True)
class DupireDiagnostics:
    """Aggregate counts of numerical interventions on a local-vol grid."""

    point_count: int
    unstable_denominator_count: int
    negative_variance_count: int
    floor_count: int
    cap_count: int
    minimum_denominator: float
    minimum_raw_variance: float
    maximum_raw_variance: float


@dataclass(frozen=True)
class LocalVolSurfaceGrid:
    """Local-volatility matrix with maturities on rows and strikes on columns."""

    maturities: NDArray[np.float64]
    strikes: NDArray[np.float64]
    local_volatilities: NDArray[np.float64]
    raw_local_variances: NDArray[np.float64]
    diagnostics: DupireDiagnostics


class DupireExtractor:
    """Extract local volatility under constant domestic/dividend rates.

    Calls are discounted prices ``C(T,K)`` generated from the supplied implied
    volatility surface. The implemented convention is

    ``sigma_loc^2 = [C_T + (r-q) K C_K + q C] / [0.5 K^2 C_KK]``.

    Derivatives use central finite differences. The time stencil shrinks to stay
    strictly inside ``T>0``; strike steps are relative and shrink to keep ``K>0``.
    SSVI's own extrapolation policy determines values outside its quote domain.
    """

    def __init__(
        self,
        surface: ImpliedVolatilitySurface,
        spot: float,
        rate: float,
        dividend_yield: float,
        config: DupireConfig | None = None,
    ) -> None:
        if not all(np.isfinite((spot, rate, dividend_yield))) or spot <= 0.0:
            raise ValueError("spot must be positive and all market inputs finite")
        self._surface = surface
        self._spot = spot
        self._rate = rate
        self._dividend_yield = dividend_yield
        self._config = config or DupireConfig()

    def call_price(self, maturity: float, strike: float) -> float:
        """Return the discounted call price used by the Dupire convention."""
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if not np.isfinite(strike) or strike <= 0.0:
            raise ValueError("strike must be finite and positive")
        forward = self._spot * np.exp((self._rate - self._dividend_yield) * maturity)
        discount = np.exp(-self._rate * maturity)
        log_moneyness = np.log(strike / forward)
        volatility = float(self._surface.implied_volatility(maturity, log_moneyness))
        return float(
            black_scholes_price(forward, strike, maturity, volatility, discount, "call")
        )

    def local_volatility(self, maturity: float, strike: float) -> LocalVolPoint:
        """Compute one regularized Dupire local-volatility point."""
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("maturity must be finite and positive")
        if not np.isfinite(strike) or strike <= 0.0:
            raise ValueError("strike must be finite and positive")
        time_step = min(self._config.time_step, 0.49 * maturity)
        strike_step = min(self._config.relative_strike_step * strike, 0.49 * strike)
        center = self.call_price(maturity, strike)
        earlier = self.call_price(maturity - time_step, strike)
        later = self.call_price(maturity + time_step, strike)
        lower = self.call_price(maturity, strike - strike_step)
        upper = self.call_price(maturity, strike + strike_step)
        derivative_time = (later - earlier) / (2.0 * time_step)
        derivative_strike = (upper - lower) / (2.0 * strike_step)
        derivative_strike_twice = (upper - 2.0 * center + lower) / strike_step**2
        numerator = (
            derivative_time
            + (self._rate - self._dividend_yield) * strike * derivative_strike
            + self._dividend_yield * center
        )
        denominator = 0.5 * strike**2 * derivative_strike_twice
        unstable = (
            not np.isfinite(denominator)
            or denominator <= self._config.denominator_floor
        )
        if unstable:
            raw_variance = np.nan
            regularized = self._config.local_variance_floor
        else:
            raw_variance = numerator / denominator
            regularized = raw_variance
        negative = bool(np.isfinite(raw_variance) and raw_variance < 0.0)
        floor_applied = bool(
            unstable
            or not np.isfinite(regularized)
            or regularized < self._config.local_variance_floor
        )
        if floor_applied:
            regularized = self._config.local_variance_floor
        cap_applied = bool(regularized > self._config.local_variance_cap)
        if cap_applied:
            regularized = self._config.local_variance_cap
        return LocalVolPoint(
            float(np.sqrt(regularized)),
            float(raw_variance),
            float(numerator),
            float(denominator),
            unstable,
            negative,
            floor_applied,
            cap_applied,
        )

    def extract_grid(
        self, maturities: ArrayLike, strikes: ArrayLike
    ) -> LocalVolSurfaceGrid:
        """Extract a validated grid and aggregate every numerical intervention."""
        times = np.asarray(maturities, dtype=float)
        strike_values = np.asarray(strikes, dtype=float)
        if times.ndim != 1 or times.size == 0 or np.any(times <= 0.0):
            raise ValueError("maturities must be a non-empty positive vector")
        if (
            strike_values.ndim != 1
            or strike_values.size == 0
            or np.any(strike_values <= 0.0)
        ):
            raise ValueError("strikes must be a non-empty positive vector")
        if np.any(~np.isfinite(times)) or np.any(~np.isfinite(strike_values)):
            raise ValueError("grid coordinates must be finite")
        local_vols = np.empty((times.size, strike_values.size))
        raw_variances = np.empty_like(local_vols)
        points: list[LocalVolPoint] = []
        for time_index, maturity in enumerate(times):
            for strike_index, strike in enumerate(strike_values):
                point = self.local_volatility(float(maturity), float(strike))
                local_vols[time_index, strike_index] = point.local_volatility
                raw_variances[time_index, strike_index] = point.raw_local_variance
                points.append(point)
        finite_raw = raw_variances[np.isfinite(raw_variances)]
        diagnostics = DupireDiagnostics(
            len(points),
            sum(point.denominator_unstable for point in points),
            sum(point.negative_variance for point in points),
            sum(point.floor_applied for point in points),
            sum(point.cap_applied for point in points),
            min(point.denominator for point in points),
            float(np.min(finite_raw)) if finite_raw.size else float("nan"),
            float(np.max(finite_raw)) if finite_raw.size else float("nan"),
        )
        return LocalVolSurfaceGrid(
            times.copy(),
            strike_values.copy(),
            local_vols,
            raw_variances,
            diagnostics,
        )
