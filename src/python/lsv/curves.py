"""Deterministic continuously-compounded rate curves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ZeroRateCurve:
    """Zero curve interpolated linearly in log discount factors.

    Times are year fractions. ``discount(t)`` extrapolates with the first or last
    continuously-compounded zero rate and returns one at time zero.
    """

    times: NDArray[np.float64]
    zero_rates: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        rates = np.asarray(self.zero_rates, dtype=float)
        if times.ndim != 1 or rates.ndim != 1 or times.size != rates.size:
            raise ValueError(
                "times and zero_rates must be one-dimensional and equal-sized"
            )
        if (
            times.size == 0
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(rates))
        ):
            raise ValueError("curve nodes must be non-empty and finite")
        if np.any(times <= 0.0) or np.any(np.diff(times) <= 0.0):
            raise ValueError("curve times must be strictly increasing and positive")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "zero_rates", rates.copy())

    @classmethod
    def flat(cls, rate: float, horizon: float = 100.0) -> ZeroRateCurve:
        """Construct a flat continuously-compounded curve."""
        if not np.isfinite(rate) or not np.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("rate must be finite and horizon must be positive")
        return cls(np.array([horizon]), np.array([rate]))

    def discount(self, time: ArrayLike) -> NDArray[np.float64]:
        """Return discount factors for non-negative year fractions."""
        value = np.asarray(time, dtype=float)
        if np.any(~np.isfinite(value)) or np.any(value < 0.0):
            raise ValueError("discount times must be finite and non-negative")
        node_log_df = -self.zero_rates * self.times
        log_df = np.interp(value, self.times, node_log_df)
        before = value < self.times[0]
        after = value > self.times[-1]
        log_df = np.where(before, -self.zero_rates[0] * value, log_df)
        log_df = np.where(after, -self.zero_rates[-1] * value, log_df)
        return np.where(value == 0.0, 1.0, np.exp(log_df))

    def zero_rate(self, time: ArrayLike) -> NDArray[np.float64]:
        """Return continuously-compounded zero rates."""
        value = np.asarray(time, dtype=float)
        discount = self.discount(value)
        safe_time = np.where(value == 0.0, 1.0, value)
        return np.asarray(
            np.where(value == 0.0, self.zero_rates[0], -np.log(discount) / safe_time),
            dtype=float,
        )


def forward_price(
    spot: float,
    maturity: ArrayLike,
    discount_curve: ZeroRateCurve,
    dividend_curve: ZeroRateCurve,
) -> NDArray[np.float64]:
    """Return equity forwards ``S0 * D_q(T) / D_r(T)``."""
    if not np.isfinite(spot) or spot <= 0.0:
        raise ValueError("spot must be finite and positive")
    time = np.asarray(maturity, dtype=float)
    return spot * dividend_curve.discount(time) / discount_curve.discount(time)
