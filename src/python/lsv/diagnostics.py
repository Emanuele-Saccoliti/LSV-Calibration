"""Static-arbitrage diagnostics for vanilla surfaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class StaticArbitrageReport:
    """Counts and magnitudes of numerical static-arbitrage violations."""

    strike_monotonicity_violations: int
    strike_convexity_violations: int
    calendar_violations: int
    maximum_monotonicity_violation: float
    maximum_convexity_violation: float
    maximum_calendar_violation: float

    @property
    def is_arbitrage_free(self) -> bool:
        """Whether no violation exceeds the configured tolerance."""
        return (
            self.strike_monotonicity_violations == 0
            and self.strike_convexity_violations == 0
            and self.calendar_violations == 0
        )


def diagnose_static_arbitrage(
    strikes: ArrayLike,
    call_prices: ArrayLike,
    total_variances: ArrayLike,
    *,
    tolerance: float = 1e-10,
) -> StaticArbitrageReport:
    """Check call monotonicity/convexity and calendar total variance.

    Price and variance matrices have shape ``(maturities, strikes)``. Convexity
    uses adjacent nonuniform-strike slopes.
    """
    strike = np.asarray(strikes, dtype=float)
    calls = np.asarray(call_prices, dtype=float)
    variances = np.asarray(total_variances, dtype=float)
    if strike.ndim != 1 or strike.size < 3 or np.any(np.diff(strike) <= 0.0):
        raise ValueError("strikes must be a strictly increasing vector of length >= 3")
    if (
        calls.ndim != 2
        or variances.shape != calls.shape
        or calls.shape[1] != strike.size
    ):
        raise ValueError("call_prices and total_variances must share shape (T, K)")
    if np.any(~np.isfinite(calls)) or np.any(~np.isfinite(variances)):
        raise ValueError("diagnostic inputs must be finite")
    price_diff = np.diff(calls, axis=1)
    slopes = price_diff / np.diff(strike)
    convex_diff = np.diff(slopes, axis=1)
    calendar_diff = np.diff(variances, axis=0)
    monotone_bad = price_diff > tolerance
    convex_bad = convex_diff < -tolerance
    calendar_bad = calendar_diff < -tolerance
    return StaticArbitrageReport(
        int(np.count_nonzero(monotone_bad)),
        int(np.count_nonzero(convex_bad)),
        int(np.count_nonzero(calendar_bad)),
        float(np.max(np.where(monotone_bad, price_diff, 0.0), initial=0.0)),
        float(np.max(np.where(convex_bad, -convex_diff, 0.0), initial=0.0)),
        float(np.max(np.where(calendar_bad, -calendar_diff, 0.0), initial=0.0)),
    )
