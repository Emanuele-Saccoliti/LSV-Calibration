"""Black--Scholes pricing under deterministic rates and dividends."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import ndtr


class OptionType(StrEnum):
    """European vanilla payoff type."""

    CALL = "call"
    PUT = "put"


def black_scholes_price(
    forward: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    volatility: ArrayLike,
    discount_factor: ArrayLike = 1.0,
    option_type: OptionType | str = OptionType.CALL,
) -> NDArray[np.float64]:
    """Price a European option from its forward and domestic discount factor.

    Volatility is annualized and maturity is a year fraction. Inputs broadcast.
    At zero maturity or volatility the discounted intrinsic value is returned.
    """
    kind = OptionType(option_type)
    fwd, k, time, vol, df = np.broadcast_arrays(
        np.asarray(forward, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(volatility, dtype=float),
        np.asarray(discount_factor, dtype=float),
    )
    if np.any(~np.isfinite([fwd, k, time, vol, df])):
        raise ValueError("Black--Scholes inputs must be finite")
    if np.any(fwd <= 0.0) or np.any(k <= 0.0):
        raise ValueError("forward and strike must be positive")
    if np.any(time < 0.0) or np.any(vol < 0.0):
        raise ValueError("maturity and volatility must be non-negative")
    if np.any(df <= 0.0) or np.any(df > 1.0 + 1e-12):
        raise ValueError("discount factors must lie in (0, 1]")
    sign = 1.0 if kind is OptionType.CALL else -1.0
    stddev = vol * np.sqrt(time)
    safe_stddev = np.where(stddev > 0.0, stddev, 1.0)
    d1 = np.log(fwd / k) / safe_stddev + 0.5 * safe_stddev
    d2 = d1 - safe_stddev
    regular = df * sign * (fwd * ndtr(sign * d1) - k * ndtr(sign * d2))
    intrinsic = df * np.maximum(sign * (fwd - k), 0.0)
    return np.asarray(np.where(stddev > 0.0, regular, intrinsic), dtype=float)


def implied_volatility(
    price: float,
    forward: float,
    strike: float,
    maturity: float,
    discount_factor: float = 1.0,
    option_type: OptionType | str = OptionType.CALL,
    *,
    lower_vol: float = 1e-9,
    upper_vol: float = 5.0,
    tolerance: float = 1e-12,
) -> float:
    """Invert a discounted Black--Scholes price using safeguarded Brent search."""
    kind = OptionType(option_type)
    values = (price, forward, strike, maturity, discount_factor)
    if not all(np.isfinite(values)):
        raise ValueError("implied-volatility inputs must be finite")
    if forward <= 0.0 or strike <= 0.0 or maturity <= 0.0:
        raise ValueError("forward, strike, and maturity must be positive")
    if discount_factor <= 0.0 or discount_factor > 1.0 + 1e-12:
        raise ValueError("discount_factor must lie in (0, 1]")
    sign = 1.0 if kind is OptionType.CALL else -1.0
    intrinsic = discount_factor * max(sign * (forward - strike), 0.0)
    maximum = discount_factor * (forward if kind is OptionType.CALL else strike)
    scale = max(1.0, maximum)
    if price < intrinsic - tolerance * scale or price >= maximum:
        raise ValueError(
            f"price {price} violates no-arbitrage bounds [{intrinsic}, {maximum})"
        )
    if price <= intrinsic + tolerance * scale:
        return 0.0

    def residual(volatility: float) -> float:
        return float(
            black_scholes_price(
                forward, strike, maturity, volatility, discount_factor, kind
            )
            - price
        )

    if residual(upper_vol) < 0.0:
        raise ValueError(f"implied volatility exceeds configured cap {upper_vol}")
    return float(brentq(residual, lower_vol, upper_vol, xtol=tolerance, rtol=1e-14))
