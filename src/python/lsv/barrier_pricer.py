"""Typed Python orchestration for C++ barrier and DNT pricing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import lsv_cpp
import numpy as np
from numpy.typing import ArrayLike


class BarrierType(StrEnum):
    """Barrier geometry."""

    LOWER = "lower"
    UPPER = "upper"
    DOUBLE = "double"


class BarrierPayoff(StrEnum):
    """Whether the maturity payout is conditional on touch or survival."""

    TOUCH = "touch"
    NO_TOUCH = "no_touch"


class Monitoring(StrEnum):
    """Barrier monitoring convention."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


@dataclass(frozen=True)
class BarrierPrice:
    """Monte Carlo estimate and 95% normal confidence interval."""

    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float]
    seed: int
    paths: int
    steps: int
    monitoring: Monitoring


def price_barrier(
    spots: ArrayLike,
    variances: ArrayLike,
    maturity: float,
    rate: float,
    payout: float,
    barrier_type: BarrierType | str,
    payoff_type: BarrierPayoff | str,
    monitoring: Monitoring | str,
    *,
    lower_barrier: float | None = None,
    upper_barrier: float | None = None,
    local_variances: ArrayLike | None = None,
    seed: int = 0,
    image_terms: int = 12,
) -> BarrierPrice:
    """Price a maturity-paid barrier claim from simulated path matrices.

    ``variances`` contains stochastic variance at path time points. For LSV,
    pass ``local_variances=L(t,S)*L(t,S)*V`` at the same points. Integrated
    interval variance is approximated by the trapezoidal rule. Continuous
    monitoring applies conditional Brownian-bridge survival weighting; discrete
    monitoring checks simulated endpoints only.
    """
    geometry = BarrierType(barrier_type)
    payoff = BarrierPayoff(payoff_type)
    monitoring_mode = Monitoring(monitoring)
    spot_paths = np.asarray(spots, dtype=float)
    variance_paths = np.asarray(variances, dtype=float)
    if spot_paths.ndim != 2 or variance_paths.shape != spot_paths.shape:
        raise ValueError("spots and variances must share shape (paths, steps+1)")
    if spot_paths.shape[0] < 2 or spot_paths.shape[1] < 2:
        raise ValueError("at least two paths and one time step are required")
    if np.any(~np.isfinite(spot_paths)) or np.any(spot_paths <= 0.0):
        raise ValueError("spot paths must be finite and positive")
    instantaneous = (
        variance_paths
        if local_variances is None
        else np.asarray(local_variances, dtype=float)
    )
    if instantaneous.shape != spot_paths.shape:
        raise ValueError("local_variances must match the spot path matrix")
    if np.any(~np.isfinite(instantaneous)) or np.any(instantaneous < 0.0):
        raise ValueError(
            "instantaneous local variances must be finite and non-negative"
        )
    if not np.isfinite(maturity) or maturity <= 0.0:
        raise ValueError("maturity must be finite and positive")
    steps = spot_paths.shape[1] - 1
    dt = maturity / steps
    interval_variances = 0.5 * (instantaneous[:, :-1] + instantaneous[:, 1:]) * dt
    lower = 0.0 if lower_barrier is None else lower_barrier
    upper = 0.0 if upper_barrier is None else upper_barrier
    raw = lsv_cpp.price_barrier(
        spot_paths,
        interval_variances,
        maturity,
        rate,
        payout,
        geometry.value,
        payoff.value,
        monitoring_mode.value,
        lower,
        upper,
        seed,
        image_terms,
    )
    return BarrierPrice(
        float(raw["estimate"]),
        float(raw["standard_error"]),
        (
            float(raw["confidence_interval_low"]),
            float(raw["confidence_interval_high"]),
        ),
        int(raw["seed"]),
        int(raw["paths"]),
        int(raw["steps"]),
        monitoring_mode,
    )


def price_dnt(
    spots: ArrayLike,
    variances: ArrayLike,
    maturity: float,
    rate: float,
    payout: float,
    lower_barrier: float,
    upper_barrier: float,
    *,
    monitoring: Monitoring | str = Monitoring.CONTINUOUS,
    local_variances: ArrayLike | None = None,
    seed: int = 0,
) -> BarrierPrice:
    """Price a maturity-paid Double No-Touch claim."""
    return price_barrier(
        spots,
        variances,
        maturity,
        rate,
        payout,
        BarrierType.DOUBLE,
        BarrierPayoff.NO_TOUCH,
        monitoring,
        lower_barrier=lower_barrier,
        upper_barrier=upper_barrier,
        local_variances=local_variances,
        seed=seed,
    )
