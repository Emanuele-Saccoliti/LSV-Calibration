"""Power-law SSVI implied total-variance surface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SSVISurface:
    """Arbitrage-controlled SSVI surface in forward log-moneyness.

    ``k = log(K/F(T))`` and ``w(k,T) = sigma_imp(k,T)^2 T``. ATM total
    variance is linearly interpolated, scales linearly to zero before the first
    expiry, and is held flat beyond the final expiry. The power-law shape is
    ``phi(theta)=eta / (theta**gamma * (1+theta)**(1-gamma))``.
    """

    maturities: NDArray[np.float64]
    atm_total_variances: NDArray[np.float64]
    rho: float
    eta: float
    gamma: float

    def __post_init__(self) -> None:
        times = np.asarray(self.maturities, dtype=float)
        theta = np.asarray(self.atm_total_variances, dtype=float)
        if times.ndim != 1 or theta.ndim != 1 or times.size != theta.size:
            raise ValueError("maturities and ATM variances must be equal-sized vectors")
        if times.size == 0 or np.any(~np.isfinite(times)) or np.any(times <= 0.0):
            raise ValueError("maturities must be finite and positive")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("maturities must be strictly increasing")
        if np.any(~np.isfinite(theta)) or np.any(theta <= 0.0):
            raise ValueError("ATM total variances must be finite and positive")
        if np.any(np.diff(theta) < -1e-12):
            raise ValueError("ATM total variance must be non-decreasing")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie strictly between -1 and 1")
        if not np.isfinite(self.eta) or self.eta <= 0.0:
            raise ValueError("eta must be finite and positive")
        if not np.isfinite(self.gamma) or not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1]")
        object.__setattr__(self, "maturities", times.copy())
        object.__setattr__(self, "atm_total_variances", theta.copy())
        violations = self.sufficient_butterfly_violations()
        if np.any(violations > 1e-10):
            raise ValueError(
                "SSVI parameters violate sufficient no-butterfly conditions; "
                f"maximum violation={float(np.max(violations)):.6g}"
            )

    def atm_total_variance(self, maturity: ArrayLike) -> NDArray[np.float64]:
        """Interpolate ATM total variance using the documented boundaries."""
        time = np.asarray(maturity, dtype=float)
        if np.any(~np.isfinite(time)) or np.any(time <= 0.0):
            raise ValueError("maturity must be finite and positive")
        theta = np.interp(time, self.maturities, self.atm_total_variances)
        before = time < self.maturities[0]
        scaled = self.atm_total_variances[0] * time / self.maturities[0]
        return np.asarray(np.where(before, scaled, theta), dtype=float)

    def phi(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the power-law SSVI shape function."""
        value = np.asarray(theta, dtype=float)
        return np.asarray(
            self.eta
            / (np.power(value, self.gamma) * np.power(1.0 + value, 1.0 - self.gamma)),
            dtype=float,
        )

    def total_variance(
        self, maturity: ArrayLike, log_moneyness: ArrayLike
    ) -> NDArray[np.float64]:
        """Return SSVI total variance for broadcastable ``(T, log(K/F))``."""
        time, k = np.broadcast_arrays(
            np.asarray(maturity, dtype=float), np.asarray(log_moneyness, dtype=float)
        )
        if np.any(~np.isfinite(k)):
            raise ValueError("log-moneyness must be finite")
        theta = self.atm_total_variance(time)
        phi = self.phi(theta)
        root = np.sqrt((phi * k + self.rho) ** 2 + 1.0 - self.rho**2)
        return np.asarray(0.5 * theta * (1.0 + self.rho * phi * k + root), dtype=float)

    def implied_volatility(
        self, maturity: ArrayLike, log_moneyness: ArrayLike
    ) -> NDArray[np.float64]:
        """Return annualized Black implied volatility."""
        time = np.asarray(maturity, dtype=float)
        return np.asarray(
            np.sqrt(self.total_variance(time, log_moneyness) / time), dtype=float
        )

    def sufficient_butterfly_violations(self) -> NDArray[np.float64]:
        """Return positive violations of two sufficient SSVI wing conditions."""
        theta = self.atm_total_variances
        phi = self.phi(theta)
        factor = 1.0 + abs(self.rho)
        first = theta * phi * factor - 4.0
        second = theta * phi**2 * factor - 4.0
        return np.maximum(np.concatenate((first, second)), 0.0)


def fit_ssvi_surface(
    maturities: ArrayLike,
    log_moneyness: ArrayLike,
    implied_volatilities: ArrayLike,
    *,
    max_evaluations: int = 10_000,
) -> SSVISurface:
    """Fit SSVI by nonlinear least squares with monotone ATM variance.

    All arrays are one-dimensional quote vectors. The objective uses total
    variance residuals and soft penalties for the sufficient butterfly bounds;
    the returned object revalidates those bounds exactly.
    """
    time, k, vol = (
        np.asarray(maturities, dtype=float),
        np.asarray(log_moneyness, dtype=float),
        np.asarray(implied_volatilities, dtype=float),
    )
    if time.ndim != 1 or k.ndim != 1 or vol.ndim != 1:
        raise ValueError("fit inputs must be one-dimensional")
    if time.size < 6 or time.size != k.size or time.size != vol.size:
        raise ValueError("at least six equal-sized SSVI quotes are required")
    if np.any(~np.isfinite([time, k, vol])):
        raise ValueError("SSVI quotes must be finite")
    if np.any(time <= 0.0) or np.any(vol <= 0.0):
        raise ValueError("maturities and implied volatilities must be positive")
    unique_times = np.unique(time)
    if unique_times.size < 2:
        raise ValueError("SSVI fitting requires at least two maturities")
    observed_w = vol**2 * time
    theta_guess = np.array(
        [
            observed_w[time == expiry][np.argmin(np.abs(k[time == expiry]))]
            for expiry in unique_times
        ]
    )
    theta_guess = np.maximum.accumulate(np.maximum(theta_guess, 1e-8))
    increments = np.diff(np.concatenate(([0.0], theta_guess)))
    x0 = np.concatenate(
        (np.log(np.maximum(increments, 1e-6)), np.array([0.0, np.log(0.5), 0.0]))
    )

    def unpack(
        parameters: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, float, float]:
        theta = np.cumsum(np.exp(parameters[: unique_times.size]))
        rho = float(np.tanh(parameters[-3]))
        eta = float(np.exp(parameters[-2]))
        gamma = float(1.0 / (1.0 + np.exp(-parameters[-1])))
        return theta, rho, eta, gamma

    def residuals(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        theta, rho, eta, gamma = unpack(parameters)
        quote_theta = np.interp(time, unique_times, theta)
        phi = eta / (quote_theta**gamma * (1.0 + quote_theta) ** (1.0 - gamma))
        model_w = (
            0.5
            * quote_theta
            * (1.0 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1.0 - rho**2))
        )
        node_phi = eta / (theta**gamma * (1.0 + theta) ** (1.0 - gamma))
        factor = 1.0 + abs(rho)
        penalties = 10.0 * np.maximum(
            np.concatenate(
                (
                    theta * node_phi * factor - 3.999,
                    theta * node_phi**2 * factor - 3.999,
                )
            ),
            0.0,
        )
        return np.concatenate((model_w - observed_w, penalties))

    result = least_squares(residuals, x0, max_nfev=max_evaluations)
    if not result.success:
        raise RuntimeError(f"SSVI optimization failed: {result.message}")
    theta, rho, eta, gamma = unpack(result.x)
    return SSVISurface(unique_times, theta, rho, eta, gamma)
