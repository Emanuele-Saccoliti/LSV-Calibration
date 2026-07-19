# Mathematical methodology

## Dupire convention

The implementation treats `C(T,K)` as the time-zero, domestically discounted
European call price. With constant continuously-compounded domestic rate `r` and
dividend yield `q`, local variance is extracted as

```text
sigma_loc(T,K)^2 =
    [C_T + (r-q) K C_K + q C] / [0.5 K^2 C_KK].
```

Calls are generated from the fitted implied-volatility surface using
`F(T)=S0 exp((r-q)T)` and `D(T)=exp(-rT)`. Central finite differences are used in
time and strike. The time stencil contracts near zero maturity so it never
evaluates at a non-positive time; the strike stencil contracts to remain at a
positive strike. Surface extrapolation is inherited from SSVI: total ATM
variance scales linearly toward zero before its first node and remains flat after
its final node.

The denominator floor, local-variance floor, and cap are configurable. Every
application is counted in `DupireDiagnostics`; negative raw variance and an
unstable density denominator are never silently discarded. Values beyond the
last SSVI maturity should generally be excluded because flat total-variance
extrapolation implies vanishing incremental variance.

## Heston pricing and calibration

The Heston characteristic function is defined for `log(S_T)` under the
risk-neutral measure with constant rates. The implementation uses the stable
`exp(-dT)` representation and selects the square-root branch with non-negative
real part. European prices use the standard Fourier `P1/P2` representation with
configurable integration cutoff and tolerances.

Calibration minimizes weighted price residuals normalized by spot. Optional
variance-swap observations use the analytic expected average variance

```text
theta + (v0-theta) * (1-exp(-kappa*T)) / (kappa*T).
```

The Feller condition is reported and is not imposed by default. Optimizer
failure raises an exception rather than returning an apparently valid parameter
set.

## Heston QE-M simulation

The C++ engine implements Andersen's quadratic-exponential scheme. At every time
step it matches the exact conditional CIR mean and variance, then selects either
the quadratic Gaussian representation or the atom-at-zero plus exponential
representation according to configurable `psi_c` (default `1.5`). Variance is
therefore non-negative without full-truncation Euler clipping.

The log-spot update uses the variance endpoints and the independent Gaussian
innovation with `gamma1=gamma2=0.5`. Its conditional drift subtracts the exact
log moment-generating function of the selected QE variance law. This is the
martingale correction in QE-M; the implementation is not log-Euler. Rates,
initial variance, path count, step count, seed, and antithetic sampling are all
explicit API inputs. Returned matrices have row-major shape
`(paths, steps + 1)`.

## Leverage fixed point

The leverage surface is represented on `(time, forward log-moneyness)` and
interpolated bilinearly, with constant boundary extrapolation. LSV simulation
freezes leverage at the start of each step and calls the same QE-M kernel used by
pure Heston; the correlated and independent spot coefficients are scaled by the
frozen leverage value.

At each global iteration the C++ engine estimates

```text
E[V_t | X_t=x] = sum(V_n K_h(X_n-x)) / sum(K_h(X_n-x)),
X_t = log(S_t/F_t),
```

with a Gaussian Nadaraya--Watson kernel. Effective sample size is
`sum(w)^2/sum(w^2)`. Low-density nodes are flagged; if the denominator falls
below its configurable floor, the nearest particle variance provides an explicit
boundary fallback. The update is exactly

```text
L_raw(t,x) = sigma_loc(t,x) / sqrt(max(E[V_t|X_t=x], epsilon)).
```

It is then smoothed across log-moneyness, damped, and clipped to configured
bounds. The convergence norm, effective sample size, low-density nodes, and clip
counts are retained for every iteration. This is particle kernel regression, not
a particle filter or density-ratio method.

## Barrier and Double No-Touch pricing

Barrier claims are paid at maturity in the currently supported convention.
Discrete monitoring checks simulated path endpoints. Continuous monitoring uses
conditional Brownian-bridge survival weighting in log spot. For a lower barrier
`b` and endpoint log spots `x,y`, interval survival is

```text
1 - exp(-2 (x-b)(y-b) / integrated_variance),
```

with the analogous upper-barrier formula. Double-barrier survival uses the exact
absorbed Brownian transition density divided by the unrestricted density,
evaluated through a configurable symmetric image series. This avoids treating
lower and upper crossings as independent events.

The interval integrated variance is the trapezoidal approximation of `V` for
Heston or `L(t,S)^2 V` for LSV. Survival probabilities are multiplied along the
path and used as conditional Monte Carlo weights. Touch uses one minus survival;
no-touch uses survival. Results contain the estimate, standard error, 95% normal
confidence interval, seed, path count, and time-step count. Rebates paid at the
touch time are not yet supported.

## Exotic outer calibration

The outer calibration keeps the local-volatility target fixed. For every
candidate stochastic-volatility parameter vector it recalibrates the leverage
fixed point, simulates LSV paths, and prices the requested DNT instruments. A
leverage/evaluation cache keyed by the full Heston parameter vector avoids
repeating identical nested work. Fixed calibration and pricing seeds provide
common random numbers across candidates.

The objective uses payout-normalized, weighted DNT price residuals. Parameters
included in the outer optimization and their bounds are explicit. The final
report contains price and implied-survival-probability residuals, residuals in
Monte Carlo standard-error units, vanilla repricing before and after, optimizer
status, evaluations, cache statistics, elapsed wall time, and quote provenance.
Every DNT quote must be marked `market` or `synthetic`; synthetic benchmark data
is never described as observed market data.

This is a noisy nested optimization even with common random numbers. Production
experiments should run particle/path/bandwidth stability studies and compare the
reported residuals with their Monte Carlo standard errors.
