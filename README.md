# LSV Calibration & Exotic Pricing Engine

This project is a hybrid Python/C++ engine for calibrating a
**Local-Stochastic Volatility (LSV)** model. It starts from vanilla option
prices, builds an arbitrage-controlled volatility surface, calibrates the LSV
dynamics, and uses the resulting model to price exotic derivatives.

The engine includes:

- validation of vanilla option quotes and construction of deterministic rate
  and dividend curves;
- Black--Scholes pricing and implied-volatility inversion;
- calibration of a power-law SSVI surface with static-arbitrage diagnostics;
- extraction of Dupire local volatility, including diagnostics for numerical
  regularization;
- semi-analytic Heston pricing through Fourier integration and weighted model
  calibration, with optional variance-swap targets;
- Andersen QE-M Heston simulation in the compiled C++ backend, with antithetic
  sampling and conditional martingale correction;
- LSV leverage calibration through particle kernel regression and fixed-point
  iteration, with convergence and low-density diagnostics;
- pricing of barrier, touch, no-touch, and double-no-touch (DNT) options with
  either discrete monitoring or continuous Brownian-bridge correction;
- nested calibration of stochastic-volatility parameters to DNT quotes, using
  cached leverage surfaces and common random numbers to reduce Monte Carlo
  noise.

## Python and C++ responsibilities

Python manages the calibration workflow and the higher-level model logic. It
is responsible for:

- loading the configuration and validating market inputs;
- generating or importing vanilla option quotes;
- Black--Scholes pricing and implied-volatility inversion;
- fitting the SSVI implied-volatility surface and checking static arbitrage;
- extracting the Dupire local-volatility surface;
- Fourier-based Heston pricing and parameter calibration;
- controlling the fixed-point iterations used to calibrate the LSV leverage
  surface;
- running the nested exotic calibration, producing diagnostics, saving results,
  and generating plots.

C++ implements the computationally intensive numerical kernels exposed to
Python through the `lsv_cpp` extension. It is responsible for:

- simulating Heston and LSV spot and variance paths with the Andersen QE-M
  scheme;
- applying antithetic sampling and conditional martingale correction;
- interpolating the leverage surface along each simulated LSV path;
- estimating the conditional variance `E[V(t) | S(t) = S]` through Gaussian
  kernel regression, which is used by the leverage calibration;
- detecting barrier events with discrete monitoring or continuous
  Brownian-bridge correction;
- computing Monte Carlo prices, standard errors, and confidence intervals for
  barrier, touch, no-touch, and DNT products.

In short, Python builds and calibrates the model, while C++ performs the most
expensive path simulation, conditional-expectation, and barrier-pricing
calculations.

## Requirements

- Python 3.11 or newer
- CMake 3.20 or newer
- a C++20 compiler

## Build and test

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -c "import lsv, lsv_cpp"
pytest -q

cmake -S . -B build/cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DLSV_BUILD_PYTHON=OFF
cmake --build build/cmake --parallel
ctest --test-dir build/cmake --output-on-failure
```

## Run the complete engine

After the installation, the entire workflow can be launch by

```bash
lsv-run
```

It generates synthetic vanilla quotes, calibrates SSVI and Heston, extracts the
Dupire surface, calibrates the LSV leverage function, simulates paths, prices a
DNT, runs the synthetic exotic outer calibration, and opens the calibrated 3D
surfaces. The terminal prints all 15 synthetic-versus-Heston vanilla prices,
the target-versus-LSV call repricing, and the DNT price with its Monte Carlo
standard error. Closing the plot window completes the command.

Outputs are stored in `results/`:

- `calibration_summary.json`: parameters and numerical diagnostics;
- `calibrated_surfaces.npz`: implied vol, local vol, and leverage arrays;
- `calibrated_surfaces_3d.png`: the three calibrated 3D surfaces.

For a headless run that only saves the chart:

```bash
lsv-run --no-show
```

For a faster smoke run:

```bash
lsv-run --quick --no-show --skip-exotic
```

The same entry point is available as `python -m lsv` and through
`python scripts/run_calibration.py`.

The standalone CMake command disables the Python extension because the editable
installation has already built it. To build the extension directly with CMake,
install pybind11 and pass its CMake package directory through `CMAKE_PREFIX_PATH`.

## Minimal Heston calibration example

The calibration routine can also be used directly from Python without running the
complete LSV workflow. The following example calibrates Heston parameters to a
small set of vanilla option prices.

```python
from lsv.heston_calibration import (
    HestonCalibrationQuote,
    calibrate_heston,
)
from lsv.heston_pricer import HestonParameters

quotes = [
    HestonCalibrationQuote(maturity=1.0, strike=90.0, price=14.2),
    HestonCalibrationQuote(maturity=1.0, strike=100.0, price=8.5),
    HestonCalibrationQuote(maturity=1.0, strike=110.0, price=4.3),
]

initial = HestonParameters(
    kappa=2.0,
    theta=0.04,
    eta=0.5,
    rho=-0.7,
    v0=0.04,
)

report = calibrate_heston(
    spot=100.0,
    rate=0.02,
    dividend_yield=0.0,
    quotes=quotes,
    initial=initial,
)

print(report.parameters)
print(report.objective_value)
```

## Mathematical conventions

- maturities are year fractions and volatilities are annualized;
- rates and dividend yields are continuously compounded;
- forwards satisfy `F(T) = S0 Dq(T) / Dr(T)`;
- Black prices are discounted by the domestic discount factor;
- SSVI uses forward log-moneyness `k = log(K/F(T))` and total variance
  `w(k,T) = sigma_imp(k,T)^2 T`;
- ATM variance scales linearly to zero before the first SSVI maturity and remains
  flat beyond the last maturity.
- Dupire uses discounted calls and includes all constant-rate drift/dividend
  terms; see [docs/methodology.md](docs/methodology.md).

## Current limitations:

- touch-time rebates are not supported;
- non-maturity payment conventions are not supported;
- the included exotic calibration benchmark is synthetic.
