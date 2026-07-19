"""Command-line entry point for the complete synthetic LSV demonstration."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import lsv_cpp
import numpy as np
from matplotlib import pyplot as plt

from lsv.barrier_pricer import Monitoring, price_dnt
from lsv.black_scholes import black_scholes_price, implied_volatility
from lsv.config import ConfigValue, load_config
from lsv.diagnostics import diagnose_static_arbitrage
from lsv.dupire import DupireConfig, DupireExtractor
from lsv.exotic_calibration import (
    DNTCalibrationQuote,
    ExoticCalibrationConfig,
    QuoteSource,
    VanillaRepricingTarget,
    calibrate_exotics,
)
from lsv.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationQuote,
    VarianceSwapQuote,
    calibrate_heston,
)
from lsv.heston_pricer import (
    HestonParameters,
    expected_average_variance,
    heston_price,
)
from lsv.leverage_calibration import (
    LeverageCalibrationConfig,
    calibrate_leverage,
)
from lsv.ssvi_surface import fit_ssvi_surface

LOGGER = logging.getLogger("lsv.cli")


@dataclass(frozen=True)
class VanillaPriceComparison:
    """One discounted vanilla price used in the terminal calibration report."""

    maturity: float
    log_moneyness: float
    strike: float
    market_price: float
    heston_price: float


@dataclass(frozen=True)
class EngineArtifacts:
    """Paths and headline results produced by one complete engine run."""

    output_directory: Path
    summary_path: Path
    surface_data_path: Path
    figure_path: Path
    heston_parameters: HestonParameters
    vanilla_prices: tuple[VanillaPriceComparison, ...]
    repricing_maturity: float
    repricing_strike: float
    target_call_price: float
    lsv_call_price: float
    dnt_price: float
    dnt_standard_error: float
    leverage_converged: bool


def _section(config: Mapping[str, ConfigValue], name: str) -> Mapping[str, ConfigValue]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"configuration section '{name}' must be a mapping")
    return value


def _float(section: Mapping[str, ConfigValue], name: str) -> float:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"configuration value '{name}' must be numeric")
    return float(value)


def _integer(section: Mapping[str, ConfigValue], name: str) -> int:
    value = section.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"configuration value '{name}' must be an integer")
    return value


def _string(section: Mapping[str, ConfigValue], name: str) -> str:
    value = section.get(name)
    if not isinstance(value, str):
        raise ValueError(f"configuration value '{name}' must be a string")
    return value


def _boolean(section: Mapping[str, ConfigValue], name: str) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        raise ValueError(f"configuration value '{name}' must be boolean")
    return value


def _float_tuple(section: Mapping[str, ConfigValue], name: str) -> tuple[float, ...]:
    value = section.get(name)
    if not isinstance(value, tuple) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"configuration value '{name}' must be a numeric list")
    return tuple(float(cast(int | float, item)) for item in value)


def _string_tuple(section: Mapping[str, ConfigValue], name: str) -> tuple[str, ...]:
    value = section.get(name)
    if not isinstance(value, tuple) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"configuration value '{name}' must be a string list")
    return tuple(cast(str, item) for item in value)


def _heston_parameters(section: Mapping[str, ConfigValue]) -> HestonParameters:
    return HestonParameters(
        _float(section, "kappa"),
        _float(section, "theta"),
        _float(section, "eta"),
        _float(section, "rho"),
        _float(section, "v0"),
    )


def _format_price_report(artifacts: EngineArtifacts) -> str:
    """Format calibration and LSV prices as a terminal-friendly table."""
    lines = [
        "",
        "Vanilla prices (discounted calls)",
        f"{'T':>6} {'log(K/F)':>10} {'Strike':>11} "
        f"{'Synthetic':>13} {'Heston':>13} {'Difference':>13}",
    ]
    for price in artifacts.vanilla_prices:
        difference = price.heston_price - price.market_price
        lines.append(
            f"{price.maturity:6.2f} {price.log_moneyness:10.3f} "
            f"{price.strike:11.4f} {price.market_price:13.6f} "
            f"{price.heston_price:13.6f} {difference:13.3e}"
        )
    lines.extend(
        (
            "",
            "LSV pricing",
            f"European call target  "
            f"(T={artifacts.repricing_maturity:.2f}, "
            f"K={artifacts.repricing_strike:.2f}): "
            f"{artifacts.target_call_price:.8f}",
            f"European call LSV     "
            f"(T={artifacts.repricing_maturity:.2f}, "
            f"K={artifacts.repricing_strike:.2f}): "
            f"{artifacts.lsv_call_price:.8f}",
            f"Call repricing error: "
            f"{artifacts.lsv_call_price - artifacts.target_call_price:+.8f}",
            f"DNT price: {artifacts.dnt_price:.8f} "
            f"(standard error {artifacts.dnt_standard_error:.8f})",
        )
    )
    return "\n".join(lines)


def _plot_surfaces(
    figure_path: Path,
    quote_times: np.ndarray,
    quote_log_moneyness: np.ndarray,
    quote_volatilities: np.ndarray,
    implied_times: np.ndarray,
    implied_log_moneyness: np.ndarray,
    implied_volatility: np.ndarray,
    local_times: np.ndarray,
    local_log_moneyness: np.ndarray,
    local_volatility: np.ndarray,
    leverage: np.ndarray,
    show: bool,
) -> None:
    implied_time_mesh, implied_moneyness_mesh = np.meshgrid(
        implied_times, implied_log_moneyness, indexing="ij"
    )
    local_time_mesh, local_moneyness_mesh = np.meshgrid(
        local_times, local_log_moneyness, indexing="ij"
    )
    figure = plt.figure(figsize=(18, 6), constrained_layout=True)
    implied_axis = figure.add_subplot(1, 3, 1, projection="3d")
    implied_axis.plot_surface(
        implied_moneyness_mesh,
        implied_time_mesh,
        implied_volatility,
        cmap="viridis",
        alpha=0.88,
    )
    implied_axis.scatter(
        quote_log_moneyness,
        quote_times,
        quote_volatilities,
        color="crimson",
        s=24,
        label="synthetic quotes",
    )
    implied_axis.set_title("Calibrated implied volatility")
    implied_axis.set_xlabel("log(K/F)")
    implied_axis.set_ylabel("maturity")
    implied_axis.set_zlabel("volatility")
    implied_axis.legend(loc="upper right")

    local_axis = figure.add_subplot(1, 3, 2, projection="3d")
    local_axis.plot_surface(
        local_moneyness_mesh,
        local_time_mesh,
        local_volatility,
        cmap="plasma",
        alpha=0.9,
    )
    local_axis.set_title("Dupire local volatility")
    local_axis.set_xlabel("log(K/F)")
    local_axis.set_ylabel("maturity")
    local_axis.set_zlabel("volatility")

    leverage_axis = figure.add_subplot(1, 3, 3, projection="3d")
    leverage_axis.plot_surface(
        local_moneyness_mesh,
        local_time_mesh,
        leverage,
        cmap="cividis",
        alpha=0.9,
    )
    leverage_axis.set_title("Calibrated leverage")
    leverage_axis.set_xlabel("log(S/F)")
    leverage_axis.set_ylabel("maturity")
    leverage_axis.set_zlabel("leverage")

    figure.savefig(figure_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def run_engine(
    config_path: str | Path = "configs/calibration.yaml",
    *,
    show_plots: bool | None = None,
    quick: bool = False,
    skip_exotic: bool = False,
    print_prices: bool = False,
) -> EngineArtifacts:
    """Execute the full reproducible synthetic calibration and pricing chain."""
    config = load_config(config_path)
    experiment = _section(config, "experiment")
    market = _section(config, "market")
    synthetic = _section(config, "synthetic_market")
    heston_settings = _section(config, "heston_calibration")
    surface_settings = _section(config, "surface")
    dupire_settings = _section(config, "dupire")
    leverage_settings = _section(config, "leverage_calibration")
    pricing_settings = _section(config, "pricing")
    exotic_settings = _section(config, "exotic_calibration")

    spot = _float(market, "spot")
    rate = _float(market, "rate")
    dividend_yield = _float(market, "dividend_yield")
    carry = rate - dividend_yield
    seed = _integer(experiment, "seed")
    output_directory = Path(_string(experiment, "output_directory"))
    output_directory.mkdir(parents=True, exist_ok=True)
    should_show = (
        _boolean(experiment, "show_plots") if show_plots is None else show_plots
    )
    LOGGER.info("Generating explicitly synthetic vanilla quotes")

    reference_parameters = _heston_parameters(_section(synthetic, "heston"))
    quote_maturities = np.asarray(_float_tuple(synthetic, "maturities"))
    quote_moneyness_nodes = np.asarray(_float_tuple(synthetic, "log_moneyness"))
    quote_times = np.repeat(quote_maturities, quote_moneyness_nodes.size)
    quote_log_moneyness = np.tile(quote_moneyness_nodes, quote_maturities.size)
    quote_forwards = spot * np.exp(carry * quote_times)
    quote_strikes = quote_forwards * np.exp(quote_log_moneyness)
    quote_discounts = np.exp(-rate * quote_times)
    quote_prices = np.array(
        [
            heston_price(
                spot,
                float(strike),
                float(maturity),
                rate,
                dividend_yield,
                reference_parameters,
            )
            for maturity, strike in zip(quote_times, quote_strikes, strict=True)
        ]
    )
    quote_volatilities = np.array(
        [
            implied_volatility(
                float(price),
                float(forward),
                float(strike),
                float(maturity),
                float(discount),
            )
            for price, forward, strike, maturity, discount in zip(
                quote_prices,
                quote_forwards,
                quote_strikes,
                quote_times,
                quote_discounts,
                strict=True,
            )
        ]
    )

    LOGGER.info("Fitting the arbitrage-controlled SSVI surface")
    ssvi = fit_ssvi_surface(quote_times, quote_log_moneyness, quote_volatilities)
    plot_times = np.linspace(
        float(np.min(quote_maturities)),
        float(np.max(quote_maturities)),
        _integer(surface_settings, "plot_maturities"),
    )
    plot_moneyness = np.linspace(
        _float(surface_settings, "minimum_log_moneyness"),
        _float(surface_settings, "maximum_log_moneyness"),
        _integer(surface_settings, "plot_log_moneyness"),
    )
    plot_time_mesh, plot_moneyness_mesh = np.meshgrid(
        plot_times, plot_moneyness, indexing="ij"
    )
    fitted_implied_volatility = ssvi.implied_volatility(
        plot_time_mesh, plot_moneyness_mesh
    )
    fixed_strikes = spot * np.exp(plot_moneyness)
    diagnostic_time_mesh, diagnostic_strike_mesh = np.meshgrid(
        plot_times, fixed_strikes, indexing="ij"
    )
    diagnostic_forwards = spot * np.exp(carry * diagnostic_time_mesh)
    diagnostic_moneyness = np.log(diagnostic_strike_mesh / diagnostic_forwards)
    diagnostic_variance = ssvi.total_variance(
        diagnostic_time_mesh, diagnostic_moneyness
    )
    diagnostic_volatility = np.sqrt(diagnostic_variance / diagnostic_time_mesh)
    diagnostic_calls = black_scholes_price(
        diagnostic_forwards,
        diagnostic_strike_mesh,
        diagnostic_time_mesh,
        diagnostic_volatility,
        np.exp(-rate * diagnostic_time_mesh),
    )
    arbitrage = diagnose_static_arbitrage(
        fixed_strikes, diagnostic_calls, diagnostic_variance, tolerance=2e-7
    )

    LOGGER.info("Extracting the Dupire local-volatility surface")
    local_maturity = _float(dupire_settings, "maturity")
    local_steps = _integer(dupire_settings, "time_steps")
    local_times = np.linspace(local_maturity / local_steps, local_maturity, local_steps)
    local_moneyness = np.linspace(
        _float(dupire_settings, "minimum_log_moneyness"),
        _float(dupire_settings, "maximum_log_moneyness"),
        _integer(dupire_settings, "log_moneyness_nodes"),
    )
    dupire = DupireExtractor(
        ssvi,
        spot,
        rate,
        dividend_yield,
        DupireConfig(
            time_step=_float(dupire_settings, "time_step"),
            relative_strike_step=_float(dupire_settings, "relative_strike_step"),
            denominator_floor=_float(dupire_settings, "denominator_floor"),
            local_variance_floor=_float(dupire_settings, "local_variance_floor"),
            local_variance_cap=_float(dupire_settings, "local_variance_cap"),
        ),
    )
    local_volatility = np.empty((local_steps, local_moneyness.size))
    dupire_floor_count = 0
    dupire_cap_count = 0
    dupire_unstable_count = 0
    for time_index, maturity in enumerate(local_times):
        forward = spot * np.exp(carry * maturity)
        for strike_index, log_moneyness in enumerate(local_moneyness):
            point = dupire.local_volatility(
                float(maturity), float(forward * np.exp(log_moneyness))
            )
            local_volatility[time_index, strike_index] = point.local_volatility
            dupire_floor_count += int(point.floor_applied)
            dupire_cap_count += int(point.cap_applied)
            dupire_unstable_count += int(point.denominator_unstable)

    LOGGER.info("Calibrating Heston to the synthetic vanilla quotes")
    initial_parameters = _heston_parameters(_section(heston_settings, "initial"))
    heston_quotes = [
        HestonCalibrationQuote(float(maturity), float(strike), float(price))
        for maturity, strike, price in zip(
            quote_times, quote_strikes, quote_prices, strict=True
        )
    ]
    heston_report = calibrate_heston(
        spot,
        rate,
        dividend_yield,
        heston_quotes,
        initial_parameters,
        variance_swap_quotes=[
            VarianceSwapQuote(
                float(maturity),
                expected_average_variance(float(maturity), reference_parameters),
                weight=2.0,
            )
            for maturity in quote_maturities
        ],
        config=HestonCalibrationConfig(
            maximum_evaluations=(
                min(30, _integer(heston_settings, "maximum_evaluations"))
                if quick
                else _integer(heston_settings, "maximum_evaluations")
            ),
            tolerance=_float(heston_settings, "tolerance"),
        ),
    )
    vanilla_prices = tuple(
        VanillaPriceComparison(
            float(maturity),
            float(log_moneyness),
            float(strike),
            float(market_price),
            float(market_price + residual),
        )
        for maturity, log_moneyness, strike, market_price, residual in zip(
            quote_times,
            quote_log_moneyness,
            quote_strikes,
            quote_prices,
            heston_report.price_residuals,
            strict=True,
        )
    )

    leverage_paths = _integer(leverage_settings, "paths")
    pricing_paths = _integer(pricing_settings, "paths")
    if quick:
        leverage_paths = min(leverage_paths, 2_000)
        pricing_paths = min(pricing_paths, 5_000)
    leverage_config = LeverageCalibrationConfig(
        paths=leverage_paths,
        iterations=_integer(leverage_settings, "iterations"),
        seed=seed,
        bandwidth=_float(leverage_settings, "bandwidth"),
        minimum_effective_sample_size=_float(
            leverage_settings, "minimum_effective_sample_size"
        ),
        damping=_float(leverage_settings, "damping"),
        smoothing_sigma=_float(leverage_settings, "smoothing_sigma"),
        leverage_minimum=_float(leverage_settings, "leverage_minimum"),
        leverage_maximum=_float(leverage_settings, "leverage_maximum"),
        convergence_tolerance=_float(leverage_settings, "convergence_tolerance"),
    )
    LOGGER.info("Calibrating the particle fixed-point leverage surface")
    leverage = calibrate_leverage(
        spot,
        rate,
        dividend_yield,
        heston_report.parameters,
        local_times,
        local_moneyness,
        local_volatility,
        config=leverage_config,
    )

    LOGGER.info("Simulating calibrated LSV paths and pricing DNT")
    pricing_seed = _integer(pricing_settings, "seed")
    simulation = lsv_cpp.simulate_lsv(
        spot,
        local_maturity,
        rate,
        dividend_yield,
        heston_report.parameters.kappa,
        heston_report.parameters.theta,
        heston_report.parameters.eta,
        heston_report.parameters.rho,
        heston_report.parameters.v0,
        pricing_paths,
        local_steps,
        pricing_seed,
        True,
        local_times,
        local_moneyness,
        leverage.leverage,
    )
    spots = np.asarray(simulation["spots"])
    variances = np.asarray(simulation["variances"])
    local_variance_paths = np.empty_like(variances)
    for time_index in range(local_steps + 1):
        path_time = time_index * local_maturity / local_steps
        row = max(time_index - 1, 0)
        coordinates = np.log(spots[:, time_index] / spot) - carry * path_time
        path_leverage = np.interp(
            coordinates,
            local_moneyness,
            leverage.leverage[row],
            left=leverage.leverage[row, 0],
            right=leverage.leverage[row, -1],
        )
        local_variance_paths[:, time_index] = (
            path_leverage**2 * variances[:, time_index]
        )
    dnt = price_dnt(
        spots,
        variances,
        local_maturity,
        rate,
        _float(pricing_settings, "payout"),
        _float(pricing_settings, "lower_barrier"),
        _float(pricing_settings, "upper_barrier"),
        monitoring=Monitoring(_string(pricing_settings, "monitoring")),
        local_variances=local_variance_paths,
        seed=pricing_seed,
    )
    terminal_call = float(
        np.mean(np.exp(-rate * local_maturity) * np.maximum(spots[:, -1] - spot, 0.0))
    )
    target_call = heston_price(
        spot,
        spot,
        local_maturity,
        rate,
        dividend_yield,
        reference_parameters,
    )

    exotic_summary: dict[str, object] | None = None
    exotic_enabled = _boolean(exotic_settings, "enabled") and not skip_exotic
    if exotic_enabled:
        LOGGER.info("Running the nested exotic outer calibration")
        exotic_leverage_paths = _integer(exotic_settings, "leverage_paths")
        exotic_pricing_paths = _integer(exotic_settings, "pricing_paths")
        if quick:
            exotic_leverage_paths = min(exotic_leverage_paths, 2_000)
            exotic_pricing_paths = min(exotic_pricing_paths, 4_000)
        exotic_report = calibrate_exotics(
            spot,
            rate,
            dividend_yield,
            heston_report.parameters,
            local_times,
            local_moneyness,
            local_volatility,
            [
                DNTCalibrationQuote(
                    local_maturity,
                    _float(pricing_settings, "lower_barrier"),
                    _float(pricing_settings, "upper_barrier"),
                    _float(pricing_settings, "payout"),
                    dnt.estimate,
                    QuoteSource.SYNTHETIC,
                )
            ],
            vanilla_targets=[VanillaRepricingTarget(local_maturity, spot, target_call)],
            leverage_config=LeverageCalibrationConfig(
                paths=exotic_leverage_paths,
                iterations=_integer(exotic_settings, "leverage_iterations"),
                seed=seed,
                bandwidth=_float(leverage_settings, "bandwidth"),
                damping=_float(leverage_settings, "damping"),
                smoothing_sigma=_float(leverage_settings, "smoothing_sigma"),
                convergence_tolerance=_float(
                    leverage_settings, "convergence_tolerance"
                ),
            ),
            config=ExoticCalibrationConfig(
                calibrated_parameters=_string_tuple(
                    exotic_settings, "calibrated_parameters"
                ),
                lower_bounds=_float_tuple(exotic_settings, "lower_bounds"),
                upper_bounds=_float_tuple(exotic_settings, "upper_bounds"),
                maximum_evaluations=_integer(exotic_settings, "maximum_evaluations"),
                tolerance=_float(exotic_settings, "tolerance"),
                finite_difference_step=_float(
                    exotic_settings, "finite_difference_step"
                ),
                pricing_paths=exotic_pricing_paths,
                pricing_seed=_integer(exotic_settings, "pricing_seed"),
                monitoring=Monitoring(_string(pricing_settings, "monitoring")),
                require_leverage_convergence=False,
            ),
        )
        exotic_summary = {
            "quote_source": exotic_report.quote_sources[0].value,
            "objective_before": exotic_report.objective_before,
            "objective_after": exotic_report.objective_after,
            "parameters": {
                "kappa": exotic_report.final_parameters.kappa,
                "theta": exotic_report.final_parameters.theta,
                "eta": exotic_report.final_parameters.eta,
                "rho": exotic_report.final_parameters.rho,
                "v0": exotic_report.final_parameters.v0,
            },
            "dnt_price_residual": exotic_report.final_dnt_residuals[0].price_residual,
            "vanilla_price_residual": exotic_report.final_vanilla_residuals[0],
            "evaluations": exotic_report.iterations,
            "elapsed_seconds": exotic_report.elapsed_seconds,
        }

    surface_data_path = output_directory / "calibrated_surfaces.npz"
    np.savez_compressed(
        surface_data_path,
        implied_times=plot_times,
        implied_log_moneyness=plot_moneyness,
        implied_volatility=fitted_implied_volatility,
        local_times=local_times,
        local_log_moneyness=local_moneyness,
        local_volatility=local_volatility,
        leverage=leverage.leverage,
    )
    figure_path = output_directory / "calibrated_surfaces_3d.png"
    summary_path = output_directory / "calibration_summary.json"
    artifacts = EngineArtifacts(
        output_directory=output_directory,
        summary_path=summary_path,
        surface_data_path=surface_data_path,
        figure_path=figure_path,
        heston_parameters=heston_report.parameters,
        vanilla_prices=vanilla_prices,
        repricing_maturity=local_maturity,
        repricing_strike=spot,
        target_call_price=target_call,
        lsv_call_price=terminal_call,
        dnt_price=dnt.estimate,
        dnt_standard_error=dnt.standard_error,
        leverage_converged=leverage.converged,
    )
    if print_prices:
        print(_format_price_report(artifacts), flush=True)
    _plot_surfaces(
        figure_path,
        quote_times,
        quote_log_moneyness,
        quote_volatilities,
        plot_times,
        plot_moneyness,
        fitted_implied_volatility,
        local_times,
        local_moneyness,
        local_volatility,
        leverage.leverage,
        should_show,
    )
    summary = {
        "experiment": _string(experiment, "name"),
        "quote_source": _string(synthetic, "quote_source"),
        "seed": seed,
        "static_arbitrage": {
            "is_arbitrage_free": arbitrage.is_arbitrage_free,
            "strike_monotonicity_violations": arbitrage.strike_monotonicity_violations,
            "strike_convexity_violations": arbitrage.strike_convexity_violations,
            "calendar_violations": arbitrage.calendar_violations,
        },
        "dupire": {
            "floor_count": dupire_floor_count,
            "cap_count": dupire_cap_count,
            "unstable_denominator_count": dupire_unstable_count,
        },
        "heston": {
            "parameters": {
                "kappa": heston_report.parameters.kappa,
                "theta": heston_report.parameters.theta,
                "eta": heston_report.parameters.eta,
                "rho": heston_report.parameters.rho,
                "v0": heston_report.parameters.v0,
            },
            "objective": heston_report.objective_value,
            "evaluations": heston_report.iterations,
            "feller_satisfied": heston_report.feller_satisfied,
            "vanilla_prices": [
                {
                    "maturity": price.maturity,
                    "log_moneyness": price.log_moneyness,
                    "strike": price.strike,
                    "synthetic_price": price.market_price,
                    "model_price": price.heston_price,
                    "residual": price.heston_price - price.market_price,
                }
                for price in vanilla_prices
            ],
        },
        "leverage": {
            "converged": leverage.converged,
            "iterations": len(leverage.history),
            "final_update_norm": leverage.history[-1].relative_update_norm,
            "low_density_count": leverage.history[-1].low_density_count,
        },
        "vanilla_repricing": {
            "target_call": target_call,
            "lsv_call": terminal_call,
            "residual": terminal_call - target_call,
        },
        "dnt": {
            "estimate": dnt.estimate,
            "standard_error": dnt.standard_error,
            "confidence_interval": list(dnt.confidence_interval),
            "paths": dnt.paths,
            "steps": dnt.steps,
            "monitoring": dnt.monitoring.value,
        },
        "exotic_calibration": exotic_summary,
        "artifacts": {
            "surface_data": str(surface_data_path),
            "figure": str(figure_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Calibration completed")
    return artifacts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lsv-run",
        description="Run the complete reproducible synthetic LSV engine.",
    )
    parser.add_argument(
        "--config", default="configs/calibration.yaml", help="YAML configuration"
    )
    parser.add_argument(
        "--no-show", action="store_true", help="save plots without opening a window"
    )
    parser.add_argument(
        "--quick", action="store_true", help="use fewer Monte Carlo paths"
    )
    parser.add_argument(
        "--skip-exotic", action="store_true", help="skip outer DNT calibration"
    )
    return parser


def main() -> int:
    """Run the command-line engine and print artifact locations."""
    arguments = _parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = run_engine(
        arguments.config,
        show_plots=False if arguments.no_show else None,
        quick=arguments.quick,
        skip_exotic=arguments.skip_exotic,
        print_prices=True,
    )
    print()
    print(f"Summary: {artifacts.summary_path}")
    print(f"3D surfaces: {artifacts.figure_path}")
    print(f"Surface data: {artifacts.surface_data_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
