"""Python orchestration package for the LSV engine.

The public API currently covers vanilla surfaces, Dupire extraction, and
semi-analytic Heston pricing and calibration.
"""

from lsv.barrier_pricer import (
    BarrierPayoff,
    BarrierPrice,
    BarrierType,
    Monitoring,
    price_barrier,
    price_dnt,
)
from lsv.black_scholes import OptionType, black_scholes_price, implied_volatility
from lsv.config import load_config
from lsv.curves import ZeroRateCurve, forward_price
from lsv.dupire import DupireConfig, DupireExtractor, LocalVolSurfaceGrid
from lsv.exotic_calibration import (
    DNTCalibrationQuote,
    ExoticCalibrationConfig,
    ExoticCalibrationReport,
    QuoteSource,
    VanillaRepricingTarget,
    calibrate_exotics,
)
from lsv.heston_calibration import (
    HestonCalibrationConfig,
    HestonCalibrationQuote,
    HestonCalibrationReport,
    VarianceSwapQuote,
    calibrate_heston,
)
from lsv.heston_pricer import HestonParameters, heston_price
from lsv.leverage_calibration import (
    LeverageCalibrationConfig,
    LeverageCalibrationResult,
    calibrate_leverage,
)
from lsv.market_data import VanillaQuote, clean_vanilla_quotes, read_vanilla_quotes
from lsv.ssvi_surface import SSVISurface, fit_ssvi_surface

__all__ = [
    "SSVISurface",
    "BarrierPayoff",
    "BarrierPrice",
    "BarrierType",
    "DupireConfig",
    "DupireExtractor",
    "DNTCalibrationQuote",
    "ExoticCalibrationConfig",
    "ExoticCalibrationReport",
    "LocalVolSurfaceGrid",
    "Monitoring",
    "HestonCalibrationConfig",
    "HestonCalibrationQuote",
    "HestonCalibrationReport",
    "HestonParameters",
    "LeverageCalibrationConfig",
    "LeverageCalibrationResult",
    "OptionType",
    "QuoteSource",
    "VanillaQuote",
    "VarianceSwapQuote",
    "VanillaRepricingTarget",
    "ZeroRateCurve",
    "__version__",
    "black_scholes_price",
    "calibrate_heston",
    "calibrate_exotics",
    "calibrate_leverage",
    "clean_vanilla_quotes",
    "fit_ssvi_surface",
    "forward_price",
    "heston_price",
    "implied_volatility",
    "load_config",
    "price_barrier",
    "price_dnt",
    "read_vanilla_quotes",
]
__version__ = "0.1.0"
