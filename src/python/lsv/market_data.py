"""Typed vanilla quote ingestion and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lsv.black_scholes import OptionType, implied_volatility


@dataclass(frozen=True)
class VanillaQuote:
    """One European quote with either volatility or bid/ask prices."""

    maturity: float
    strike: float
    option_type: OptionType
    implied_vol: float | None = None
    bid: float | None = None
    ask: float | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("quote maturity must be finite and positive")
        if not np.isfinite(self.strike) or self.strike <= 0.0:
            raise ValueError("quote strike must be finite and positive")
        object.__setattr__(self, "option_type", OptionType(self.option_type))
        has_vol = self.implied_vol is not None
        has_prices = self.bid is not None or self.ask is not None
        if has_vol == has_prices:
            raise ValueError("provide exactly one of implied_vol or a bid/ask pair")
        if has_vol:
            implied_vol = self.implied_vol
            if (
                implied_vol is None
                or not np.isfinite(implied_vol)
                or implied_vol <= 0.0
            ):
                raise ValueError("implied_vol must be finite and positive")
        if has_prices:
            if self.bid is None or self.ask is None:
                raise ValueError("both bid and ask are required")
            if not np.isfinite(self.bid) or not np.isfinite(self.ask):
                raise ValueError("bid and ask must be finite")
            if self.bid < 0.0 or self.ask < self.bid:
                raise ValueError("quotes require 0 <= bid <= ask")


def read_vanilla_quotes(path: str | Path) -> list[VanillaQuote]:
    """Read and validate vanilla quotes from CSV.

    Required columns are maturity, strike, option_type and either implied_vol or
    both bid and ask. Unexpected columns are retained by pandas but ignored.
    """
    quote_path = Path(path)
    if not quote_path.is_file():
        raise FileNotFoundError(f"quote file does not exist: {quote_path}")
    frame = pd.read_csv(quote_path)
    required = {"maturity", "strike", "option_type"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"quote file is missing columns: {sorted(missing)}")
    has_vol = "implied_vol" in frame
    has_prices = {"bid", "ask"}.issubset(frame.columns)
    if has_vol == has_prices:
        raise ValueError("quote file needs either implied_vol or bid and ask columns")
    quotes: list[VanillaQuote] = []
    for index, row in frame.iterrows():
        try:
            quotes.append(
                VanillaQuote(
                    maturity=float(row["maturity"]),
                    strike=float(row["strike"]),
                    option_type=OptionType(str(row["option_type"]).lower()),
                    implied_vol=float(row["implied_vol"]) if has_vol else None,
                    bid=float(row["bid"]) if has_prices else None,
                    ask=float(row["ask"]) if has_prices else None,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid quote at CSV row {index + 2}: {exc}") from exc
    if not quotes:
        raise ValueError("quote file contains no quotes")
    return clean_vanilla_quotes(quotes)


def clean_vanilla_quotes(quotes: list[VanillaQuote]) -> list[VanillaQuote]:
    """Sort quotes and collapse exact duplicates, rejecting conflicting ones.

    A contract is identified by maturity, strike, and option type. Conflicting
    observations for one contract must be resolved upstream rather than averaged
    silently. The returned order is deterministic.
    """
    if not quotes:
        raise ValueError("at least one vanilla quote is required")
    unique: dict[tuple[float, float, OptionType], VanillaQuote] = {}
    for quote in quotes:
        key = (quote.maturity, quote.strike, quote.option_type)
        existing = unique.get(key)
        if existing is not None and existing != quote:
            raise ValueError(
                "conflicting duplicate quote for "
                f"maturity={quote.maturity}, strike={quote.strike}, "
                f"option_type={quote.option_type.value}"
            )
        unique[key] = quote
    return [
        unique[key]
        for key in sorted(unique, key=lambda item: (item[0], item[1], item[2].value))
    ]


def quote_implied_vol(
    quote: VanillaQuote,
    forward: float,
    discount_factor: float,
) -> float:
    """Return supplied volatility or invert the bid/ask midpoint."""
    if quote.implied_vol is not None:
        return quote.implied_vol
    if quote.bid is None or quote.ask is None:
        raise AssertionError("validated price quote has no bid/ask pair")
    return implied_volatility(
        0.5 * (quote.bid + quote.ask),
        forward,
        quote.strike,
        quote.maturity,
        discount_factor,
        quote.option_type,
    )
