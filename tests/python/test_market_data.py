from pathlib import Path

import numpy as np
import pytest
from lsv.curves import ZeroRateCurve, forward_price
from lsv.market_data import VanillaQuote, clean_vanilla_quotes, read_vanilla_quotes


def test_curve_discount_and_forward_conventions() -> None:
    domestic = ZeroRateCurve.flat(0.03)
    dividend = ZeroRateCurve.flat(0.01)
    assert domestic.discount(2.0) == pytest.approx(np.exp(-0.06))
    assert forward_price(100.0, 2.0, domestic, dividend) == pytest.approx(
        100.0 * np.exp(0.04)
    )


def test_malformed_quote_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="0 <= bid <= ask"):
        VanillaQuote(1.0, 100.0, "call", bid=2.0, ask=1.0)  # type: ignore[arg-type]


def test_csv_quote_validation_reports_row(tmp_path: Path) -> None:
    path = tmp_path / "quotes.csv"
    path.write_text(
        "maturity,strike,option_type,implied_vol\n1.0,-100,call,0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="CSV row 2"):
        read_vanilla_quotes(path)


def test_cleaning_is_sorted_and_rejects_conflicting_duplicates() -> None:
    later = VanillaQuote(2.0, 100.0, "call", implied_vol=0.22)  # type: ignore[arg-type]
    earlier = VanillaQuote(1.0, 100.0, "call", implied_vol=0.20)  # type: ignore[arg-type]
    assert clean_vanilla_quotes([later, earlier, earlier]) == [earlier, later]
    conflicting = VanillaQuote(1.0, 100.0, "call", implied_vol=0.21)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="conflicting duplicate"):
        clean_vanilla_quotes([earlier, conflicting])
