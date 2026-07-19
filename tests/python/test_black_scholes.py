import numpy as np
import pytest
from lsv.black_scholes import black_scholes_price, implied_volatility


def test_call_put_parity_and_implied_vol_repricing() -> None:
    forward = 103.0
    strike = 97.0
    maturity = 1.7
    discount = 0.96
    volatility = 0.31
    call = float(
        black_scholes_price(forward, strike, maturity, volatility, discount, "call")
    )
    put = float(
        black_scholes_price(forward, strike, maturity, volatility, discount, "put")
    )
    assert call - put == pytest.approx(discount * (forward - strike), abs=1e-12)
    recovered = implied_volatility(call, forward, strike, maturity, discount, "call")
    assert recovered == pytest.approx(volatility, abs=1e-11)


def test_vector_prices_decrease_and_are_convex_in_strike() -> None:
    strikes = np.linspace(60.0, 140.0, 81)
    calls = black_scholes_price(100.0, strikes, 1.0, 0.2, 0.98)
    assert np.all(np.diff(calls) <= 0.0)
    assert np.all(np.diff(calls, n=2) >= -1e-12)


def test_implied_vol_rejects_price_outside_bounds() -> None:
    with pytest.raises(ValueError, match="no-arbitrage bounds"):
        implied_volatility(101.0, 100.0, 100.0, 1.0)
