# stock_tools.py
import json
import math
import re
from datetime import datetime, timezone

import yfinance as yf


def _validate_symbol(symbol: str) -> str:
    symbol = symbol.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", symbol):
        raise ValueError("Provide a ticker such as AAPL, MSFT, or RELIANCE.NS.")
    return symbol


def _number(value):
    """Convert pandas/numpy values to JSON-safe numbers."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def get_latest_stock_price(symbol: str) -> dict:
    """Get the latest available one-minute closing price.

    This is a price bar, not a guaranteed real-time trade quote.
    Includes extended-hours data when Yahoo provides it.
    """
    try:
        symbol = _validate_symbol(symbol)
        ticker = yf.Ticker(symbol)

        # Five days allows a result during weekends and market holidays.
        prices = ticker.history(
            period="5d",
            interval="1m",
            prepost=True,
            auto_adjust=False,
            actions=False,
            timeout=15,
            raise_errors=True,
        ).dropna(subset=["Close"])

        if prices.empty:
            return {
                "ok": False,
                "symbol": symbol,
                "error": "NO_DATA",
                "message": "No prices returned. Check the ticker or try later.",
            }

        timestamp = prices.index[-1]
        latest = prices.iloc[-1]
        metadata = ticker.get_history_metadata() or {}

        return {
            "ok": True,
            "symbol": symbol,
            "price": _number(latest["Close"]),
            "currency": metadata.get("currency"),
            "bar_time": timestamp.isoformat(),
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "interval": "1m",
            "includes_extended_hours": True,
            "source": "Yahoo Finance via yfinance",
        }

    except ValueError as exc:
        return {"ok": False, "error": "INVALID_INPUT", "message": str(exc)}
    except Exception as exc:
        return {
            "ok": False,
            "error": "DATA_FETCH_FAILED",
            "message": str(exc),
        }
