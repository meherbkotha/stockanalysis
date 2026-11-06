import math
import re
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
from langchain.tools import tool


def _symbol(value: str) -> str:
    value = value.strip().upper()
    if not re.fullmatch(r"[A-Z0-9.^=-]{1,25}", value):
        raise ValueError("Use a ticker such as GOOGL, AAPL, or RELIANCE.NS.")
    return value


def _number(value):
    """Return a JSON-safe number; missing values become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _daily_prices(symbol, period):
    prices = yf.Ticker(symbol).history(
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        timeout=15,
        raise_errors=True,
    )
    prices = prices.dropna(subset=["Close"]).sort_index()

    if prices.empty:
        raise ValueError("Yahoo returned no historical prices.")

    return prices


def get_stock_history(symbol: str, period: str = "3mo") -> dict:
    """Fetch adjusted daily OHLCV stock history from Yahoo Finance.
    Use for historical price questions and trends.
    period: 1mo, 3mo, 6mo, 1y, or 2y.
    """
    try:
        symbol = _symbol(symbol)
        if period not in {"1mo", "3mo", "6mo", "1y", "2y"}:
            raise ValueError("period must be 1mo, 3mo, 6mo, 1y, or 2y.")

        prices = _daily_prices(symbol, period)
        rows = []

        for timestamp, row in prices.iterrows():
            rows.append({
                "date": timestamp.date().isoformat(),
                "open": _number(row["Open"]),
                "high": _number(row["High"]),
                "low": _number(row["Low"]),
                "close": _number(row["Close"]),
                "volume": _number(row["Volume"]),
            })

        return {
            "ok": True,
            "symbol": symbol,
            "period": period,
            "interval": "1d",
            "adjusted_prices": True,
            "prices": rows,
            "retrieved_at": _now(),
            "source": "Yahoo Finance via yfinance",
            "note": "The latest daily bar may still be in progress.",
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_technical_indicators(symbol: str) -> dict:
    """Calculate SMA20, SMA50, SMA200 and Wilder RSI14 from
    two years of Yahoo adjusted daily closing prices.
    Use for moving-average, momentum, and technical-analysis questions.
    """
    try:
        symbol = _symbol(symbol)
        prices = _daily_prices(symbol, "2y")
        close = prices["Close"].astype(float)

        # Exact Wilder initialization: mean of the first 14 changes,
        # followed by Wilder's recursive smoothing.
        rsi = None
        if len(close) >= 15:
            changes = close.diff().iloc[1:]
            gains = changes.clip(lower=0)
            losses = -changes.clip(upper=0)

            average_gain = float(gains.iloc[:14].mean())
            average_loss = float(losses.iloc[:14].mean())

            for gain, loss in zip(gains.iloc[14:], losses.iloc[14:]):
                average_gain = (average_gain * 13 + float(gain)) / 14
                average_loss = (average_loss * 13 + float(loss)) / 14

            if average_gain == 0 and average_loss == 0:
                rsi = 50.0  # Explicit convention for a flat series.
            elif average_loss == 0:
                rsi = 100.0
            else:
                rsi = 100 - 100 / (1 + average_gain / average_loss)

        averages = {
            f"sma_{window}": (
                _number(close.iloc[-window:].mean())
                if len(close) >= window else None
            )
            for window in (20, 50, 200)
        }

        return {
            "ok": True,
            "symbol": symbol,
            "as_of_bar_date": close.index[-1].date().isoformat(),
            "adjusted_close": _number(close.iloc[-1]),
            **averages,
            "rsi_14": _number(rsi),
            "rsi_method": "Wilder smoothing; flat-series RSI = 50",
            "observations": len(close),
            "missing_indicators": [
                name for name, value in {**averages, "rsi_14": rsi}.items()
                if value is None
            ],
            "retrieved_at": _now(),
            "source": "Calculated from Yahoo Finance adjusted daily prices",
            "note": (
                "The latest daily bar may still be in progress. "
                "Indicators can change until that session closes."
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _statement_records(frame, fields):
    """Keep each statement's own period dates and missing values."""
    if frame is None or frame.empty:
        return []

    records = []
    for column in sorted(frame.columns, reverse=True)[:4]:
        record = {"period_end": pd.Timestamp(column).date().isoformat()}
        for output_name, yahoo_name in fields.items():
            record[output_name] = (
                _number(frame.loc[yahoo_name, column])
                if yahoo_name in frame.index else None
            )
        records.append(record)

    return records


def get_company_financials(
    symbol: str,
    frequency: str = "quarterly",
) -> dict:
    """Fetch up to four available income statements, balance sheets,
    and cash-flow statements from Yahoo Finance.
    frequency must be quarterly or annual. Values are reported amounts,
    not forecasts. Missing fields are returned as null.
    """
    try:
        symbol = _symbol(symbol)
        if frequency not in {"quarterly", "annual"}:
            raise ValueError("frequency must be quarterly or annual.")

        ticker = yf.Ticker(symbol)
        errors = {}
        statements = {}

        requests = {
            "income_statements": (
                ticker.get_income_stmt,
                {
                    "revenue": "TotalRevenue",
                    "operating_income": "OperatingIncome",
                    "net_income": "NetIncome",
                    "diluted_eps": "DilutedEPS",
                },
            ),
            "balance_sheets": (
                ticker.get_balance_sheet,
                {
                    "total_assets": "TotalAssets",
                    "total_liabilities": "TotalLiabilitiesNetMinorityInterest",
                    "stockholders_equity": "StockholdersEquity",
                    "total_debt": "TotalDebt",
                    "cash_and_short_term_investments":
                        "CashCashEquivalentsAndShortTermInvestments",
                },
            ),
            "cash_flow_statements": (
                ticker.get_cashflow,
                {
                    "operating_cash_flow": "OperatingCashFlow",
                    "capital_expenditure": "CapitalExpenditure",
                    "free_cash_flow": "FreeCashFlow",
                },
            ),
        }

        # Preserve partial results if one statement request fails.
        for name, (fetch, fields) in requests.items():
            try:
                statements[name] = _statement_records(
                    fetch(freq=frequency), fields
                )
                if not statements[name]:
                    errors[name] = "No statement data returned."
            except Exception as exc:
                statements[name] = []
                errors[name] = str(exc)

        for record in statements["income_statements"]:
            revenue = record["revenue"]
            income = record["net_income"]
            record["net_profit_margin_pct"] = (
                100 * income / revenue
                if income is not None and revenue not in (None, 0)
                else None
            )

        # Reporting currency can differ from the stock's trading currency.
        currency = None
        try:
            currency = ticker.get_info().get("financialCurrency")
        except Exception as exc:
            errors["reporting_currency"] = str(exc)

        has_data = any(statements.values())

        return {
            "ok": has_data,
            "partial": has_data and bool(errors),
            "symbol": symbol,
            "frequency": frequency,
            "reporting_currency": currency,
            **statements,
            "errors": errors,
            "retrieved_at": _now(),
            "source": "Yahoo Finance via yfinance",
            "note": (
                "Period end is not the publication date. "
                "Amounts are in currency units, not millions; EPS is per share. "
                "Financial statements describe past performance, not future growth."
            ),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}