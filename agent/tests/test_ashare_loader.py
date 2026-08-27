"""Tests for the Ashare dual-source A-share loader.

Defends four observable contracts:
  1. non-daily intervals are rejected (loader is daily-only);
  2. Tencent qfqday is the primary source and its rows map to OHLCV correctly;
  3. when Tencent is unavailable, Sina raw bars are re-adjusted with qfq.js
     factors — division by the cumulative factor, so the return across an
     ex-date is the true move rather than the mechanical dividend gap;
  4. a symbol whose Sina factors are missing is dropped, never returned
     unadjusted (ex-dividend gaps would fabricate cross-sectional returns).
"""

from __future__ import annotations

import json

import pytest

from backtest.loaders import ashare_loader
from backtest.loaders.ashare_loader import DataLoader


@pytest.fixture()
def loader(monkeypatch) -> DataLoader:
    """Loader with the shared cache bypassed so fetch lambdas always run."""
    monkeypatch.setattr(
        ashare_loader,
        "cached_loader_fetch",
        lambda **kwargs: kwargs["fetch"](),
    )
    return DataLoader()


def _tx_payload(klines: list) -> dict:
    return {"code": 0, "data": {"sh600519": {"qfqday": klines}}}


class _Resp:
    def __init__(self, payload: dict | list, text: str = "") -> None:
        self._payload = payload
        self.text = text or json.dumps(payload)

    def json(self):
        return self._payload


def test_intraday_request_returns_empty(loader, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        loader, "_fetch_one", lambda code, s, e: calls.append(code) or None
    )
    assert loader.fetch(["600519.SH"], "2026-01-01", "2026-01-31", interval="1h") == {}
    assert calls == []


def test_tencent_primary_maps_ohlcv(loader, monkeypatch) -> None:
    klines = [
        ["2026-01-05", "10.0", "10.5", "11.0", "9.5", "1000"],
        ["2026-01-06", "10.5", "10.2", "10.8", "10.1", "900"],
    ]

    def fake_get(url, **kwargs):
        return _Resp(_tx_payload(klines))

    monkeypatch.setattr(ashare_loader, "throttled_get", fake_get)
    result = loader.fetch(["600519.SH"], "2026-01-01", "2026-01-31")
    df = result["600519.SH"]
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # Tencent row order is [date, open, close, high, low, volume].
    assert df.loc["2026-01-05", "close"] == 10.5
    assert df.loc["2026-01-05", "high"] == 11.0


def test_tencent_long_window_chunks_backwards(loader, monkeypatch) -> None:
    """A chunked walk merges, dedupes, and stops once start is reached."""
    calls: list[str] = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if len(calls) == 1:
            rows = [[f"2026-01-{d}", "10", "10", "10", "10", "1"] for d in ("06", "07", "08")]
            return _Resp(_tx_payload(rows))
        if len(calls) == 2:
            rows = [[f"2026-01-{d}", "9", "9", "9", "9", "1"] for d in ("02", "05")]
            return _Resp(_tx_payload(rows))
        raise AssertionError("should stop after reaching start_date")

    monkeypatch.setattr(ashare_loader, "throttled_get", fake_get)
    monkeypatch.setattr(ashare_loader, "_TENCENT_CHUNK", 3)
    result = loader.fetch(["600519.SH"], "2026-01-01", "2026-01-08")
    df = result["600519.SH"]
    assert len(df) == 5
    assert len(calls) == 2  # stopped because oldest 2026-01-02 <= start


def test_sina_fallback_applies_qfq_factors(loader, monkeypatch) -> None:
    """qfq = raw / f with f cumulative-from-today (1.0 latest, rising into the past).

    Raw bars show a mechanical -7.3% gap across the ex-date; the adjustment
    must turn it into the true +2% move, and volume is re-scaled so turnover
    stays invariant.
    """
    def fake_get(url, **kwargs):
        if "ifzq.gtimg.cn" in url:
            raise ConnectionError("tencent down")
        if "getKLineData" in url:
            return _Resp([
                {"day": "2026-01-05", "open": "11.0", "high": "11.2",
                 "low": "10.8", "close": "11.0", "volume": "1000"},
                {"day": "2026-01-06", "open": "10.1", "high": "10.3",
                 "low": "10.0", "close": "10.2", "volume": "2000"},
            ])
        if "qfq.js" in url:
            # Factor 1.1 before the ex-date, 1.0 from 2026-01-06 onward.
            body = {"total": 2, "data": [
                {"d": "1990-01-01", "f": "1.1"},
                {"d": "2026-01-06", "f": "1.0"},
            ]}
            return _Resp(body, text=f"var x={json.dumps(body)}")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(ashare_loader, "throttled_get", fake_get)
    result = loader.fetch(["600519.SH"], "2026-01-01", "2026-01-31")
    df = result["600519.SH"]
    # 2026-01-05: close 11.0 / 1.1 = 10.0; 2026-01-06 unchanged at 10.2.
    assert df.loc["2026-01-05", "close"] == pytest.approx(10.0)
    assert df.loc["2026-01-06", "close"] == pytest.approx(10.2)
    # Return across the ex-date is the real move (+2%), not the raw -7.3%.
    assert df["close"].iloc[-1] / df["close"].iloc[0] - 1 == pytest.approx(0.02)
    # Volume scaled up by the factor so price x volume (turnover) is invariant.
    assert df.loc["2026-01-05", "volume"] == pytest.approx(1100.0)


def test_sina_without_factors_drops_symbol(loader, monkeypatch) -> None:
    def fake_get(url, **kwargs):
        if "ifzq.gtimg.cn" in url:
            raise ConnectionError("tencent down")
        if "getKLineData" in url:
            return _Resp([
                {"day": "2026-01-05", "open": "11.0", "high": "11.2",
                 "low": "10.8", "close": "11.0", "volume": "1000"},
            ])
        if "qfq.js" in url:
            raise ConnectionError("factor endpoint down")
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(ashare_loader, "throttled_get", fake_get)
    assert loader.fetch(["600519.SH"], "2026-01-01", "2026-01-31") == {}


def test_non_ashare_code_rejected(loader) -> None:
    assert loader.fetch(["AAPL.US"], "2026-01-01", "2026-01-31") == {}
