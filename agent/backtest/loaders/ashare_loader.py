"""Ashare loader: free, no-auth A-share daily OHLCV via a dual-source design.

Port of the single-file ``Ashare`` helper (https://github.com/mpquant/Ashare),
adapted to the repo's loader protocol and correctness bar.  Ashare pairs two
unauthenticated public endpoints — Sina's K-line API and Tencent's fqkline API —
so a single source being rate-limited or unreachable degrades to the other.

Adjustment policy (why this is not a verbatim port):
  * Primary = Tencent ``qfqday``.  Tencent returns natively forward-adjusted
    (前复权) prices; verified continuous across the 600519.SH 2024-06-19
    ex-dividend date (+0.75% true move, no dividend gap).  This is the safe
    default and matches the ``qfq`` convention akshare/tencent already use.
  * Fallback = Sina raw K-lines + ``qfq.js`` corporate-action factors.  Sina's
    K-line API returns UNADJUSTED prices, so they are re-adjusted here; the
    loader refuses to return a symbol whose factors are missing rather than
    handing back contaminated prices.

Only daily bars are served (Ashare's minute endpoints are out of scope for the
backtest/factor pipeline), mirroring ``tencent_loader``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders._http import resolve_min_interval, throttled_get
from backtest.loaders.base import cached_loader_fetch, validate_date_range
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_SINA_KLINE_URL = (
    "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
    "CN_MarketData.getKLineData"
)
_SINA_QFQ_URL = "https://finance.sina.com.cn/realstock/company/{sym}/qfq.js"

_TENCENT_HOST_KEY = "tencent"
_SINA_HOST_KEY = "sina"
_MIN_INTERVAL_ENV = "VIBE_TRADING_ASHARE_MIN_INTERVAL"
_DEFAULT_MIN_INTERVAL = 0.5

# Tencent's fqkline endpoint caps ``count`` at 800 rows; anything larger errors
# out (``param error``). Long windows are walked backwards in chunks of this
# size, anchored on ``end_date``.
_TENCENT_CHUNK = 800
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
)

_DAILY_ALIASES = frozenset({"1d", "d", "day", "daily"})


def _is_a_share(code: str) -> bool:
    return bool(code) and code.upper().endswith((".SZ", ".SH"))


def _prefixed(code: str) -> Optional[str]:
    """Map ``600519.SH`` -> ``sh600519``; ``000001.SZ`` -> ``sz000001``."""
    parts = code.upper().split(".")
    if len(parts) != 2:
        return None
    symbol, suffix = parts
    if suffix == "SH":
        return f"sh{symbol}"
    if suffix == "SZ":
        return f"sz{symbol}"
    return None


@register
class DataLoader:
    """Ashare dual-source A-share daily OHLCV loader (free, HTTP, no auth)."""

    name = "ashare"
    markets = {"a_share"}
    requires_auth = False

    def is_available(self) -> bool:
        """Always available — plain HTTP against public quote endpoints."""
        return True

    def __init__(self) -> None:
        pass

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        validate_date_range(start_date, end_date)
        del fields

        if interval.strip().lower() not in _DAILY_ALIASES:
            logger.warning(
                "ashare supports daily bars only; rejecting interval=%s", interval
            )
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                df = cached_loader_fetch(
                    source=self.name,
                    symbol=code,
                    timeframe=interval,
                    start_date=start_date,
                    end_date=end_date,
                    fields=None,
                    fetch=lambda code=code: self._fetch_one(code, start_date, end_date),
                )
                if df is not None and not df.empty:
                    result[code] = df
            except Exception as exc:
                logger.warning("ashare failed for %s: %s", code, exc)
        return result

    # ------------------------------------------------------------------
    # Per-symbol dual-source fetch
    # ------------------------------------------------------------------
    def _fetch_one(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        if not _is_a_share(code):
            return None
        sym = _prefixed(code)
        if sym is None:
            return None

        # Primary: Tencent natively-adjusted qfqday.
        df = self._fetch_tencent(sym, start_date, end_date)
        if df is not None and not df.empty:
            return df

        # Fallback: Sina raw bars re-adjusted with qfq.js factors.
        logger.info("ashare: tencent unavailable for %s, trying sina", code)
        return self._fetch_sina(sym, start_date, end_date)

    # ------------------------------------------------------------------
    # Tencent: natively forward-adjusted qfqday (chunked for long windows)
    # ------------------------------------------------------------------
    def _fetch_tencent(
        self, sym: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        min_interval = resolve_min_interval(_MIN_INTERVAL_ENV, _DEFAULT_MIN_INTERVAL)
        chunks: list[pd.DataFrame] = []
        cursor_end = end_date
        start_ts = pd.Timestamp(start_date)

        for _ in range(30):  # hard cap: 30 x 800 bars ≈ 96 years
            url = (
                f"{_TENCENT_KLINE_URL}?param={sym},day,"
                f"{start_date},{cursor_end},{_TENCENT_CHUNK},qfq"
            )
            try:
                resp = throttled_get(
                    url,
                    host_key=_TENCENT_HOST_KEY,
                    min_interval=min_interval,
                    headers={"User-Agent": _BROWSER_UA, "Referer": "https://web.ifzq.gtimg.cn/"},
                    timeout=20,
                )
                data = resp.json()
            except Exception as exc:
                logger.warning("ashare tencent request failed for %s: %s", sym, exc)
                return None

            stock = (data.get("data") or {})
            if not isinstance(stock, dict) or sym not in stock:
                return None
            klines = stock[sym].get("qfqday") or stock[sym].get("day")
            if not klines:
                return None

            chunks.append(_rows_to_frame(klines))
            oldest = pd.Timestamp(klines[0][0])
            if oldest <= start_ts or len(klines) < _TENCENT_CHUNK:
                break
            # Next chunk ends the day before this chunk's oldest bar.
            cursor_end = (oldest - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if not chunks:
            return None
        df = pd.concat(chunks)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return _finalize(df, start_date, end_date)

    # ------------------------------------------------------------------
    # Sina: raw K-lines + qfq.js corporate-action factors
    # ------------------------------------------------------------------
    def _fetch_sina(
        self, sym: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        min_interval = resolve_min_interval(_MIN_INTERVAL_ENV, _DEFAULT_MIN_INTERVAL)
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        # Generous datalen: Sina returns the trailing N trading days.
        datalen = max(400, int((end_ts - start_ts).days * 0.75))
        try:
            resp = throttled_get(
                _SINA_KLINE_URL,
                host_key=_SINA_HOST_KEY,
                min_interval=min_interval,
                params={
                    "symbol": sym, "scale": 240, "ma": 5, "datalen": datalen,
                },
                headers={"User-Agent": _BROWSER_UA},
                timeout=20,
            )
            bars = resp.json()
        except Exception as exc:
            logger.warning("ashare sina request failed for %s: %s", sym, exc)
            return None
        if not isinstance(bars, list) or not bars:
            return None

        df = pd.DataFrame(bars)
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["day"] = pd.to_datetime(df["day"])
        df = (
            df.set_index("day")
            .sort_index()[["open", "high", "low", "close", "volume"]]
            .dropna(subset=["open", "high", "low", "close"])
        )
        if df.empty:
            return None

        adjusted = self._apply_sina_qfq(sym, df, min_interval)
        if adjusted is None:
            return None
        return _finalize(adjusted, start_date, end_date)

    def _apply_sina_qfq(
        self, sym: str, df: pd.DataFrame, min_interval: float
    ) -> Optional[pd.DataFrame]:
        """Re-adjust Sina's unadjusted bars to forward-adjusted (前复权).

        Returns ``None`` when factors are unavailable — the symbol is dropped
        rather than benched on contaminated prices (matches ``cn_adjust``).
        """
        try:
            resp = throttled_get(
                _SINA_QFQ_URL.format(sym=sym),
                host_key=_SINA_HOST_KEY,
                min_interval=min_interval,
                headers={
                    "User-Agent": _BROWSER_UA,
                    "Referer": "https://finance.sina.com.cn/",
                },
                timeout=20,
            )
            match = re.search(r"=(\{.*\})", resp.text, re.DOTALL)
            if not match:
                return None
            entries = json.loads(match.group(1)).get("data") or []
        except Exception as exc:
            logger.warning("ashare sina qfq factor failed for %s: %s", sym, exc)
            return None
        if not entries:
            return None

        factor = pd.Series(
            {
                pd.Timestamp(e["d"]): float(e["f"])
                for e in entries
                if pd.Timestamp(e["d"]).year > 1950 and float(e["f"]) > 0
            }
        ).sort_index()
        if factor.empty:
            return None

        # Sina's qfq.js factor is cumulative-from-today: it equals 1.0 at the
        # latest ex-date and RISES going back in time (e.g. 600519.SH: 1.0 at
        # 2026-06-26, 1.0969 before 2024-06-19, 8.88 at IPO). Forward-adjusted
        # (前复权) prices are therefore ``raw / f`` — old bars scale DOWN, the
        # latest bar is unchanged, and ex-date returns lose the dividend gap
        # (verified against Tencent's native qfqday: 1521.50/1.0969 = 1387.16
        # on 600519.SH 2024-06-18, mean |return diff| 0.05% over 628 days).
        ratio = factor.reindex(df.index).ffill().fillna(factor.iloc[0])
        out = df.copy()
        for col in ("open", "high", "low", "close"):
            out[col] = out[col] / ratio
        if "volume" in out.columns:
            # Keep turnover (price x volume) invariant under the re-scaling.
            out["volume"] = out["volume"] * ratio
        return out


def _rows_to_frame(klines: list) -> pd.DataFrame:
    """Tencent row ``[date, open, close, high, low, volume, ...]`` -> OHLCV frame."""
    rows = []
    for k in klines:
        if len(k) >= 6:
            rows.append(
                {
                    "trade_date": k[0],
                    "open": float(k[1]),
                    "close": float(k[2]),
                    "high": float(k[3]),
                    "low": float(k[4]),
                    "volume": float(k[5]),
                }
            )
    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    return (
        df.set_index("trade_date")
        .sort_index()[["open", "high", "low", "close", "volume"]]
        .dropna(subset=["open", "high", "low", "close"])
    )


def _finalize(
    df: pd.DataFrame, start_date: str, end_date: str
) -> Optional[pd.DataFrame]:
    window = df.loc[start_date:end_date]
    ohlc = window[["open", "high", "low", "close"]]
    window = window[(ohlc > 0).all(axis=1)]
    return window if not window.empty else None
