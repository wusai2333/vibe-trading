"""gtja191_171 relative-strength screener.

POSITIONING (read before using): the factor does NOT beat an equal-weight
basket net of costs on the full CSI300 universe. Its real, reproducible
property is a MONOTONIC cross-sectional rank — the highest-factor quintile
outperforms the lowest. So this tool is a RELATIVE-RANK FILTER: given a
candidate pool, it orders names by factor strength and hands back the
relatively stronger ones. It makes no absolute-alpha or outperformance claim.

Factor: gtja191_171, smoothed with a rolling mean to cut ranking churn.
Data: ashare dual-source loader (forward-adjusted). Universe: a code list
file (one JSON with a "codes" array) or the bundled CSI300 roster.

Usage:
    python gtja171_screener.py [--universe PATH] [--top N] [--smooth DAYS]
    Defaults: bundled CSI300 roster, top 15, smooth 10 trading days.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = Path(__file__).resolve().parent / "csi300_cons.json"


def _suffixed(code: str) -> str | None:
    if code.startswith(("6", "9")):
        return f"{code}.SH"
    if code.startswith(("0", "2", "3")):
        return f"{code}.SZ"
    return None


def _load_universe(path: Path) -> list[str]:
    codes = json.loads(path.read_text(encoding="utf-8"))["codes"]
    return sorted({_suffixed(c) for c in codes} - {None})


def _build_panel(loader, codes: list[str], start: str, end: str):
    fetched = loader.fetch(codes, start, end)
    fields = ["open", "high", "low", "close", "volume"]
    panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in fields}
    panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
    return panel, fetched


def run_screen(universe_path: Path, top_n: int, smooth: int, lookback_days: int) -> dict:
    from backtest.loaders.registry import resolve_loader
    from src.factors.registry import get_default_registry

    codes = _load_universe(universe_path)
    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    loader = resolve_loader("a_share")
    panel, fetched = _build_panel(loader, codes, start, end)

    reg = get_default_registry()
    factor_raw = reg.compute("gtja191_171", panel)
    factor = factor_raw if smooth <= 1 else factor_raw.rolling(smooth).mean()

    close = panel["close"]
    last = factor.iloc[-1].dropna()
    if last.empty:
        raise RuntimeError("factor produced no values; check the data panel")

    ranked = last.sort_values(ascending=False)
    picks = ranked.head(top_n)

    # Cross-sectional percentile + a 20d momentum column for context only.
    ret20 = close.iloc[-1] / close.iloc[-21] - 1

    def _finite(x, nd):
        """Round to nd digits, returning None for NaN/inf (strict JSON)."""
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return round(v, nd) if np.isfinite(v) else None

    rows = []
    for i, (sym, val) in enumerate(picks.items(), 1):
        rows.append({
            "rank": i,
            "symbol": sym,
            "factor": _finite(val, 4),
            "pct_rank": _finite((last < val).mean() * 100, 1),
            "mom_20d_pct": _finite(ret20.get(sym, np.nan) * 100, 1) if sym in ret20.index else None,
            "last_close": _finite(close[sym].iloc[-1], 2) if sym in close.columns else None,
        })

    return {
        "as_of": end,
        "universe": {"requested": len(codes), "fetched": len(fetched),
                     "start": start, "smooth_days": smooth, "factor": "gtja191_171"},
        "positioning": "relative-rank filter only; no absolute-alpha claim",
        "top_picks": rows,
        "pool_stats": {
            "names_ranked": int(len(last)),
            "factor_median": round(float(last.median()), 4),
            "factor_max": round(float(last.max()), 4),
            "factor_min": round(float(last.min()), 4),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="gtja191_171 relative-strength screener")
    ap.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE,
                    help="JSON file with a 'codes' array (default: bundled CSI300)")
    ap.add_argument("--top", type=int, default=15, help="names to return")
    ap.add_argument("--smooth", type=int, default=10, help="rolling-mean window in trading days")
    ap.add_argument("--lookback", type=int, default=400, help="calendar days of history to fetch")
    args = ap.parse_args()

    result = run_screen(args.universe, args.top, args.smooth, args.lookback)
    print(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False))

    out = Path(__file__).resolve().parent / "screener_latest.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    print(f"\nSAVED {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
