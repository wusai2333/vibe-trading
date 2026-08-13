"""CSI300 sector rotation screener (relative ranking, no alpha claim).

Ranks the ten CSI 300 sector indices (000928-000937) by trailing momentum.
Data: Tencent fqkline HTTP API (no token). The screener is a RELATIVE-RANK
filter: it orders sectors by 1m/3m momentum and flags the strongest, not a
forecast of future returns.

Usage:
    python sector_screener.py [--lookback-days 250] [--top 3]
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import requests

SECTORS = {
    "sh000928": "能源",
    "sh000929": "材料",
    "sh000930": "工业",
    "sh000931": "可选消费",
    "sh000932": "主要消费",
    "sh000933": "医药卫生",
    "sh000934": "金融地产",
    "sh000935": "信息技术",
    "sh000936": "电信服务",
    "sh000937": "公用事业",
}
_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://web.ifzq.gtimg.cn/"}


def _fetch_closes(code: str, lookback: int) -> list[float]:
    """Return the trailing close series for one index via Tencent fqkline."""
    count = max(60, lookback + 30)  # pad for non-trading days
    params = {"param": f"{code},day,,,{count},qfq"}
    resp = requests.get(_URL, params=params, headers=_HEADERS, timeout=15, allow_redirects=True)
    resp.raise_for_status()
    stk = json.loads(resp.content)["data"][code]
    rows = stk.get("qfqday") or stk.get("day") or []
    return [float(r[2]) for r in rows]


def _momentum(closes: list[float], n: int) -> float | None:
    if len(closes) <= n:
        return None
    return round((closes[-1] / closes[-1 - n] - 1) * 100, 2)


def run(lookback_days: int, top_n: int) -> dict:
    rows = []
    for code, name in SECTORS.items():
        try:
            closes = _fetch_closes(code, lookback_days)
        except Exception as exc:
            rows.append({"code": code, "sector": name, "error": str(exc)[:80]})
            continue
        rows.append({
            "code": code,
            "sector": name,
            "close": closes[-1],
            "mom_5d_pct": _momentum(closes, 5),
            "mom_20d_pct": _momentum(closes, 20),
            "mom_60d_pct": _momentum(closes, 60),
        })
        time.sleep(0.5)  # stay polite to the host

    scored = [r for r in rows if "mom_60d_pct" in r and r["mom_60d_pct"] is not None]
    # Composite score: equal-weight blend of 20d and 60d momentum ranks.
    for key in ("mom_20d_pct", "mom_60d_pct"):
        order = sorted(scored, key=lambda r: r[key], reverse=True)  # rank 1 = strongest
        for rank, r in enumerate(order):
            r[f"{key}_rank"] = rank + 1
    for r in scored:
        r["score"] = round((r["mom_20d_pct_rank"] + r["mom_60d_pct_rank"]) / 2, 1)
    scored.sort(key=lambda r: r["score"])

    return {
        "positioning": "relative sector-rotation ranking; no return forecast",
        "sectors_ranked": len(scored),
        "top_sectors": scored[:top_n],
        "all_sectors": scored,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="CSI300 sector rotation screener")
    ap.add_argument("--lookback-days", type=int, default=250)
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    result = run(args.lookback_days, args.top)
    print(json.dumps(result, ensure_ascii=False, indent=1))

    out = Path(__file__).resolve().parent / "sector_screener_latest.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nSAVED {out}", flush=True)


if __name__ == "__main__":
    main()
