"""Composite-factor screener: IC-weighted 12-factor blend, stock + sector views.

Factor set: the 12 factors selected in csi300_multifactor.py (top-|IR|,
capped at 4 per zoo). Blend: cross-sectional z-score then IC-weighted sum
(weights from the full-sample IR, sign included).

Two views in one run:
  1. Stock view  — top-N names by composite score, with sector labels.
  2. Sector view — average composite score per CSI300 sector (relative
     sector strength), ranked.

Positioning: relative-rank filter only. No absolute-alpha claim.

Usage:
    python composite_screener.py [--top 15] [--smooth 10]
Requires: reports/data/csi300_panel.pkl (build_csi300_panel.py)
          /tmp/stock2sector.json is rebuilt on the fly from csindex.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))

import numpy as np
import pandas as pd

# The 12 factors + weights from the multi-factor study (csi300_multifactor.json).
BLEND = [
    ("gtja191_171", 0.260), ("alpha101_083", 0.257), ("alpha101_054", 0.242),
    ("qlib158_ksft2", -0.242), ("qlib158_ksft", -0.241), ("qlib158_klow", -0.240),
    ("gtja191_111", 0.222), ("alpha101_060", 0.211), ("alpha101_042", 0.205),
    ("gtja191_178", -0.196), ("qlib158_kmid", -0.190), ("gtja191_150", -0.168),
]

DATA_DIR = Path(__file__).resolve().parent
SECTORS = {
    "000928": "能源", "000929": "材料", "000930": "工业", "000931": "可选消费",
    "000932": "主要消费", "000933": "医药卫生", "000934": "金融地产",
    "000935": "信息技术", "000936": "电信服务", "000937": "公用事业",
}


def _finite(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _stock2sector() -> dict[str, str]:
    """Rebuild stock -> CSI sector mapping from csindex sector indices."""
    import akshare as ak
    mapping: dict[str, str] = {}
    for code, name in SECTORS.items():
        try:
            df = ak.index_stock_cons_weight_csindex(symbol=code)
            for c in df["成分券代码"].astype(str).str.zfill(6):
                mapping[c] = name
        except Exception:
            pass
        time.sleep(0.2)
    return mapping


def _names() -> dict[str, str]:
    cons = json.loads((DATA_DIR / "csi300_cons.json").read_text(encoding="utf-8"))
    return dict(zip(cons["codes"], cons["names"]))


def run(top_n: int, smooth: int) -> dict:
    import pickle
    panel = pickle.load(open(DATA_DIR / "csi300_panel.pkl", "rb"))
    close = panel["close"]
    ret20 = close.iloc[-1] / close.iloc[-21] - 1

    from src.factors.registry import get_default_registry
    reg = get_default_registry()

    def zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu, sd = df.mean(axis=1), df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    wsum = sum(abs(w) for _, w in BLEND)
    blend = None
    for aid, w in BLEND:
        f = reg.compute(aid, panel)
        if smooth > 1:
            f = f.rolling(smooth).mean()
        term = zscore(f) * (w / wsum)
        blend = term if blend is None else blend + term

    last = blend.iloc[-1].dropna()
    ranked = last.sort_values(ascending=False)
    picks = ranked.head(top_n)

    stock2sector = _stock2sector()
    names = _names()

    stock_rows = []
    for i, (sym, val) in enumerate(picks.items(), 1):
        code = sym.split(".")[0]
        stock_rows.append({
            "rank": i,
            "symbol": sym,
            "name": names.get(code, ""),
            "sector": stock2sector.get(code, "未分类"),
            "score": _finite(val, 4),
            "pct_rank": _finite((last < val).mean() * 100, 1),
            "mom_20d_pct": _finite(ret20.get(sym, np.nan) * 100, 1) if sym in ret20.index else None,
            "last_close": _finite(close[sym].iloc[-1], 2) if sym in close.columns else None,
        })

    # Sector view: average composite score per sector (relative strength).
    sector_score = pd.Series(
        {sym: stock2sector.get(sym.split(".")[0], "未分类") for sym in last.index}
    )
    sect = last.groupby(sector_score).agg(["mean", "count"])
    sect = sect[sect["count"] >= 5].sort_values("mean", ascending=False)
    sector_rows = [
        {
            "sector": name,
            "avg_score": _finite(row["mean"], 4),
            "names_ranked": int(row["count"]),
            "in_top_picks": sum(1 for r in stock_rows if r["sector"] == name),
        }
        for name, row in sect.iterrows()
    ]

    return {
        "as_of": str(close.index[-1].date()),
        "blend": {
            "factors": len(BLEND), "smooth_days": smooth,
            "weighting": "IC-weighted (full-sample IR, sign included)",
        },
        "positioning": "relative-rank filter only; no absolute-alpha claim",
        "pool_stats": {"names_scored": int(len(last)),
                       "score_median": _finite(last.median(), 4)},
        "top_picks": stock_rows,
        "sector_ranking": sector_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Composite-factor screener (stock + sector)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--smooth", type=int, default=10)
    args = ap.parse_args()

    result = run(args.top, args.smooth)
    print(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False))
    out = DATA_DIR / "composite_screener_latest.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
    print(f"\nSAVED {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
