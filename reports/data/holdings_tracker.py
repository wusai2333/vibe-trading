"""Daily factor tracking for the personal holdings portfolio.

Scores each holding with the stable-5 rolling-weight blend (same calibration
as stable5_screener.py: trailing-252d weights, rolling(10, min_periods=6),
cross-sectional percentile within the CSI300 panel). Appends one row per
holding per run to holdings_history.csv so percentile trends are visible
across days.

Usage:
    python holdings_tracker.py            # score today, append history
    python holdings_tracker.py --report   # also print trend from history
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

DATA_DIR = Path(__file__).resolve().parent
HISTORY = DATA_DIR / "holdings_history.csv"

HOLDINGS = {
    "600011.SH": ("华能国际", 25),
    "601899.SH": ("紫金矿业", 20),
    "000657.SZ": ("中钨高新", 15),
    "600795.SH": ("国电电力", 10),
    "000408.SZ": ("藏格矿业", 8),
    "000975.SZ": ("中金黄金", 8),
    "601061.SH": ("中信金属", 7),
}
STABLE_IDS = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060"]
TRAIN = 252


def _finite(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def run() -> dict:
    import pickle
    panel = pickle.load(open(DATA_DIR / "csi300_panel.pkl", "rb"))
    close = panel["close"]
    fwd = close.pct_change().shift(-1)
    days = close.index

    # fetch holdings not in the CSI300 panel
    missing = [s for s in HOLDINGS if s not in close.columns]
    if missing:
        from backtest.loaders.registry import resolve_loader
        fetched = resolve_loader("a_share").fetch(
            missing, "2018-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
        for f in ["open", "high", "low", "close", "volume"]:
            for k, v in fetched.items():
                panel[f][k] = v[f]
        panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
        close = panel["close"]
        fwd = close.pct_change().shift(-1)
        days = close.index

    from src.factors.registry import get_default_registry
    reg = get_default_registry()

    def zscore(df):
        mu, sd = df.mean(axis=1), df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
           for a in STABLE_IDS}
    ic = {a: pd.Series([fac[a].loc[t].corr(fwd.loc[t]) for t in days[-TRAIN - 1:-1]],
                       index=days[-TRAIN - 1:-1]) for a in STABLE_IDS}

    def ir_of(s):
        s = s.dropna()
        return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

    irs = {a: ir_of(ic[a]) for a in STABLE_IDS}
    wsum = sum(abs(v) for v in irs.values())
    weights = {a: v / wsum for a, v in irs.items()}
    blend = sum(fac[a] * weights[a] for a in STABLE_IDS)

    last = blend.iloc[-1].dropna()
    ret5 = close.iloc[-1] / close.iloc[-6] - 1
    ret20 = close.iloc[-1] / close.iloc[-21] - 1

    rows = []
    for sym, (name, wt) in HOLDINGS.items():
        v = blend.loc[days[-1], sym] if sym in blend.columns else np.nan
        pct = float((last < v).mean()) * 100 if pd.notna(v) else None
        rows.append({
            "date": str(days[-1].date()),
            "symbol": sym,
            "name": name,
            "weight_pct": wt,
            "score": _finite(v, 4),
            "pct_rank": _finite(pct, 1),
            "mom_5d_pct": _finite(ret5.get(sym, np.nan) * 100, 1) if sym in ret5.index else None,
            "mom_20d_pct": _finite(ret20.get(sym, np.nan) * 100, 1) if sym in ret20.index else None,
            "close": _finite(close[sym].iloc[-1], 2) if sym in close.columns else None,
        })

    scored = [(r["pct_rank"], r["weight_pct"]) for r in rows if r["pct_rank"] is not None]
    wpct = sum(p * w for p, w in scored) / sum(w for _, w in scored) if scored else None
    result = {"as_of": str(days[-1].date()),
              "weights": {k: round(v, 3) for k, v in weights.items()},
              "weighted_pct_rank": _finite(wpct, 1),
              "holdings": sorted(rows, key=lambda r: -(r["pct_rank"] or -1))}

    # append history
    df_new = pd.DataFrame(rows)
    if HISTORY.exists():
        df_old = pd.read_csv(HISTORY)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=["date", "symbol"], keep="last")
    else:
        df_all = df_new
    df_all.to_csv(HISTORY, index=False)
    return result


def report() -> None:
    if not HISTORY.exists():
        print("no history yet")
        return
    df = pd.read_csv(HISTORY)
    dates = sorted(df["date"].unique())
    print(f"history: {len(dates)} days ({dates[0]} -> {dates[-1]})")
    for sym, (name, wt) in HOLDINGS.items():
        sub = df[df["symbol"] == sym].sort_values("date")
        if sub.empty:
            continue
        pcts = [f"{p:.0f}" if pd.notna(p) else "-" for p in sub["pct_rank"]]
        print(f"{name:6s} {wt:>2d}%  分位轨迹: {' -> '.join(pcts)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Daily factor tracking for holdings")
    ap.add_argument("--report", action="store_true", help="print percentile trend from history")
    args = ap.parse_args()

    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False))
    print(f"\nSAVED {HISTORY}", file=sys.stderr)
    if args.report:
        print("\n=== 分位轨迹 ===")
        report()


if __name__ == "__main__":
    main()
