"""Build + cache the full CSI300 panel (ashare loader) for reuse."""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
import pandas as pd

cons = json.load(open("reports/data/csi300_cons.json"))
def suffixed(c):
    if c.startswith(("6", "9")): return f"{c}.SH"
    if c.startswith(("0", "2", "3")): return f"{c}.SZ"
    return None
CODES = sorted({suffixed(c) for c in cons["codes"]} - {None})
print(f"universe: {len(CODES)}", file=sys.stderr)

from backtest.loaders.registry import resolve_loader
t0 = time.time()
fetched = resolve_loader("a_share").fetch(CODES, "2018-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
print(f"fetched {len(fetched)}/{len(CODES)} in {time.time()-t0:.0f}s", file=sys.stderr)
fields = ["open", "high", "low", "close", "volume"]
panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in fields}
panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0

# Patch any stocks missing the final day's bar (the tencent chunked fetch can
# drop the in-progress last bar for some names). Re-fetch a short window and
# fill only the missing cells so the panel's last row is complete.
_missing = panel["close"].iloc[-1][panel["close"].iloc[-1].isna()].index.tolist()
if _missing:
    print(f"patching {len(_missing)} stocks missing {panel['close'].index[-1].date()}", file=sys.stderr)
    _patch = resolve_loader("a_share").fetch(_missing, (panel["close"].index[-1] - pd.Timedelta(days=45)).strftime("%Y-%m-%d"), pd.Timestamp.today().strftime("%Y-%m-%d"))
    for _f in fields:
        for _k, _v in _patch.items():
            if _k in panel[_f].columns and not _v.empty:
                _common = panel[_f].index.intersection(_v.index)
                for _idx in _common:
                    if pd.isna(panel[_f].loc[_idx, _k]) and pd.notna(_v.loc[_idx, _f]):
                        panel[_f].loc[_idx, _k] = _v.loc[_idx, _f]
    panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
    print(f"after patch, still missing last day: {panel['close'].iloc[-1].isna().sum()}", file=sys.stderr)

# Scrub corrupted price segments (dual-source stitching can splice in bars
# whose level is off by 10-50x; they become NaN, same as suspensions).
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from panel_scrub import scrub
_scrub_report = scrub(panel)

# ---- data-quality gate ----
# Scrub only masks level dislocations; residual impossible one-day returns
# (bad splices the level check can't see) are counted here so corruption is
# never silent. Limit-aware: main board 10%, ChiNext/STAR 20%; first-day-back
# from a LONG suspension is exempt (no price limit on such resumptions).
_close = panel["close"]; _days = _close.index
_ret = _close.pct_change()
_lim = pd.DataFrame(0.10, index=_days, columns=_close.columns)
_star = [c for c in _close.columns if c.startswith("688")]
_gem = [c for c in _close.columns if c.startswith("30")]
if _star: _lim[_star] = 0.20
if _gem: _lim.loc[_days >= pd.Timestamp("2020-08-24"), _gem] = 0.20
_first_back = _close.notna() & _close.shift(1).isna()
_long_gap = _first_back & _close.shift(20).isna()
# "clearly impossible" = >10pp beyond the board limit (small 12-15% splices are
# a tolerated residual class the backtest return-guard zeroes in P&L). A fully
# reconciled panel leaves ~7 such cells (incl. the known 601088.SH +54%); a
# spike well above that signals fresh corruption.
_impossible = (_ret.abs() > _lim + 0.10) & ~_long_gap
_n_impossible = int(_impossible.sum().sum())
_quality = {"scrub_masked_cells": _scrub_report["masked_cells"],
            "impossible_returns_remaining": _n_impossible,
            "verdict": "PASS" if _n_impossible <= 10 else "RECONCILE_RECOMMENDED"}
print(f"DATA QUALITY: scrub masked {_scrub_report['masked_cells']} cells; "
      f"{_n_impossible} impossible one-day returns remain -> {_quality['verdict']}",
      file=sys.stderr)
if _quality["verdict"] != "PASS":
    print("  -> run `python reports/data/panel_reconcile.py` to repair "
          "(Sina+Baostock arbitration, ~10 min)", file=sys.stderr)

panel["_meta"] = {"universe": "csi300-ashare", "constituent_count": len(fetched),
                  "price_adjustment": "qfq", "survivorship_bias": True,
                  "scrub": _scrub_report, "quality": _quality}

pickle.dump(panel, open("reports/data/csi300_panel.pkl", "wb"))
print(f"panel cached: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)
