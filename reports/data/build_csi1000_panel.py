"""Build + cache the CSI1000 panel (ashare loader), chunked and resumable.

Usage:
  python build_csi1000_panel.py --chunk N    # fetch batch N of 4 (250 codes), save partial pkl
  python build_csi1000_panel.py --assemble   # merge partials, patch last day, scrub, quality gate
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import pandas as pd

DATA = Path(__file__).resolve().parent
N_CHUNKS, CHUNK_SIZE = 4, 250

cons = json.load(open(DATA / "csi1000_cons.json"))
def suffixed(c):
    if c.startswith(("6", "9")): return f"{c}.SH"
    if c.startswith(("0", "2", "3")): return f"{c}.SZ"
    return None
CODES = sorted({suffixed(c) for c in cons["codes"]} - {None})

from backtest.loaders.registry import resolve_loader
FIELDS = ["open", "high", "low", "close", "volume"]
START = "2018-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

if "--chunk" in sys.argv:
    ci = int(sys.argv[sys.argv.index("--chunk") + 1])
    part = DATA / f"csi1000_part_{ci}.pkl"
    if part.exists():
        print(f"chunk {ci} already fetched, skip", file=sys.stderr)
        sys.exit(0)
    codes = CODES[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
    print(f"chunk {ci}: {len(codes)} codes", file=sys.stderr, flush=True)
    t0 = time.time()
    fetched = resolve_loader("a_share").fetch(codes, START, END)
    print(f"fetched {len(fetched)}/{len(codes)} in {time.time()-t0:.0f}s", file=sys.stderr, flush=True)
    panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in FIELDS}
    pickle.dump(panel, open(part, "wb"))
    print(f"SAVED {part}", file=sys.stderr)
    sys.exit(0)

if "--assemble" not in sys.argv:
    print("need --chunk N or --assemble", file=sys.stderr)
    sys.exit(1)

# ---- assemble ----
parts = []
for ci in range(N_CHUNKS):
    p = DATA / f"csi1000_part_{ci}.pkl"
    if not p.exists():
        print(f"MISSING {p}", file=sys.stderr)
        sys.exit(1)
    parts.append(pickle.load(open(p, "rb")))
panel = {f: pd.concat([pt[f] for pt in parts], axis=1) for f in FIELDS}
panel = {f: df.loc[:, ~df.columns.duplicated()] for f, df in panel.items()}
panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
print(f"merged: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)

# patch stocks missing the final bar (same as csi500 build)
_missing = panel["close"].iloc[-1][panel["close"].iloc[-1].isna()].index.tolist()
if _missing:
    print(f"patching {len(_missing)} stocks missing {panel['close'].index[-1].date()}", file=sys.stderr, flush=True)
    _patch = resolve_loader("a_share").fetch(_missing, (panel["close"].index[-1] - pd.Timedelta(days=45)).strftime("%Y-%m-%d"), END)
    for _f in FIELDS:
        for _k, _v in _patch.items():
            if _k in panel[_f].columns and not _v.empty:
                _common = panel[_f].index.intersection(_v.index)
                for _idx in _common:
                    if pd.isna(panel[_f].loc[_idx, _k]) and pd.notna(_v.loc[_idx, _f]):
                        panel[_f].loc[_idx, _k] = _v.loc[_idx, _f]
    panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
    print(f"after patch, still missing last day: {panel['close'].iloc[-1].isna().sum()}", file=sys.stderr)

sys.path.insert(0, str(DATA))
from panel_scrub import scrub
_scrub_report = scrub(panel)

_close = panel["close"]; _days = _close.index
_ret = _close.pct_change()
_lim = pd.DataFrame(0.10, index=_days, columns=_close.columns)
_star = [c for c in _close.columns if c.startswith("688")]
_gem = [c for c in _close.columns if c.startswith("30")]
if _star: _lim[_star] = 0.20
if _gem: _lim.loc[_days >= pd.Timestamp("2020-08-24"), _gem] = 0.20
_first_back = _close.notna() & _close.shift(1).isna()
_long_gap = _first_back & _close.shift(20).isna()
_impossible = (_ret.abs() > _lim + 0.10) & ~_long_gap
_n_impossible = int(_impossible.sum().sum())
_quality = {"scrub_masked_cells": _scrub_report["masked_cells"],
            "impossible_returns_remaining": _n_impossible,
            "verdict": "PASS" if _n_impossible <= 10 else "RECONCILE_RECOMMENDED"}
print(f"DATA QUALITY: scrub masked {_scrub_report['masked_cells']} cells; "
      f"{_n_impossible} impossible one-day returns remain -> {_quality['verdict']}", file=sys.stderr)

panel["_meta"] = {"universe": "csi1000-ashare", "constituent_count": panel["close"].shape[1],
                  "price_adjustment": "qfq", "survivorship_bias": True,
                  "scrub": _scrub_report, "quality": _quality}
pickle.dump(panel, open(DATA / "csi1000_panel.pkl", "wb"))
print(f"panel cached: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)