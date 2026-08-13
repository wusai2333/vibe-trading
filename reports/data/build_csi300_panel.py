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

panel["_meta"] = {"universe": "csi300-ashare", "constituent_count": len(fetched),
                  "price_adjustment": "qfq", "survivorship_bias": True}

pickle.dump(panel, open("reports/data/csi300_panel.pkl", "wb"))
print(f"panel cached: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)
