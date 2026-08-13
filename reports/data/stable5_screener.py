"""Stable-5 screener: stock picks + sector ranking.

Uses the LAST rolling-training window's weights (no future information):
blend signal on the final day = IC-weighted z-scores of the 5 sign-stable
factors, weights fitted on the trailing 252 trading days.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA_DIR = Path(__file__).resolve().parent
panel = pickle.load(open(DATA_DIR / "csi300_panel.pkl", "rb"))
close = panel["close"]
fwd = close.pct_change().shift(-1)
days = close.index

STABLE_IDS = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060"]
TRAIN = 252

from src.factors.registry import get_default_registry
reg = get_default_registry()

def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

fac = {aid: zscore(reg.compute(aid, panel).rolling(10, min_periods=6).mean()) for aid in STABLE_IDS}

# weights from the trailing-252d window ending at the last day
ic = {aid: pd.Series([fac[aid].loc[t].corr(fwd.loc[t]) for t in days[-TRAIN - 1:-1]],
                     index=days[-TRAIN - 1:-1]) for aid in STABLE_IDS}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
irs = {aid: ir_of(ic[aid]) for aid in STABLE_IDS}
wsum = sum(abs(v) for v in irs.values())
weights = {aid: v / wsum for aid, v in irs.items()}
print("final-window weights:", {k: round(v, 3) for k, v in weights.items()}, file=sys.stderr)

blend = sum(fac[aid] * weights[aid] for aid in STABLE_IDS)
last = blend.iloc[-1].dropna()
ret20 = close.iloc[-1] / close.iloc[-21] - 1

# ---- sector mapping (csindex sector indices) ----
SECTORS = {"000928": "能源", "000929": "材料", "000930": "工业", "000931": "可选消费",
           "000932": "主要消费", "000933": "医药卫生", "000934": "金融地产",
           "000935": "信息技术", "000936": "电信服务", "000937": "公用事业"}
import akshare as ak
def _load_sector_map() -> dict:
    """Stock -> CSI sector mapping with a 7-day local cache (constituents
    change quarterly, so the cache is safe and avoids slow csindex calls)."""
    cache = DATA_DIR / "stock2sector_cache.json"
    if cache.exists():
        age_days = (time.time() - cache.stat().st_mtime) / 86400
        if age_days < 7:
            return json.load(open(cache))
    mapping = {}
    for code, name in SECTORS.items():
        try:
            df = ak.index_stock_cons_weight_csindex(symbol=code)
            for c in df["成分券代码"].astype(str).str.zfill(6):
                mapping[c] = name
        except Exception:
            pass
        time.sleep(0.2)
    if mapping:
        json.dump(mapping, open(cache, "w"), ensure_ascii=False)
    return mapping

stock2sector = _load_sector_map()

cons = json.load(open(DATA_DIR / "csi300_cons.json"))
names = dict(zip(cons["codes"], cons["names"]))

def finite(x, nd):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None

# ---- stock view ----
picks = last.sort_values(ascending=False).head(20)
stock_rows = []
for i, (sym, val) in enumerate(picks.items(), 1):
    code = sym.split(".")[0]
    stock_rows.append({
        "rank": i, "symbol": sym, "name": names.get(code, ""),
        "sector": stock2sector.get(code, "未分类"),
        "score": finite(val, 4),
        "pct_rank": finite((last < val).mean() * 100, 1),
        "mom_20d_pct": finite(ret20.get(sym, np.nan) * 100, 1) if sym in ret20.index else None,
        "last_close": finite(close[sym].iloc[-1], 2) if sym in close.columns else None,
    })

# ---- sector view ----
sect = last.groupby(pd.Series({s: stock2sector.get(s.split(".")[0], "未分类") for s in last.index}))
sect_rows = [{"sector": n, "avg_score": finite(g.mean(), 4), "names_scored": int(len(g)),
              "in_top20": sum(1 for r in stock_rows if r["sector"] == n)}
             for n, g in sect if len(g) >= 5]
sect_rows.sort(key=lambda r: -(r["avg_score"] or 0))

out = {"as_of": str(days[-1].date()),
       "factors": STABLE_IDS,
       "weights": {k: round(v, 3) for k, v in weights.items()},
       "weight_window": {"start": str(days[-TRAIN - 1].date()), "end": str(days[-1].date())},
       "positioning": "relative-rank filter only; no absolute-alpha claim",
       "names_scored": int(len(last)),
       "top_picks": stock_rows,
       "sector_ranking": sect_rows}
out_path = DATA_DIR / "stable5_screener_latest.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False))
print(f"\nSAVED {out_path}", file=sys.stderr)
