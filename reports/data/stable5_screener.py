"""Stable-7 screener: stock picks + sector ranking.

Uses the LAST rolling-training window's weights (no future information):
blend signal on the final day = IC-weighted z-scores of the 7 production
factors, weights fitted on the trailing 252 trading days.

History: launched as stable-5; upgraded to stable-7 on 2026-08-17 (user
approved) by adding limit_dist (limit-up microstructure, confirmed_alive)
and vol_ivol60 (idiosyncratic volatility, reversed -> negative weight):
constrained OOS 30.3%/yr vs 25.3% for stable-5, Sharpe 1.08, 2026YTD +19.0%
vs -2.1%. See reports/2026-08-17_因子挖掘_涨跌停与波动率.md. Filename kept
so the daily three-piece commands stay unchanged.

2026-08-21 (user approved option A): blend in a FIXED 10% EP sleeve -
signal = 0.9 * IR-weighted(stable-7) + 0.1 * z(fund_earnings_yield).
EP passed gate-3 despite IC 0.0163 < the 0.02 bar (first counterexample);
validated: fixed 10% -> constrained OOS 31.9%/Sharpe 1.17 vs 29.4%/1.04,
gain concentrated in bull regimes, neutral outside. See
reports/2026-08-21_CST_practice_and_EP.md)
"""
import sys, json, pickle, warnings, time
import requests
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd

DATA_DIR = Path(__file__).resolve().parent
panel = pickle.load(open(DATA_DIR / "csi300_panel.pkl", "rb"))
close = panel["close"]
fwd = close.pct_change().shift(-1)
days = close.index
_fund = pickle.load(open(DATA_DIR / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        # ffill past the cache build date: quarterly values persist until the
        # next announcement, so carrying the last known value is PIT-safe
        panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()

STABLE_IDS = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
              "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN = 252
EP_W = 0.10  # fixed EP sleeve, user-approved 2026-08-21 (option A)

from src.factors.registry import get_default_registry
reg = get_default_registry()

def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

fac = {aid: zscore(reg.compute(aid, panel).rolling(10, min_periods=6).mean()) for aid in STABLE_IDS}
ep = zscore(reg.compute("fund_earnings_yield", panel))  # PIT: pubDate ffill

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

blend = (1 - EP_W) * sum(fac[aid] * weights[aid] for aid in STABLE_IDS) + EP_W * ep
last = blend.iloc[-1].dropna()
ret20 = close.iloc[-1] / close.iloc[-21] - 1

# ---- sector mapping (csindex sector indices) ----
SECTORS = {"000928": "能源", "000929": "材料", "000930": "工业", "000931": "可选消费",
           "000932": "主要消费", "000933": "医药卫生", "000934": "金融地产",
           "000935": "信息技术", "000936": "电信服务", "000937": "公用事业"}
import akshare as ak
def _load_sector_map() -> dict:
    """Stock -> CSI sector mapping with a 30-day local cache (constituents
    change quarterly, so the cache is safe; csindex calls are slow/flaky,
    so keep the TTL comfortably longer than the refresh cadence)."""
    cache = DATA_DIR / "stock2sector_cache.json"
    if cache.exists():
        age_days = (time.time() - cache.stat().st_mtime) / 86400
        if age_days < 30:
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

# ---- hq20 exposure gate（S10，2026-08-24 用户拍板上生产）----
def refresh_index(last_day) -> pd.Series:
    """沪深300 日线：sina 全史覆写缓存；sina 滞后时用腾讯实时补当日（前收吻合才补）。"""
    p = DATA_DIR / "csi300_index_daily.csv"
    try:
        import akshare as ak
        ak.stock_zh_index_daily(symbol="sh000300").to_csv(p, index=False)
    except Exception as e:
        print(f"WARNING: 指数抓取失败，用旧缓存: {e}", file=sys.stderr)
    df = pd.read_csv(p, parse_dates=["date"])
    if pd.Timestamp(df["date"].iloc[-1]) < last_day:
        try:
            f = requests.get("https://qt.gtimg.cn/q=sh000300", timeout=10).content.decode("gbk").split("~")
            if abs(float(f[4]) - float(df["close"].iloc[-1])) / float(f[4]) < 1e-4:
                row = {"date": last_day, "open": float(f[5]), "high": float(f[33]),
                       "low": float(f[34]), "close": float(f[3]), "volume": float(f[6])}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.to_csv(p, index=False)
        except Exception as e:
            print(f"WARNING: 当日指数补全失败，用缓存: {e}", file=sys.stderr)
    return df.set_index("date")["close"]

def hq20_gate(idx: pd.Series) -> dict:
    r1 = idx.pct_change(fill_method=None)
    def hk(k):
        vk = idx.pct_change(k, fill_method=None).rolling(120).var()
        return np.log(vk / r1.rolling(120).var()) / (2 * np.log(k))
    H = (pd.concat([hk(k) for k in (2, 4, 8, 16)], axis=1)
         .mean(axis=1).rolling(5).mean().clip(-0.5, 1.5))
    thr = H.rolling(252).quantile(0.20)
    h, t = float(H.iloc[-1]), float(thr.iloc[-1])
    half = h < t
    return {"rule": "hq20 (S10)", "H": round(h, 3), "threshold_20pct": round(t, 3),
            "exposure": 0.5 if half else 1.0,
            "note": "H 落近252日最低20%分位 -> 建议半仓" if half else "正常仓位"}

hgate = hq20_gate(refresh_index(days[-1]))
print(f"hq20 gate: H={hgate['H']} thr={hgate['threshold_20pct']} -> exposure={hgate['exposure']}",
      file=sys.stderr)

out = {"as_of": str(days[-1].date()),
       "factors": STABLE_IDS + ["fund_earnings_yield"],
       "hgate": hgate,
       "weights": {k: round(v * (1 - EP_W), 3) for k, v in weights.items()},
       "ep_fixed_weight": EP_W,
       "weight_window": {"start": str(days[-TRAIN - 1].date()), "end": str(days[-1].date())},
       "positioning": "relative-rank filter only; no absolute-alpha claim",
       "names_scored": int(len(last)),
       "top_picks": stock_rows,
       "sector_ranking": sect_rows}
out_path = DATA_DIR / "stable5_screener_latest.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False))
print(f"\nSAVED {out_path}", file=sys.stderr)
