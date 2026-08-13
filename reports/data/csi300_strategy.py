"""gtja191_171 cross-sectional strategy on the FULL CSI300 universe.

Panel: ashare loader (dual-source qfq). Universe: current CSI300
constituents (akshare index_stock_cons). Note: point-in-time membership is
NOT available with the free token, so this uses the current roster (the
same survivorship caveat as before, now stated explicitly).

Runs both the raw top-5 variant (for comparison with the 30-name result)
and the smooth20/top15 variant. Reports both against an equal-weight
buy-and-hold benchmark of the same universe.
"""
import sys, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
import numpy as np, pandas as pd

cons = json.load(open("/tmp/csi300_cons.json"))
def suffixed(code: str) -> str | None:
    if code.startswith(("6", "9")): return f"{code}.SH"
    if code.startswith(("0", "2", "3")): return f"{code}.SZ"
    return None
CODES = sorted({suffixed(c) for c in cons["codes"]} - {None})
print(f"universe: {len(CODES)} symbols", file=sys.stderr)

from backtest.loaders.registry import resolve_loader
loader = resolve_loader("a_share")
t0 = time.time()
fetched = loader.fetch(CODES, "2018-01-01", "2025-12-31")
print(f"fetched {len(fetched)}/{len(CODES)} in {time.time()-t0:.0f}s", file=sys.stderr)

fields = ["open", "high", "low", "close", "volume"]
panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in fields}
panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0

from src.factors.registry import get_default_registry
factor_raw = get_default_registry().compute("gtja191_171", panel)
close = panel["close"]
fwd = close.pct_change().shift(-1)
COST, REBAL = 0.001, 5

def backtest(factor: pd.DataFrame, top_n: int) -> dict:
    w = pd.DataFrame(0.0, index=factor.index, columns=factor.columns)
    last = None
    for i, t in enumerate(factor.index):
        if last is None or i % REBAL == 0:
            rowv = factor.loc[t].dropna()
            if len(rowv) >= top_n:
                last = set(rowv.nlargest(top_n).index)
        if last:
            w.loc[t, list(last)] = 1.0 / top_n
    gross = (w * fwd).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= w.sum(axis=1).gt(0).idxmax()]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    return {
        "total_pct": round(float(eq.iloc[-1] - 1) * 100, 1),
        "cagr_pct": round(cagr * 100, 1),
        "sharpe": round(cagr / vol, 2) if vol > 0 else None,
        "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
        "ann_turnover": round(float(turn[eq.index].sum() / yrs), 1),
    }

# benchmark: equal-weight universe buy & hold
bn_ret = fwd.mean(axis=1).shift(1).fillna(0.0)
bn = (1 + bn_ret).cumprod()

results = {}
for name, factor, top_n in [
    ("raw_top5", factor_raw, 5),
    ("smooth20_top15", factor_raw.rolling(20).mean(), 15),
    ("smooth10_top10", factor_raw.rolling(10).mean(), 10),
]:
    results[name] = backtest(factor, top_n)
    print(f"{name}: {results[name]}", flush=True)

# benchmark stats
yrs = (bn.index[-1] - bn.index[0]).days / 365.25
bn_cagr = float(bn.iloc[-1] ** (1 / yrs) - 1)
bn_vol = float(bn_ret.std() * np.sqrt(252))
results["benchmark_equal_weight"] = {
    "total_pct": round(float(bn.iloc[-1] - 1) * 100, 1),
    "cagr_pct": round(bn_cagr * 100, 1),
    "sharpe": round(bn_cagr / bn_vol, 2),
    "max_dd_pct": round(float(((bn / bn.cummax()) - 1).min()) * 100, 1),
    "ann_turnover": 0.0,
}

final = {
    "description": "gtja191_171 on full CSI300 (current constituents), ashare qfq panel",
    "universe": {"requested": len(CODES), "fetched": len(fetched),
                 "days": int(len(close)), "start": str(close.index[0].date()),
                 "end": str(close.index[-1].date())},
    "results": results,
    "caveat": "current-roster survivorship bias; free token has no point-in-time membership",
}
print(json.dumps(final, ensure_ascii=False, indent=1))
json.dump(final, open("/tmp/csi300_backtest.json", "w"), ensure_ascii=False, indent=1)
print("SAVED /tmp/csi300_backtest.json", file=sys.stderr)
