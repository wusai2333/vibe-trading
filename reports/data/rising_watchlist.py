"""爬升观察榜（2026-08-26，用户需求：更早看到"进入中"的股票）。

官方 Top20 不变（生产信号）。本榜补充两样：
  1. Top50 扩展榜：今日 blend 排名 1-50
  2. 爬升榜：近 5 个交易日排名改善最多的股（正在往 Top20 里进的）
用与 screener 相同的 stable-7+EP blend（最后一窗 IR 权重，对近几日打分排名）。
诚实提示：这是更早的可见性，不是消除滞后——排名本身仍是滞后因子。
"""
import pickle, sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent

panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(panel["close"].index).reindex(columns=panel["close"].columns).ffill()
close, volume = panel["close"], panel["volume"]
days = close.index
fwd = close.pct_change().shift(-1)  # 与 screener 一致（默认 fill）

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN = 252
EP_W = 0.10
from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
ep = zscore(reg.compute("fund_earnings_yield", panel))
# 最后一窗 IR 权重（与 screener 完全一致：Pearson IC）
ic = {a: pd.Series([fac[a].loc[t].corr(fwd.loc[t]) for t in days[-TRAIN - 1:-1]],
                   index=days[-TRAIN - 1:-1]) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
irs = {a: ir_of(ic[a]) for a in STABLE7}
wsum = sum(abs(v) for v in irs.values()) or 1.0
wts = {a: v / wsum for a, v in irs.items()}
blend = (1 - EP_W) * sum(fac[a] * wts[a] for a in STABLE7) + EP_W * ep

# 近 N 日排名轨迹
N = 6
recent = days[-N:]
ranks = blend.loc[recent].rank(axis=1, ascending=False)
n_stocks = close.notna().sum(axis=1)  # 当日有效股数（排名归一参考）
today = recent[-1]
r_today = ranks.loc[today]
r_5ago = ranks.loc[recent[0]]
delta = r_5ago - r_today   # 正=排名上升（数字变小）

# 名称映射（与 screener 同源）
cons = json.load(open(DATA / "csi300_cons.json"))
names = dict(zip(cons["codes"], cons["names"]))
def nm(sym):
    return names.get(sym.split(".")[0], sym.split(".")[0])

out = {"as_of": str(today.date()), "top50": [], "risers": []}
# Top50
t50 = r_today.dropna().sort_values().head(50)
for sym, rk in t50.items():
    d = delta.get(sym, np.nan)
    out["top50"].append({"rank": int(rk), "symbol": sym, "name": nm(sym),
                         "d_rank_5d": (int(round(d)) if pd.notna(d) else None)})
# 爬升榜：当前排名 21-80 且 5 日排名改善最大
pool = r_today.dropna()
band = pool[(pool >= 21) & (pool <= 80)]
cand = delta.loc[band.index].dropna().sort_values(ascending=False).head(15)
for sym, d in cand.items():
    if d <= 0:
        continue
    out["risers"].append({"rank_today": int(pool[sym]), "symbol": sym, "name": nm(sym),
                          "d_rank_5d": int(round(d)), "rank_5d_ago": int(r_5ago.get(sym, np.nan))})
json.dump(out, open(DATA / "rising_watchlist_latest.json", "w"), ensure_ascii=False, indent=1)

print(f"as_of {today.date()}  （Δrank 正=5日内排名上升）")
print("== Top50 扩展榜（↑/↓=较5日前变化）==")
for it in out["top50"]:
    d = it["d_rank_5d"]
    mark = f" ↑{d}" if (d is not None and d >= 3) else (f" ↓{-d}" if (d is not None and d <= -3) else "")
    print(f"  {it['rank']:>2} {it['name']:8s}{mark}")
print("== 爬升观察榜（现排名21-80，5日爬升最快）==")
for it in out["risers"]:
    print(f"  今#{it['rank_today']:<2} {it['name']:8s} 5日前#{it['rank_5d_ago']} → 上升{it['d_rank_5d']}名")
