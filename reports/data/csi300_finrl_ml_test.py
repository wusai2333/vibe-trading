"""FinRL-style ML factor screen on A-shares (2026-08-21).

Borrow FinRL-Trading's use-case-2 recipe (ml_bucket_selection.py): tree
ensemble on fundamental levels + fundamental momentum + price momentum,
walk-forward, predict forward returns, rank stocks. Mapped to our data:

  features (as of t, all PIT):
    levels : fund:roe / fund:gross_profitability / fund:bp / fund:asset_growth
             (baostock, visible from pubDate, ffill — same discipline as
             FinRL's datadate->tradedate rule), cross-sectional z-score
    fund mom: QoQ diff of roe and gp
    price mom: ret_1q (21d), ret_4q (84d), ret_accel (ret_1q - prior ret_1q)
  target   : 63d forward log return, anomalous windows masked
  model    : RandomForest(n=100, depth=6)  [FinRL: 200/8, halved for speed]
  walk-fwd : retrain every 63d; train rows end at S-63 (target horizon
             shift — same lookahead fix class as the monthly-IR bug)

Evaluations (xgb/lr test precedent — ML scores can't enter the zoo):
  E1 daily-IC of the score (relate to the 0.02 incremental bar)
  E2 constrained top-15/5d backtest of the score alone
  E3 50/50 blend with stable-7 signal, same backtest
  E4 FinRL-faithful quarterly top-25 rebalance (their use case 2)
Baseline: frozen stable-7 constrained = 29.4% / 1.04 / -34.5%.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, volume = panel["close"], panel["volume"]
days = close.index
fwd1 = close.pct_change().shift(-1)
ret = close.pct_change()
fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))

TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2019-01-01")

# ---- tradability + guard (verbatim from frozen pipeline) ----
tradable = close.notna() & volume.fillna(0).gt(0)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
fwd1_clean = fwd1.mask(anomalous.shift(-1).fillna(False), 0.0)
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

# ---- features ----
def cz(df):  # cross-sectional zscore
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

feats = {
    "roe": cz(fund["fund:roe"].reindex(days).reindex(columns=close.columns)),
    "gp": cz(fund["fund:gross_profitability"].reindex(days).reindex(columns=close.columns)),
    "bp": cz(fund["fund:bp"].reindex(days).reindex(columns=close.columns)),
    "ag": cz(fund["fund:asset_growth"].reindex(days).reindex(columns=close.columns)),
    "roe_chg": cz(fund["fund:roe"].reindex(days).reindex(columns=close.columns).diff(63)),
    "gp_chg": cz(fund["fund:gross_profitability"].reindex(days).reindex(columns=close.columns).diff(63)),
    "ret_1q": cz(close.pct_change(21)),
    "ret_4q": cz(close.pct_change(84)),
    "ret_accel": cz(close.pct_change(21) - close.pct_change(21).shift(21)),
}
FKEYS = list(feats)

# ---- target: 63d fwd log return, mask windows containing anomalous days ----
fwd63 = np.log(close.shift(-63) / close)
bad_win = anomalous.rolling(63).max().shift(-62).fillna(False).astype(bool)
fwd63 = fwd63.mask(bad_win | close.shift(-63).isna())

# ---- walk-forward RF score ----
score = pd.DataFrame(np.nan, index=days, columns=close.columns)
X_all = pd.DataFrame({k: f.stack() for k, f in feats.items()})
y_all = fwd63.stack()
df_all = X_all.join(y_all.rename("y")).dropna()
df_all["day"] = df_all.index.get_level_values(0)
n_blocks = 0
for start in range(TRAIN, len(days), RETRAIN):
    S = days[start]
    tr = df_all[df_all["day"] <= S - pd.Timedelta(days=90)]  # ~63 trading days
    tr = tr[tr["day"] >= S - pd.Timedelta(days=365 * 3)]     # 3y window cap
    if len(tr) < 5000:
        continue
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
    rf.fit(tr[FKEYS].values, tr["y"].values)
    blk = days[start:start + RETRAIN]
    Xb = X_all.loc[blk[0]:blk[-1]]
    Xb = Xb.dropna()
    if len(Xb) == 0:
        continue
    pred = pd.Series(rf.predict(Xb[FKEYS].values), index=Xb.index).unstack()
    score.loc[pred.index, pred.columns] = pred
    n_blocks += 1
    print(f"block {n_blocks}: train={len(tr)} @ {S.date()}", file=sys.stderr)

# ---- E1: daily IC ----
ic = pd.Series({t: score.loc[t].corr(fwd1_clean.loc[t], method="spearman")
                for t in days[days >= OOS_START]}).dropna()
e1 = {"daily_ic_mean": round(float(ic.mean()), 4), "ic_ir": round(float(ic.mean() / ic.std()), 3) if ic.std() else None}

def stats_from(net: pd.Series, label: str) -> dict:
    eq = (1 + net).cumprod()
    eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    mdd = float(((eq / eq.cummax()) - 1).min())
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "max_dd_pct": round(mdd * 100, 1), "calmar": round(cagr / abs(mdd), 2) if mdd < 0 else None}

def constrained_bt(sig: pd.DataFrame, top_n: int = TOP_N, rebal: int = REBAL) -> pd.Series:
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % rebal == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= top_n:
                desired = set(rowv.nlargest(top_n).index)
                locked, keep = set(), held & desired
                for s in held - desired:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= top_n: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), top_n)
    gross = (w * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

# ---- stable-7 signal (for blend) ----
from src.factors.registry import get_default_registry
reg = get_default_registry()
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
fac7 = {a: cz(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(pd.Series([fac7[a].loc[t].corr(fwd1_clean.loc[t]) for t in win], index=win))
           for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    sig7.loc[days[start:start + RETRAIN]] = sum(fac7[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)

print("backtests...", file=sys.stderr)
blend = cz(cz(sig7) * 0.5 + cz(score) * 0.5)
results = [
    stats_from(constrained_bt(sig7), "stable7_baseline"),
    stats_from(constrained_bt(score), "E2_finrl_ml_alone"),
    stats_from(constrained_bt(blend), "E3_blend_50_50"),
    stats_from(constrained_bt(score, top_n=25, rebal=63), "E4_quarterly_top25"),
]
out = {"description": "FinRL-style walk-forward RF on fundamentals+momentum, A-share CSI300",
       "spec": {"features": FKEYS, "target": "63d fwd log ret", "model": "RF 100/d6",
                "retrain": "63d", "train_window": "3y cap, horizon-shifted",
                "blocks": n_blocks},
       "E1_daily_ic": e1, "results": results}
json.dump(out, open(DATA / "csi300_finrl_ml.json", "w"), ensure_ascii=False, indent=1)
pickle.dump(score, open(DATA / "finrl_ml_score.pkl", "wb"))
print(json.dumps(out, ensure_ascii=False, indent=1))
