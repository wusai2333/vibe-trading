"""majiajue recipe on our bench (2026-08-21, user: 重新跑一下实验).

Their claim: HS300, XGBoost binary classifier (label = next-month return
>= 3%), monthly rebalance, factors {PE, PB, MktValue, LFLO, NIAP, MA10},
risk control by HS300 vol threshold -> 11.54%/yr, MaxDD -17.91% (2016-19).

Faithful port to our panel + gates:
  factors: ep(=1/PE direction), bp, log(mktcap), ni_yoy, amihud illiq, ma10dist
  model A (faithful): XGB classifier, label = fwd21d >= 3%
  model B (our way):  XGB regressor, label = z(fwd21d)  [ablation: does the
                      binary label choice matter?]
  walk-forward: 36m train, last 6m validation (early stop), predict 1m, step 1
  selection: top-15 by pred, constrained engine, 10bps
  risk control (their vol-threshold class): NOT added — closed line D1/D8,
  reported gross of it.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()
ret = close.pct_change()
fwd21 = close.shift(-21) / close - 1

TRAIN_M, VALID_M, TOP_N, COST = 36, 6, 15, 0.001
OOS_START = pd.Timestamp("2019-01-01")

tradable = close.notna() & volume.fillna(0).gt(0)
lim = pd.DataFrame(0.10, index=days, columns=close.columns)
star = [c for c in close.columns if c.startswith("688")]
gem = [c for c in close.columns if c.startswith("30")]
if star: lim[star] = 0.20
if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
first_back = close.notna() & close.shift(1).isna()
long_gap = first_back & close.shift(20).isna()
anomalous = (ret.abs() > lim + 0.02) & ~long_gap
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))
fwd1_clean = ret.shift(-1).mask(anomalous.shift(-1).fillna(False), 0.0)
bad_win = anomalous.rolling(21).max().shift(-20).fillna(False).astype(bool)
fwd21 = fwd21.mask(bad_win | close.shift(-21).isna())

from src.factors.registry import get_default_registry
reg = get_default_registry()
print("building features...", file=sys.stderr, flush=True)
ep = reg.compute("fund_earnings_yield", panel)
bp = panel["fund:bp"]
logmkt = np.log(close * panel["fund:shares_diluted"].replace(0, np.nan))
ni = panel["fund:net_income"]
ni_yoy = (ni / ni.shift(252) - 1).replace([np.inf, -np.inf], np.nan)
amount = (volume * panel["vwap"]).replace(0, np.nan)
illiq = (ret.abs() / amount).rolling(20).mean() * 1e6
ma10d = close / close.rolling(10).mean() - 1
FEATS = {"ep": ep, "bp": bp, "logmkt": logmkt, "ni_yoy": ni_yoy,
         "illiq": illiq, "ma10d": ma10d}

from itertools import groupby
rebal = pd.DatetimeIndex([list(g)[-1] for _, g in groupby(days, key=lambda d: (d.year, d.month))])
rows = []
for f, df in FEATS.items():
    s = df.loc[rebal].stack()
    rows.append(s.rename(f))
X = pd.concat(rows, axis=1)
X.index.names = ["date", "stock"]
y = fwd21.loc[rebal].stack().rename("y")
y.index.names = ["date", "stock"]
dfm = X.join(y).dropna(subset=["y"]).reset_index()
dfm["date"] = pd.to_datetime(dfm["date"])
# winsorize growth tails (their platform factors come pre-cleaned)
dfm["ni_yoy"] = dfm["ni_yoy"].clip(dfm["ni_yoy"].quantile(0.01), dfm["ni_yoy"].quantile(0.99))
feat_cols = list(FEATS)
months = sorted(dfm["date"].unique())
print(f"rows={len(dfm)} months={len(months)}", file=sys.stderr, flush=True)

def walk(mode):
    preds = []
    i = 0
    while i + TRAIN_M < len(months):
        tr_m = months[i:i + TRAIN_M]
        te_m = months[i + TRAIN_M:i + TRAIN_M + 1]
        if not te_m: break
        fit_m, val_m = tr_m[:-VALID_M], tr_m[-VALID_M:]
        tr = dfm[dfm.date.isin(fit_m)]
        val = dfm[dfm.date.isin(val_m)]
        te = dfm[dfm.date.isin(te_m)]
        if mode == "clf":
            tr_lab = (tr["y"] >= 0.03).astype(int)
            val_lab = (val["y"] >= 0.03).astype(int)
            dtr = xgb.DMatrix(tr[feat_cols], label=tr_lab)
            dval = xgb.DMatrix(val[feat_cols], label=val_lab)
            m = xgb.train({"objective": "binary:logistic", "eval_metric": "auc",
                           "max_depth": 4, "eta": 0.05, "subsample": 0.8,
                           "colsample_bytree": 0.8, "min_child_weight": 100,
                           "seed": 7, "nthread": 4},
                          dtr, num_boost_round=800, evals=[(dval, "v")],
                          early_stopping_rounds=60, verbose_eval=False)
            p = m.predict(xgb.DMatrix(te[feat_cols]), iteration_range=(0, m.best_iteration + 1))
        else:
            def zc(s):
                return (s - s.mean()) / (s.std() + 1e-9) if s.std() > 0 else s * 0
            tr_lab = tr.groupby("date")["y"].transform(zc).clip(-4, 4)
            val_lab = val.groupby("date")["y"].transform(zc).clip(-4, 4)
            dtr = xgb.DMatrix(tr[feat_cols], label=tr_lab)
            dval = xgb.DMatrix(val[feat_cols], label=val_lab)
            m = xgb.train({"objective": "reg:squarederror", "eval_metric": "rmse",
                           "max_depth": 4, "eta": 0.05, "subsample": 0.8,
                           "colsample_bytree": 0.8, "min_child_weight": 100,
                           "seed": 7, "nthread": 4},
                          dtr, num_boost_round=800, evals=[(dval, "v")],
                          early_stopping_rounds=60, verbose_eval=False)
            p = m.predict(xgb.DMatrix(te[feat_cols]), iteration_range=(0, m.best_iteration + 1))
        preds.append(te.assign(pred=p)[["date", "stock", "pred", "y"]])
        i += 1
    return pd.concat(preds, ignore_index=True)

def stats(net, label):
    eq = (1 + net).cumprod(); eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    exb = net[~net.index.year.isin([2024, 2025])]
    eqx = (1 + exb).cumprod()
    shx = None
    if len(exb) > 50 and exb.std() > 0:
        yx = max((eqx.index[-1] - eqx.index[0]).days / 365.25, 1e-9)
        shx = round(float(eqx.iloc[-1] ** (1 / yx) - 1) / (exb.std() * np.sqrt(252)), 2)
    return {"label": label, "cagr_pct": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "sharpe_ex_bull2425": shx,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

def bt(pred):
    w = pd.DataFrame(0.0, index=days, columns=close.columns)
    held = set()
    reb_idx = {t: i for i, t in enumerate(days)}
    for t, grp in pred.groupby("date"):
        if t not in reb_idx: continue
        i = reb_idx[t]
        end = days[min(i + 21, len(days) - 1)]
        top = grp.nlargest(TOP_N, "pred")["stock"].tolist()
        # apply at t with constraint mechanics (one-shot month block)
        dset = set(top)
        locked, keep = set(), held & dset
        for s in held - dset:
            if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
        buys = []
        for s in grp.sort_values("pred", ascending=False)["stock"]:
            if len(keep) + len(locked) + len(buys) >= TOP_N: break
            if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
            buys.append(s)
        held = keep | locked | set(buys)
        if held:
            w.loc[t:end, list(held)] = 1.0 / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    return gross - turn * 2 * COST

results = {}
for mode in ["clf", "reg"]:
    print(f"walk-forward {mode}...", file=sys.stderr, flush=True)
    pred = walk(mode)
    ic = pred.groupby("date").apply(
        lambda s: spearmanr(s["pred"], s["y"]).correlation if s["pred"].std() > 0 else np.nan)
    results[mode] = {"monthly_ic": round(float(ic.mean()), 4),
                     "ic_t": round(float(ic.mean() / ic.std() * np.sqrt(ic.notna().sum())), 2),
                     "sleeve": stats(bt(pred), f"majiajue_{mode}_top15")}
json.dump(results, open(DATA / "csi300_majiajue_test.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))
