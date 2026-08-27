"""Model the pybroker-mined factors (2026-08-21, user request).

Gate-3 failed as incremental blends, but the user asks: what if we MODEL on
them instead (linear / nonlinear / LGBM)? Three variants, all aligned to the
production mechanism (5d prediction horizon, top-15, constrained engine):

  E1 Ridge on 5 pybroker factors
  E2 LGBM on 5 pybroker factors
  E3 LGBM on stable-7 + 5 pybroker (12 factors) — the "add them in" version

Walk-forward: 3y trailing train (rows every 5d, horizon-shifted by 5d),
6m validation slice for LGBM early stopping, retrain every 63d like
production. Baseline: stable-7 IR-weighted signal.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]; volume = panel["volume"]
days = close.index
fwd1 = close.pct_change().shift(-1)
ret = close.pct_change()
fwd5 = close.pct_change(5).shift(-5)

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
PYB5 = ["pybroker_pvfit20", "pybroker_pvi20", "pybroker_nvi20",
        "pybroker_qtrend20", "pybroker_ltrend20"]
TRAIN_D, RETRAIN, TOP_N, REBAL, COST = 756, 63, 15, 5, 0.001
OOS_START = pd.Timestamp("2021-02-09")  # first model block (3y warmup from 2018-01)

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
fwd5 = fwd5.mask(anomalous.rolling(5).max().shift(-4).fillna(False).astype(bool))
tradable = tradable & ~anomalous
limit_up = tradable & (ret >= lim - 0.002)
limit_down = tradable & (ret <= -(lim - 0.002))

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", file=sys.stderr, flush=True)
fac = {}
for a in STABLE7:
    fac[a] = zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
for a in PYB5:
    fac[a] = zscore(reg.compute(a, panel))

def build_features(feat_names):
    rows = []
    for a in feat_names:
        s = fac[a].stack()
        rows.append(s.rename(a))
    X = pd.concat(rows, axis=1)
    X.index.names = ["date", "stock"]
    y = fwd5.stack().rename("y")
    y.index.names = ["date", "stock"]
    return X.join(y)

def walk_predict(feat_names, model_kind):
    df = build_features(feat_names)
    pred = pd.DataFrame(np.nan, index=days, columns=close.columns)
    starts = list(range(TRAIN_D, len(days), RETRAIN))
    for si, start in enumerate(starts):
        S = days[start]
        blk = days[start:min(start + RETRAIN, len(days))]
        tr_end = S - pd.Timedelta(days=7)  # horizon shift (5 trading days ~ 7 cal)
        tr_dates = [d for d in days[start - TRAIN_D:start] if d <= tr_end][::REBAL]
        tr = df[df.index.get_level_values("date").isin(tr_dates)].dropna()
        if len(tr) < 10000:
            continue
        te = df[df.index.get_level_values("date").isin(blk)]
        if model_kind == "ridge":
            m = Ridge(alpha=100.0)
            m.fit(tr[feat_names].values, tr["y"].values)
            for d in blk:
                xd = te.loc[d].dropna() if d in te.index.get_level_values("date") else None
                if xd is None or len(xd) < TOP_N: continue
                p = m.predict(xd[feat_names].values)
                pred.loc[d, xd.index.get_level_values("stock")] = p
        else:
            val_dates = [d for d in tr_dates[-13:]][::1]
            val = tr[tr.index.get_level_values("date").isin(val_dates)]
            trn = tr[~tr.index.get_level_values("date").isin(val_dates)]
            dtr = lgb.Dataset(trn[feat_names].values, label=trn["y"].values)
            dval = lgb.Dataset(val[feat_names].values, label=val["y"].values)
            m = lgb.train({"objective": "regression", "metric": "rmse",
                           "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
                           "colsample_bytree": 0.8, "min_child_samples": 100,
                           "seed": 7, "num_threads": 4},
                          dtr, num_boost_round=500, valid_sets=[dval],
                          callbacks=[lgb.early_stopping(40, verbose=False),
                                     lgb.log_evaluation(0)])
            for d in blk:
                if d not in te.index.get_level_values("date"): continue
                xd = te.loc[d].dropna()
                if len(xd) < TOP_N: continue
                p = m.predict(xd[feat_names].values, num_iteration=m.best_iteration)
                pred.loc[d, xd.index.get_level_values("stock")] = p
        if si % 5 == 0:
            print(f"  {model_kind} block {si}/{len(starts)} @ {S.date()}", file=sys.stderr, flush=True)
    return pred

def constrained_bt(pred):
    w = pd.DataFrame(0.0, index=days, columns=close.columns)
    held = set()
    for i, t in enumerate(days):
        if i % REBAL == 0:
            rowv = pred.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(rowv.nlargest(TOP_N).index)
                locked, keep = set(), held & dset
                for s in held - dset:
                    if not tradable.at[t, s] or limit_down.at[t, s]: locked.add(s)
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd1_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    eq = (1 + net).cumprod(); eq = eq[eq.index >= OOS_START]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net[eq.index].std() * np.sqrt(252))
    exb = net[~net.index.year.isin([2024, 2025])]
    shx = None
    if len(exb) > 50 and exb.std() > 0:
        eqx = (1 + exb).cumprod()
        yx = max((eqx.index[-1] - eqx.index[0]).days / 365.25, 1e-9)
        shx = round(float(eqx.iloc[-1] ** (1 / yx) - 1) / (exb.std() * np.sqrt(252)), 2)
    return {"cagr_pct": round(cagr * 100, 1), "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "sharpe_ex_bull2425": shx,
            "max_dd_pct": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "yearly_pct": {str(y): round(float(eq[eq.index.year == y].iloc[-1] /
                                               eq[eq.index.year == y].iloc[0] - 1) * 100, 1)
                           for y in sorted(set(eq.index.year))}}

# baseline: stable-7 IR-weighted signal (production mechanics sans EP)
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0
sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(252, len(days), RETRAIN):
    win = days[start - 252:start - 1]
    irs = {a: ir_of(pd.Series([fac[a].loc[t].corr(fwd1_clean.loc[t]) for t in win], index=win)) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    wts = {a: v / wsum for a, v in irs.items()}
    sig7.loc[days[start:start + RETRAIN]] = sum(fac[a].loc[days[start:start + RETRAIN]] * wts[a] for a in STABLE7)

results = {}
print("baseline...", file=sys.stderr, flush=True)
results["stable7_baseline"] = constrained_bt(sig7)
print("E1 ridge pyb5...", file=sys.stderr, flush=True)
results["E1_ridge_pyb5"] = constrained_bt(walk_predict(PYB5, "ridge"))
print("E2 lgbm pyb5...", file=sys.stderr, flush=True)
results["E2_lgbm_pyb5"] = constrained_bt(walk_predict(PYB5, "lgbm"))
print("E3 lgbm 12fac...", file=sys.stderr, flush=True)
results["E3_lgbm_12fac"] = constrained_bt(walk_predict(STABLE7 + PYB5, "lgbm"))
json.dump(results, open(DATA / "csi300_pybroker_modeling.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(results, ensure_ascii=False, indent=1))
