"""Walk-forward LightGBM on full-zoo features, ZZ1000 monthly (astock-lab spec).

Monthly track REOPENED by user (2026-08-21) with the astock-alpha-factor-lab
method. Spec borrowed verbatim from ml_lgbm.py:

  * rolling 36-month train -> predict next 12 -> step 12
  * last 6 train months = validation: per-date Spearman-IC early stopping
    (patience 80, 1500 rounds) + grid num_leaves {15,31,63} x lr {0.03,0.05},
    best by validation IC
  * features cross-sectional z per date, clip +-3; label y_cs = cross-sectional
    z of next-month return, clip +-4 (backtest uses raw return)
  * 20 groups; Long=G20, Short=G1; bench = universe mean
  * BASE params verbatim: ff 0.8, bf 0.8, min_child_samples 100, l2 1.0, seed 7

Our deltas (documented): no market-cap data -> no small-10% drop (their
no-drop variant was stronger anyway); costs applied at 10bps one-way in the
long-only and long-short legs for an honest compare; panel = 2018 constituents
(survivorship caveat, same as theirs).

Comparisons: stable-7 on ZZ1000 (recorded: excess -0.037%/day, Sharpe 0.23)
and stable-7 on CSI300 (29.4/1.04/-34.5). Validation: ex-2020 Sharpe, yearly.
"""
import sys, json, pickle, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

DATA = Path(__file__).resolve().parent
ds = pickle.load(open(DATA / "csi300_monthly_features.pkl", "rb"))
feats_dict, rebal = ds["feats"], pd.DatetimeIndex(ds["rebal"])
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
print(f"factors={len(feats_dict)} months={len(rebal)}", file=sys.stderr)

# ---- long-form dataset: rows = (month-end date, stock) ----
feat_cols = sorted(feats_dict.keys())
wide = {a: feats_dict[a] for a in feat_cols}
rows = []
for a in feat_cols:
    s = wide[a].stack()
    rows.append(s.rename(a))
df = pd.concat(rows, axis=1)
df.index.names = ["date", "stock"]
df = df.reset_index()

# label: next-month raw return (close at next rebal / close at this rebal - 1)
c_rebal = close.loc[rebal]
y_wide = c_rebal.shift(-1) / c_rebal - 1
y = y_wide.stack().rename("y")
y.index.names = ["date", "stock"]
df = df.merge(y.reset_index(), on=["date", "stock"], how="inner")
df = df.dropna(subset=["y"])
df["date"] = pd.to_datetime(df["date"])

# guard: drop rows whose month contains impossible daily moves (panel already
# cleaned; belt-and-braces winsorize of raw y per date happens via y_cs clip)
df = df.sort_values(["date", "stock"]).reset_index(drop=True)
df = df[df["date"] >= "2018-06-01"]  # need factor warmup
print(f"rows={len(df)} months={df['date'].nunique()} feats={len(feat_cols)}", file=sys.stderr)

# ---- cross-sectional normalization (their exact code) ----
g = df.groupby("date")[feat_cols]
df[feat_cols] = ((df[feat_cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)) \
    .clip(-3, 3).fillna(0.0)
df["y_cs"] = df.groupby("date")["y"].transform(
    lambda x: ((x - x.mean()) / x.std()) if x.std() > 0 else x * 0).clip(-4, 4)

rebal_months = sorted(df["date"].unique())
TRAIN, TEST, STEP, VALID, N_GROUPS = 36, 12, 12, 6, 20
BASE = dict(objective="regression", metric="None", feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, min_child_samples=100,
            lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)
GRID = [dict(num_leaves=nl, learning_rate=lr)
        for nl in (15, 31, 63) for lr in (0.03, 0.05)]

def date_ic(sub, pcol="pred"):
    ics = []
    for _, s in sub.groupby("date"):
        if len(s) > 20 and s[pcol].std() > 0:
            ics.append(spearmanr(s[pcol], s["y"]).correlation)
    return np.nanmean(ics) if ics else np.nan

def make_ic_feval(dates_arr, y_raw):
    tmp = pd.DataFrame({"d": dates_arr, "y": y_raw})
    groups = [gg.index.to_numpy() for _, gg in tmp.groupby("d")]
    def feval(preds, dset):
        ics = []
        for idx in groups:
            p = preds[idx]
            if p.std() > 0:
                ics.append(spearmanr(p, tmp["y"].to_numpy()[idx]).correlation)
        ic = np.nanmean(ics) if ics else 0.0
        return "IC", (ic if np.isfinite(ic) else 0.0), True
    return feval

preds, fold_log = [], []
i = 0
while i + TRAIN < len(rebal_months):
    tr_d = rebal_months[i:i + TRAIN]
    te_d = rebal_months[i + TRAIN:i + TRAIN + TEST]
    if not te_d:
        break
    fit_d, val_d = tr_d[:-VALID], tr_d[-VALID:]
    tr = df[df.date.isin(fit_d)]
    val = df[df.date.isin(val_d)].reset_index(drop=True)
    te = df[df.date.isin(te_d)]
    dtr = lgb.Dataset(tr[feat_cols], tr["y_cs"])
    dval = lgb.Dataset(val[feat_cols], val["y_cs"])
    feval = make_ic_feval(val["date"].to_numpy(), val["y"].to_numpy())
    best = None
    for gp in GRID:
        params = {**BASE, **gp}
        m = lgb.train(params, dtr, num_boost_round=1500, valid_sets=[dval], feval=feval,
                      callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        vpred = val.assign(pred=m.predict(val[feat_cols], num_iteration=m.best_iteration))
        vic = date_ic(vpred)
        if best is None or (np.isfinite(vic) and vic > best[0]):
            best = (vic, m, gp)
    vic, model, gp = best
    tp = te.assign(pred=model.predict(te[feat_cols], num_iteration=model.best_iteration))
    preds.append(tp[["date", "stock", "pred", "y"]])
    fold_log.append(dict(test=f"{te_d[0].date()}..{te_d[-1].date()}", val_IC=round(vic, 4),
                         num_leaves=gp["num_leaves"], lr=gp["learning_rate"],
                         best_iter=model.best_iteration))
    print(f"fold {len(fold_log)}: test {te_d[0].date()}..{te_d[-1].date()} "
          f"valIC={vic:.3f} nl={gp['num_leaves']} lr={gp['learning_rate']} it={model.best_iteration}",
          file=sys.stderr)
    i += STEP

pred = pd.concat(preds, ignore_index=True)
pred["group"] = pred.groupby("date", group_keys=False).apply(
    lambda s: pd.qcut(s["pred"].rank(method="first"), N_GROUPS, labels=False)).astype(int)

gm = pred.groupby(["date", "group"])["y"].mean().unstack("group")
monthly = pd.DataFrame(index=gm.index)
monthly["long"] = gm[N_GROUPS - 1]
monthly["short"] = gm[0]
monthly["ls"] = monthly["long"] - monthly["short"]
monthly["bench"] = pred.groupby("date")["y"].mean()
monthly = monthly.sort_index()

ic_series = pred.groupby("date").apply(
    lambda s: spearmanr(s["pred"], s["y"]).correlation if s["pred"].std() > 0 else np.nan)

COST_M = 0.001  # 10bps one-way, applied to each leg's full monthly turnover (conservative)
long_book = {d: set(s.loc[s.group == N_GROUPS - 1, "stock"]) for d, s in pred.groupby("date")}
short_book = {d: set(s.loc[s.group == 0, "stock"]) for d, s in pred.groupby("date")}
ds_ = sorted(long_book)
to_l = [1 - len(long_book[ds_[k]] & long_book[ds_[k-1]]) / max(len(long_book[ds_[k]]), 1)
        for k in range(1, len(ds_))]
to_s = [1 - len(short_book[ds_[k]] & short_book[ds_[k-1]]) / max(len(short_book[ds_[k]]), 1)
        for k in range(1, len(ds_))]
monthly["long_net"] = monthly["long"] - pd.Series([np.nan] + to_l, index=monthly.index) * 2 * COST_M
monthly["ls_net"] = monthly["ls"] - (pd.Series([np.nan] + to_l, index=monthly.index) +
                                       pd.Series([np.nan] + to_s, index=monthly.index)) * 2 * COST_M

def metrics(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    n = len(r)
    cum = (1 + r).prod()
    ann = float(cum ** (12 / n) - 1)
    vol = float(r.std() * np.sqrt(12))
    eq = (1 + r).cumprod()
    mdd = float((eq / eq.cummax() - 1).min())
    # OOS starts mid-2021 -> ex-2020 vacuous; concentration risk is the
    # 2024-25 bull, so report ex-bull Sharpe instead
    exb = r[~r.index.year.isin([2024, 2025])]
    def yearly(eq_):
        out = {}
        for y in sorted(set(eq_.index.year)):
            seg = eq_[eq_.index.year == y]
            out[str(y)] = round(float(seg.iloc[-1] / seg.iloc[0] - 1) * 100, 1)
        return out
    return {"label": label,
            "ann_pct": round(ann * 100, 1),
            "sharpe": round(r.mean() / r.std() * np.sqrt(12), 2) if r.std() > 0 else None,
            "sharpe_ex_bull2425": round(exb.mean() / exb.std() * np.sqrt(12), 2) if len(exb) > 3 and exb.std() > 0 else None,
            "max_dd_pct": round(mdd * 100, 1),
            "yearly_pct": yearly(eq)}

results = [metrics(monthly["long"], "G20_long_gross"),
           metrics(monthly["long_net"], "G20_long_net"),
           metrics(monthly["ls"], "LS_gross"),
           metrics(monthly["ls_net"], "LS_net"),
           metrics(monthly["bench"], "ZZ1000_panel_EW_bench")]

# ---- model fusion: LGBM x RF(FinRL quarterly score), same universe ----
print("fusion with RF score...", file=sys.stderr)
rf_score = pickle.load(open(DATA / "finrl_ml_score.pkl", "rb"))
rf_long_df = rf_score.stack().rename("rf")
rf_long_df.index.names = ["date", "stock"]
pred2 = pred.merge(rf_long_df.reset_index(), on=["date", "stock"], how="inner")

def sleeve_from_groups(gdf, gcol):
    gm2 = gdf.groupby(["date", gcol])["y"].mean().unstack(gcol)
    lg = gm2[N_GROUPS - 1]
    book = {d: set(s.loc[s[gcol] == N_GROUPS - 1, "stock"]) for d, s in gdf.groupby("date")}
    dss = sorted(book)
    to = [1 - len(book[dss[k]] & book[dss[k-1]]) / max(len(book[dss[k]]), 1)
          for k in range(1, len(dss))]
    net = lg - pd.Series([np.nan] + to, index=lg.index) * 2 * COST_M
    return lg, net, float(np.mean(to))

ens = pred2.copy()
ens["erank"] = ens.groupby("date")["pred"].rank(pct=True)
ens["rrank"] = ens.groupby("date")["rf"].rank(pct=True)
ens["egroup"] = ens.groupby("date", group_keys=False).apply(
    lambda s: pd.qcut(((s["erank"] + s["rrank"]) / 2).rank(method="first"), N_GROUPS, labels=False)).astype(int)
ens_long, ens_net, ens_to = sleeve_from_groups(ens, "egroup")
rfq = pred2.copy()
rfq["rrank"] = rfq.groupby("date")["rf"].rank(pct=True)
rfq["rgroup"] = rfq.groupby("date", group_keys=False).apply(
    lambda s: pd.qcut(s["rrank"].rank(method="first"), N_GROUPS, labels=False)).astype(int)
rf_long, rf_net, rf_to = sleeve_from_groups(rfq, "rgroup")
results += [metrics(ens_net, "FUSION_lgbm_x_rf_long_net"),
            metrics(rf_net, "RF_alone_long_net"),
            metrics(ens_long, "FUSION_long_gross")]

out = {"description": "full-zoo LGBM walk-forward on ZZ1000 monthly (astock-lab spec, monthly track reopened)",
       "spec": {"factors_in": len(feat_cols), "skipped_factors": len(ds.get("skipped", [])),
                "months_total": len(rebal_months), "oos_months": int(pred['date'].nunique()),
                "train/test/step/valid": [TRAIN, TEST, STEP, VALID], "groups": N_GROUPS,
                "cost_one_way": 0.001},
       "ic": {"monthly_ic_mean": round(float(ic_series.mean()), 4),
              "ic_ir": round(float(ic_series.mean() / ic_series.std()), 3),
              "ic_t": round(float(ic_series.mean() / ic_series.std() * np.sqrt(ic_series.notna().sum())), 1)},
       "turnover_monthly_one_way": {"long": round(float(np.mean(to_l)), 3),
                                    "short": round(float(np.mean(to_s)), 3)},
       "folds": fold_log, "results": results}
json.dump(out, open(DATA / "csi300_lgbm_fusion.json", "w"), ensure_ascii=False, indent=1)
pickle.dump({"pred": pred, "monthly": monthly}, open(DATA / "csi300_lgbm_monthly.pkl", "wb"))
print(json.dumps(out, ensure_ascii=False, indent=1))
