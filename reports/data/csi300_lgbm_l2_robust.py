"""L2 障碍标签稳健性复核（2026-08-24，用户拍板"跑"）。

预注册网格：障碍倍数 σ ∈ {0.75, 1.0, 1.5} × 种子 {7, 13, 21} = 9 跑，
另加 L0 对照 × 3 种子（同折同网格，公平横比）。
存活判据（全部预注册）：
  a) 9 个 L2 的 net Sharpe 均值 >= L0 三种子均值 - 0.05
  b) 9 个里 ex-bull < L0_exb均值 - 0.05 的最多 1 个
  c) 无灾难跑（net Sharpe < 0.6）
  d) σ 扫描不翻脸：不允许某 σ 档三种子均值在 Sharpe 和 ex-bull 上双输 L0
全过 -> 入策略库候选；任何一条不过 -> L2 判死，ML 赛道重新关闭。
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
days = close.index

feat_cols = sorted(feats_dict.keys())
df = pd.concat([feats_dict[a].stack().rename(a) for a in feat_cols], axis=1)
df.index.names = ["date", "stock"]
df = df.reset_index()
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2018-06-01"].sort_values(["date", "stock"]).reset_index(drop=True)
g = df.groupby("date")[feat_cols]
df[feat_cols] = ((df[feat_cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan))     .clip(-3, 3).fillna(0.0)

c_rebal = close.loc[rebal]
y_wide = c_rebal.shift(-1) / c_rebal - 1
ret = close.pct_change(fill_method=None)

def barrier_labels(mult: float) -> pd.DataFrame:
    sig21 = ret.rolling(21).std().loc[rebal] * np.sqrt(21)
    lab = pd.DataFrame(0.0, index=rebal[:-1], columns=close.columns)
    for k in range(len(rebal) - 1):
        wdays = days[(days > rebal[k]) & (days <= rebal[k + 1])]
        if len(wdays) == 0:
            continue
        cw = close.loc[wdays].values
        base = close.loc[rebal[k]].values
        sdm = sig21.iloc[k].values * mult
        up, dn = base * (1 + sdm), base * (1 - sdm)
        hit_up = (cw >= up[None, :]).argmax(0)
        hit_dn = (cw <= dn[None, :]).argmax(0)
        any_up = (cw >= up[None, :]).any(0)
        any_dn = (cw <= dn[None, :]).any(0)
        v = np.zeros(cw.shape[1])
        v[any_up & (~any_dn | (hit_up <= hit_dn))] = 1.0
        v[any_dn & (~any_up | (hit_dn < hit_up))] = -1.0
        lab.iloc[k] = v
    return lab

TRAIN, TEST, STEP, VALID, N_GROUPS = 36, 12, 12, 6, 20
BASE = dict(objective="regression", metric="None", feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, min_child_samples=100,
            lambda_l2=1.0, verbose=-1, num_threads=4)
GRID = [dict(num_leaves=nl, learning_rate=lr)
        for nl in (15, 31, 63) for lr in (0.03, 0.05)]
COST_M = 0.001

def date_ic(sub):
    ics = [spearmanr(s["pred"], s["y"]).correlation
           for _, s in sub.groupby("date") if len(s) > 20 and s["pred"].std() > 0]
    return np.nanmean(ics) if ics else np.nan

def make_ic_feval(dates_arr, y_raw):
    tmp = pd.DataFrame({"d": dates_arr, "y": y_raw})
    groups = [gg.index.to_numpy() for _, gg in tmp.groupby("d")]
    def feval(preds, dset):
        ics = [spearmanr(preds[idx], tmp["y"].to_numpy()[idx]).correlation
               for idx in groups if preds[idx].std() > 0]
        ic = np.nanmean(ics) if ics else 0.0
        return "IC", (ic if np.isfinite(ic) else 0.0), True
    return feval

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    exb = r[~r.index.year.isin([2024, 2025])]
    return {"ann_pct": round(float(eq.iloc[-1] ** (12 / len(r)) - 1) * 100, 1),
            "sharpe": round(r.mean() / r.std() * np.sqrt(12), 2) if r.std() > 0 else None,
            "sharpe_ex_bull": round(exb.mean() / exb.std() * np.sqrt(12), 2)
            if len(exb) > 3 and exb.std() > 0 else None,
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}

def run(d2: pd.DataFrame, seed: int) -> dict:
    rebal_months = sorted(d2["date"].unique())
    preds = []
    i = 0
    while i + TRAIN < len(rebal_months):
        tr_d = rebal_months[i:i + TRAIN]
        te_d = rebal_months[i + TRAIN:i + TRAIN + TEST]
        if not te_d:
            break
        fit_d, val_d = tr_d[:-VALID], tr_d[-VALID:]
        tr = d2[d2.date.isin(fit_d)]
        val = d2[d2.date.isin(val_d)].reset_index(drop=True)
        te = d2[d2.date.isin(te_d)]
        dtr = lgb.Dataset(tr[feat_cols], tr["y_cs"])
        dval = lgb.Dataset(val[feat_cols], val["y_cs"])
        feval = make_ic_feval(val["date"].to_numpy(), val["ylab"].to_numpy())
        best = None
        for gp in GRID:
            m = lgb.train({**BASE, **gp, "seed": seed}, dtr, num_boost_round=1500,
                          valid_sets=[dval], feval=feval,
                          callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
            vpred = val.assign(pred=m.predict(val[feat_cols], num_iteration=m.best_iteration))
            vic = date_ic(vpred)
            if best is None or (np.isfinite(vic) and vic > best[0]):
                best = (vic, m)
        _, model = best
        tp = te.assign(pred=model.predict(te[feat_cols], num_iteration=model.best_iteration))
        preds.append(tp[["date", "stock", "pred", "y"]])
        i += STEP
    pred = pd.concat(preds, ignore_index=True)
    pred["group"] = pred.groupby("date", group_keys=False).apply(
        lambda s: pd.qcut(s["pred"].rank(method="first"), N_GROUPS, labels=False)).astype(int)
    gm = pred.groupby(["date", "group"])["y"].mean().unstack("group")
    long_r = gm[N_GROUPS - 1].sort_index()
    book = {d: set(s.loc[s.group == N_GROUPS - 1, "stock"]) for d, s in pred.groupby("date")}
    dts = sorted(book)
    to = [1 - len(book[dts[k]] & book[dts[k - 1]]) / max(len(book[dts[k]]), 1)
          for k in range(1, len(dts))]
    long_net = long_r - pd.Series([np.nan] + to, index=long_r.index) * 2 * COST_M
    return {"net": metrics(long_net), "gross": metrics(long_r)}

# ---- configs ----
y_true = y_wide.stack().rename("y").rename_axis(["date", "stock"]).reset_index()
rows = []
for label, mult in [("L0", None)] + [(f"L2_s{s}", s) for s in (0.75, 1.0, 1.5)]:
    ylab_w = y_wide if mult is None else barrier_labels(mult)
    ylab = ylab_w.stack().rename("ylab").rename_axis(["date", "stock"]).reset_index()
    d2 = df.merge(ylab, on=["date", "stock"], how="inner")            .merge(y_true, on=["date", "stock"], how="inner").dropna(subset=["ylab", "y"])
    d2 = d2.assign(y_cs=d2.groupby("date")["ylab"].transform(
        lambda x: ((x - x.mean()) / x.std()) if x.std() > 0 else x * 0).clip(-4, 4))
    for seed in (7, 13, 21):
        r = run(d2, seed)
        rows.append({"label": label, "seed": seed, **r})
        print(f"{label} seed={seed}: net {r['net']['ann_pct']}%/{r['net']['sharpe']}/"
              f"exb {r['net']['sharpe_ex_bull']}/dd {r['net']['max_dd_pct']}%",
              file=sys.stderr, flush=True)

res = pd.DataFrame(rows)
l0 = res[res.label == "L0"]
l2 = res[res.label != "L0"]
verdict = {
    "a_sharpe": bool(l2["net"].map(lambda d: d["sharpe"]).mean() >=
                     l0["net"].map(lambda d: d["sharpe"]).mean() - 0.05),
    "b_exbull_outliers": int(sum(l2["net"].map(lambda d: d["sharpe_ex_bull"]) <
                                 l0["net"].map(lambda d: d["sharpe_ex_bull"]).mean() - 0.05)),
    "c_disasters": int(sum(l2["net"].map(lambda d: d["sharpe"]) < 0.6)),
    "d_sigma_flip": {s: {
        "sharpe": round(float(g["net"].map(lambda d: d["sharpe"]).mean()), 3),
        "exbull": round(float(g["net"].map(lambda d: d["sharpe_ex_bull"]).mean()), 3)}
        for s, g in l2.groupby(l2.label.str.extract(r"s([d.]+)")[0])},
}
out = {"description": "L2 障碍标签稳健性复核（σ×种子网格 + L0 三种子对照）",
       "verdict_checks": verdict, "rows": rows}
json.dump(out, open(DATA / "csi300_lgbm_l2_robust.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
