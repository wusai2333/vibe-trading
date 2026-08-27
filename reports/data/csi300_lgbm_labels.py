"""ML 赛道重启：换标签实验（2026-08-24，用户拍板；vectorbt labels/ 启发）。

三轮 ML 实验（LGBM/RF/Ridge）都用同一标签=下月收益。本实验检验"标签本身"
是否被忽略的变量。基座=S7 口径（CSI300 月频 walk-forward LGBM，435 因子）。

预注册 4 个标签（跑之前定死）：
  L0 base   下月收益（S7 原样，对照）
  L1 trend  下月收益 / 下月已实现波动（TRENDLB 族：奖励平滑趋势，惩罚震荡大涨）
  L2 barrier 下月内先触碰 +1σ_m 障=+1、-1σ_m 障=-1、均未触=0（LEXLB 族）
  L3 revert -下月收益 × sign(z60 超涨度)（MEANLB 族：奖励超涨回归）

评估纪律：**所有变体一律按真实下月收益计盈亏**（标签只进训练目标，
不进 P&L），long=预测 top 组（~14 只），10bps 单边，同网格同折。
PASS 判据（vs L0）：net Sharpe >= L0+0.10 且 ex-bull >= L0_exb-0.05。
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
print(f"factors={len(feats_dict)} months={len(rebal)}", file=sys.stderr)

# ---- long-form features (build once) ----
feat_cols = sorted(feats_dict.keys())
df = pd.concat([feats_dict[a].stack().rename(a) for a in feat_cols], axis=1)
df.index.names = ["date", "stock"]
df = df.reset_index()
df["date"] = pd.to_datetime(df["date"])
df = df[df["date"] >= "2018-06-01"].sort_values(["date", "stock"]).reset_index(drop=True)

# ---- label ingredients from daily panel ----
c_rebal = close.loc[rebal]
y_wide = (c_rebal.shift(-1) / c_rebal - 1)                      # next-month return
ret = close.pct_change(fill_method=None)

# L1: realized vol within next month per stock（日 t 属于其右侧 rebal 月）
def vol_next_month() -> pd.DataFrame:
    m_idx = np.searchsorted(rebal.values, days.values, side="left") - 1
    r = ret.assign(_m=m_idx)
    vols, nds = {}, {}
    for k, gg in r.groupby("_m"):
        if k < 0:
            continue
        vols[k] = gg.drop(columns="_m").std()
        nds[k] = max(len(gg), 5)
    vol = pd.DataFrame(vols).T.mul(np.sqrt(pd.Series(nds)), axis=0)
    vol.index = rebal[:len(vol)]
    return vol.reindex(rebal)

vm = vol_next_month()
labels = {"L0_base": y_wide}
labels["L1_trend"] = y_wide / vm.clip(lower=0.02)

# L2: barrier first-touch within next month (+-1 monthly vol of trailing 21d)
def barrier_labels() -> pd.DataFrame:
    sig21 = ret.rolling(21).std().loc[rebal] * np.sqrt(21)
    lab = pd.DataFrame(0.0, index=rebal[:-1], columns=close.columns)
    for k in range(len(rebal) - 1):
        wdays = days[(days > rebal[k]) & (days <= rebal[k + 1])]
        if len(wdays) == 0:
            continue
        cw = close.loc[wdays].values
        base = close.loc[rebal[k]].values
        sdm = sig21.iloc[k].values
        up = base * (1 + sdm)
        dn = base * (1 - sdm)
        hit_up = (cw >= up[None, :]).argmax(0)
        hit_dn = (cw <= dn[None, :]).argmax(0)
        any_up = (cw >= up[None, :]).any(0)
        any_dn = (cw <= dn[None, :]).any(0)
        v = np.zeros(cw.shape[1])
        v[any_up & (~any_dn | (hit_up <= hit_dn))] = 1.0
        v[any_dn & (~any_up | (hit_dn < hit_up))] = -1.0
        lab.iloc[k] = v
    return lab

labels["L2_barrier"] = barrier_labels()

# L3: overextension reversion
mom60 = (close / close.shift(60) - 1).loc[rebal]
z60 = mom60.sub(mom60.mean(axis=1), axis=0).div(mom60.std(axis=1).replace(0, np.nan), axis=0)
labels["L3_revert"] = -y_wide * np.sign(z60)

# ---- walk-forward (S7/astock spec) ----
TRAIN, TEST, STEP, VALID, N_GROUPS = 36, 12, 12, 6, 20
BASE = dict(objective="regression", metric="None", feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, min_child_samples=100,
            lambda_l2=1.0, verbose=-1, num_threads=4, seed=7)
GRID = [dict(num_leaves=nl, learning_rate=lr)
        for nl in (15, 31, 63) for lr in (0.03, 0.05)]
COST_M = 0.001

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

def metrics(r: pd.Series) -> dict:
    r = r.dropna()
    n = len(r)
    cum = (1 + r).prod()
    ann = float(cum ** (12 / n) - 1) if n else 0.0
    eq = (1 + r).cumprod()
    exb = r[~r.index.year.isin([2024, 2025])]
    return {"ann_pct": round(ann * 100, 1),
            "sharpe": round(r.mean() / r.std() * np.sqrt(12), 2) if r.std() > 0 else None,
            "sharpe_ex_bull": round(exb.mean() / exb.std() * np.sqrt(12), 2)
            if len(exb) > 3 and exb.std() > 0 else None,
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}

def run_label(name: str, y_lab: pd.DataFrame) -> dict:
    d2 = df.merge(y_lab.stack().rename("ylab").rename_axis(["date", "stock"]).reset_index(),
                  on=["date", "stock"], how="inner").dropna(subset=["ylab"])
    # actual next-month return for P&L (always true returns)
    d2 = d2.merge(y_wide.stack().rename("y").rename_axis(["date", "stock"]).reset_index(),
                  on=["date", "stock"], how="inner").dropna(subset=["y"])
    g = d2.groupby("date")[feat_cols]
    d2[feat_cols] = ((d2[feat_cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan))         .clip(-3, 3).fillna(0.0)
    d2["y_cs"] = d2.groupby("date")["ylab"].transform(
        lambda x: ((x - x.mean()) / x.std()) if x.std() > 0 else x * 0).clip(-4, 4)
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
            m = lgb.train({**BASE, **gp}, dtr, num_boost_round=1500, valid_sets=[dval],
                          feval=feval, callbacks=[lgb.early_stopping(80, verbose=False),
                                                   lgb.log_evaluation(0)])
            vpred = val.assign(pred=m.predict(val[feat_cols], num_iteration=m.best_iteration))
            vic = date_ic(vpred)
            if best is None or (np.isfinite(vic) and vic > best[0]):
                best = (vic, m, gp)
        vic, model, gp = best
        tp = te.assign(pred=model.predict(te[feat_cols], num_iteration=model.best_iteration))
        preds.append(tp[["date", "stock", "pred", "y"]])
        print(f"  {name} fold {i//STEP+1}: valIC_lab={vic:.3f}", file=sys.stderr)
        i += STEP
    pred = pd.concat(preds, ignore_index=True)
    pred["group"] = pred.groupby("date", group_keys=False).apply(
        lambda s: pd.qcut(s["pred"].rank(method="first"), N_GROUPS, labels=False)).astype(int)
    gm = pred.groupby(["date", "group"])["y"].mean().unstack("group")
    long_r = gm[N_GROUPS - 1].sort_index()
    bench_r = pred.groupby("date")["y"].mean().sort_index()
    book = {d: set(s.loc[s.group == N_GROUPS - 1, "stock"]) for d, s in pred.groupby("date")}
    dts = sorted(book)
    to = [1 - len(book[dts[k]] & book[dts[k - 1]]) / max(len(book[dts[k]]), 1)
          for k in range(1, len(dts))]
    long_net = long_r - pd.Series([np.nan] + to, index=long_r.index) * 2 * COST_M
    ic_vs_ret = pred.groupby("date").apply(
        lambda s: spearmanr(s["pred"], s["y"]).correlation if s["pred"].std() > 0 else np.nan)
    return {"label": name, "long_gross": metrics(long_r), "long_net": metrics(long_net),
            "bench": metrics(bench_r),
            "ic_pred_vs_true_ret": round(float(ic_vs_ret.mean()), 4),
            "turnover_monthly": round(float(np.mean(to)), 3),
            "oos_months": int(pred["date"].nunique())}

out = {"description": "ML 赛道重启：换标签实验（S7 口径，4 标签对照）",
       "pass_rule": "net Sharpe >= L0+0.10 且 ex-bull >= L0_exb-0.05",
       "labels": {k: run_label(k, v) for k, v in labels.items()}}
json.dump(out, open(DATA / "csi300_lgbm_labels.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
