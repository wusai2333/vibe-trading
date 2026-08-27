"""Hurst regime 降险层实测（2026-08-24，S9 vt15 之后的第二候选，用户拍板）。

想法来自 backtrader 的 hurst.py（只借公式，不抄 GPL 代码）：
指数 Hurst 指数 H<0.5 = 均值回归/震荡市（动量因子的死地），H>=0.5 = 趋势市。
H 估计：分数布朗运动方差标度 Var(r_k)=k^(2H)Var(r_1)，k∈{2,4,8,16} 取均值，
滚动 120d 方差，再 5d 平滑。判据沿用 S9 预注册三条（8月改善>=1.5pp、
全期 Sharpe>=基线-0.10、exBull>=基线-0.10），对照含 vt15 以判"替代还是叠加"。

变体：
  none      对照（生产口径 top15/5d/10bps）
  vt15      S9 参照
  h50/h48   H < 0.5 / 0.48 -> 半仓
  hq20      H < 滚动252 20分位 -> 半仓（自适应阈）
  vth       vt15 × h50 叠加
所有 exposure 只用 t 日收盘及以前信息（决策在 t 收盘，次日生效，无未来函数）。
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
EP_W = 0.10
OOS_START = pd.Timestamp("2019-01-01")
BULL = ("2024-01-01", "2025-12-31")


def metrics(net: pd.Series) -> dict:
    net = net[net.index >= OOS_START]
    eq = (1 + net).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net.std() * np.sqrt(252))
    ex = net[(net.index < BULL[0]) | (net.index > BULL[1])]
    exeq = (1 + ex).cumprod()
    exyrs = max((exeq.index[-1] - exeq.index[0]).days / 365.25, 1e-9)
    excagr = float(exeq.iloc[-1] ** (1 / exyrs) - 1)
    exvol = float(ex.std() * np.sqrt(252))
    return {"cagr": round(cagr * 100, 1),
            "sharpe": round(cagr / vol, 2) if vol > 0 else None,
            "maxdd": round(float(((eq / eq.cummax()) - 1).min()) * 100, 1),
            "sharpe_ex_bull": round(excagr / exvol, 2) if exvol > 0 else None}


def subperiod(net: pd.Series, lo: str, hi: str) -> dict:
    seg = net[(net.index >= lo) & (net.index <= hi)]
    eq = (1 + seg).cumprod()
    return {"ret": round(float(eq.iloc[-1] - 1) * 100, 2) if len(eq) else None,
            "maxdd": round(float(((eq / eq.cummax()) - 1).min()) * 100, 2) if len(eq) else None}


def main():
    panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
    close, volume = panel["close"], panel["volume"]
    days = close.index
    fwd = close.pct_change(fill_method=None).shift(-1)
    ret = close.pct_change(fill_method=None)

    _fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
    for _k, _v in _fund.items():
        if _k.startswith("fund:"):
            panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()

    tradable = close.notna() & volume.fillna(0).gt(0)
    lim = pd.DataFrame(0.10, index=days, columns=close.columns)
    star = [c for c in close.columns if c.startswith("688")]
    gem = [c for c in close.columns if c.startswith("30")]
    if star: lim[star] = 0.20
    if gem: lim.loc[days >= pd.Timestamp("2020-08-24"), gem] = 0.20
    first_back = close.notna() & close.shift(1).isna()
    long_gap = first_back & close.shift(20).isna()
    anomalous = (ret.abs() > lim + 0.02) & ~long_gap
    fwd_clean = fwd.mask(anomalous.shift(-1).fillna(False), 0.0)
    tradable = tradable & ~anomalous
    limit_up = tradable & (ret >= lim - 0.002)
    limit_down = tradable & (ret <= -(lim - 0.002))

    from src.factors.registry import get_default_registry
    reg = get_default_registry()

    def zscore(df):
        mu, sd = df.mean(axis=1), df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    print("computing factors...", flush=True)
    fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
           for a in STABLE7}
    ep = zscore(reg.compute("fund_earnings_yield", panel))
    rf = fwd.rank(axis=1)
    ic = {a: fac[a].rank(axis=1).corrwith(rf, axis=1) for a in STABLE7}

    def ir_of(s):
        s = s.dropna()
        return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

    print("walk-forward signal...", flush=True)
    sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(ic[a].loc[win]) for a in STABLE7}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        block = days[start:start + RETRAIN]
        sig7.loc[block] = sum(fac[a].loc[block] * (irs[a] / wsum) for a in STABLE7)
    sig = (1 - EP_W) * sig7 + EP_W * ep

    # ---- exposure 序列（全部只用 t 收盘及以前信息）----
    idx = (pd.read_csv(DATA / "csi300_index_daily.csv", parse_dates=["date"])
           .set_index("date")["close"].reindex(days).ffill())
    idx_ret = idx.pct_change(fill_method=None)
    idx_vol20 = idx_ret.rolling(20).std() * np.sqrt(252)
    disp = ret.rank(axis=1).std(axis=1).rolling(20).mean()  # 截面离散度代理
    disp_flag = disp > disp.rolling(252).quantile(0.80)

    def hurst_k(k: int) -> pd.Series:
        rk = idx.pct_change(k, fill_method=None)
        v1 = idx_ret.rolling(120).var()
        vk = rk.rolling(120).var()
        return np.log(vk / v1) / (2 * np.log(k))

    H = (pd.concat([hurst_k(k) for k in (2, 4, 8, 16)], axis=1)
         .mean(axis=1).rolling(5).mean().clip(-0.5, 1.5))
    haug = H[(H.index >= "2026-07-15") & (H.index <= "2026-08-24")]
    print(f"Hurst: 全期中位 {H.median():.3f}，P(H<0.5)={(H < 0.5).mean()*100:.0f}%；"
          f"7/15-8/24 区间 {haug.min():.3f}~{haug.max():.3f}，"
          f"8月均值 {H[H.index >= '2026-08-01'].mean():.3f}")

    exposures = {"none": pd.Series(1.0, index=days)}
    exposures["vt15"] = (0.15 / idx_vol20).clip(upper=1.0).fillna(1.0)
    for thr, name in ((0.50, "h50"), (0.48, "h48")):
        exposures[name] = pd.Series(np.where(H < thr, 0.5, 1.0), index=days).fillna(1.0)
    for q in (0.10, 0.15, 0.20, 0.25, 0.30):
        exposures[f"hq{int(q*100)}"] = pd.Series(
            np.where(H < H.rolling(252).quantile(q), 0.5, 1.0), index=days).fillna(1.0)

    # ---- 约束回测（gate23 同构 + exposure 钩子）----
    def backtest(exp: pd.Series) -> pd.Series:
        w = pd.DataFrame(0.0, index=days, columns=close.columns)
        held = set()
        for i, t in enumerate(days):
            if i % REBAL == 0:
                rowv = sig.loc[t].dropna()
                if len(rowv) >= TOP_N:
                    dset = set(rowv.nlargest(TOP_N).index)
                    keep = held & dset
                    locked = {s for s in held - dset
                              if not tradable.at[t, s] or limit_down.at[t, s]}
                    buys = []
                    for s in rowv.sort_values(ascending=False).index:
                        if len(keep) + len(locked) + len(buys) >= TOP_N:
                            break
                        if s in held or not tradable.at[t, s] or limit_up.at[t, s]:
                            continue
                        buys.append(s)
                    held = keep | locked | set(buys)
            if held:
                w.loc[t, list(held)] = exp.at[t] / max(len(held), TOP_N)
        gross = (w * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
        turn = (w.diff().abs().sum(axis=1) / 2).fillna(0.0).shift(1).fillna(0.0)
        return gross - turn * 2 * COST

    base = backtest(exposures["none"])
    bm = metrics(base)
    print(f"\n{'variant':8s} {'CAGR':>6s} {'Sharpe':>7s} {'MaxDD':>7s} {'exBull':>7s} "
          f"{'8月26':>7s} {'8月25':>7s}  判定")
    print("-" * 70)

    def line(name, net, m):
        a26, a25 = subperiod(net, "2026-08-01", "2026-08-31"), subperiod(net, "2025-08-01", "2025-08-31")
        if name == "none":
            verdict = "对照"
        else:
            ok = (a26["ret"] - subperiod(base, "2026-08-01", "2026-08-31")["ret"] >= 1.5
                  and m["sharpe"] >= bm["sharpe"] - 0.10
                  and m["sharpe_ex_bull"] >= bm["sharpe_ex_bull"] - 0.10)
            verdict = "PASS" if ok else "fail"
        print(f"{name:8s} {m['cagr']:>5.1f}% {m['sharpe']:>7.2f} {m['maxdd']:>6.1f}% "
              f"{m['sharpe_ex_bull']:>7.2f} {a26['ret']:>+6.2f}% {a25['ret']:>+6.2f}%  {verdict}")

    line("none", base, bm)
    for name in ["vt15", "h50", "hq10", "hq15", "hq20", "hq25", "hq30"]:
        net = backtest(exposures[name])
        line(name, net, metrics(net))

    # 8 月内最痛 5 日回撤对比（2026-08）
    print("\n2026-08 段内最大回撤（净值）：")
    for name in ["none", "vt15", "h50", "hq20"]:
        net = base if name == "none" else backtest(exposures[name])
        print(f"  {name:8s} {subperiod(net, '2026-08-01', '2026-08-31')['maxdd']:+.2f}%")


if __name__ == "__main__":
    main()
