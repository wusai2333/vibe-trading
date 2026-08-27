"""闸3 覆盖层：overextension 降权能否在不伤收益的前提下压低动量崩溃回撤（2026-08-26）。

基线 = 生产 stable-7+EP 约束引擎（Top15/5日调仓/10bps，walk-forward IR 权重）。
变体 = 同一信号，选股前对 extension_z 高于 1.28（约前10%）的候选按 λ 扣分：
      score_adj = blend - λ × max(extension_z - 1.28, 0)，λ ∈ {0.25, 0.5, 1.0}
检验：CAGR/Sharpe/MaxDD/剔牛市 是否保持，且最惨的 5 日超额（-6.32/-3.29pp 那类）是否被压住。
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent

panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
# EP 所需基金列
_fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
for _k, _v in _fund.items():
    if _k.startswith("fund:"):
        panel[_k] = _v.reindex(panel["close"].index).reindex(columns=panel["close"].columns).ffill()

close, volume = panel["close"], panel["volume"]
days = close.index
fwd = close.pct_change(fill_method=None).shift(-1)
ret = close.pct_change(fill_method=None)
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

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
TRAIN, RETRAIN, TOP_N, REBAL, COST = 252, 63, 15, 5, 0.001
EP_W = 0.10
OOS = pd.Timestamp("2019-01-01")

from src.factors.registry import get_default_registry
reg = get_default_registry()
def zscore(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

print("computing factors...", flush=True)
fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
ep = zscore(reg.compute("fund_earnings_yield", panel))
ic = {a: fac[a].rank(axis=1).corrwith(fwd_clean.rank(axis=1), axis=1) for a in STABLE7}
def ir_of(s):
    s = s.dropna()
    return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
for start in range(TRAIN, len(days), RETRAIN):
    win = days[start - TRAIN:start - 1]
    irs = {a: ir_of(ic[a].loc[win]) for a in STABLE7}
    wsum = sum(abs(v) for v in irs.values()) or 1.0
    blk = days[start:start + RETRAIN]
    sig7.loc[blk] = sum(fac[a].loc[blk] * (irs[a] / wsum) for a in STABLE7)
blend = (1 - EP_W) * sig7 + EP_W * ep

# overextension 维度
ma20 = close.rolling(20).mean()
ext = close / ma20 - 1
ext_z = zscore(ext)
THR = 1.28

def backtest(sig):
    w = pd.DataFrame(0.0, index=sig.index, columns=sig.columns)
    held = set()
    for i, t in enumerate(sig.index):
        if i % REBAL == 0:
            rowv = sig.loc[t].dropna()
            if len(rowv) >= TOP_N:
                dset = set(rowv.nlargest(TOP_N).index)
                keep = held & dset
                locked = {s for s in held - dset if not tradable.at[t, s] or limit_down.at[t, s]}
                buys = []
                for s in rowv.sort_values(ascending=False).index:
                    if len(keep) + len(locked) + len(buys) >= TOP_N: break
                    if s in held or not tradable.at[t, s] or limit_up.at[t, s]: continue
                    buys.append(s)
                held = keep | locked | set(buys)
        if held:
            w.loc[t, list(held)] = 1.0 / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    bench = fwd_clean.fillna(0).mean(axis=1).shift(1).fillna(0.0)
    return net[days >= OOS], bench[days >= OOS], turn[days >= OOS].mean()

def metrics(net, bench):
    exc = net - bench
    eq = (1 + net).cumprod()
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    vol = float(net.std() * np.sqrt(252))
    mdd = float((eq / eq.cummax() - 1).min())
    exb = net[~net.index.year.isin([2024, 2025])]
    exb_eq = (1 + exb).cumprod()
    exb_yrs = max((exb_eq.index[-1] - exb_eq.index[0]).days / 365.25, 1e-9)
    exb_cagr = float(exb_eq.iloc[-1] ** (1 / exb_yrs) - 1)
    exb_sharpe = exb_cagr / (exb.std() * np.sqrt(252)) if exb.std() else 0
    # 最惨 5 日累计超额
    exc5 = exc.rolling(5).sum()
    return {"cagr": cagr, "sharpe": cagr / vol if vol else 0, "mdd": mdd,
            "exb_sharpe": exb_sharpe, "worst5_exc": float(exc5.min())}

print("backtesting...", flush=True)
results = {}
net0, bench, to0 = backtest(blend)
results["baseline"] = (metrics(net0, bench), to0)
for lam in [0.25, 0.5, 1.0]:
    pen = lam * (ext_z - THR).clip(lower=0)
    netl, _, tol = backtest(blend - pen)
    results[f"pen_{lam}"] = (metrics(netl, bench), tol)

print(f"{'variant':10s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'exBull':>7s} {'worst5exc':>10s} {'turn':>6s}")
for name, (m, to) in results.items():
    print(f"{name:10s} {m['cagr']*100:6.1f}% {m['sharpe']:7.2f} {m['mdd']*100:6.1f}% "
          f"{m['exb_sharpe']:7.2f} {m['worst5_exc']*100:9.2f}% {to*100:5.1f}%")
