"""hq 闸门灵敏度（2026-08-26）：阈值分位 q ∈ {0.10,0.15,0.20(现产),0.25,0.30}。

同一生产信号（stable-7+EP 约束引擎），闸门 exposure = 0.5 if H < roll252 分位(q) else 1.0。
H 与生产 screener 同口径（sina 指数、方差标度 k∈{2,4,8,16}、120d 窗、5d 平滑）。
看：回撤/最惨5日是否随 q 收紧而压，代价多少 CAGR/Sharpe，以及 q=0.20 是平台还是刀锋。
"""
import pickle, sys, warnings
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

# H（生产同口径）
idx = pd.read_csv(DATA / "csi300_index_daily.csv", parse_dates=["date"]).set_index("date")["close"]
idx = idx.reindex(days).ffill()
ir_ = idx.pct_change(fill_method=None)
def hk(k):
    vk = idx.pct_change(k, fill_method=None).rolling(120).var()
    return np.log(vk / ir_.rolling(120).var()) / (2 * np.log(k))
H = pd.concat([hk(k) for k in (2, 4, 8, 16)], axis=1).mean(axis=1).rolling(5).mean().clip(-0.5, 1.5)

def backtest(sig, expo):
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
            w.loc[t, list(held)] = expo.at[t] / max(len(held), TOP_N)
    gross = (w.fillna(0) * fwd_clean.fillna(0)).sum(axis=1).shift(1).fillna(0.0)
    turn = (w.diff().abs().sum(axis=1) / 2).fillna(0).shift(1).fillna(0.0)
    net = gross - turn * 2 * COST
    bench = fwd_clean.fillna(0).mean(axis=1).shift(1).fillna(0.0)
    return net[days >= OOS], bench[days >= OOS]

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
    return {"cagr": cagr, "sharpe": cagr / vol if vol else 0, "mdd": mdd,
            "exb_sharpe": exb_sharpe, "worst5": float(exc.rolling(5).sum().min())}

print("backtesting...", flush=True)
rows = []
# no-gate 对照
net0, bench = backtest(blend, pd.Series(1.0, index=days))
m0 = metrics(net0, bench)
rows.append(("no_gate", m0, 1.0, 0))
for q in [0.10, 0.15, 0.20, 0.25, 0.30]:
    thr = H.rolling(252).quantile(q)
    gate_on = (H < thr).fillna(False)
    expo = pd.Series(np.where(gate_on, 0.5, 1.0), index=days)
    netl, _ = backtest(blend, expo)
    m = metrics(netl, bench)
    avg_exp = float(expo[days >= OOS].mean())
    flips = int((gate_on[days >= OOS].astype(int).diff().abs()).sum())
    rows.append((f"q{int(q*100)}", m, avg_exp, flips))

print(f"{'variant':8s} {'CAGR':>7s} {'Sharpe':>7s} {'MaxDD':>7s} {'exBull':>7s} {'worst5':>8s} {'avgExp':>7s} {'flips':>6s}")
for name, m, ae, fl in rows:
    print(f"{name:8s} {m['cagr']*100:6.1f}% {m['sharpe']:7.2f} {m['mdd']*100:6.1f}% "
          f"{m['exb_sharpe']:7.2f} {m['worst5']*100:7.2f}% {ae:7.2f} {fl:6d}")
