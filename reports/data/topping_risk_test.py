"""高位回调陷阱检验（2026-08-26，用户方法论洞察）。

用户洞察：涨了很久、开始回调的股票，滞后动量因子反而给高分——因子在顶部最危险时
喊"强"。这是动量崩溃的微观机制。检验两件事：
  A. 因子层：overextension/rollover 维度本身有没有预测力（IC）
  B. 条件层：在 stable-7 高分股内部，"高位+开始回调"的那批是否系统性跑输
     ——若跑输，证明现有排序确实被这类股欺骗，需要扩展评估

定义（全 close-based，信号 t 收盘，收益 t+1 起）：
  extension     = close/MA20 - 1            高于20日均线的幅度
  dist_peak     = close/rollmax(close,20)-1  距20日峰值（负=已回落）
  ret5          = close/close.shift(5)-1     近5日收益
  topping_risk  = z(extension) × (-z(ret5))  高位×近期走弱（越高越危险）
  decel         = ret5 - ret20/4             动量减速（短动量弱于长动量）
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
days = close.index
ret5 = close / close.shift(5) - 1
ret20 = close / close.shift(20) - 1
ma20 = close.rolling(20).mean()
extension = close / ma20 - 1
dist_peak = close / close.rolling(20).max() - 1

def z(df):
    mu, sd = df.mean(axis=1), df.std(axis=1)
    return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

topping_risk = z(extension) * (-z(ret5))
decel = ret5 - ret20 / 4
fwd5 = close.pct_change(5, fill_method=None).shift(-5)

def ic_series(f, y):
    s = f.rank(axis=1).corrwith(y.rank(axis=1), axis=1)
    return s[days >= pd.Timestamp("2019-01-01")].dropna()

print("== A. 因子层：各维度对 fwd5 的 rank-IC（负=该维度高者未来跑输）")
for name, f in [("extension", extension), ("dist_peak", dist_peak), ("ret5", ret5),
                ("topping_risk", topping_risk), ("decel", decel)]:
    s = ic_series(f, fwd5)
    t = s.mean() / (s.std() / np.sqrt(len(s)))
    yr = s.groupby(s.index.year).mean()
    print(f"  {name:14s} IC {s.mean():+.4f}  t {t:+.2f}  分年正负 {(yr>0).sum()}/{len(yr)}")

# == B. 条件层：stable-7 高分股内部，高位回调者是否跑输 ==
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
from src.factors.registry import get_default_registry
reg = get_default_registry()
fac = {a: z(reg.compute(a, panel).rolling(10, min_periods=6).mean()) for a in STABLE7}
# 用最近可得 IR 权重近似（等权简化，足以做条件分层）
blend = sum(fac[a] for a in STABLE7) / len(STABLE7)

print("== B. 条件层：stable-7 高分（前20%）股内部，按 topping_risk 分层 ==")
oos = days >= pd.Timestamp("2019-01-01")
fwd_ret5 = fwd5.copy()
rows = []
for t in days[oos]:
    sc = blend.loc[t].dropna()
    if len(sc) < 60:
        continue
    top = sc[sc >= sc.quantile(0.8)].index
    tr = topping_risk.loc[t, top].dropna()
    fr = fwd_ret5.loc[t, tr.index].dropna()
    common = tr.index.intersection(fr.index)
    if len(common) < 30:
        continue
    hi = tr[common] >= tr[common].median()   # 高位回调组
    rows.append({"hi_trap": fr[common[hi.values]].mean(),
                 "lo_safe": fr[common[~hi.values]].mean()})
rdf = pd.DataFrame(rows).dropna()
spread = (rdf["lo_safe"] - rdf["hi_trap"])
print(f"  样本天数 {len(rdf)}")
print(f"  高位回调组 fwd5 均值: {rdf['hi_trap'].mean()*100:+.3f}%")
print(f"  其余高分组   fwd5 均值: {rdf['lo_safe'].mean()*100:+.3f}%")
print(f"  差值(safe-trap): {spread.mean()*100:+.3f}%/5日, t {spread.mean()/(spread.std()/np.sqrt(len(spread))):+.2f}, 为正天数占比 {(spread>0).mean()*100:.0f}%")
