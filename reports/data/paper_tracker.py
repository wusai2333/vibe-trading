"""因子模拟盘跟踪（2026-08-26 13:00 盘中建仓，实时价锁定）。

价格法净值：持仓 i 在 t 日价值 = w_i × close_i[t]/entry_i，现金 50% 恒定。
盘中建仓的首日即含 13:00→收盘 这段。基准：
  沪深300 从 13:00 点位 4600.57 起（与组合同时点，公平）；
  全池等权从 08-26 收盘起（无盘中池数据，首日基准=0，次日起可比）。
每次运行：面板到最新交易日 -> 算净值 -> 追加 paper_history.csv -> 打印摘要。
v1 固定持仓；调仓/闸门变动手动改 paper_portfolio.json。
"""
import json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
DATA = Path(__file__).resolve().parent
PP = DATA / "paper_portfolio.json"

pp = json.load(open(PP))
panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close = panel["close"]
days = close.index
inception = pd.Timestamp(pp["inception"])

if inception not in days:
    print(f"面板尚无建仓日 {pp['inception']} 数据，等今晚重建后再跑")
    sys.exit(0)

# 持仓净值（从建仓日起，含首日盘中→收盘）
w_cash = pp.get("cash_pct", 0.0) / 100.0
eq = pd.Series(w_cash, index=days[days >= inception])
for p in pp["positions"]:
    sym, w_i, ep = p["symbol"], p["weight_pct"] / 100.0, p["entry_price"]
    if ep and sym in close.columns:
        px = close.loc[eq.index, sym]
        eq = eq + w_i * (px / ep).fillna(1.0)   # 缺数据日按持平
# 建仓时点净值恰为 1.0（现金+各仓按入场价计），直接作净值，保留首日 13:00→收盘涨跌
port = eq / (w_cash + sum(p["weight_pct"] / 100.0 for p in pp["positions"]))

# 基准1：沪深300 从 13:00 点位
idx = pd.read_csv(DATA / "csi300_index_daily.csv", parse_dates=["date"]).set_index("date")["close"]
b_entry = pp.get("benchmark_entry", {}).get("csi300")
idx_eq = (idx.reindex(eq.index) / b_entry) if b_entry else None

# 基准2：全池等权 从建仓日收盘
pool_ret = close.pct_change(fill_method=None).mean(axis=1)
pool_eq = (1 + pool_ret.reindex(eq.index).fillna(0)).cumprod()
pool_eq = pool_eq / pool_eq.iloc[0]

last = eq.index[-1]
hist = pd.DataFrame({"port": port, "csi300": idx_eq, "pool_ew": pool_eq})
hist.index.name = "date"
hf = DATA / "paper_history.csv"
if hf.exists():
    old = pd.read_csv(hf, index_col=0, parse_dates=True)
    hist = pd.concat([old, hist[~hist.index.isin(old.index)]])
hist.to_csv(hf)

n_days = len(eq)
print(f"模拟盘 {pp['inception']} 13:00 建仓 -> {last.date()}（第 {n_days} 个交易日，仓位 {(1-w_cash)*100:.0f}%）")
print(f"{'':10s} {'今日':>8s} {'累计':>9s} {'vs300':>9s}")
d_ret = port.iloc[-1] / port.iloc[-2] - 1 if n_days > 1 else port.iloc[-1] - 1
print(f"{'模拟盘':10s} {d_ret*100:+7.2f}% {(port.iloc[-1]-1)*100:+8.2f}% "
      f"{(port.iloc[-1]-idx_eq.iloc[-1])*100:+8.2f}pp" if idx_eq is not None else "")
if idx_eq is not None:
    i_ret = idx_eq.iloc[-1] / idx_eq.iloc[-2] - 1 if n_days > 1 else idx_eq.iloc[-1] - 1
    print(f"{'沪深300':10s} {i_ret*100:+7.2f}% {(idx_eq.iloc[-1]-1)*100:+8.2f}%")
p_ret = pool_eq.iloc[-1] / pool_eq.iloc[-2] - 1 if n_days > 1 else pool_eq.iloc[-1] - 1
print(f"{'全池等权':10s} {p_ret*100:+7.2f}% {(pool_eq.iloc[-1]-1)*100:+8.2f}%")
print("\n个股（自建仓价起）:")
for p in sorted(pp["positions"], key=lambda x: -x["weight_pct"]):
    sym = p["symbol"]
    if sym in close.columns and p["entry_price"]:
        r_since = close.loc[last, sym] / p["entry_price"] - 1
        print(f"  {p['name']:6s} {p['weight_pct']:>4.1f}%仓  入场{p['entry_price']:>8.2f} 现{close.loc[last,sym]:>8.2f}  {r_since*100:+6.2f}%  贡献{r_since*p['weight_pct']/100*100:+5.2f}pp")
