"""起涨点（launch-point）因子挖掘（2026-08-26，用户需求）。

起涨点因子 = 触发型/状态型信号，用 t 日收盘前信息预测 t+1 起的短期延续，
目标是比滞后动量更早捕捉启动。全部 close-based，无未来函数（信号在 t 收盘，
收益从 t+1 计）。与 zoo 的差别：这些是"事件触发 + 形态状态"，不是纯尾随窗口。

8 个候选：
  F1 vol_ignition      今日量 / 前20日基础量（量能点燃）
  F2 breakout_int      收盘超前20日最高价的幅度（新高突破强度）
  F3 vol_contract      近期波动/前期波动 取负（缩量蓄势/卷曲，高=收缩）
  F4 quiet_ignition    今日涨幅 × 前15日基础安静度（安静底座后的启动）
  F5 fresh_break       站上20日高点的新鲜度（刚突破得分高）
  F6 vp_ignition       收益×量比（带方向的量价点燃）
  F7 gap_up            今日开盘跳空（open/prev_close-1）
  F8 obv_slope         OBV 5日斜率（归一化，资金流入拐点）

筛选：对 fwd1/fwd3/fwd5 的日度截面 rank-IC（均值+t 值，2019-01 起），
并与 stable-7 七因子做相关（>0.5 判克隆）。|IC|>=0.02 且 |t|>=2 且非克隆 -> 进闸1。
"""
import pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent

panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
close, open_ = panel["close"], panel["open"]
high, low, volume = panel["high"], panel["low"], panel["volume"]
days = close.index
ret = close.pct_change(fill_method=None)
eps = 1e-9

# ---- candidates ----
vol_base = volume.shift(1).rolling(20).mean()
F1 = volume / (vol_base + eps)

hh20 = high.shift(1).rolling(20).max()
F2 = close / (hh20 + eps) - 1

vol_recent = ret.rolling(10).std()
vol_prior = ret.shift(10).rolling(30).std()
F3 = -(vol_recent / (vol_prior + eps))

quiet = 1.0 / (ret.shift(1).rolling(15).std() + eps)
F4 = np.maximum(ret, 0) * quiet

ll20 = low.shift(1).rolling(20).min()
above = (close > hh20).astype(float)
consec = above.copy()
for _ in range(1, 60):  # 连续站上天数（上限60）
    consec = above * (consec.shift(1).fillna(0) + 1)
F5 = above / (consec + eps)

F6 = ret * F1

F7 = open_ / close.shift(1) - 1

obv = (np.sign(ret).fillna(0) * volume).cumsum()
F8 = obv.diff(5) / (volume.rolling(20).mean() * 5 + eps)

cands = {"F1_vol_ignition": F1, "F2_breakout_int": F2, "F3_vol_contract": F3,
         "F4_quiet_ignition": F4, "F5_fresh_break": F5, "F6_vp_ignition": F6,
         "F7_gap_up": F7, "F8_obv_slope": F8}

# ---- forward returns ----
fwd = {h: close.pct_change(h, fill_method=None).shift(-h) for h in (1, 3, 5)}

def rank_ic(f, y):
    s = f.rank(axis=1).corrwith(y.rank(axis=1), axis=1)
    s = s[days >= pd.Timestamp("2019-01-01")].dropna()
    return float(s.mean()), float(s.mean() / (s.std() / np.sqrt(len(s)))) if s.std() else 0.0

# ---- stable-7 correlation (clone gate) ----
STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
from src.factors.registry import get_default_registry
reg = get_default_registry()
fac7 = {a: reg.compute(a, panel).rolling(10, min_periods=6).mean() for a in STABLE7}

print(f"{'factor':18s} {'IC1':>7s} {'t1':>5s} {'IC3':>7s} {'t3':>5s} {'IC5':>7s} {'t5':>5s}  {'maxCorr':>7s}  verdict")
for name, f in cands.items():
    ics = [rank_ic(f, fwd[h]) for h in (1, 3, 5)]
    # clone check vs stable-7 (rank corr, OOS mean)
    mc = 0.0
    for a in STABLE7:
        c = f.rank(axis=1).corrwith(fac7[a].rank(axis=1), axis=1)
        c = c[days >= pd.Timestamp("2019-01-01")].dropna()
        mc = max(mc, abs(float(c.mean())) if len(c) else 0.0)
    # best horizon by |t|
    best = max(ics, key=lambda x: abs(x[1]))
    icb, tb = best
    clone = mc >= 0.5
    alive = abs(icb) >= 0.02 and abs(tb) >= 2.0 and not clone
    verdict = ("ALIVE->gate1" if alive else ("clone" if clone else "dead"))
    print(f"{name:18s} {ics[0][0]:+7.4f} {ics[0][1]:+5.1f} {ics[1][0]:+7.4f} {ics[1][1]:+5.1f} "
          f"{ics[2][0]:+7.4f} {ics[2][1]:+5.1f}  {mc:7.3f}  {verdict}")
