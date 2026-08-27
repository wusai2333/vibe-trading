"""八月因子审计：stable-7+EP 信号是否与收益反向？

用户质疑（2026-08-24）：整个 8 月因子与表现相反。
检验方法：按生产口径（252d 滚动 IR 加权、63d 重训、rolling(10,6) 平滑、
EP 固定 10%）重建 walk-forward 信号，逐日 rank IC，回答三个问题：
  1. 8 月逐日 IC 是否为系统性负值（还是个别回撤日）
  2. 8 月 IC 在 2019 年以来月度 IC 分布中处于什么位置
  3. 哪些因子在 8 月翻转（对比全期 IC）
只诊断，不改生产。
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
TRAIN, RETRAIN, EP_W = 252, 63, 0.10


def main():
    panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
    close, volume = panel["close"], panel["volume"]
    days = close.index
    fwd = close.pct_change(fill_method=None).shift(-1)
    tradable = close.notna() & volume.fillna(0).gt(0)
    fwd = fwd.mask(~tradable.shift(-1).fillna(False))

    _fund = pickle.load(open(DATA / "fund_cache.pkl", "rb"))
    for _k, _v in _fund.items():
        if _k.startswith("fund:"):
            panel[_k] = _v.reindex(days).reindex(columns=close.columns).ffill()

    from src.factors.registry import get_default_registry
    reg = get_default_registry()

    def zscore(df):
        mu, sd = df.mean(axis=1), df.std(axis=1)
        return df.sub(mu, axis=0).div(sd.replace(0, np.nan), axis=0)

    print("computing factors...", flush=True)
    fac = {a: zscore(reg.compute(a, panel).rolling(10, min_periods=6).mean())
           for a in STABLE7}
    ep = zscore(reg.compute("fund_earnings_yield", panel))

    # 逐日 rank IC（因子 vs 次曰收益）
    rf = fwd.rank(axis=1)
    ic = {a: fac[a].rank(axis=1).corrwith(rf, axis=1) for a in STABLE7}

    # walk-forward IR 加权（与 gate23 同构，向量化：IR 直接取 IC 窗口均值/波动）
    def ir_of(s):
        s = s.dropna()
        return float(s.mean() / s.std()) if len(s) >= 60 and s.std() else 0.0

    sig7 = pd.DataFrame(np.nan, index=days, columns=close.columns)
    for start in range(TRAIN, len(days), RETRAIN):
        win = days[start - TRAIN:start - 1]
        irs = {a: ir_of(ic[a].loc[win]) for a in STABLE7}
        wsum = sum(abs(v) for v in irs.values()) or 1.0
        block = days[start:start + RETRAIN]
        sig7.loc[block] = sum(fac[a].loc[block] * (irs[a] / wsum) for a in STABLE7)
    blend = (1 - EP_W) * sig7 + EP_W * ep
    icb = blend.rank(axis=1).corrwith(rf, axis=1)

    oos = icb[days >= pd.Timestamp("2019-01-01")].dropna()
    aug = icb[(days >= "2026-08-01") & (days <= "2026-08-24")].dropna()

    print("\n== 1. 八月逐日 blend IC ==")
    for t, v in aug.items():
        print(f"{t.date()}  {v:+.4f}" + ("  <<<" if abs(v) >= 0.05 else ""))
    tstat = aug.mean() / (aug.std() / np.sqrt(len(aug)))
    print(f"8月均值 {aug.mean():+.4f}  t={tstat:+.2f}  负IC天数 {(aug < 0).sum()}/{len(aug)}")

    print("\n== 2. 月度平均 IC：2026 各月 ==")
    s26 = oos[oos.index >= "2026-01-01"]
    m26 = s26.groupby(s26.index.to_period("M")).agg(["mean", "count"])
    for p, row in m26.iterrows():
        print(f"{p}  {row['mean']:+.4f}")
    hist_abs = (oos.abs() >= 0.3).mean()
    print(f"（全期 |日IC|>=0.3 的天数占比 {hist_abs*100:.1f}%，8月 {(aug.abs()>=0.3).mean()*100:.0f}%）")

    print("\n== 3. 八月在历史月度 IC 分布中的位置（2019-01 起）==")
    mall = oos.groupby(oos.index.to_period("M")).mean()
    rank = (mall < mall.loc["2026-08"]).mean()
    print(f"2026-08 月均 IC {mall.loc['2026-08']:+.4f}，历史分位 {rank*100:.1f}%（0%=最差月）")
    print("历史最差 5 个月:")
    for p, v in mall.nsmallest(5).items():
        print(f"  {p}  {v:+.4f}")
    print(f"历史月均 IC 分布: 均值 {mall.mean():+.4f}, std {mall.std():.4f}, 负月占比 {(mall<0).mean()*100:.0f}%")

    print("\n== 4. 分因子：8月 IC vs 全期 IC ==")
    print(f"{'factor':16s} {'8月IC':>8s} {'全期IC':>8s}  状态")
    ic["__EP__"] = ep.rank(axis=1).corrwith(rf, axis=1)
    for a in STABLE7 + ["__EP__"]:
        ica = ic[a]
        ica_oos = ica[days >= "2019-01-01"].dropna()
        a8 = ica[(days >= "2026-08-01") & (days <= "2026-08-24")].dropna()
        flip = "翻转!" if np.sign(a8.mean()) != np.sign(ica_oos.mean()) else ""
        print(f"{a:16s} {a8.mean():+8.4f} {ica_oos.mean():+8.4f}  {flip}")


if __name__ == "__main__":
    main()
