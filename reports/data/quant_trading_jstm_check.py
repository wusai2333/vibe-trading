"""je-suis-tm/quant-trading 概念克隆性检验。

该仓库 12 个单资产 TA 策略（美股/期货/期权），概念全部落在经典 TA 族。
检验其最有辨识度的 3 个概念在 CSI300 面板上是否与 zoo 现有因子克隆
（闸 2 判据 |rho|>=0.5 即克隆），并给出原始 rank IC。只作诊断，不入 zoo。

概念：
  AO   Awesome Oscillator = SMA5(median)-SMA34(median), median=(H+L)/2
  HA   Heikin-Ashi 蜡烛方向 = HA_close - HA_open（递归 HA_open）
  DPOS Donchian 位置 = (C-LL20)/(HH20-LL20)（PSAR/突破类的位置等价物）
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "agent"))
DATA = Path(__file__).resolve().parent

STABLE7 = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow",
           "alpha101_060", "limit_dist", "vol_ivol60"]
COMPARE = STABLE7 + ["qlib158_roc5", "qlib158_roc20", "qlib158_rsv20",
                     "qlib158_imax20", "lit_resmom20", "momentum_mom40"]


def main():
    panel = pickle.load(open(DATA / "csi300_panel.pkl", "rb"))
    o, h, l, c = panel["open"], panel["high"], panel["low"], panel["close"]

    # --- candidates ---
    med = (h + l) / 2
    ao = med.rolling(5).mean() - med.rolling(34).mean()

    ha_c = (o + h + l + c) / 4
    ha_o = ha_c.copy() * np.nan
    ha_o.iloc[0] = o.iloc[0]
    for i in range(1, len(ha_c)):  # 递归 HA 开盘，2096 行循环可接受
        ha_o.iloc[i] = (ha_o.iloc[i - 1] + ha_c.iloc[i - 1]) / 2
    ha = ha_c - ha_o

    ll, hh = l.rolling(20).min(), h.rolling(20).max()
    dpos = (c - ll) / (hh - ll).replace(0, np.nan)
    cands = {"AO": ao, "HA": ha, "DPOS": dpos}

    # --- zoo factors ---
    from src.factors.registry import get_default_registry
    reg = get_default_registry()
    zoo = {a: reg.compute(a, panel) for a in COMPARE}

    # --- 5d forward return for IC ---
    fwd5 = c.pct_change(5, fill_method=None).shift(-5)

    def spearman(a: pd.DataFrame, b: pd.DataFrame) -> float:
        return float(a.rank(axis=1).corrwith(b.rank(axis=1), axis=1).mean())

    print(f"{'cand':6s} {'max|rho|':>9s} {'vs':22s} {'rankIC_5d':>10s}")
    for name, f in cands.items():
        rhos = {a: spearman(f, z) for a, z in zoo.items()}
        best = max(rhos, key=lambda a: abs(rhos[a]))
        ic = spearman(f, fwd5)
        verdict = "CLONE" if abs(rhos[best]) >= 0.5 else "novel?"
        print(f"{name:6s} {abs(rhos[best]):9.3f} {best:22s} {ic:10.4f}  {verdict}")


if __name__ == "__main__":
    main()
