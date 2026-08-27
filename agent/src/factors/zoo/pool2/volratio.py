# ============================================================
# 中文名称: 波动率状态因子 (VOLRATIO)
# 简要说明: 5 日波动率 / 60 日波动率：短期波动相对长期是放大还是收缩。
# 典型用途: 波动率状态切换信号；收缩后突破 vs 放大后均值回归，方向由检验定。
# ============================================================
"""pool2 VOLRATIO: short-to-long volatility ratio (5d/60d).

Volatility regime state: expansion (>1) vs contraction (<1) relative to
the trailing quarter. Regime-change information not present in volatility
level factors (rvol20/ivol60).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_volratio',
    'nickname': '5d/60d volatility ratio',
    'theme': ['volatility'],
    'formula_latex': r'\sigma_5(r)/\sigma_{60}(r)',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Vol regime state; sign decided by bench.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 5-day over 60-day return volatility ratio."""
    ret = panel['close'].pct_change()
    short = ret.rolling(5, min_periods=4).std()
    long = ret.rolling(60, min_periods=40).std()
    out = short / long
    return out.replace([np.inf, -np.inf], np.nan)
