# ============================================================
# 中文名称: PS流动性 (PSLIQ60)
# 简要说明: 收益对滞后符号成交量的滚动回归系数：量价冲击的可逆性。
# 典型用途: 流动性风险溢价。Pastor-Stambaugh (2003) 的日频简化。
# ============================================================
"""lit psliq60: rolling coefficient of return on lagged signed volume.

Source: Pastor & Stambaugh (2003) liquidity, daily simplified: regress
today's return on yesterday's volume-normalized signed return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'lit_psliq60',
    'nickname': '60d Pastor-Stambaugh liquidity proxy',
    'theme': ['liquidity'],
    'formula_latex': r'\gamma_i:\ r_{t} = \gamma sgn(r_{t-1})V_{t-1}/\bar V + \epsilon',
    'columns_required': ['close', 'volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Signed-volume impact coefficient, rolling 60d.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the rolling signed-volume impact coefficient."""
    ret = panel['close'].pct_change()
    vol = panel['volume']
    sv = (np.sign(ret) * vol / vol.rolling(20, min_periods=15).mean()).shift(1)
    coef = ret.rolling(60, min_periods=40).cov(sv).div(sv.rolling(60, min_periods=40).var(), axis=0)
    return coef.replace([np.inf, -np.inf], np.nan)
