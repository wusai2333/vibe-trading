# ============================================================
# 中文名称: 成交量加权动量 (VWMOM20)
# 简要说明: 20 日收益按成交量加权：放量日的涨跌权重更大。
# 典型用途: 量价动量，与普通动量互补。
# ============================================================
"""pool3 vwmom20: 20-day volume-weighted momentum.

Source: volume-price momentum family (GF Securities volume-price factor
research). Returns on high-volume days carry more information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_vwmom20',
    'nickname': '20d volume-weighted momentum',
    'theme': ['momentum'],
    'formula_latex': r'\frac{\sum_{20} r_t V_t}{\sum_{20} V_t}',
    'columns_required': ['close', 'volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'Volume-weighted 20d return.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 20-day volume-weighted momentum."""
    ret = panel['close'].pct_change()
    vol = panel['volume']
    out = (ret * vol).rolling(20, min_periods=15).sum() / vol.rolling(20, min_periods=15).sum()
    return out.replace([np.inf, -np.inf], np.nan)
