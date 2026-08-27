# ============================================================
# 中文名称: 聪明钱因子日频近似 (SMM20)
# 简要说明: 成交量加权绝对收益：大成交量日贡献的振幅占比，近似机构参与度。聪明钱因子（开源金工 2016）的日频降级版。
# 典型用途: 机构活动强度。预期方向由检验定。
# ============================================================
"""pool3 smm20: 20d volume-weighted absolute return (smart-money daily proxy).

Source: smart-money factor daily approximation; Kaiyuan Securities
microstructure series (2016). The original uses minute data; this is the
daily downgrade (volume-weighted |return|).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_smm20',
    'nickname': '20d volume-weighted absolute return (smart-money daily proxy)',
    'theme': ['volume'],
    'formula_latex': r'\frac{\sum_{20}|r_t|V_t}{\sum_{20}V_t}',
    'columns_required': ['close', 'volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'Volume-weighted |ret| over 20d.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 20-day volume-weighted absolute return."""
    ret = panel['close'].pct_change()
    vol = panel['volume']
    out = (ret.abs() * vol).rolling(20, min_periods=15).sum() / vol.rolling(20, min_periods=15).sum()
    return out.replace([np.inf, -np.inf], np.nan)
