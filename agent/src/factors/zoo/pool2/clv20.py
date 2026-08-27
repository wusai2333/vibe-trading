# ============================================================
# 中文名称: 20日收盘位置因子 (CLV20)
# 简要说明: 收盘价在过去 20 日高低区间中的相对位置（随机指标 %K 的 20 日版）。
# 典型用途: 接近区间顶部 = 强势延续 or 超买，方向由检验定。alpha101_060 是 1 日日内版。
# ============================================================
"""pool2 CLV20: close location within trailing 20-day range.

Stochastic-%K style position of today's close inside the 20-day high-low
range. The 1-day intraday analogue is alpha101_060; this is the multi-day
trend-position variant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_clv20',
    'nickname': 'close location in 20d range',
    'theme': ['momentum'],
    'formula_latex': r'\frac{c_t - \min_{20} c}{\max_{20} c - \min_{20} c}',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 20,
    'notes': 'Bounded 0-1; degenerate flat ranges -> NaN.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return close position within the trailing 20-day range."""
    close = panel['close']
    lo = close.rolling(20, min_periods=15).min()
    hi = close.rolling(20, min_periods=15).max()
    out = (close - lo) / (hi - lo)
    return out.replace([np.inf, -np.inf], np.nan)
