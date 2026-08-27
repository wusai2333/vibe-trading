# ============================================================
# 中文名称: 上涨天数占比因子 (UPDAYS20)
# 简要说明: 过去 20 日中上涨天数的占比：路径平滑的动量（青蛙跳锅效应）。
# 典型用途: Da-Gurkovych-Warner (2014)：连续小涨比跳涨更有持续性。与动量水平值互补。
# ============================================================
"""pool2 UPDAYS20: fraction of up days over trailing 20 days.

Path-dependent momentum: smooth persistent climbs (many small up-days) vs
jumpy paths. Related to the frog-in-the-pan effect (Da, Gurkovych, Warner
2014) where continuous information arrival predicts momentum continuation.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'pool2_updays20',
    'nickname': '20d up-day fraction',
    'theme': ['momentum'],
    'formula_latex': r'\frac{1}{20}\sum_{k=t-19}^{t} 1[r_k>0]',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'Counts ordinary up days; NOT limit-up counts (limit zoo).',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing 20-day fraction of positive-return days."""
    ret = panel['close'].pct_change()
    return (ret > 0).rolling(20, min_periods=15).mean()
