# Mined from edtechre/pybroker vect.py (MIT), 2026-08-21: 线性趋势强度.
"""pybroker_ltrend20: formula = \\hat{b}_1(\\log\\mathrm{close}_t \\sim t, 20)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    'id': 'pybroker_ltrend20',
    'nickname': '线性趋势强度 (PyBroker vect)',
    'theme': ['momentum'],
    'formula_latex': '\\hat{b}_1(\\log\\mathrm{close}_t \\sim t, 20)',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 20,
    'notes': (
        '20 日窗口 log(close) 对时间的线性拟合斜率，按窗口收益波动归一（PyBroker linear_trend 核）：尺度自由的动量强度（对照项）。'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pybroker_ltrend20 on the supplied panel."""
    y = np.log(panel['close'])
    t = np.arange(20, dtype=float)
    t -= t.mean()
    def lin(w):
        if np.isnan(w).any():
            return np.nan
        return np.polyfit(t, w, 1)[0]
    slope = y.rolling(20).apply(lin, raw=True)
    scale = y.diff().rolling(20).std()
    return zscore(slope / scale.replace(0, np.nan))
