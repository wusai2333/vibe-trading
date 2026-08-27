# Mined from edtechre/pybroker vect.py (MIT), 2026-08-21: 二次趋势曲率.
"""pybroker_qtrend20: formula = \\hat{b}_2(\\log\\mathrm{close}_t \\sim t + t^2, 20)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    'id': 'pybroker_qtrend20',
    'nickname': '二次趋势曲率 (PyBroker vect)',
    'theme': ['momentum'],
    'formula_latex': '\\hat{b}_2(\\log\\mathrm{close}_t \\sim t + t^2, 20)',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 20,
    'notes': (
        '20 日窗口 log(close) 对时间的二次多项式拟合的二次项系数，按窗口收益波动归一（PyBroker quadratic_trend 的 Legendre 二次核）：动量的加速度/曲率。'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pybroker_qtrend20 on the supplied panel."""
    y = np.log(panel['close'])
    t = np.arange(20, dtype=float)
    t -= t.mean()
    def quad(w):
        if np.isnan(w).any():
            return np.nan
        c = np.polyfit(t, w, 2)
        return c[0]
    curv = y.rolling(20).apply(quad, raw=True)
    scale = y.diff().rolling(20).std()
    return zscore(curv / scale.replace(0, np.nan))
