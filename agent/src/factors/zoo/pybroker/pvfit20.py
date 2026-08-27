# Mined from edtechre/pybroker vect.py (MIT), 2026-08-21: 量价回归斜率.
"""pybroker_pvfit20: formula = \\beta_{20}(\\log\\mathrm{close},\\log(\\mathrm{volume}+1))."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    'id': 'pybroker_pvfit20',
    'nickname': '量价回归斜率 (PyBroker vect)',
    'theme': ['momentum'],
    'formula_latex': '\\beta_{20}(\\log\\mathrm{close},\\log(\\mathrm{volume}+1))',
    'columns_required': ['close','volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 20,
    'notes': (
        '20 日窗口 log(close) 对 log(volume+1) 的回归斜率：量价同向趋势的强度（PyBroker price_volume_fit 的斜率核）。'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pybroker_pvfit20 on the supplied panel."""
    x = np.log(panel['volume'] + 1.0)
    y = np.log(panel['close'])
    slope = y.rolling(20).cov(x) / x.rolling(20).var()
    return zscore(slope)
