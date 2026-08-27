# Mined from edtechre/pybroker vect.py (MIT), 2026-08-21: 正成交量指数.
"""pybroker_pvi20: formula = \\sum_{d\\in\\uparrow vol,20} r_d / (\\sqrt{20}\\,\\sigma_{250})."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    'id': 'pybroker_pvi20',
    'nickname': '正成交量指数 (PyBroker vect)',
    'theme': ['momentum'],
    'formula_latex': '\\sum_{d\\in\\uparrow vol,20} r_d / (\\sqrt{20}\\,\\sigma_{250})',
    'columns_required': ['close','volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 250,
    'notes': (
        '近 20 日中成交量放大日的对数收益之和，除以 sqrt(20) 与 250 日收益波动（PyBroker normalized_positive_volume_index）：放量上涨的累积确认。'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pybroker_pvi20 on the supplied panel."""
    close = panel['close']
    volume = panel['volume']
    logret = np.log(close / close.shift(1))
    vol_up = (volume > volume.shift(1)).astype(float)
    vol_up[volume.shift(1).isna() | volume.isna()] = np.nan
    tot = (logret * vol_up).rolling(20).sum() / np.sqrt(20.0)
    denom = logret.rolling(250, min_periods=120).std()
    return zscore(tot / denom)
