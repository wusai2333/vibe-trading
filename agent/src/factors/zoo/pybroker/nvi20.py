# Mined from edtechre/pybroker vect.py (MIT), 2026-08-21: 负成交量指数.
"""pybroker_nvi20: formula = \\sum_{d\\in\\downarrow vol,20} r_d / (\\sqrt{20}\\,\\sigma_{250})."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    'id': 'pybroker_nvi20',
    'nickname': '负成交量指数 (PyBroker vect)',
    'theme': ['momentum'],
    'formula_latex': '\\sum_{d\\in\\downarrow vol,20} r_d / (\\sqrt{20}\\,\\sigma_{250})',
    'columns_required': ['close','volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 250,
    'notes': (
        '近 20 日中成交量缩小日的对数收益之和，除以 sqrt(20) 与 250 日收益波动（PyBroker normalized_negative_volume_index）：缩量日的收益特征。'
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return pybroker_nvi20 on the supplied panel."""
    close = panel['close']
    volume = panel['volume']
    logret = np.log(close / close.shift(1))
    vol_dn = (volume < volume.shift(1)).astype(float)
    vol_dn[volume.shift(1).isna() | volume.isna()] = np.nan
    tot = (logret * vol_dn).rolling(20).sum() / np.sqrt(20.0)
    denom = logret.rolling(250, min_periods=120).std()
    return zscore(tot / denom)
