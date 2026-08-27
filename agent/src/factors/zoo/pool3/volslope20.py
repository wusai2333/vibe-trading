# ============================================================
# 中文名称: 成交量趋势 (VOLSLOPE20)
# 简要说明: log(volume) 的 20 日线性斜率：量能趋势性放大/萎缩。
# 典型用途: 量在价先假设的量能趋势腿。
# ============================================================
"""pool3 volslope20: 20-day linear slope of log volume.

Source: volume-trend factor family (classic volume-price analysis; Haitong
volume-price factor series). Trending expansion/contraction of activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_volslope20',
    'nickname': '20d log-volume slope',
    'theme': ['volume'],
    'formula_latex': r'\mathrm{slope}_{20}(\log V_t)',
    'columns_required': ['volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 20,
    'notes': 'Linear slope of log volume over 20d.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the 20-day linear slope of log volume."""
    y = np.log(panel['volume'].replace(0, np.nan))
    x = np.arange(20, dtype=float)

    def _slope(w):
        if np.isnan(w).any():
            return np.nan
        return np.polyfit(x, w, 1)[0]

    return y.rolling(20, min_periods=20).apply(_slope, raw=True)
