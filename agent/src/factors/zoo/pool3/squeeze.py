# ============================================================
# 中文名称: 振幅压缩 (SQUEEZE)
# 简要说明: 5 日均振幅 / 60 日均振幅：短期振幅相对长期的压缩/扩张状态。
# 典型用途: 波动率状态切换（布林挤压族）。
# ============================================================
"""pool3 squeeze: 5-day over 60-day mean normalized range.

Source: volatility-regime / Bollinger-squeeze family. Range-based (high-low)
amplitude state, distinct from the close-to-close volratio factor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_squeeze',
    'nickname': '5d/60d range amplitude squeeze',
    'theme': ['volatility'],
    'formula_latex': r'\frac{\mathrm{mean}_5((h-l)/c)}{\mathrm{mean}_{60}((h-l)/c)}',
    'columns_required': ['high', 'low', 'close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 60,
    'notes': 'Range-based amplitude regime state.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 5-day over 60-day mean normalized range."""
    rng = (panel['high'] - panel['low']) / panel['close']
    out = rng.rolling(5, min_periods=4).mean() / rng.rolling(60, min_periods=40).mean()
    return out.replace([np.inf, -np.inf], np.nan)
