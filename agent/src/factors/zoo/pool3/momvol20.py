# ============================================================
# 中文名称: 波动率调整动量 (MOMVOL20)
# 简要说明: 20日动量 / 20日波动率（MAR 比率风格）：路径质量好的动量，高质量动量因子家族。
# 典型用途: 区分平滑趋势与高波动冲高。
# ============================================================
"""pool3 momvol20: risk-adjusted 20-day momentum (MAR style).

Source: high-quality momentum family (QuantsPlaybook momentum construction
research; Daniel-Moskowitz momentum quality literature). Momentum scaled by
its own path volatility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_momvol20',
    'nickname': '20d momentum over 20d volatility',
    'theme': ['momentum'],
    'formula_latex': r'\frac{c_t/c_{t-20}-1}{\sigma_{20}(r)}',
    'columns_required': ['close'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 21,
    'notes': 'MAR-style momentum quality.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 20-day momentum divided by 20-day volatility."""
    ret = panel['close'].pct_change()
    mom = panel['close'].pct_change(20)
    out = mom / ret.rolling(20, min_periods=15).std()
    return out.replace([np.inf, -np.inf], np.nan)
