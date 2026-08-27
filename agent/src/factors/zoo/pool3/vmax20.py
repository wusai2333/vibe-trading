# ============================================================
# 中文名称: 量能尖峰集中度 (VMAX20)
# 简要说明: 20 日最大单日成交量 / 均量：放量事件集中度。
# 典型用途: 事件驱动/异动监测腿。
# ============================================================
"""pool3 vmax20: max daily volume over mean volume, trailing 20 days.

Source: volume-price factor family (volume spike concentration). One big
volume day relative to the norm flags an information event.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_vmax20',
    'nickname': '20d volume spike concentration',
    'theme': ['volume'],
    'formula_latex': r'\frac{\max_{20} V_t}{\mathrm{mean}_{20} V_t}',
    'columns_required': ['volume'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 20,
    'notes': 'Volume spike concentration.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return max daily volume over mean volume, trailing 20 days."""
    vol = panel['volume']
    out = vol.rolling(20, min_periods=15).max() / vol.rolling(20, min_periods=15).mean()
    return out.replace([np.inf, -np.inf], np.nan)
