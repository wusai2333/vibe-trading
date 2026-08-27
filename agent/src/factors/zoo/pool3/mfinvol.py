# ============================================================
# 中文名称: 杠杆波动率 (MFINVOL)
# 简要说明: 融资余额 5 日增速的 60 日标准差：杠杆资金进出的稳定性。
# 典型用途: 两融数据扩展腿。PIT 右移一日。
# ============================================================
"""pool3 mfinvol: 60-day std of 5-day margin-financing balance growth.

Source: margin-data extension (this project, 2026-08). Stability of leveraged
money flow: erratic margin churn vs smooth accumulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'pool3_mfinvol',
    'nickname': 'margin financing growth volatility',
    'theme': ['sentiment'],
    'formula_latex': r'\sigma_{60}\bigl(\Delta_5 B^{fin}\bigr)',
    'columns_required': ['margin:fin_balance'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 66,
    'notes': 'PIT shift(1) inside; margin data published after close.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return 60-day std of the 5-day margin-financing balance growth."""
    bal = panel['margin:fin_balance'].shift(1)
    return bal.pct_change(5).rolling(60, min_periods=40).std()
