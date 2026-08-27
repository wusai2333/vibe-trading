# ============================================================
# 中文名称: 预告类型排序 (ETYPE)
# 简要说明: 业绩预告类型有序打分：预增4/扭亏3/略增2/续盈1/不确定0/续亏-1/略减-2/预减-3/首亏-4。
# 典型用途: 盈余事件质量分层。
# ============================================================
"""event etype: ordinal score of the earnings-preview category (PIT).

Source: exchange earnings-preview categories, ordered by profitability
implication (pre-increase best, first-loss worst).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'event_type',
    'nickname': 'earnings preview category score',
    'theme': ['growth'],
    'formula_latex': r'\mathrm{score}(\mathrm{preview\_type})',
    'columns_required': ['event:type'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': 'Ordinal category score, PIT ffill.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest known preview category score."""
    return panel['event:type']
