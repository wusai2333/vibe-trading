# ============================================================
# 中文名称: 预告环比动量 (EMOM)
# 简要说明: 本期预告变动幅度 - 上期预告变动幅度：盈余惊喜的加速度。
# 典型用途: 业绩趋势拐点识别。
# ============================================================
"""event emom: preview surprise minus previous-period surprise (PIT).

Source: earnings momentum family (SUE-drift literature); acceleration of
the surprise across consecutive previews.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'event_mom',
    'nickname': 'preview surprise acceleration',
    'theme': ['growth'],
    'formula_latex': r'\mathrm{surprise}_t - \mathrm{surprise}_{t-1}',
    'columns_required': ['event:mom'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': 'PIT ffill from announcement date.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest known surprise acceleration."""
    return panel['event:mom']
