# ============================================================
# 中文名称: 业绩预告超预期幅度 (SURP)
# 简要说明: 最新业绩预告的业绩变动幅度（%），公告日起生效（PIT）。
# 典型用途: 盈余惊喜事件因子。
# ============================================================
"""event surp: latest earnings-preview surprise percentage (PIT).

Source: exchange earnings previews via eastmoney (fetched 2026-08-20),
effective from the announcement date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'event_surp',
    'nickname': 'earnings preview surprise %',
    'theme': ['growth'],
    'formula_latex': r'\mathrm{surprise}_{latest}',
    'columns_required': ['event:surprise'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': 'PIT event panel; ffill from announcement date.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest known earnings-preview surprise."""
    return panel['event:surprise']
