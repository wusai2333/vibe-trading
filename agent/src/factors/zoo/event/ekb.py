# ============================================================
# 中文名称: 快报确认 (EKB)
# 简要说明: 业绩快报净利润同比 - 同期预告变动幅度：快报对预告的超越/确认。
# 典型用途: 盈余兑现质量。
# ============================================================
"""event ekb: express-report growth minus same-period preview surprise (PIT).

Source: earnings express vs preview comparison; a positive value means the
express beat the earlier preview.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'event_kb',
    'nickname': 'express-beats-preview confirmation',
    'theme': ['growth'],
    'formula_latex': r'g^{kb} - \mathrm{surprise}^{yg}',
    'columns_required': ['event:kb'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 20,
    'min_warmup_bars': 1,
    'notes': 'PIT ffill from express announcement date.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return the latest known express confirmation value."""
    return panel['event:kb']
