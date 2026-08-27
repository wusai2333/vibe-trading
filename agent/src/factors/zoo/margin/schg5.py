# ============================================================
# 中文名称: 融券余量5日增速 (SCHG5)
# 简要说明: 融券余量的 5 日增长率：做空/对冲资金动向。A 股融券池小、噪声大，作为情绪对照腿。
# 典型用途: 空头情绪。PIT 右移一日。预期负向（做空增多=看空），检验定。
# ============================================================
"""margin schg5: 5d securities-lending quantity growth.

PIT: margin data for day d is published after the d close, so every
    series is shifted by one trading day — the value used at time t is the
    margin state of t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'margin_schg5',
    'nickname': '5d securities-lending quantity growth',
    'theme': ['sentiment'],
    'formula_latex': r'\frac{Q^{short}_{t-1}}{Q^{short}_{t-6}}-1',
    'columns_required': ['margin:short_qty'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 8,
    'notes': 'PIT shift(1) inside; short side, noisy on A-shares.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return PIT-safe 5-day growth of securities lending quantity."""
    qty = panel['margin:short_qty'].shift(1)
    out = qty.pct_change(5)
    return out.replace([np.inf, -np.inf], np.nan)
