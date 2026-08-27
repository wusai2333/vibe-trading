# ============================================================
# 中文名称: 融资余额20日增速 (FCHG20)
# 简要说明: 融资余额的 20 日增长率：中期杠杆资金趋势。
# 典型用途: 与 fchg5 同族不同周期，检验决定谁留下。PIT 右移一日。
# ============================================================
"""margin fchg20: 20d margin-financing balance growth.

PIT: margin data for day d is published after the d close, so every
    series is shifted by one trading day — the value used at time t is the
    margin state of t-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__alpha_meta__ = {
    'id': 'margin_fchg20',
    'nickname': '20d margin-financing balance growth',
    'theme': ['sentiment'],
    'formula_latex': r'\frac{B^{fin}_{t-1}}{B^{fin}_{t-21}}-1',
    'columns_required': ['margin:fin_balance'],
    'universe': ['equity_cn'],
    'frequency': ['1d'],
    'decay_horizon': 5,
    'min_warmup_bars': 23,
    'notes': 'PIT shift(1) inside.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return PIT-safe 20-day growth of margin financing balance."""
    bal = panel['margin:fin_balance'].shift(1)
    out = bal.pct_change(20)
    return out.replace([np.inf, -np.inf], np.nan)
