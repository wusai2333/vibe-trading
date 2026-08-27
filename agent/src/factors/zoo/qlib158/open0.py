# Adapted from microsoft/qlib@d5379c520f66a39953bad76234a7019a72796fd0:qlib/contrib/data/handler.py
# (Apache-2.0). Copyright (c) Microsoft Corporation.
# ============================================================
# 中文名称: 开盘收盘比
# 简要说明: 开盘价相对收盘价的比值，qlib Alpha158 价格归一化族。
# 典型用途: 日内价格结构特征，与其他量价因子组合使用。
# ============================================================
"""qlib158 OPEN0: formula = \\mathrm{open} / \\mathrm{close}."""
from __future__ import annotations

import pandas as pd
from src.factors.base import safe_div

__alpha_meta__ = {
    'id': 'qlib158_open0',
    'theme': ['microstructure'],
    'formula_latex': '\\mathrm{open} / \\mathrm{close}',
    'columns_required': ['open','close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk', 'equity_in', 'equity_kr'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 1,
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return qlib158 OPEN0 on the supplied OHLCV panel."""
    return safe_div(panel['open'], panel['close'])
