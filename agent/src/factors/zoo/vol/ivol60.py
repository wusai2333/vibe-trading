# ============================================================
# 中文名称: 特质波动率（60日）
# 简要说明: 对市场（全池等权）收益做 60 日滚动回归，残差的标准差。
# 典型用途: IVOL 异象：剔除市场暴露后，特质波动高的股票未来收益更低
#          （A 股与美股均有文献支持）。与 RVOL 的区别：剥离了 beta，
#          度量的是"纯个股噪声/分歧"，和大盘涨跌无关。预期负 IC。
# ============================================================
"""vol IVOL60: 60-day idiosyncratic volatility (market-model residual std).

Rolling 60-day regression of each stock's daily return on the equal-weight
panel mean return; the factor is the standard deviation of the residuals.
Unlike raw realized volatility it strips out market exposure (beta), so it
measures stock-specific noise/disagreement rather than systematic risk. The
IVOL anomaly (high IVOL -> lower future returns) is documented in both the
US and China. Expected negative IC; use sign-aware.
"""
from __future__ import annotations

import pandas as pd

__alpha_meta__ = {
    'id': 'vol_ivol60',
    'nickname': '60-day idiosyncratic volatility',
    'theme': ['volatility'],
    'formula_latex': r'\sigma\bigl(r_i - \hat\beta_i r_m\bigr),\ 60d',
    'columns_required': ['close'],
    'universe': ['equity_us', 'equity_cn', 'equity_hk'],
    'frequency': ['1d'],
    'decay_horizon': 1,
    'min_warmup_bars': 61,
    'notes': 'Market = equal-weight panel mean return; concurrent beta.',
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return trailing-60d idiosyncratic volatility on the supplied panel."""
    ret = panel['close'].pct_change()
    r_m = ret.mean(axis=1)
    win = ret.rolling(60, min_periods=40)
    beta = win.cov(r_m).div(r_m.rolling(60, min_periods=40).var(), axis=0)
    resid = ret.sub(beta.mul(r_m, axis=0))
    return resid.rolling(60, min_periods=40).std()
