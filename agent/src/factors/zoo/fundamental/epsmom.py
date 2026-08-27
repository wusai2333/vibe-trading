"""Fundamental earnings-momentum factor (PIT-safe, from TTM EPS drift).

Hypothesis: price momentum is weak/reversed in A-shares, but EARNINGS
momentum (slow drift in fundamentals) may survive because it is anchored to
announcements rather than sentiment. Measures the semi-annual change in
TTM EPS as known at each date (pubDate-aligned upstream in the fund cache).
"""

from __future__ import annotations

import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    "id": "fund_epsmom",
    "nickname": "Earnings momentum - semi-annual TTM EPS change",
    "theme": ["growth", "momentum"],
    "formula_latex": r"\mathrm{zscore}_{x}\bigl(\mathrm{EPS}_{TTM,t} / \mathrm{EPS}_{TTM,t-126} - 1\bigr)",
    "columns_required": ["fund:net_income"],
    "universe": ["equity_us", "equity_cn", "equity_hk"],
    "frequency": ["1d"],
    "decay_horizon": 21,
    "min_warmup_bars": 127,
    "notes": (
        "TTM EPS ratio vs ~6 months earlier (both PIT-visible values from the "
        "fund cache). Tests whether earnings momentum survives in CSI300 where "
        "price momentum is reversed. Sign flips on negative EPS bases are "
        "left to the cross-sectional z-score; NaN when the base is ~0."
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return cross-sectional z-scored semi-annual TTM EPS change."""
    eps = panel["fund:net_income"]
    base = eps.shift(126)
    ratio = eps / base.where(base.abs() > 1e-6) - 1
    return zscore(ratio)
