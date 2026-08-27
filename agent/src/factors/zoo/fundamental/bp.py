"""Fundamental book-to-market value factor (daily, PIT by construction)."""

from __future__ import annotations

import pandas as pd

from src.factors.base import zscore

__alpha_meta__ = {
    "id": "fund_bp",
    "nickname": "Book-to-market (1/PB)",
    "theme": ["value"],
    "formula_latex": r"\mathrm{zscore}_{x}(1/\mathrm{PB}_{MRQ})",
    "columns_required": ["fund:bp"],
    "universe": ["equity_us", "equity_cn", "equity_hk"],
    "frequency": ["1d"],
    "decay_horizon": 252,
    "min_warmup_bars": 1,
    "notes": (
        "Daily book-to-market from the fundamental panel (baostock pbMRQ). "
        "PIT by construction: the PB ratio uses the latest reported book "
        "value at each date. Classic value factor, complementing the "
        "earnings-yield factor with the balance-sheet side."
    ),
}


def compute(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return cross-sectional z-scored book-to-market."""
    return zscore(panel["fund:bp"])
