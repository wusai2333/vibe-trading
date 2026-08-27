"""Build the PIT-safe fundamental panel from Baostock (free, no token).

Unblocks the fundamental zoo (4 factors previously skipped for lack of data)
and adds a book-to-market factor. Fetches for all CSI300 names:

  * daily valuation : peTTM, pbMRQ            (query_history_k_data_plus)
  * quarterly profit: roeAvg, gpMargin, epsTTM, netProfit   (query_profit_data)
  * quarterly growth: YOYAsset                (query_growth_data)

Point-in-time rule: a quarter's values become visible on its ``pubDate``
(announcement date), never ``statDate`` (period end) — values are forward
filled day by day from pubDate, so no look-ahead leaks into factors.

Output columns (daily, aligned to csi500_panel.pkl calendar & columns):
  fund:roe                  <- roeAvg (quarterly ROE, PIT)
  fund:gross_profitability  <- gpMargin (gross margin, PIT)
  fund:asset_growth         <- YOYAsset (YoY asset growth, PIT)
  fund:net_income           <- epsTTM  (see note)
  fund:shares_diluted       <- 1.0     (see note)
  fund:bp                   <- 1/pbMRQ (daily book-to-market)

Note on earnings_yield: the zoo factor computes net_income/(close*shares).
Setting net_income=epsTTM and shares=1 makes it exactly EP(TTM)=epsTTM/close,
algebraically identical to earnings/market-cap — documented shortcut because
Baostock's TTM earnings arrive per-share.

Run time ~30-60 min (288 names x 35 quarters x 2 quarterly tables, sequential
single TCP connection). Resumable: progress checkpoints to a temp file.
"""
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent
PANEL = DATA / "csi500_panel.pkl"
CACHE = DATA / "fund_cache_csi500.pkl"
TMP = DATA / "fund_cache_csi500_partial.pkl"

START, END = "2018-01-01", pd.Timestamp.today().strftime("%Y-%m-%d")
Q_START = [(2017, 4)]  # one quarter before panel start for warmup
Q_END = (2026, 2)


def quarters():
    y, q = Q_START[0]
    while (y, q) <= Q_END:
        yield y, q
        q += 1
        if q > 4:
            y, q = y + 1, 1


def bs_code(code: str) -> str:
    sym, suf = code.split(".")
    return ("sh." if suf == "SH" else "sz.") + sym


def collect_rows(rs) -> list:
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(dict(zip(rs.fields, rs.get_row_data())))
    return rows


def main() -> None:
    panel = pickle.load(open(PANEL, "rb"))
    close = panel["close"]
    codes = sorted(close.columns)
    print(f"universe: {len(codes)} names, {close.index[0].date()}..{close.index[-1].date()}",
          file=sys.stderr)

    import baostock as bs
    lg = bs.login()
    if lg.error_code != "0":
        sys.exit(f"baostock login failed: {lg.error_msg}")

    # resume support
    done: dict = {}
    if TMP.exists():
        done = pickle.load(open(TMP, "rb"))
        print(f"resuming: {len(done)} names already fetched", file=sys.stderr)

    qlist = list(quarters())
    t0 = time.time()
    for i, code in enumerate(codes):
        if code in done:
            continue
        bsc = bs_code(code)
        rec = {"val": None, "profit": [], "growth": []}
        try:
            rs = bs.query_history_k_data_plus(
                bsc, "date,peTTM,pbMRQ",
                start_date=START, end_date=END, frequency="d")
            rec["val"] = collect_rows(rs)
            for y, q in qlist:
                rs = bs.query_profit_data(code=bsc, year=y, quarter=q)
                rec["profit"].extend(collect_rows(rs))
                rs = bs.query_growth_data(code=bsc, year=y, quarter=q)
                rec["growth"].extend(collect_rows(rs))
        except Exception as exc:  # noqa: BLE001 - keep going, record failure
            print(f"  {code}: fetch error {exc}", file=sys.stderr)
            rec["error"] = str(exc)
        done[code] = rec
        if (i + 1) % 20 == 0:
            pickle.dump(done, open(TMP, "wb"))
            el = time.time() - t0
            eta = el / (i + 1 - sum(1 for c in codes[:i + 1] if c in done and c != code)) \
                * (len(codes) - i - 1) if i else 0
            print(f"  {i + 1}/{len(codes)} done, {el / 60:.1f} min elapsed",
                  file=sys.stderr, flush=True)

    bs.logout()
    pickle.dump(done, open(TMP, "wb"))
    print(f"fetch done in {(time.time() - t0) / 60:.1f} min; assembling...", file=sys.stderr)

    # ---- assemble daily PIT frames on the panel calendar ----
    days = close.index
    frames = {k: pd.DataFrame(index=days, columns=codes, dtype=float)
              for k in ("roe", "gp", "asset_growth", "eps_ttm", "bp")}

    def pit_series(events: list, field: str) -> pd.Series:
        """Latest quarter value visible at each date (visible from pubDate)."""
        pts = {}
        for row in events:
            try:
                pub = pd.Timestamp(row.get("pubDate"))
                val = float(row.get(field))
            except (TypeError, ValueError):
                continue
            if pd.isna(val) or pub.year < 1990:
                continue
            pts[pub] = val  # later pubDate for same announcement wins
        if not pts:
            return pd.Series(dtype=float)
        s = pd.Series(pts).sort_index()
        return s.reindex(days, method="ffill")

    n_err = 0
    for code in codes:
        rec = done.get(code) or {}
        if rec.get("error"):
            n_err += 1
        # daily valuation -> bp
        val = rec.get("val") or []
        if val:
            v = pd.DataFrame(val)
            v["date"] = pd.to_datetime(v["date"])
            v = v.set_index("date").reindex(days)
            pb = pd.to_numeric(v["pbMRQ"], errors="coerce")
            frames["bp"][code] = (1.0 / pb.replace(0, np.nan)).astype(float)
        # quarterly PIT series
        prof = rec.get("profit") or []
        frames["roe"][code] = pit_series(prof, "roeAvg")
        frames["gp"][code] = pit_series(prof, "gpMargin")
        frames["eps_ttm"][code] = pit_series(prof, "epsTTM")
        frames["asset_growth"][code] = pit_series(rec.get("growth") or [], "YOYAsset")

    out = {
        "fund:roe": frames["roe"],
        "fund:gross_profitability": frames["gp"],
        "fund:asset_growth": frames["asset_growth"],
        "fund:net_income": frames["eps_ttm"],   # see module docstring
        "fund:shares_diluted": pd.DataFrame(1.0, index=days, columns=codes),
        "fund:bp": frames["bp"],
        "_meta": {"source": "baostock", "built": pd.Timestamp.today().strftime("%Y-%m-%d"),
                  "pit_rule": "quarterly values visible from pubDate, ffill daily",
                  "fetch_errors": n_err,
                  "coverage_last_day": {k: int(f.iloc[-1].notna().sum())
                                        for k, f in frames.items()}},
    }
    pickle.dump(out, open(CACHE, "wb"))
    TMP.unlink(missing_ok=True)
    print(json.dumps(out["_meta"], ensure_ascii=False, indent=1, default=str))
    print(f"SAVED {CACHE}", file=sys.stderr)


if __name__ == "__main__":
    main()
