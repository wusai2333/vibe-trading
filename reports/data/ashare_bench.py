"""A-share gtja191 bench via vibe-trading engine + akshare (sina) panel."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")

import pandas as pd
import numpy as np

CODES = [
    "600519", "601318", "600036", "000333", "000858",
    "601166", "600276", "601398", "601288", "600030",
    "600887", "601012", "601888", "000651", "600028",
    "601628", "600000", "601088", "601857", "600009",
    "601899", "002594", "600585", "300750", "601658",
    "600048", "601138", "601668", "000001", "000002",
]

def sina_sym(code: str) -> str:
    return ("sh" if code.startswith(("6", "9")) else "sz") + code

def build_panel() -> dict:
    import akshare as ak
    fetched = {}
    for i, code in enumerate(CODES, 1):
        df = None
        for attempt in range(5):
            try:
                df = ak.stock_zh_a_daily(symbol=sina_sym(code), adjust="qfq",
                                         start_date="20180101", end_date="20251231")
                break
            except Exception as e:
                print(f"  retry {attempt+1} {code}: {e}", file=sys.stderr)
                time.sleep(3 + attempt * 2)
        if df is None or df.empty:
            print(f"  SKIP {code}", file=sys.stderr); continue
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        for c in ("open", "high", "low", "close", "volume", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[["open", "high", "low", "close", "volume", "amount"]].dropna(
            subset=["open", "high", "low", "close"])
        if len(df) > 200:
            fetched[code] = df
        print(f"  [{i}/{len(CODES)}] {code}: {len(df)} rows", file=sys.stderr)
        time.sleep(0.8)  # stay polite to sina

    fields = ["open", "high", "low", "close", "volume", "amount"]
    panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in fields}
    # sina: amount in CNY, volume in shares -> vwap CNY/share
    panel["vwap"] = panel["amount"] / panel["volume"].replace(0, np.nan)
    panel["_meta"] = {
        "universe": "csi300-akshare30",
        "survivorship_bias": True,
        "pit_membership": False,
        "degraded": True,
        "constituent_source": "vibe-trading csi300 fallback roster, akshare/sina qfq data",
        "constituent_count": len(fetched),
        "price_adjustment": "qfq",
    }
    return panel

import src.tools.alpha_bench_tool as abt
_PANEL = build_panel()
print(f"panel built: {_PANEL['close'].shape[1]} names x {len(_PANEL['close'])} days", file=sys.stderr)

abt._load_universe_panel = lambda universe, period: _PANEL

from src.factors.bench_runner import run_bench
result = run_bench(zoo="gtja191", universe="csi300-akshare30",
                   period="2018-01-01/2025-12-31", top=20)

json.dump(result, open("/tmp/ashare_bench_result.json", "w"),
          ensure_ascii=False, default=str, indent=1)
print("RESULT_SAVED", file=sys.stderr)
print(json.dumps({k: result[k] for k in ("status", "n_alphas_tested", "n_skipped",
                                          "wall_seconds") if k in result}, ensure_ascii=False))
