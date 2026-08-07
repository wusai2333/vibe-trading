"""gtja191 re-bench with the new ashare loader (registry path)."""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")

import numpy as np
import pandas as pd

CODES = [
    "600519.SH", "601318.SH", "600036.SH", "000333.SZ", "000858.SZ",
    "601166.SH", "600276.SH", "601398.SH", "601288.SH", "600030.SH",
    "600887.SH", "601012.SH", "601888.SH", "000651.SZ", "600028.SH",
    "601628.SH", "600000.SH", "601088.SH", "601857.SH", "600009.SH",
    "601899.SH", "002594.SZ", "600585.SH", "300750.SZ", "601658.SH",
    "600048.SH", "601138.SH", "601668.SH", "000001.SZ", "000002.SZ",
]

from backtest.loaders.registry import resolve_loader
loader = resolve_loader("a_share")
print("loader:", loader.name, file=sys.stderr)

fetched = loader.fetch(CODES, "2018-01-01", "2025-12-31")
print(f"fetched {len(fetched)}/{len(CODES)} symbols", file=sys.stderr)

fields = ["open", "high", "low", "close", "volume"]
panel = {f: pd.DataFrame({c: df[f] for c, df in fetched.items()}) for f in fields}
# ashare has no amount; vwap approximated as OHLC mean (only affects amount-using alphas)
panel["vwap"] = sum(panel[f] for f in ("open", "high", "low", "close")) / 4.0
panel["_meta"] = {
    "universe": "csi300-ashare30",
    "survivorship_bias": True,
    "pit_membership": False,
    "degraded": True,
    "constituent_source": "vibe-trading csi300 fallback roster, ashare dual-source qfq data",
    "constituent_count": len(fetched),
    "price_adjustment": "qfq",
}

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.bench_runner import run_bench
result = run_bench(zoo="gtja191", universe="csi300-ashare30",
                   period="2018-01-01/2025-12-31", top=20)
json.dump(result, open("/tmp/ashare_bench_via_loader.json", "w"),
          ensure_ascii=False, default=str, indent=1)
print(json.dumps({k: result[k] for k in ("status", "n_alphas_tested", "n_skipped",
                                          "alive", "reversed", "dead")}, ensure_ascii=False))
