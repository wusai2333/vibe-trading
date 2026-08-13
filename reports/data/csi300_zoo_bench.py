"""Full factor zoo bench on the cached CSI300 panel.

Runs every registered zoo (gtja191, alpha101, qlib158, academic, ...) through
vibe-trading's own bench_runner against the cached ashare panel. Outputs per-zoo
alive/dead counts and a cross-zoo top-30 ranked by IR.
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")

panel = pickle.load(open("reports/data/csi300_panel.pkl", "rb"))
print(f"panel loaded: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

from src.factors.registry import get_default_registry
from src.factors.bench_runner import run_bench

reg = get_default_registry()
zoos = sorted({reg.get(aid).zoo for aid in reg.list()})
print("zoos:", zoos, file=sys.stderr)

all_rows = []
summary = {}
for z in zoos:
    t0 = time.time()
    r = run_bench(zoo=z, universe="csi300-ashare", period="2018-01-01/2025-12-31", top=20)
    summary[z] = {
        "tested": r.get("n_alphas_tested"), "skipped": r.get("n_skipped"),
        "alive": r.get("alive"), "reversed": r.get("reversed"), "dead": r.get("dead"),
        "wall_seconds": round(time.time() - t0, 1),
    }
    for row in r.get("rows", []):
        if isinstance(row, dict) and row.get("ir") is not None:
            row["zoo"] = z
            all_rows.append(row)
    print(f"{z}: {summary[z]}", flush=True)

all_rows.sort(key=lambda r: -abs(r["ir"]))
result = {
    "universe": {"names": int(panel["close"].shape[1]), "days": int(len(panel["close"]))},
    "zoo_summary": summary,
    "top30_by_abs_ir": [
        {k: r.get(k) for k in ("id", "zoo", "ir", "ic_mean", "ic_positive_ratio", "ic_count", "_category")}
        for r in all_rows[:30]
    ],
    "multiple_testing_per_zoo": {},
}
json.dump(result, open("reports/data/csi300_zoo_bench.json", "w"), ensure_ascii=False, indent=1, default=str)
print("\nSAVED reports/data/csi300_zoo_bench.json", file=sys.stderr)
print(json.dumps(summary, ensure_ascii=False))
