"""Fill the two engine gaps: strict bench + head-to-head compare on the stable-5.

1. bench_runner_strict: IC + same-universe random control + train/test OOS
   split, run over the 3 zoos that own the stable-5 factors. Extracts the
   stable-5 rows and reports whether they graduate to confirmed_alive.
2. compare_runner.compare_alphas: head-to-head IR ranking of the stable-5.

Both run on the cached CSI300 panel (no re-fetch).
"""
import sys, json, pickle, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/wusai/Vibe-Trading/agent")

panel = pickle.load(open("reports/data/csi300_panel.pkl", "rb"))
print(f"panel: {panel['close'].shape[1]} names x {len(panel['close'])} days", file=sys.stderr)

import src.tools.alpha_bench_tool as abt
abt._load_universe_panel = lambda universe, period: panel

STABLE = ["gtja191_171", "alpha101_083", "alpha101_042", "qlib158_klow", "alpha101_060"]
ZOOS = ["gtja191", "alpha101", "qlib158"]
OOS_SPLIT = "2022-06-30"

# ---------- 1. strict bench (random control + OOS) ----------
from src.factors.bench_runner_strict import run_bench_strict
strict_rows = {}
for z in ZOOS:
    t0 = time.time()
    r = run_bench_strict(zoo=z, universe="csi300-ashare",
                         period="2018-01-01/2025-12-31",
                         random_control=True, n_random_seeds=5,
                         oos_split=OOS_SPLIT, top=20)
    for row in r.get("rows", []):
        if isinstance(row, dict) and row.get("id") in STABLE:
            strict_rows[row["id"]] = row
    print(f"strict {z}: tested={r.get('n_alphas_tested')} "
          f"confirmed_alive={r.get('confirmed_alive')} wall={time.time()-t0:.0f}s", flush=True)

# ---------- 2. head-to-head compare ----------
from src.factors.compare_runner import compare_alphas
cmp = compare_alphas(STABLE, universe="csi300-ashare",
                     period="2018-01-01/2025-12-31", sort="ir")

out = {
    "stable_factors": STABLE,
    "oos_split": OOS_SPLIT,
    "strict_bench": strict_rows,
    "compare": cmp,
}
json.dump(out, open("reports/data/csi300_strict_compare.json", "w"),
          ensure_ascii=False, indent=1, default=str)

print("\n=== STRICT (stable-5) ===")
for aid in STABLE:
    row = strict_rows.get(aid)
    if not row:
        print(f"{aid}: NOT TESTED"); continue
    print(f"{aid}: cat={row.get('_category')} ir={row.get('ir')} "
          f"ic={row.get('ic_mean')} rand_ic={row.get('random_ic_mean')} "
          f"oos_ir={row.get('oos_ir')}")
print("\n=== COMPARE ranking ===")
for r in cmp.get("ranking", []):
    print(f"#{r['rank']} {r['id']} ir={r['ir']} ic={r['ic_mean']}")
print("winner:", cmp.get("winner"))
