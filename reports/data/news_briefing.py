"""每日新闻面简报：因子推荐 Top10 + 个人持仓。

流程：东财个股新闻(近3天) -> qwen 逐股评估(利好/利空/中性) -> Markdown 简报。
输出 reports/data/news_briefing_latest.md（另存日期副本）。

铁律：新闻评估仅供人读参考，**绝不回流进因子信号/模型**。
LLM 配置读 agent/.env（DASHSCOPE_API_KEY/BASE_URL、LANGCHAIN_MODEL_NAME）。

Usage:
    python news_briefing.py             # 拉新闻+评估+渲染
    python news_briefing.py --selftest  # 离线自检（合成数据走渲染链路）
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

DATA_DIR = Path(__file__).resolve().parent
REPO = DATA_DIR.parents[1]
SCREENER_JSON = DATA_DIR / "stable5_screener_latest.json"
NEWS_DAYS = 3
MAX_NEWS_PER_STOCK = 8
TOP_N_PICKS = 10

EMOJI = {-2: "🔴🔴", -1: "🔴", 0: "⚪", 1: "🟢", 2: "🟢🟢"}


def load_env() -> dict:
    env = {}
    for line in (REPO / "agent" / ".env").read_text().splitlines():
        m = re.match(r"^([A-Z_]+)=(.*)$", line.strip())
        if m:
            env[m.group(1)] = m.group(2).strip().strip("'\"")
    return env


def llm_config() -> tuple:
    """(base_url, key, model)：优先 dsh 的 IdeaLab（内部免费），回退 agent/.env token plan。"""
    key = os.environ.get("IDEALAB_API_KEY", "")
    if not key:
        zshenv = Path.home() / ".zshenv"
        if zshenv.exists():
            m = re.search(r"^export IDEALAB_API_KEY=(\S+)", zshenv.read_text(), re.M)
            if m:
                key = m.group(1)
    if key:
        return ("https://idealab.alibaba-inc.com/api/openai/v1", key,
                "Peach-07-17-DogFooding")
    env = load_env()
    return (env["DASHSCOPE_BASE_URL"].rstrip("/"), env["DASHSCOPE_API_KEY"],
            env.get("LANGCHAIN_MODEL_NAME", "qwen-max"))


def universe() -> list[dict]:
    """Top10 推荐 + 持仓，去重合并，ETF 跳过个股新闻。"""
    picks = json.loads(SCREENER_JSON.read_text())["top_picks"][:TOP_N_PICKS]
    sys.path.insert(0, str(DATA_DIR))
    from holdings_tracker import HOLDINGS

    items: dict[str, dict] = {}
    for p in picks:
        items[p["symbol"]] = {"symbol": p["symbol"], "name": p["name"],
                              "tag": f"推荐#{p['rank']}"}
    for sym, (name, wt) in HOLDINGS.items():
        if sym.startswith(("51", "56", "15")):  # ETF：无个股新闻源
            continue
        if sym in items:
            items[sym]["tag"] += f" + 持仓{wt}%"
        else:
            items[sym] = {"symbol": sym, "name": name, "tag": f"持仓{wt}%"}
    return list(items.values())


def fetch_news(code6: str, cache: dict) -> list[dict]:
    """近 NEWS_DAYS 天新闻，按标题去重，限 MAX_NEWS_PER_STOCK 条。"""
    if code6 in cache:
        rows = cache[code6]
    else:
        import akshare as ak
        try:
            df = ak.stock_news_em(symbol=code6)
        except Exception as e:  # 单股失败不阻断全场
            print(f"  ! {code6} 新闻抓取失败: {e}")
            return []
        rows = [{"t": str(r["发布时间"])[:16], "title": r["新闻标题"],
                 "src": r["文章来源"]}
                for _, r in df.iterrows()]
        cache[code6] = rows
        time.sleep(0.5)  # 温和限速
    cutoff = datetime.now() - timedelta(days=NEWS_DAYS)
    seen, out = set(), []
    for r in rows:
        try:
            ts = datetime.strptime(r["t"], "%Y-%m-%d %H:%M")
        except ValueError:
            continue
        if ts < cutoff or r["title"] in seen:
            continue
        seen.add(r["title"])
        out.append(r)
        if len(out) >= MAX_NEWS_PER_STOCK:
            break
    return out


def llm_eval(cfg: tuple, name: str, code: str, news: list[dict]) -> dict:
    base, key, model = cfg
    prompt = (
        f"你是A股新闻面分析助手。以下是「{name}({code})」最近{NEWS_DAYS}天的新闻。"
        "评估对该股短期(1-5个交易日)的影响，只输出一个JSON对象："
        '{"score": -2到2整数(-2重大利空,2重大利好), "verdict": "一句话结论不超过30字", '
        '"risk": "一句话风险提示，没有则null"}\n新闻：\n'
        + "\n".join(f"{i+1}. [{r['t']}] {r['title']}（{r['src']}）"
                     for i, r in enumerate(news))
    )
    r = requests.post(
        base + "/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model,
              "messages": [{"role": "user", "content": prompt}],
              "temperature": 0.1},
        timeout=120)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", text, re.S)
    d = json.loads(m.group(0) if m else text)
    return {"score": max(-2, min(2, int(d["score"]))),
            "verdict": str(d.get("verdict", ""))[:60],
            "risk": d.get("risk")}


def render(as_of: str, results: list[dict]) -> str:
    L = [f"# 新闻面简报 {as_of}", "",
         "> 数据：东财个股新闻(近3天) + LLM 评估；仅供人读参考，不进入因子信号。", ""]
    for section, tag_pat in (("持仓", "持仓"), ("因子推荐 Top10", "推荐")):
        grp = sorted((x for x in results if tag_pat in x["tag"]),
                     key=lambda x: -x["score"])
        if not grp:
            continue
        L.append(f"## {section}")
        for x in grp:
            e = EMOJI.get(x["score"], "⚪")
            L.append(f"### {e} {x['score']:+d} {x['name']} {x['symbol']}（{x['tag']}）")
            if x["score"] == 99:  # 无新闻占位
                L.append("- 近3天无个股新闻")
            else:
                L.append(f"- 结论：{x['verdict']}")
                if x.get("risk"):
                    L.append(f"- 风险：{x['risk']}")
                for n in x["news"]:
                    L.append(f"  - [{n['t'][5:16]}] {n['title']}")
            L.append("")
    return "\n".join(L)


def run():
    cfg = llm_config()
    print(f"LLM: {cfg[2]} @ {cfg[0].split('//')[1].split('/')[0]}")
    cache_file = DATA_DIR / f"news_cache_{datetime.now():%Y-%m-%d}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    results = []
    for it in universe():
        code6 = it["symbol"].split(".")[0]
        print(f"* {it['name']} {it['symbol']}（{it['tag']}）")
        news = fetch_news(code6, cache)
        if not news:
            results.append({**it, "score": 99, "news": []})
            continue
        try:
            ev = llm_eval(cfg, it["name"], code6, news)
        except Exception as e:
            print(f"  ! LLM 评估失败: {e}")
            ev = {"score": 0, "verdict": f"评估失败（{type(e).__name__}）", "risk": None}
        results.append({**it, **ev, "news": news})
    cache_file.write_text(json.dumps(cache, ensure_ascii=False))
    as_of = datetime.now().strftime("%Y-%m-%d")
    md = render(as_of, results)
    (DATA_DIR / "news_briefing_latest.md").write_text(md)
    (DATA_DIR / f"news_briefing_{as_of}.md").write_text(md)
    print(f"\n已写出 news_briefing_latest.md（{len(results)} 只）")


def selftest():
    fake = [{"symbol": "600011.SH", "name": "华能国际", "tag": "持仓25%",
             "score": 1, "verdict": "股东增持，偏多", "risk": None,
             "news": [{"t": "2026-08-24 10:33", "title": "一致行动人增持1.47亿元",
                       "src": "证券时报"}]},
            {"symbol": "300394.SZ", "name": "天孚通信", "tag": "推荐#1",
             "score": 99, "news": []}]
    md = render("2026-08-24", fake)
    assert "🟢 +1 华能国际" in md and "一致行动人增持" in md
    assert "近3天无个股新闻" in md and "## 持仓" in md
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run()
