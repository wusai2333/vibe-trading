"""Xingye Yinxi (000426.SZ) data collection + indicators + valuation."""
import json, sys, warnings, datetime as dt
warnings.filterwarnings("ignore")
import akshare as ak
import pandas as pd
import numpy as np

out = {"fetched_at": dt.datetime.now().isoformat(timespec="seconds"), "symbol": "000426"}

def safe(name, fn):
    try:
        out[name] = fn()
        print(f"OK   {name}", file=sys.stderr)
    except Exception as e:
        out[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"FAIL {name}: {type(e).__name__}: {e}", file=sys.stderr)

# 1) Daily bars ~1.5y, qfq
def bars():
    df = ak.stock_zh_a_hist(symbol="000426", period="daily", adjust="qfq",
                            start_date=(dt.date.today() - dt.timedelta(days=550)).strftime("%Y%m%d"),
                            end_date=dt.date.today().strftime("%Y%m%d"))
    df.columns = [c.strip() for c in df.columns]
    return df.to_dict(orient="list")
safe("bars", bars)

# 2) Valuation tail (PE/PB/mcap)
def valuation():
    df = ak.stock_value_em(symbol="000426")
    return df.tail(3).to_dict(orient="list")
safe("valuation_tail", valuation)

# 3) PE/PB 3-year history for percentiles
def val_hist():
    res = {}
    for label, ind in (("pe3", "市盈率(TTM)"), ("pb3", "市净率")):
        try:
            df = ak.stock_zh_valuation_baidu(symbol="000426", indicator=ind, period="近三年")
            v = df["value"].astype(float).dropna()
            cur = float(v.iloc[-1])
            res[label] = {"now": cur, "pct": round(float((v < cur).mean()) * 100, 1),
                          "min": float(v.min()), "max": float(v.max()),
                          "median": round(float(v.median()), 2),
                          "p25": round(float(v.quantile(0.25)), 2),
                          "p75": round(float(v.quantile(0.75)), 2)}
        except Exception as e:
            res[label] = {"error": str(e)[:120]}
    return res
safe("val_hist", val_hist)

# 4) Financials by report period
def financials():
    df = ak.stock_financial_abstract_ths(symbol="000426", indicator="按报告期")
    return df.tail(6).to_dict(orient="list")
safe("financials", financials)

# 5) Tin & silver futures (main contracts) — 60d window for trend
def futures():
    res = {}
    for key, sym in (("tin", "sn0"), ("silver", "ag0")):
        try:
            df = ak.futures_zh_daily_sina(symbol=sym)
            df = df.tail(60)
            res[key] = {"tail5": df.tail(5).to_dict(orient="list"),
                        "close_60d_ago": float(df["close"].iloc[0]),
                        "close_now": float(df["close"].iloc[-1]),
                        "chg_60d_pct": round(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100, 1)}
        except Exception as e:
            res[key] = {"error": str(e)[:120]}
    return res
safe("futures", futures)

# 6) Analyst profit forecasts (eastmoney)
def forecasts():
    res = {}
    for fn, kw in (("stock_profit_forecast_em", {}),):
        try:
            f = getattr(ak, fn)
            df = f(symbol="000426", **kw) if kw else f(symbol="000426")
            res[fn] = df.tail(12).to_dict(orient="list")
        except Exception as e:
            res[fn] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
    return res
safe("forecasts", forecasts)

# ---- indicators ----
if isinstance(out.get("bars"), dict) and out["bars"].get("收盘"):
    b = pd.DataFrame(out["bars"])
    close = b["收盘"].astype(float); high = b["最高"].astype(float)
    low = b["最低"].astype(float); vol = b["成交量"].astype(float)

    def rsi(s, n=14):
        d = s.diff()
        up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
        dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
        return 100 - 100 / (1 + up / dn)

    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()

    out["indicators"] = {
        "last_date": str(b["日期"].iloc[-1]),
        "last_close": float(close.iloc[-1]),
        "chg_1d_pct": round(float(close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
        "chg_5d_pct": round(float(close.iloc[-1] / close.iloc[-6] - 1) * 100, 2),
        "chg_20d_pct": round(float(close.iloc[-1] / close.iloc[-21] - 1) * 100, 2),
        "chg_60d_pct": round(float(close.iloc[-1] / close.iloc[-61] - 1) * 100, 2),
        "chg_250d_pct": round(float(close.iloc[-1] / close.iloc[-251] - 1) * 100, 2) if len(close) > 251 else None,
        "ma5": round(float(close.rolling(5).mean().iloc[-1]), 2),
        "ma10": round(float(close.rolling(10).mean().iloc[-1]), 2),
        "ma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
        "ma60": round(float(close.rolling(60).mean().iloc[-1]), 2),
        "ma120": round(float(close.rolling(120).mean().iloc[-1]), 2),
        "ma250": round(float(close.rolling(250).mean().iloc[-1]), 2) if len(close) >= 250 else None,
        "rsi14": round(float(rsi(close).iloc[-1]), 1),
        "macd": round(float(macd.iloc[-1]), 3), "macd_signal": round(float(sig.iloc[-1]), 3),
        "macd_hist": round(float((macd - sig).iloc[-1]), 3),
        "macd_hist_prev5": [round(float(x), 3) for x in (macd - sig).iloc[-6:-1]],
        "high_250d": float(high.iloc[-250:].max()), "low_250d": float(low.iloc[-250:].min()),
        "high_60d": float(high.iloc[-60:].max()), "low_60d": float(low.iloc[-60:].min()),
        "off_250d_high_pct": round(float(close.iloc[-1] / high.iloc[-250:].max() - 1) * 100, 1),
        "vol_ratio_5v20": round(float(vol.iloc[-5:].mean() / vol.iloc[-20:].mean()), 2),
        "ann_vol_pct": round(float(close.pct_change().iloc[-250:].std() * np.sqrt(242) * 100), 1),
        "max_drawdown_250d_pct": round(float(((close.iloc[-250:] / close.iloc[-250:].cummax()) - 1).min()) * 100, 1),
        "recent_20d": [[str(d), float(c)] for d, c in zip(b["日期"].iloc[-20:], close.iloc[-20:])],
    }

json.dump(out, open("/tmp/xingye_data.json", "w"), ensure_ascii=False, default=str)
print("DONE", file=sys.stderr)
