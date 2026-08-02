#!/usr/bin/env python3
"""
Modern Edge Portfolio - daily backtest data generator
=====================================================

Pulls DAILY total-return (adjusted-close) prices for the current Modern Edge
lineup and a 60/40 benchmark, reconstructs daily net-of-fee equity curves with
quarterly rebalancing, and writes both a CSV and a JSON that the marketing
charts consume.

You do NOT run this yourself. GitHub Actions runs it (the runner can reach
Yahoo Finance); you download the files it produces in data/ and attach them
back in chat.

Current lineup (50 / 25 / 25):
  Equity 50%        VOO 26  QQQ 12  VEA 6  IEMG 6
  Fixed Income 25%  SGOV 10 IEF 10  SCHP 5
  Alternatives 25%  DBMF 10 SGOL 5  BITB 5  CAOS 5

Compliance notes are emitted into the JSON so nothing about proxy splicing is
hidden. Anything spliced makes the series HYPOTHETICAL/BACKTESTED under SEC
Marketing Rule 206(4)-1 and needs the standard hypothetical disclosures.
"""

import json
import sys
import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

MODERN_EDGE_WEIGHTS = {
    "VOO": 0.26, "QQQ": 0.12, "VEA": 0.06, "IEMG": 0.06,
    "SGOV": 0.10, "IEF": 0.10, "SCHP": 0.05,
    "DBMF": 0.10, "SGOL": 0.05, "BITB": 0.05, "CAOS": 0.05,
}

BENCHMARK_WEIGHTS = {"VOO": 0.40, "VXUS": 0.20, "BND": 0.40}

# Proxy tickers used to extend a holding's history BEFORE its own inception.
# Each proxy series is price-scaled to the real series at the splice date so the
# handoff is continuous. Every splice is recorded in the JSON's "proxies" block.
PROXIES = {
    "SGOV": "BIL",      # SGOV launched 2020-05; BIL (SPDR 1-3mo T-Bill) before
    "DBMF": "WTMF",     # DBMF launched 2019-05; WTMF (WisdomTree Mgd Futures) before
    "BITB": "BTC-USD",  # BITB launched 2024-01; BTC spot before
}

# CAOS pre-inception handling.
# CAOS is the SUCCESSOR to the Arin Large Cap Theta Fund (mutual fund ticker
# AVOLX), launched Aug 2013 under Arin Risk Advisors, converted to an ETF on
# 2023-03-06 with the SAME investment objective. Portfolio managers Lawrence
# Lempert and Joseph DeSipio ran the predecessor fund since Aug 2013 and CAOS
# since 2023 — continuous management, same strategy. Using AVOLX's real NAV as
# the pre-2023 CAOS history is therefore predecessor-fund data, not a generic
# proxy. Modes:
#   "AVOLX" -> splice the real AVOLX NAV before 2023-03-06 (default). Disclose:
#              (a) predecessor mutual fund, different fee/tax treatment; and
#              (b) the AVOLX-era payoff was somewhat MORE convex than today's CAOS.
#   "cash"  -> hold the 5% sleeve in the SGOV/BIL proxy before 2023-03 (most
#              conservative; understates the crash hedge pre-2023).
#   "start" -> begin the long window at CAOS's first real ETF date (cleanest).
CAOS_MODE = "AVOLX"
CAOS_INCEPTION = "2023-03-06"

QUARTERLY_FEE = 0.002        # 0.2% per quarter = 0.8% annualized, ME only
REBALANCE = "QS"             # quarter start
START_VALUE = 100.0

# Two windows. "long" is set early and AUTO-TRIMS to the first date where all
# eleven sleeves have data (post-splice). In practice the binding constraint is
# BTC-USD (Yahoo history begins ~Sept 2014), NOT CAOS/AVOLX (Aug 2013) — so the
# portfolio series realistically begins in late 2014 even with the AVOLX splice.
# "short" is the clean 4-year marketing window.
WINDOWS = {
    "long":  {"start": "2013-08-01"},
    "short": {"start": "2022-01-03"},
}

OUT_JSON = "data/modern_edge_backtest.json"
OUT_CSV_PREFIX = "data/modern_edge_daily"  # -> _long.csv, _short.csv


# ----------------------------------------------------------------------------
# DATA
# ----------------------------------------------------------------------------

def all_tickers():
    t = set(MODERN_EDGE_WEIGHTS) | set(BENCHMARK_WEIGHTS) | set(PROXIES.values())
    if CAOS_MODE == "AVOLX":
        t.add("AVOLX")
    return sorted(t)


def download(tickers, start):
    """Daily adjusted-close (total return). Returns a wide DataFrame."""
    raw = yf.download(
        tickers, start=start, auto_adjust=True,
        progress=False, group_by="column", threads=True,
    )
    px = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
    px = px.dropna(how="all").sort_index()
    return px


def splice(px, real, proxy):
    """Extend `real` backward with `proxy`, chain-scaled at the boundary.

    Does NOT require a shared date. Scales the proxy so its last value before the
    real series begins lines up with the first real value (continuous handoff).
    """
    if real not in px or proxy not in px:
        return px
    s_real, s_proxy = px[real].dropna(), px[proxy].dropna()
    if s_real.empty or s_proxy.empty:
        return px
    first = s_real.index[0]
    pre = s_proxy[s_proxy.index < first]
    if pre.empty:
        return px
    scale = s_real.loc[first] / pre.iloc[-1]   # boundary scale, no shared date needed
    merged = pd.concat([pre * scale, s_real])
    px[real] = merged.reindex(px.index)
    return px


def prepare(px, splice_log):
    for real, proxy in PROXIES.items():
        before = px[real].first_valid_index() if real in px else None
        px = splice(px, real, proxy)
        after = px[real].first_valid_index() if real in px else None
        if before != after:
            splice_log.append(
                {"holding": real, "proxy": proxy,
                 "proxy_covers_before": str(before)[:10] if before else None,
                 "note": f"{proxy} spliced (price-scaled) before {real} inception"})

    # CAOS handling
    if "CAOS" in px:
        real_start = px["CAOS"].first_valid_index()
        if CAOS_MODE == "cash":
            filler = "SGOV" if px.get("SGOV") is not None else "BIL"
            fill = px[filler]
            stub = fill[fill.index < real_start]
            if not stub.empty:
                anchor = px["CAOS"].loc[real_start]
                stub_scaled = stub / fill.loc[stub.index[-1]] * anchor
                px["CAOS"] = pd.concat([stub_scaled, px["CAOS"].dropna()]).reindex(px.index)
                splice_log.append(
                    {"holding": "CAOS", "proxy": filler, "mode": "cash-stub",
                     "note": ("CAOS 5% sleeve held in cash proxy before 2023-03 "
                              "conversion; understates crash hedge pre-2023")})
        elif CAOS_MODE == "AVOLX":
            px = splice(px, "CAOS", "AVOLX")
            splice_log.append(
                {"holding": "CAOS", "proxy": "AVOLX", "mode": "predecessor-fund-splice",
                 "note": ("Pre-2023-03-06 = real NAV of the Arin Large Cap Theta Fund "
                          "(AVOLX), the CAOS predecessor mutual fund. Same objective "
                          "and same portfolio managers (Lempert, DeSipio) since Aug "
                          "2013. DISCLOSE: (1) predecessor mutual fund with different "
                          "fee/tax treatment; (2) AVOLX-era payoff was somewhat more "
                          "convex than today's ETF-wrapper CAOS.")})
        # "start" mode: no fill; window trimmed to CAOS real data below
    return px


# ----------------------------------------------------------------------------
# PORTFOLIO MATH
# ----------------------------------------------------------------------------

def portfolio_curve(px, weights, start, fee_per_q):
    """Daily curve, quarterly rebalance to target weights, quarterly fee."""
    cols = list(weights)
    sub = px[cols].loc[start:].dropna(how="any")
    if sub.empty:
        return None
    rets = sub.pct_change().fillna(0.0)
    rb_dates = set(pd.date_range(sub.index[0], sub.index[-1], freq=REBALANCE)
                   .map(lambda d: sub.index[sub.index.get_indexer([d], method="bfill")[0]]))
    rb_dates.add(sub.index[0])

    w = pd.Series(weights, dtype=float)
    value = START_VALUE
    out = []
    for i, dt in enumerate(sub.index):
        if dt in rb_dates and i > 0:
            w = pd.Series(weights, dtype=float)  # snap back to targets
        day_ret = float((w * rets.loc[dt]).sum())
        value *= (1 + day_ret)
        # drift weights with the day's returns
        grown = w * (1 + rets.loc[dt])
        w = grown / grown.sum()
        if fee_per_q and dt in rb_dates and i > 0:
            value *= (1 - fee_per_q)  # quarterly advisory fee, ME only
        out.append(value)
    return pd.Series(out, index=sub.index)


def drawdown(curve):
    peak = curve.cummax()
    return (curve - peak) / peak


def stats(curve):
    daily = curve.pct_change().dropna()
    n = len(curve)
    years = n / 252.0
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    vol = daily.std() * np.sqrt(252)
    sharpe = (cagr - 0.045) / vol if vol else 0
    mdd = drawdown(curve).min()
    return {"cagr_pct": round(cagr * 100, 2), "vol_pct": round(vol * 100, 2),
            "sharpe": round(sharpe, 2), "max_drawdown_pct": round(mdd * 100, 2),
            "final_value_of_100": round(curve.iloc[-1], 2),
            "start": str(curve.index[0])[:10], "end": str(curve.index[-1])[:10],
            "trading_days": n}


# ----------------------------------------------------------------------------
# RUN
# ----------------------------------------------------------------------------

def run():
    earliest = min(w["start"] for w in WINDOWS.values())
    pull_start = "2015-01-01"  # pull extra history so proxies have room
    print(f"Downloading {len(all_tickers())} tickers from {pull_start} ...")
    px = download(all_tickers(), pull_start)

    splice_log = []
    px = prepare(px, splice_log)

    result = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
              "lineup": MODERN_EDGE_WEIGHTS, "benchmark": BENCHMARK_WEIGHTS,
              "advisory_fee_annual_pct": 0.8, "rebalance": "quarterly",
              "caos_mode": CAOS_MODE, "proxies": splice_log,
              "compliance": ("Any spliced series is HYPOTHETICAL/BACKTESTED under "
                             "SEC Marketing Rule 206(4)-1 and requires hypothetical "
                             "disclosures. Net-of-fee applies to Modern Edge only."),
              "windows": {}}

    for name, cfg in WINDOWS.items():
        start = cfg["start"]
        if CAOS_MODE == "start" and "CAOS" in px:
            cstart = px["CAOS"].first_valid_index()
            if cstart is not None and str(cstart)[:10] > start:
                start = str(cstart)[:10]

        me_net = portfolio_curve(px, MODERN_EDGE_WEIGHTS, start, QUARTERLY_FEE)
        me_gross = portfolio_curve(px, MODERN_EDGE_WEIGHTS, start, 0.0)
        bench = portfolio_curve(px, BENCHMARK_WEIGHTS, start, 0.0)
        if me_net is None or bench is None:
            print(f"  [{name}] skipped (insufficient data)")
            continue

        idx = me_net.index.intersection(bench.index)
        me_net, me_gross, bench = me_net[idx], me_gross[idx], bench[idx]
        df = pd.DataFrame({
            "date": [str(d)[:10] for d in idx],
            "me_net_value": me_net.round(4).values,
            "me_gross_value": me_gross.round(4).values,
            "benchmark_value": bench.round(4).values,
            "me_drawdown_pct": (drawdown(me_net) * 100).round(3).values,
            "benchmark_drawdown_pct": (drawdown(bench) * 100).round(3).values,
        })
        csv_path = f"{OUT_CSV_PREFIX}_{name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [{name}] {len(df)} rows -> {csv_path}")

        result["windows"][name] = {
            "modern_edge_net": stats(me_net),
            "modern_edge_gross": stats(me_gross),
            "benchmark_6040": stats(bench),
            "dates": df["date"].tolist(),
            "me_net": me_net.round(4).tolist(),
            "benchmark": bench.round(4).tolist(),
            "me_drawdown_pct": (drawdown(me_net) * 100).round(3).tolist(),
            "benchmark_drawdown_pct": (drawdown(bench) * 100).round(3).tolist(),
        }

    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {OUT_JSON}")
    return result


if __name__ == "__main__":
    try:
        run()
    except Exception as e:  # noqa
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
