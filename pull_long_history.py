#!/usr/bin/env python3
# =============================================================================
# pull_long_history.py
#
# Pulls long-history proxy series for the Modern Edge (no-Bitcoin) sleeves and
# the 60/40 benchmark, back through the 2008 crisis (and the dot-com tail where
# data allows), so we can run an extended safe-withdrawal-rate bootstrap that
# actually contains real bear markets.
#
# Output: modern_edge_long_history_monthly.csv  (monthly TOTAL returns, one
#         column per sleeve/benchmark proxy) + a coverage log printed to screen.
#
# Run it, then send me the CSV back (same as you did with the price file).
#
#   pip install yfinance pandas numpy
#   python pull_long_history.py
#
# NOTE ON MANAGED FUTURES (read this): there is no clean managed-futures ETF or
# fund on Yahoo before ~2010, and managed futures is the exact sleeve that earns
# its keep in 2008. To reach 2008 you need a trend/CTA index. Two options:
#   (A) Download the SG Trend Index (free monthly data, SocGen Prime Services)
#       or the BarclayHedge BTOP50, save it next to this script as
#       'trend_index.csv' with two columns: date,value  (value = index level
#       OR monthly return in decimal — the script auto-detects). If present, it
#       is used and the backtest can span ~2000-present.
#   (B) If that file is absent, the script falls back to WTMF (starts ~2011),
#       which means the common window will start ~2011 and WON'T test 2008.
# Everything here is proxy/backtested data with the usual limitations; the point
# is to test the thesis through 2008, not to represent live performance.
# =============================================================================

import sys, os
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("Please run:  pip install yfinance pandas numpy")

START = "1997-01-01"   # download from here; actual coverage is logged below

# ---- Proxy map: logical sleeve -> ordered list of Yahoo tickers to try -------
# Modern Edge (no-Bitcoin, retiree weights) and the 60/40 benchmark.
# CAOS is intentionally proxied by T-bills for the whole long window (its live
# AVOLX/CAOS history is already captured in the recent daily backtest).
PROXIES = {
    "US_LARGE":        ["VFINX", "SPY"],          # VOO  / 60-40 US equity  (S&P 500 TR)
    "US_NASDAQ":       ["QQQ", "ONEQ"],           # QQQ  (Nasdaq-100, from 1999-03)
    "DEV_INTL":        ["FSPSX", "EFA", "VTMGX"], # VEA  (MSCI EAFE)
    "EM":              ["VEIEX", "VWO", "EEM"],   # IEMG (emerging markets)
    "INT_TREASURY":    ["VFITX", "IEF"],          # IEF  (intermediate Treasury)
    "TIPS":            ["VIPSX", "SCHP", "TIP"],  # SCHP (TIPS, from 2000-06)
    "GOLD":            ["GC=F", "GLD", "IAU"],    # SGOL (gold)
    "BENCH_INTL":      ["VGTSX", "VXUS"],         # VXUS (total international)
    "BENCH_AGG":       ["VBMFX", "BND", "AGG"],   # BND  (US aggregate bond)
}
MGD_FUTURES_FALLBACK = ["WTMF", "DBMF"]           # DBMF; only if trend_index.csv absent
TRHEE_MO_TBILL       = "^IRX"                      # 13-week T-bill yield (SGOV + CAOS proxy)

# Weights are documented here for reference; portfolio construction happens on my end.
ME_WEIGHTS = {"US_LARGE":.31,"US_NASDAQ":.12,"DEV_INTL":.06,"EM":.06,"TBILL":.10,
              "INT_TREASURY":.10,"TIPS":.05,"MGD_FUTURES":.10,"GOLD":.05,"CAOS":.05}
BENCH_WEIGHTS = {"US_LARGE":.40,"BENCH_INTL":.20,"BENCH_AGG":.40}


def monthly_total_return(tickers):
    """Try each ticker; return monthly total-return series from the first that works."""
    for tk in tickers:
        try:
            df = yf.download(tk, start=START, auto_adjust=True, progress=False)
            if df is None or len(df) == 0:
                continue
            px = df["Close"]
            if isinstance(px, pd.DataFrame):      # yfinance sometimes returns a frame
                px = px.iloc[:, 0]
            m = px.resample("ME").last().pct_change().dropna()
            if len(m) > 12:
                print(f"    {tk:8s} OK  {m.index.min().date()} -> {m.index.max().date()}  ({len(m)} mo)")
                return m.rename(tk), tk
        except Exception as e:
            print(f"    {tk:8s} failed: {e}")
    return None, None


def tbill_monthly():
    """Monthly T-bill total return from ^IRX (13-week discount yield, in percent)."""
    df = yf.download(TRHEE_MO_TBILL, start=START, auto_adjust=True, progress=False)
    y = df["Close"]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    ym = y.resample("ME").last() / 100.0          # decimal annualized yield
    r = (1 + ym.shift(1)) ** (1/12) - 1            # use prior month-end yield
    r = r.dropna()
    print(f"    {TRHEE_MO_TBILL:8s} OK  {r.index.min().date()} -> {r.index.max().date()}  ({len(r)} mo)  [T-bill; used for SGOV and CAOS]")
    return r.rename("TBILL")


def trend_index_monthly():
    """Use trend_index.csv (SG Trend / BTOP50) if present; else fall back to a fund."""
    path = "trend_index.csv"
    if os.path.exists(path):
        raw = pd.read_csv(path)
        raw.columns = [c.strip().lower() for c in raw.columns]
        raw["date"] = pd.to_datetime(raw["date"])
        raw = raw.set_index("date").sort_index()
        val = raw["value"].astype(float)
        # auto-detect: index levels (values >> 1) vs monthly returns (small around 0)
        if val.abs().median() > 1.5:
            r = val.resample("ME").last().pct_change().dropna()
            kind = "index levels"
        else:
            r = val.resample("ME").last().dropna()
            kind = "monthly returns"
        print(f"    trend_index.csv OK  ({kind})  {r.index.min().date()} -> {r.index.max().date()}  ({len(r)} mo)  [DBMF proxy]")
        return r.rename("MGD_FUTURES")
    print("    trend_index.csv NOT found -> falling back to a fund proxy.")
    print("    *** WARNING: without a trend index this will NOT reach 2008. ***")
    m, tk = monthly_total_return(MGD_FUTURES_FALLBACK)
    return m.rename("MGD_FUTURES") if m is not None else None


def main():
    print("Downloading long-history proxies (monthly total returns)...\n")
    cols = {}

    for sleeve, tickers in PROXIES.items():
        print(f"  {sleeve}:")
        s, used = monthly_total_return(tickers)
        if s is None:
            print(f"    !! no data for {sleeve} — check tickers")
        else:
            cols[sleeve] = s

    print("  TBILL (SGOV + CAOS proxy):")
    cols["TBILL"] = tbill_monthly()

    print("  MGD_FUTURES (DBMF proxy):")
    mf = trend_index_monthly()
    if mf is not None:
        cols["MGD_FUTURES"] = mf

    # CAOS = T-bills for the entire long window (per instruction)
    cols["CAOS"] = cols["TBILL"].rename("CAOS")

    data = pd.concat(cols.values(), axis=1)
    data.index.name = "date"

    # Report common-history window (all Modern Edge sleeves present)
    me_cols = [c for c in ME_WEIGHTS if c in data.columns]
    common = data[me_cols].dropna()
    print("\n" + "=" * 66)
    if len(common):
        print(f"Common history where ALL Modern Edge sleeves exist:")
        print(f"   {common.index.min().date()}  ->  {common.index.max().date()}   ({len(common)} months)")
        covers_2008 = common.index.min() <= pd.Timestamp("2007-10-01")
        print(f"   Covers the 2008 crisis: {'YES' if covers_2008 else 'NO  (need trend_index.csv)'}")
    else:
        print("No common window — a sleeve is missing. See warnings above.")
    print("=" * 66)

    out = "modern_edge_long_history_monthly.csv"
    data.to_csv(out, float_format="%.6f")
    print(f"\nSaved {out}  ({data.shape[0]} rows x {data.shape[1]} columns)")
    print("Columns:", ", ".join(data.columns))
    print("\nSend this CSV back and I'll run the extended safe-withdrawal bootstrap.")


if __name__ == "__main__":
    main()
