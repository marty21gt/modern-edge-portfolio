#!/usr/bin/env python3
"""
Modern Edge - Long-History Monthly Builder
==========================================
Produces  data/modern_edge_long_history_monthly.csv : monthly TOTAL-RETURN series
for each portfolio sleeve, using long-history proxies, back as far as the data
allows (~2000, capped by TIPS/gold).

You never run this locally. GitHub Actions runs it on GitHub's servers, which have
the internet access needed to reach Yahoo Finance. You just click "Run workflow".

It reads  trend_index.csv  (BTOP50 managed-futures monthly returns) from the repo,
because that one series is NOT available on Yahoo. Commit that file to the repo first.

Output columns are named by the live ETF each sleeve uses today (VOO, QQQ, ...), so
this monthly file lines up column-for-column with the daily components file. The real
proxy instrument behind each column is written to
data/modern_edge_long_history_proxymap.csv  for provenance / disclosures.
"""

import os, sys
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. The workflow installs it automatically.")
    sys.exit(1)

# --- sleeve ETF  ->  ordered list of Yahoo proxy candidates (first with data wins) ---
SLEEVE_PROXY = {
    "VOO":  ["VFINX"],           # S&P 500 index fund (1976)
    "QQQ":  ["QQQ"],             # Nasdaq-100 (1999) - catches the dot-com bust
    "VEA":  ["FSPSX", "VGTSX"],  # Developed ex-US / EAFE (1997)
    "IEMG": ["VEIEX"],           # Emerging markets (1994)
    "IEF":  ["VFITX"],           # Intermediate-term Treasury (1991)
    "SCHP": ["VIPSX"],           # TIPS (2000)  <-- usually the binding constraint
    "SGOL": ["GC=F"],            # Gold futures (~2000)
}
BENCH_PROXY = {
    "BENCH_US":   ["VFINX"],     # 60/40 U.S. equity (1976)
    "BENCH_INTL": ["VGTSX"],     # 60/40 international (1996)
    "BENCH_BOND": ["VBMFX"],     # 60/40 total bond (1986)
}
TBILL_TICKER = "^IRX"            # 13-week T-bill yield (%) -> cash return for SGOV & CAOS sleeves
TREND_CANDIDATES = ["trend_index.csv", "data/trend_index.csv"]  # BTOP50 -> DBMF sleeve

RETIREE_SLEEVES = ["VOO","QQQ","VEA","IEMG","SGOV","IEF","SCHP","DBMF","SGOL","CAOS"]

# retiree no-Bitcoin weights (BITB's 5% moved into VOO) and 60/40 weights, for the sanity check
RETIREE_W = {"VOO":.31,"QQQ":.12,"VEA":.06,"IEMG":.06,"SGOV":.10,"IEF":.10,
             "SCHP":.05,"DBMF":.10,"SGOL":.05,"CAOS":.05}
BENCH_W   = {"BENCH_US":.40,"BENCH_INTL":.20,"BENCH_BOND":.40}
FEE = 0.008


def _close_series(df):
    """Return a clean 1-D Close series from a yfinance frame (handles MultiIndex)."""
    if df is None or len(df) == 0:
        return None
    px = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.dropna()


def monthly_tr(candidates):
    """Monthly total returns from month-end adjusted closes; first candidate with data wins."""
    for t in candidates:
        try:
            df = yf.download(t, start="1975-01-01", auto_adjust=True,
                             progress=False, threads=False)
            px = _close_series(df)
            if px is None or len(px) < 24:
                continue
            m = px.resample("ME").last()
            r = m.pct_change().dropna()
            if len(r) > 12:
                r.name = None
                return r, t
        except Exception as e:
            print(f"    {t}: {e}")
    return None, None


def tbill_monthly():
    try:
        df = yf.download(TBILL_TICKER, start="1975-01-01", auto_adjust=True,
                         progress=False, threads=False)
        y = _close_series(df)
        if y is None:
            return None
        y = y.resample("ME").last()          # annualized yield, in percent
        r = (y.shift(1) / 100.0) / 12.0        # earn prior month's yield over the month
        return r.dropna()
    except Exception as e:
        print("  T-bill error:", e)
        return None


def trend_monthly():
    path = next((p for p in TREND_CANDIDATES if os.path.exists(p)), None)
    if path is None:
        print("  !! trend_index.csv not found in repo root or data/ — DBMF sleeve will be missing.")
        return None
    td = pd.read_csv(path)
    dcol = "date" if "date" in td.columns else td.columns[0]
    vcol = "value" if "value" in td.columns else td.columns[1]
    td[dcol] = pd.to_datetime(td[dcol])
    s = td.set_index(dcol)[vcol].astype(float)     # already monthly RETURNS (decimal)
    s.index = s.index + pd.offsets.MonthEnd(0)     # snap to month-end
    return s.sort_index()


def port_stats(returns, weights, fee=0.0, label=""):
    have = [c for c in weights if c in returns.columns]
    if len(have) < len(weights):
        return None
    w = pd.Series({k: weights[k] for k in have})
    w = w / w.sum()
    sub = returns[have].dropna()
    pr = sub.mul(w, axis=1).sum(axis=1)            # monthly rebalanced
    if fee:
        pr = (1 + pr) * ((1 - fee) ** (1/12)) - 1
    n = len(pr)
    cum = (1 + pr).cumprod()
    cagr = cum.iloc[-1] ** (12/n) - 1
    vol = pr.std() * np.sqrt(12)
    dd = (cum / cum.cummax() - 1).min()
    print(f"   {label:28s} {cagr*100:5.2f}% CAGR / {vol*100:5.2f}% vol / {dd*100:6.2f}% maxDD"
          f"   [{sub.index.min().date()}→{sub.index.max().date()}]")


def main():
    os.makedirs("data", exist_ok=True)
    cols, prov = {}, []
    print("Downloading long-history proxies (monthly total returns)...")

    for etf, cands in {**SLEEVE_PROXY, **BENCH_PROXY}.items():
        r, used = monthly_tr(cands)
        if r is None:
            print(f"  {etf:10s} NO DATA (tried {cands})")
            continue
        cols[etf] = r
        prov.append({"column": etf, "proxy": used,
                     "start": str(r.index.min().date()), "obs": int(len(r))})
        print(f"  {etf:10s} <- {used:7s}  from {r.index.min().date()}  ({len(r)} mo)")

    tb = tbill_monthly()
    if tb is not None:
        cols["SGOV"] = tb
        cols["CAOS"] = tb.copy()
        for etf in ("SGOV", "CAOS"):
            prov.append({"column": etf, "proxy": "^IRX T-bill",
                         "start": str(tb.index.min().date()), "obs": int(len(tb))})
        print(f"  SGOV/CAOS  <- ^IRX     from {tb.index.min().date()}  ({len(tb)} mo)")

    mf = trend_monthly()
    if mf is not None:
        cols["DBMF"] = mf
        prov.append({"column": "DBMF", "proxy": "BTOP50 (trend_index.csv)",
                     "start": str(mf.index.min().date()), "obs": int(len(mf))})
        print(f"  DBMF       <- BTOP50  from {mf.index.min().date()}  ({len(mf)} mo)")

    data = pd.concat(cols, axis=1).sort_index()
    data.index.name = "date"

    have = [c for c in RETIREE_SLEEVES if c in data.columns]
    common = data[have].dropna()
    print("\n" + "=" * 66)
    if len(common):
        starts = {c: data[c].dropna().index.min() for c in have}
        binder = max(starts, key=starts.get)
        print("Common history where ALL retiree sleeves exist:")
        print(f"   {common.index.min().date()}  ->  {common.index.max().date()}   ({len(common)} months)")
        print(f"   Covers 2008 GFC : {'YES' if common.index.min() <= pd.Timestamp('2007-10-31') else 'NO'}")
        print(f"   Covers dot-com  : {'YES' if common.index.min() <= pd.Timestamp('2000-09-30') else 'PARTIAL / NO'}")
        print(f"   Binding sleeve  : {binder}  (starts {starts[binder].date()})")
        print("\n   Sanity check (monthly-rebalanced, common window):")
        port_stats(common, RETIREE_W, fee=FEE, label="Retiree no-BTC (net 0.8%)")
        port_stats(data[[c for c in BENCH_W if c in data.columns]].dropna(), BENCH_W, 0.0,   "60/40 (gross)")
        port_stats(data[[c for c in BENCH_W if c in data.columns]].dropna(), BENCH_W, FEE, "60/40 (net 0.8%)")
    else:
        print("No common window — a sleeve is missing. See messages above.")
    print("=" * 66)

    out = "data/modern_edge_long_history_monthly.csv"
    data.to_csv(out, float_format="%.6f")
    pd.DataFrame(prov).to_csv("data/modern_edge_long_history_proxymap.csv", index=False)
    print(f"\nSaved {out}  ({data.shape[0]} rows x {data.shape[1]} cols)")
    print("Saved data/modern_edge_long_history_proxymap.csv")
    print("\nDownload the monthly CSV from the data/ folder and attach it back in chat.")


if __name__ == "__main__":
    main()
