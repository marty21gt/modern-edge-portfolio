#!/usr/bin/env python3
"""
Modern Edge - Long-History Monthly Builder  (v2 - extended proxies)
==================================================================
Produces  data/modern_edge_long_history_monthly.csv : monthly TOTAL-RETURN series
for each portfolio sleeve, pushing history back toward ~1997.

You never run this locally - GitHub Actions runs it. Click "Run workflow".

v2 changes (the four proxy fixes):
  1. VEA (developed intl)  -> VGTSX (1996), NOT FSPSX (Yahoo only has FSPSX from 2011,
     which was silently capping the whole file at 2011). VGTSX is total-intl, so it
     carries a little emerging-markets overlap with the IEMG sleeve - noted for disclosure.
  2. QQQ (growth)          -> QQQ spliced over the Nasdaq-100 index ^NDX (1985). ^NDX is a
     PRICE index (no dividends), so pre-1999 returns understate by ~0.5%/yr - disclose.
  3. SCHP (TIPS)           -> VIPSX spliced over VFITX (nominal intermediate Treasury) for
     the 1997-2000 gap, since US TIPS did not exist before 1997. Disclosed proxy bridge.
  4. SGOL (gold)           -> spot-gold file if present (data/gold_spot.csv), else GC=F.
     GATE TO 1997: Yahoo's gold (GC=F) only starts 2000, so WITHOUT a committed spot-gold
     file the window still caps at ~2000. Drop a gold_spot.csv in the repo (date,price)
     to reach ~1997 - same pattern as trend_index.csv for managed futures.

Reads from repo:  trend_index.csv (BTOP50 -> DBMF),  optional gold_spot.csv (-> SGOL).
"""

import os, sys
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed (the workflow installs it).")
    sys.exit(1)

FEE = 0.008
RETIREE_W = {"VOO":.31,"QQQ":.12,"VEA":.06,"IEMG":.06,"SGOV":.10,"IEF":.10,
             "SCHP":.05,"DBMF":.10,"SGOL":.05,"CAOS":.05}
BENCH_W   = {"BENCH_US":.40,"BENCH_INTL":.20,"BENCH_BOND":.40}


def _close(df):
    if df is None or len(df) == 0:
        return None
    px = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.dropna()


def monthly_returns(ticker):
    """Month-end total returns for one Yahoo ticker (auto-adjusted)."""
    try:
        df = yf.download(ticker, start="1975-01-01", auto_adjust=True,
                         progress=False, threads=False)
        px = _close(df)
        if px is None or len(px) < 24:
            return None
        return px.resample("ME").last().pct_change().dropna()
    except Exception as e:
        print(f"    {ticker}: {e}")
        return None


def splice(*series):
    """Priority splice of monthly-return series: first has priority, later fill the gaps."""
    series = [s for s in series if s is not None]
    if not series:
        return None
    out = series[0].copy()
    for s in series[1:]:
        out = out.combine_first(s)
    return out.sort_index()


def tbill():
    df = yf.download("^IRX", start="1975-01-01", auto_adjust=True, progress=False, threads=False)
    y = _close(df)
    if y is None:
        return None
    y = y.resample("ME").last()
    return ((y.shift(1) / 100.0) / 12.0).dropna()


def from_file(candidates, is_return=None):
    """Read a committed CSV (date,value). Auto-detect return vs price unless told."""
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return None, None
    d = pd.read_csv(path)
    dc = "date" if "date" in d.columns else d.columns[0]
    vc = "value" if "value" in d.columns else ("price" if "price" in d.columns else d.columns[1])
    d[dc] = pd.to_datetime(d[dc])
    s = d.set_index(dc)[vc].astype(float)
    s.index = s.index + pd.offsets.MonthEnd(0)
    s = s.sort_index()
    if is_return is None:  # heuristic: returns are small & can be negative
        is_return = (s.abs().mean() < 0.5) and (s.min() < 0)
    return (s if is_return else s.resample("ME").last().pct_change().dropna()), path


def main():
    os.makedirs("data", exist_ok=True)
    cols, prov = {}, []
    print("Building long-history monthly series (v2)...\n")

    def add(name, series, source):
        if series is None or len(series) < 12:
            print(f"  {name:10s} NO DATA ({source})"); return
        cols[name] = series
        prov.append({"column": name, "proxy": source,
                     "start": str(series.index.min().date()), "obs": int(len(series))})
        print(f"  {name:10s} <- {source:30s} from {series.index.min().date()} ({len(series)} mo)")

    add("VOO",  monthly_returns("VFINX"), "VFINX (S&P 500)")
    add("QQQ",  splice(monthly_returns("QQQ"), monthly_returns("^NDX")), "QQQ / ^NDX price (1985)")
    add("VEA",  monthly_returns("VGTSX"), "VGTSX total-intl (1996; EM overlap)")
    add("IEMG", monthly_returns("VEIEX"), "VEIEX emerging (1994)")
    add("IEF",  monthly_returns("VFITX"), "VFITX int. Treasury (1991)")
    add("SCHP", splice(monthly_returns("VIPSX"), monthly_returns("VFITX")), "VIPSX / VFITX bridge pre-2000")

    gold, gsrc = from_file(["gold_spot.csv", "data/gold_spot.csv"])
    if gold is not None:
        add("SGOL", gold, f"spot gold ({os.path.basename(gsrc)})")
    else:
        add("SGOL", monthly_returns("GC=F"), "GC=F futures (2000) <- add gold_spot.csv for 1997")

    add("BENCH_US",   monthly_returns("VFINX"),  "VFINX (S&P 500)")
    add("BENCH_INTL", monthly_returns("VGTSX"),  "VGTSX total-intl (1996)")
    add("BENCH_BOND", monthly_returns("VBMFX"),  "VBMFX total bond (1986)")

    tb = tbill()
    if tb is not None:
        cols["SGOV"] = tb; cols["CAOS"] = tb.copy()
        for n in ("SGOV", "CAOS"):
            prov.append({"column": n, "proxy": "^IRX T-bill", "start": str(tb.index.min().date()), "obs": int(len(tb))})
        print(f"  SGOV/CAOS  <- ^IRX T-bill                  from {tb.index.min().date()} ({len(tb)} mo)")

    mf, msrc = from_file(["trend_index.csv", "data/trend_index.csv"], is_return=True)
    if mf is not None:
        add("DBMF", mf, f"BTOP50 ({os.path.basename(msrc)})")
    else:
        print("  !! trend_index.csv not found - DBMF sleeve missing.")

    data = pd.concat(cols, axis=1).sort_index()
    data.index.name = "date"

    have = [c for c in RETIREE_W if c in data.columns]
    common = data[have].dropna()
    print("\n" + "=" * 68)
    if len(common):
        starts = {c: data[c].dropna().index.min() for c in have}
        binder = max(starts, key=starts.get)
        print(f"Common retiree window : {common.index.min().date()} -> {common.index.max().date()} ({len(common)} mo)")
        print(f"Binding sleeve        : {binder} (starts {starts[binder].date()})")
        print(f"Covers dot-com (2000) : {'YES' if common.index.min() <= pd.Timestamp('2000-09-30') else 'PARTIAL/NO'}")
        print(f"Covers 2008 GFC       : {'YES' if common.index.min() <= pd.Timestamp('2007-10-31') else 'NO'}")

        def stats(w, fee, label):
            src = common if set(w) <= set(common.columns) else data[[c for c in w if c in data.columns]].dropna()
            wv = pd.Series({k: w[k] for k in w if k in src.columns}); wv /= wv.sum()
            pr = src[wv.index].mul(wv, axis=1).sum(axis=1)
            pr = (1 + pr) * ((1 - fee) ** (1/12)) - 1
            cum = (1 + pr).cumprod(); n = len(pr)
            print(f"   {label:26s} {(cum.iloc[-1]**(12/n)-1)*100:5.2f}% CAGR / "
                  f"{pr.std()*np.sqrt(12)*100:5.2f}% vol / {((cum/cum.cummax())-1).min()*100:6.2f}% maxDD")
        print("\n   Sanity (monthly-rebalanced, common window):")
        stats(RETIREE_W, FEE, "Retiree no-BTC net 0.8%")
        stats(BENCH_W, FEE, "60/40 net 0.8%")
        stats(BENCH_W, 0.0, "60/40 gross")
    print("=" * 68)

    out = "data/modern_edge_long_history_monthly.csv"
    data.to_csv(out, float_format="%.6f")
    pd.DataFrame(prov).to_csv("data/modern_edge_long_history_proxymap.csv", index=False)
    print(f"\nSaved {out}  ({data.shape[0]} x {data.shape[1]})")
    print("Download it from data/ and attach it back in chat.")


if __name__ == "__main__":
    main()
