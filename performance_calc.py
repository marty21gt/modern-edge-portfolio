#!/usr/bin/env python3
"""
Modern Edge - Performance calc (quarterly rebalance)
====================================================
Reproduces the Performance tab of Modern_Edge_Stress_Test_1996-2026.xlsx.

Input : modern_edge_components_prices.csv  (daily ETF adjusted-close, 2013-2026)
Method: daily returns, QUARTERLY rebalancing (reset to target the first trading
        day of each quarter, drift within the quarter), net of 0.80%.

Run:    python performance_calc.py modern_edge_components_prices.csv

--------------------------------------------------------------------------
FIX 2026-09 — non-trading rows must be dropped BEFORE differencing.
The price file carries weekend rows as blanks. pandas <3.0 defaulted
pct_change(fill_method='pad'), which forward-filled them so Monday computed
against Friday. pandas 3.0 changed that default to None, so every Monday
after a blank weekend became NaN and was silently dropped by the later
dropna() -- 508 of 3,434 trading days, 438 of them Mondays, including
2020-03-16 (VOO -11.74%) and 2020-03-09 (VOO -7.72%).
That understated max drawdown by ~7pp and overstated CAGR by ~1.5pp.
Dropping NaN rows per column-set first makes the result version-independent.
--------------------------------------------------------------------------
"""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "modern_edge_components_prices.csv"
FEE = 0.008
RETIREE = {"VOO":.31,"QQQ":.12,"VEA":.06,"IEMG":.06,"SGOV":.10,"IEF":.10,
           "SCHP":.05,"DBMF":.10,"SGOL":.05,"CAOS":.05}          # BITB's 5% -> VOO
BENCH   = {"VOO":.40,"VXUS":.20,"BND":.40}

# expected output for the shipped price file - guards against silent regressions
EXPECTED = {"Modern Edge (net)": (0.0869, 0.0954, -0.1826),
            "60/40 (net)":       (0.0750, 0.1004, -0.2199),
            "60/40 (gross)":     (0.0837, 0.1004, -0.2149)}
TOL = 0.0010

def daily_returns(px, cols):
    """Drop non-trading (blank) rows FIRST, then difference. Never pct_change() the raw frame."""
    return px[cols].dropna(how="any").pct_change().dropna(how="any")

def quarterly(px, weights, fee=FEE):
    cols = list(weights); tgt = np.array([weights[c] for c in cols])
    r = daily_returns(px, cols)
    qstart = set(r.groupby(r.index.to_period("Q")).head(1).index)
    w = tgt.copy(); out = []
    for dt, row in zip(r.index, r[cols].values):
        if dt in qstart: w = tgt.copy()          # rebalance to target
        pr = float(np.dot(w, row)); out.append(pr)
        w = w * (1 + row) / (1 + pr)              # drift within quarter
    pr = pd.Series(out, index=r.index)
    pr = (1 + pr) * ((1 - fee) ** (1/252)) - 1     # net of fee
    cum = (1 + pr).cumprod(); n = len(pr)
    return cum.iloc[-1] ** (252/n) - 1, pr.std()*np.sqrt(252), ((cum/cum.cummax())-1).min(), n

def main():
    px = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
    ref = daily_returns(px, list(BENCH))
    print(f"window {ref.index.min().date()} -> {ref.index.max().date()}  |  "
          f"{len(ref)} trading days  |  quarterly rebal\n")
    ok = True
    for label, w, fee in [("Modern Edge (net)", RETIREE, FEE),
                          ("60/40 (net)", BENCH, FEE),
                          ("60/40 (gross)", BENCH, 0.0)]:
        c, v, d, n = quarterly(px, w, fee)
        flag = ""
        if label in EXPECTED:
            e = EXPECTED[label]
            if max(abs(c-e[0]), abs(v-e[1]), abs(d-e[2])) > TOL:
                flag = f"   <-- DRIFT from expected {e[0]*100:.2f}/{e[1]*100:.2f}/{e[2]*100:.2f}"
                ok = False
        print(f"  {label:20s} {c*100:5.2f}% CAGR / {v*100:5.2f}% vol / {d*100:6.2f}% maxDD{flag}")
    print("\n  regression check: PASS" if ok else "\n  regression check: FAIL - investigate before publishing")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
