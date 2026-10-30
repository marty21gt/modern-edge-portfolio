#!/usr/bin/env python3
"""
Modern Edge - does the failure-rate gap survive a change of regime?
====================================================================
survival_sim.py builds 30-year retirements by resampling 12-month blocks from the
whole 1996-2026 history. This script asks the narrower question behind the claim
"the gap between the two approaches is the durable finding": if the next 30 years
looked like only ONE stretch of that history, does the 60/40 still run out more often?

For each data window of WIN years (stepping one year at a time), it runs the exact
survival_sim.py machinery - same weights, 0.80% fee, quarterly rebalancing, 12-month
circular blocks, $1M, 4% + 2.5% inflation, single restorable 10% guardrail cut at 6% -
but draws every block only from that window.

Run:  python regime_survival.py modern_edge_long_history_monthly.csv [WIN=10] [SIMS=20000]
"""
import sys
import numpy as np
import pandas as pd
import survival_sim as ss

CSV  = sys.argv[1] if len(sys.argv) > 1 else "modern_edge_long_history_monthly.csv"
WIN  = int(sys.argv[2]) if len(sys.argv) > 2 else 10
SIMS = int(sys.argv[3]) if len(sys.argv) > 3 else 20_000

def run(Rme, Rbn):
    L = len(Rme); rng = np.random.default_rng(ss.SEED); nblk = ss.HORIZON // ss.BLOCK
    starts = rng.integers(0, L, size=(SIMS, nblk))
    bidx = ((starts[:, :, None] + np.arange(ss.BLOCK)[None, None, :]) % L).reshape(SIMS, ss.HORIZON)
    ME, BN = ss.qtr_paths(Rme, ss.RET_W, bidx), ss.qtr_paths(Rbn, ss.BEN_W, bidx)
    sM, sB, gM = ss.simulate(ME), ss.simulate(BN), ss.simulate(ME, True)
    return sM["fail"], sB["fail"], gM["fail"]

def main():
    m = pd.read_csv(CSV, parse_dates=["date"]).set_index("date").sort_index()
    m = m[ss.RET_S + ss.BEN_S].dropna()
    rows = []
    # full window first, as the anchor that should reproduce the published figures
    full = run(m[ss.RET_S].values, m[ss.BEN_S].values)
    rows.append(("1996-05", m.index[-1].strftime("%Y-%m"), len(m), *full))
    n = 12 * WIN
    s = 0
    while s + n <= len(m):
        w = m.iloc[s:s + n]
        rows.append((w.index[0].strftime("%Y-%m"), w.index[-1].strftime("%Y-%m"), len(w),
                     *run(w[ss.RET_S].values, w[ss.BEN_S].values)))
        s += 12
    df = pd.DataFrame(rows, columns=["from", "to", "months", "me_static", "bn_static", "me_full_process"])
    df["gap_pts"] = (df.bn_static - df.me_static) * 100
    df["process_gap_pts"] = (df.bn_static - df.me_full_process) * 100
    df.to_csv(f"regime_survival_{WIN}yr.csv", index=False)

    show = df.copy()
    for c in ["me_static", "bn_static", "me_full_process"]:
        show[c] = (show[c] * 100).map("{:5.1f}%".format)
    for c in ["gap_pts", "process_gap_pts"]:
        show[c] = show[c].map("{:+5.1f}".format)
    print(f"{WIN}-year data windows, {SIMS:,} paths each (first row = full history)\n")
    print(show.to_string(index=False))
    sub = df.iloc[1:]
    print(f"\n60/40 static fails more often than ME static:        {(sub.gap_pts > 0).sum()} of {len(sub)}")
    print(f"60/40 static fails more than full ME process:         {(sub.process_gap_pts > 0).sum()} of {len(sub)}")
    print(f"windows where neither static approach fails (<0.5%):  {((sub.me_static < .005) & (sub.bn_static < .005)).sum()}")

if __name__ == "__main__":
    main()
