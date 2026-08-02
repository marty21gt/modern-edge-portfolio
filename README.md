# Modern Edge — Daily Backtest Data

This repo pulls **real daily total-return prices** for the current Modern Edge
lineup and a 60/40 benchmark, and builds the data the marketing charts need.
**You never run Python.** GitHub's servers do the work; you download the result.

## Current lineup (50 / 25 / 25)

| Sleeve | Holdings |
|---|---|
| Equity 50% | VOO 26 · QQQ 12 · VEA 6 · IEMG 6 |
| Fixed Income 25% | SGOV 10 · IEF 10 · SCHP 5 |
| Alternatives 25% | DBMF 10 · SGOL 5 · BITB 5 · CAOS 5 |

Fee: 0.8%/yr (0.2%/qtr), applied to Modern Edge only. Quarterly rebalancing.

## One-time setup (about 5 minutes, all in the browser)

1. Create the repo: on github.com click **New** → name it `modern-edge-backtest`
   → **Create repository**.
2. Add these files. For each one: **Add file → Create new file**, type the path
   (e.g. `.github/workflows/backtest.yml`), paste the contents, **Commit**.
   - `generate_backtest.py`
   - `requirements.txt`
   - `.github/workflows/backtest.yml`
3. Give Actions write access: **Settings → Actions → General →
   Workflow permissions → Read and write permissions → Save**.

## Run it

- Go to the **Actions** tab → **Modern Edge Backtest** → **Run workflow**.
- Wait ~1–2 minutes. When it finishes, the `data/` folder will contain:
  - `modern_edge_daily_long.csv` — daily series from ~June 2019
  - `modern_edge_daily_short.csv` — clean 4-year window from Jan 2022
  - `modern_edge_backtest.json` — same data + summary stats + proxy disclosures
- Download any file from the repo (open it → **Download raw**) and **attach it
  back in chat**. I'll wire it into the page HTMLs.

It also re-runs automatically on the 1st of each month.

## CAOS history (set to the predecessor-fund splice)

CAOS is the **successor to the Arin Large Cap Theta Fund** (mutual fund ticker
AVOLX), launched **August 2013** and converted to an ETF on **March 6, 2023**
with the same investment objective. The same portfolio managers (Lawrence
Lempert, Joseph DeSipio) ran the predecessor fund since Aug 2013 and CAOS since
2023 — continuous management, same strategy. So the pre-2023 series is the *real
predecessor-fund NAV*, not a generic proxy. `CAOS_MODE` is set to `"AVOLX"`.

Two disclosures this requires on any chart:
1. Pre-2023 data reflects a **predecessor mutual fund** with different fee and
   tax treatment than the current ETF.
2. The **AVOLX-era payoff was somewhat more convex** than today's ETF-wrapper
   CAOS, which now leans toward stable positive carry — so the pre-2023 slice
   may make the hedge look punchier in a crash than current CAOS would deliver.

Note: even with AVOLX back to 2013, the **portfolio** series can't start before
~late 2014, because BTC-USD (the BITB proxy) is the binding constraint on Yahoo.
The long window auto-trims to the earliest date all eleven sleeves have data.

Alternatives if you'd rather: `"cash"` (hold the sleeve in cash pre-2023, most
conservative) or `"start"` (begin at CAOS's real ETF date, no predecessor data).

## Compliance

Any spliced series (SGOV←BIL, DBMF←WTMF, BITB←BTC, and CAOS depending on mode)
is **hypothetical/backtested** under SEC Marketing Rule 206(4)-1 and needs the
standard hypothetical disclosures — not just a footnote. Every splice is logged
in the JSON's `proxies` block so nothing is hidden. Your Kwanti single-account
stats remain the actual track record; this is a model of the strategy.
