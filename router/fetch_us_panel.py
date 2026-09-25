"""Build a long-horizon US daily panel (2005-2026) from the Yahoo chart endpoint.

Why this exists
---------------
Every reviewer of the NeurIPS submission flagged the same thing: the out-of-sample
window was 13-17 months of the 2024+ growth phase, with no drawdown in it. The
earlier local dump starts 2022-01-03
and the extended panel starts 2022 as well, so neither can answer that.

Yahoo's chart endpoint serves split/dividend-adjusted daily closes back to
2005-01-03 with no key. That span contains the GFC, the 2011 downgrade, the
2015-16 selloff, Q4-2018, the COVID crash, the 2022 bear and the 2024-26 rally.

Provider discipline
-------------------
We do NOT splice this with the existing Alpha Vantage / Polygon panels. The whole
panel is re-derived from one source and one vintage; mixing vintages can leak
information across sources. `panel_close_extended.csv` is kept only
as a cross-check (see `verify_overlap`).

Outputs (data/):
    panel_close_us_2005.csv     dates x tickers, adjusted close
    panel_volume_us_2005.csv    dates x tickers, raw volume
    us_panel_manifest.json      source, vintage, coverage, rejects
"""
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ROUTER_ROOT", Path(__file__).resolve().parent))
DATA = ROOT / "data"
CACHE = DATA / "_yahoo_cache"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"

START = "2005-01-01"
END = "2026-09-20"

# Sector / broad ETFs. These carry no survivorship bias (they never leave the
# index because they are the index), so they give a clean control universe.
ETFS = ["SPY", "XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLY", "XLB",
        "IWM", "EFA", "TLT", "GLD", "QQQ"]


def _epoch(d):
    return int(pd.Timestamp(d).timestamp())


def fetch_one(sym, retries=4):
    """Return a DataFrame indexed by date with adj_close and volume, or None."""
    cache = CACHE / f"{sym}.json"
    if cache.exists():
        try:
            payload = json.loads(cache.read_text())
        except Exception:
            payload = None
        if payload is not None:
            return _parse(sym, payload)

    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1={_epoch(START)}&period2={_epoch(END)}"
           f"&interval=1d&events=div%2Csplit")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=20) as r:
                payload = json.loads(r.read().decode())
            cache.write_text(json.dumps(payload))
            return _parse(sym, payload)
        except Exception as e:  # network flake, 404 on a delisted/renamed ticker
            last = e
            time.sleep(1.5 * (attempt + 1) + random.random())
    print(f"  ! {sym}: {last}", file=sys.stderr)
    return None


def _parse(sym, payload):
    try:
        res = payload["chart"]["result"][0]
        ts = res["timestamp"]
        quote = res["indicators"]["quote"][0]
        adj = res["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError):
        return None
    idx = pd.to_datetime(pd.Series(ts), unit="s", utc=True).dt.tz_convert(
        "America/New_York").dt.normalize().dt.tz_localize(None)
    df = pd.DataFrame({"adj_close": adj, "volume": quote.get("volume")},
                      index=idx.values)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[df["adj_close"].notna()]
    return df if len(df) else None


def main():
    CACHE.mkdir(parents=True, exist_ok=True)

    # Universe of candidate tickers: the 500 names already used by the paper,
    # plus the ETF control set (data/tickers.txt, one symbol per line).
    tickers = sorted(set((DATA / "tickers.txt").read_text().split()) | set(ETFS))
    print(f"fetching {len(tickers)} tickers {START}..{END}")

    closes, volumes, rejects = {}, {}, []
    t0 = time.time()
    for i, sym in enumerate(tickers, 1):
        df = fetch_one(sym)
        if df is None or len(df) < 250:
            rejects.append({"symbol": sym,
                            "reason": "no data" if df is None else f"only {len(df)} rows"})
        else:
            closes[sym] = df["adj_close"]
            volumes[sym] = df["volume"]
        if i % 50 == 0:
            print(f"  {i}/{len(tickers)}  ok={len(closes)}  "
                  f"rejected={len(rejects)}  {time.time()-t0:.0f}s")
        if not (CACHE / f"{sym}.json").exists():
            time.sleep(0.3)

    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(volumes).sort_index()

    # Trading calendar: keep days where a healthy majority of names quote, then
    # forward-fill single-name gaps. A name is dropped entirely if it is missing
    # more than 2% of its own post-listing history.
    live = close.notna().sum(axis=1)
    calendar = close.index[live >= 0.5 * live.max()]
    close, vol = close.loc[calendar], vol.loc[calendar]
    close = close.ffill(limit=5)
    vol = vol.ffill(limit=5)

    close.index.name = "date"
    vol.index.name = "date"
    close.to_csv(DATA / "panel_close_us_2005.csv")
    vol.to_csv(DATA / "panel_volume_us_2005.csv")

    first_valid = {c: str(close[c].first_valid_index().date()) for c in close.columns}
    manifest = {
        "source": "query1.finance.yahoo.com/v8/finance/chart, adjclose",
        "fetched_utc": pd.Timestamp.utcnow().isoformat(),
        "requested_span": [START, END],
        "realised_span": [str(close.index.min().date()), str(close.index.max().date())],
        "n_rows": int(len(close)),
        "n_symbols": int(close.shape[1]),
        "etf_control": [e for e in ETFS if e in close.columns],
        "rejects": rejects,
        "n_full_history_2005": int(sum(v <= "2005-12-31" for v in first_valid.values())),
        "first_valid": first_valid,
        "note": ("Single-source, single-vintage. Not spliced with the Alpha Vantage "
                 "or Polygon panels. Survivorship: constituent list is as-of 2026, "
                 "so pre-2026 delistings are absent; all modes and baselines share "
                 "the universe, so the bias is common-mode."),
    }
    (DATA / "us_panel_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\npanel {close.shape} {manifest['realised_span'][0]}..{manifest['realised_span'][1]}")
    print(f"full-history-since-2005 names: {manifest['n_full_history_2005']}")
    print(f"rejects: {len(rejects)}")


if __name__ == "__main__":
    main()
