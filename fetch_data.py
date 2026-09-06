#!/usr/bin/env python3
"""
fetch_data.py — Download historical JPY → BDT daily rates plus the two USD legs.

Data source: free fawazahmed0/currency-api package served via jsDelivr CDN.
  https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@<YYYY.M.D>/v1/currencies/usd.json

Why fetch `usd.json` instead of `jpy.json`?
  The JPY → BDT rate factors exactly into two USD legs:

      JPY/BDT  =  (BDT per USD) / (JPY per USD)

  `usd.json` contains BOTH `usd.bdt` (BDT per 1 USD) and `usd.jpy` (JPY per 1 USD),
  so one request per day gives us the target rate *and* the two cross-rate legs
  that drive it. The extra columns (usd_to_jpy, usd_to_bdt) are used as LSTM
  features: Bangladesh manages the Taka against the USD, while JPY/USD is the
  volatile market leg — together they decompose almost all of the JPY→BDT move.

  We first read the jsDelivr "tags" endpoint, which maps dates -> npm versions, so
  we know exactly which version to query for every day, then fetch each day in
  parallel with retries and write the result to a CSV.

Usage:
    python3 fetch_data.py [--start 2024-03-02] [--end 2026-08-27] [--out data/jpy_bdt_daily.csv]
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

TAGS_URL = "https://data.jsdelivr.com/v1/packages/npm/@fawazahmed0/currency-api"
FILE_TMPL = "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{ver}/v1/currencies/usd.json"

TIMEOUT = 30
MAX_WORKERS = 8
MAX_RETRIES = 4


def iso(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def get_tags() -> dict[str, str]:
    r = requests.get(TAGS_URL, timeout=TIMEOUT)
    r.raise_for_status()
    tags = r.json().get("tags", {})
    # Keep only real date keys like "2024-03-02"; drop special tags ("latest").
    return {k: v for k, v in tags.items() if len(k) == 10 and k[4] == "-" and k[7] == "-"}


def fetch_one(date_str: str, version: str) -> tuple[str, float | None, float | None]:
    """Return (date_str, jpy_per_usd, bdt_per_usd). Nones on failure (after retries)."""
    url = FILE_TMPL.format(ver=version)
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            if r.status_code == 404:
                return date_str, None, None
            r.raise_for_status()
            data = r.json()
            jpy = data["usd"]["jpy"]
            bdt = data["usd"]["bdt"]
            if not isinstance(jpy, (int, float)) or not isinstance(bdt, (int, float)):
                return date_str, None, None
            return date_str, float(jpy), float(bdt)
        except Exception as e:  # noqa: BLE001
            wait = 1.5 * (attempt + 1)
            print(f"  [retry {attempt + 1}/{MAX_RETRIES}] {date_str} {version}: {e}", file=sys.stderr)
            time.sleep(wait)
    return date_str, None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-03-02")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--out", default="data/jpy_bdt_daily.csv")
    args = ap.parse_args()

    end = dt.date.today() if not args.end else iso(args.end)
    start = iso(args.start)
    if end <= start:
        raise SystemExit("--end must be after --start")

    print(f"Resolving per-day npm versions from jsDelivr …")
    tags = get_tags()
    print(f"  found {len(tags)} tagged days")

    days = [d for d in daterange(start, end)]
    print(f"Fetching {len(days)} days ({start} … {end}) with {MAX_WORKERS} workers …")

    rows: dict[str, tuple[float | None, float | None]] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {
            ex.submit(fetch_one, d.isoformat(), tags[d.isoformat()]): d.isoformat()
            for d in days
            if d.isoformat() in tags
        }
        missing_dates = [d.isoformat() for d in days if d.isoformat() not in tags]
        done = 0
        for fut in as_completed(futs):
            date_str, jpy, bdt = fut.result()
            rows[date_str] = (jpy, bdt)
            done += 1
            if done % 100 == 0 or done == len(futs):
                print(f"  {done}/{len(futs)} days done")

    ok = sum(1 for v in rows.values() if v[0] is not None and v[1] is not None)
    fail = [d for d, v in rows.items() if v[0] is None or v[1] is None]
    print(f"Fetched {ok} rate pairs, {len(fail)} failed, {len(missing_dates)} missing versions")

    # Forward-fill missing/failed days with the last known rate (rates are flat
    # on weekends/holidays anyway). Keeps a continuous daily series for the model.
    last: tuple[float, float] | None = None
    filled = 0
    series: list[tuple[dt.date, float, float]] = []
    for d in days:
        r = rows.get(d.isoformat())
        if r is None or r[0] is None or r[1] is None:
            r = last
            if r is not None:
                filled += 1
        if r is not None:
            last = r
            series.append((d, r[0], r[1]))

    print(f"Forward-filled {filled} days (weekends/holidays/gaps). Final series: {len(series)} rows")

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "jpy_to_bdt", "bdt_to_jpy", "usd_to_jpy", "usd_to_bdt"])
        for d, jpy, bdt in series:
            jb = bdt / jpy
            w.writerow([d.isoformat(), f"{jb:.6f}", f"{1.0 / jb:.6f}", f"{jpy:.6f}", f"{bdt:.6f}"])
    print(f"Saved -> {args.out}")
    if series:
        d, jpy, bdt = series[0]
        print(f"  first: {d}  JPY/BDT={bdt / jpy:.4f}")
        d, jpy, bdt = series[-1]
        print(f"  last : {d}  JPY/BDT={bdt / jpy:.4f}")


if __name__ == "__main__":
    main()
