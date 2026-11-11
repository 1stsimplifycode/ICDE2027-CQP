"""Download official NSE cash-market bhavcopies (UDiFF format) for a date range.

Each trading day's file is stored verbatim (zip) in data/raw/bhav/ and a manifest row with the SHA-256 is appended.
Non-trading days return 404 and are recorded as such. Resumable: existing files are skipped.
Usage: python src/fetch_bhav.py 2024-07-08 2026-11-06
"""
from __future__ import annotations

import csv
import hashlib
import os
import sys
import time
from datetime import date, timedelta

import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "bhav")
MANIFEST = os.path.join(ROOT, "data", "raw", "bhav_manifest.csv")
URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d}_F_0000.csv.zip"
HDR = {"User-Agent": "Mozilla/5.0", "Accept": "*/*"}  # a longer UA string is stalled by the CDN


ALL_DAYS = "--all-days" in sys.argv  # weekend special sessions (e.g. Budget Saturday, Muhurat)


def main(start: str, end: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    seen = set()
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            seen = {r["date"] for r in csv.DictReader(f)}
    new = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    w = csv.writer(mf)
    if new:
        w.writerow(["date", "status", "bytes", "sha256", "fetched_utc"])
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    while d <= e:
        ds = d.strftime("%Y%m%d")
        if (d.weekday() < 5 or ALL_DAYS) and ds not in seen:
            p = os.path.join(OUT, f"{ds}.csv.zip")
            code = "000"
            for attempt in range(4):
                # NSE's CDN stalls python-requests; curl is used (TLS/HTTP fingerprint), see research/execution_log.md
                r = subprocess.run(["curl", "-s", "--max-time", "40", "-A", HDR["User-Agent"], "-o", p,
                                    "-w", "%{http_code}", URL.format(d=ds)], capture_output=True, text=True)
                code = r.stdout.strip() or "000"
                if code in ("200", "404"):
                    break
                time.sleep(2 ** attempt)
            body = open(p, "rb").read() if os.path.exists(p) else b""
            if code == "200" and body[:2] == b"PK":
                w.writerow([ds, 200, len(body), hashlib.sha256(body).hexdigest(),
                            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
            else:
                if os.path.exists(p):
                    os.remove(p)
                w.writerow([ds, code, 0, "", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
            mf.flush()
            time.sleep(0.7)
        d += timedelta(days=1)
    mf.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
