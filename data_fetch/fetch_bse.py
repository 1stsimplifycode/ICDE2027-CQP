"""Download official BSE equity bhavcopies (UDiFF format) for a date range.

Stored gzip-compressed in data/raw/bse/<YYYYMMDD>.csv.gz; the manifest SHA-256 is of the *uncompressed* bytes as served.
Usage: python src/fetch_bse.py 2024-07-08 2026-11-06
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import os
import subprocess
import sys
import time
from datetime import date, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "raw", "bse")
MANIFEST = os.path.join(ROOT, "data", "raw", "bse_manifest.csv")
URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{d}_F_0000.CSV"


ALL_DAYS = "--all-days" in sys.argv  # weekend special sessions (e.g. Budget Saturday, Muhurat)


def main(start: str, end: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    seen = {r["date"] for r in csv.DictReader(open(MANIFEST))} if os.path.exists(MANIFEST) else set()
    new = not os.path.exists(MANIFEST)
    mf = open(MANIFEST, "a", newline="")
    w = csv.writer(mf)
    if new:
        w.writerow(["date", "status", "bytes", "sha256", "fetched_utc"])
    d, e = date.fromisoformat(start), date.fromisoformat(end)
    tmp = os.path.join(OUT, "_tmp.csv")
    while d <= e:
        ds = d.strftime("%Y%m%d")
        if (d.weekday() < 5 or ALL_DAYS) and ds not in seen:
            code = "000"
            for attempt in range(4):
                r = subprocess.run(["curl", "-s", "--max-time", "60", "-A", "Mozilla/5.0", "-H",
                                    "Referer: https://www.bseindia.com/", "-o", tmp, "-w", "%{http_code}",
                                    URL.format(d=ds)], capture_output=True, text=True)
                code = r.stdout.strip() or "000"
                if code in ("200", "404"):
                    break
                time.sleep(2 ** attempt)
            body = open(tmp, "rb").read() if os.path.exists(tmp) else b""
            ok = code == "200" and body.startswith(b"TradDt")
            if ok:
                with gzip.open(os.path.join(OUT, f"{ds}.csv.gz"), "wb") as g:
                    g.write(body)
            w.writerow([ds, 200 if ok else (code if code != "200" else "nonbhav"), len(body) if ok else 0,
                        hashlib.sha256(body).hexdigest() if ok else "",
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())])
            mf.flush()
            time.sleep(0.7)
        d += timedelta(days=1)
    if os.path.exists(tmp):
        os.remove(tmp)
    mf.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
