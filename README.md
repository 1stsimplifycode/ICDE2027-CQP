# ICDE 2027 submission: Certified threshold alerts + shared-evidence acquisition (SEAQ)

- Paper: `ICDE2027_Research_Submission.pdf` (source `paper/main.tex`, `paper/refs.bib`; numbers in `paper/numbers.tex` and `paper/tab_acq.tex` are generated).
- Code: `src/cert.py` (operator), `src/acquire.py` (policies), `src/workload.py`, `src/run_acq.py`, `src/run_stream.py`, `src/analyze.py`, `src/make_numbers.py`, `src/make_table.py`.
- Protocol: `DEV_LOG.md` (dev-only tuning, freeze, hashes, post-freeze loader bug fix D2).
- Data: public NSE/BSE bhavcopies, not redistributed; `data_fetch/` holds the download scripts, the panel builder and SHA-256 checksums. The code expects the derived parquet files at the ICRITO-2027 path set in `src/workload.py` (edit `ICRITO`).

Reproduce (about 10 minutes on a laptop):
```
cd src
python run_acq.py real 8 --tag main
python run_acq.py synth 12 --seed 3 --tag main
python run_acq.py real 12 --costs hetero --tag hetero
python run_stream.py 7
python make_numbers.py && python make_table.py
python run_fanout.py && python verify_theory.py && python make_extra.py
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## CQP prototype (DuckDB-backed, round 5)
```
python3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # pinned; duckdb 1.1.3
.venv/Scripts/python -m pytest -q cqp/test_cqp.py      # 8 tests
.venv/Scripts/python cqp/demo.py                         # deterministic demo -> cqp/demo_out/
.venv/Scripts/python cqp/bench.py 120                    # real panel (needs exchange files, see data_fetch/)
```
Verified on 2026-11-11 in a freshly created venv. Docker image (Dockerfile) not built or tested.
