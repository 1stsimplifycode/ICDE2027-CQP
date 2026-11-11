# Certified Threshold Alerts over Incomplete Multi-Source Streams — ICDE 2027 artifact

Paper: `ICDE2027_Research_Submission_FINAL.pdf` (LaTeX source in `paper/`; all tables and graphs are generated from `results/`).
Authors: Adishree Gupta, Akshat, Shylaja S S (Dept. of CSE, PES University, Bengaluru).

## Layout
| Path | Contents |
|---|---|
| `cqp/` | CQP: DuckDB-backed certified decision-query processor (`engine.py`), SQL SEAQ + exact planner (`acquire_sql.py`), tests (`test_cqp.py`, `test_numeric.py`), deterministic demo (`demo.py`, outputs in `cqp/demo_out/`), benchmark (`bench.py`, `bench_repeat.sh`, `make_repeat_table.py`) |
| `src/` | Algorithms and experiments: operator (`cert.py`), policies (`acquire.py`), exact GR planners (`wcdp.py`, `wcdp_sp.py`), workloads, experiment runners, verification (`verify_exact.py`, `verify_theory.py`), table/graph generators |
| `results/` | Raw outputs (JSON) and logs of every reported experiment |
| `paper/` | LaTeX source, bibliography, generated tables and graphs |
| `data_fetch/` | Download scripts and SHA-256 checksums for the public NSE/BSE files (data not redistributed) |
| `MANIFEST.sha256` | Checksums of all artifact files |
| `CLAIMS_EVIDENCE.md`, `AUDIT_MATRIX.csv`, `FINAL_REPORT.md` | Claim-to-evidence map, audit matrix, executed-command report |
| `DEV_LOG.md`, `DUCKDB_COMPAT.md` | Development/freeze log, DuckDB version notes |

## Quick start (no external data needed)
Tested with Python 3.12 and the pinned `requirements.txt` (DuckDB 1.1.3; DuckDB 2.0.0 is known to fail, see `DUCKDB_COMPAT.md`).
```
python3.12 -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt          # Linux/macOS: .venv/bin/python
.venv/Scripts/python -m pytest -q cqp/test_cqp.py cqp/test_numeric.py   # 16 tests
.venv/Scripts/python cqp/demo.py                                  # prints DEMO OK, writes cqp/demo_out/
python src/verify_exact.py                                        # exact-arithmetic check of Theorem 2 (0 mismatches)
python src/wcdp_sp.py                                             # value-indexed DP and ILP vs brute force
```

## Real-panel experiments (need the public exchange files)
Download the NSE/BSE bhavcopies with `data_fetch/` and build the derived panels; set the path `ICRITO` in `src/workload.py`.
```
PY=.venv/Scripts/python bash cqp/bench_repeat.sh   # Table VIII / Fig. 3: 5 runs x 3 modes, 60 days
python cqp/make_repeat_table.py && python src/make_graphs.py
cd src && python run_acq.py real 8 --tag main && python run_acq.py synth 12 --seed 3 --tag main \
  && python run_acq.py real 12 --costs hetero --tag hetero && python run_stream.py 7 \
  && python run_wc.py && python run_rollup.py && python run_fanout.py
python make_numbers.py && python make_table.py && python make_extra.py && python make_wc.py && python make_rollup.py
cd ../paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
The Dockerfile has not been built or tested.
