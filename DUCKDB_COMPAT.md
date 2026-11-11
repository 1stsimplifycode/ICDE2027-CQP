# DuckDB compatibility report
- Verified: DuckDB 1.1.3 on Python 3.12.x (fresh venv `.venv_clean`, `requirements.txt`): 16/16 tests, demo, 15 benchmark runs.
- Failing: DuckDB 2.0.0 on Python 3.14 (system interpreter). Earlier `cqp/bench.py 120` on the full panel raised
  `_duckdb.InternalException: INTERNAL Error: dict-surviving cache: global dictionary column 3 changed identity after being
  seeded for dictionary caching` inside the delta `UPDATE ... FROM` / `CREATE TEMP TABLE upd AS SELECT ... JOIN decisions`
  statement. The same code passed all small tests under 2.0.0. Switching the VARCHAR level column to an integer code and
  materialising the update set did not remove it. DuckDB reports this class as an internal assertion failure.
- Not done: a minimal standalone reproduction; testing other DuckDB versions (no Python 3.14 wheels for 1.x; 1.4.1 failed to build).
- Policy: pin duckdb==1.1.3; no compatibility claim for other versions.
