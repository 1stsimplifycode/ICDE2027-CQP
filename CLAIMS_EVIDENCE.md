# Claim -> evidence map (all numbers in the PDF are macros/tables generated from these files)

| Claim (section) | Evidence file | Generator |
|---|---|---|
| Zero-budget certification, imputation error (Tab. I, V-B) | results/acq_real_main.json | src/make_numbers.py |
| SEAQ best AUC in all 9 cells, paired CIs (Tab. II) | results/acq_{real_main,synth_main,real_hetero}.json | src/make_table.py (src/analyze.py) |
| Stream replay: wrong-at-emission, delay, reversals, throughput (Tab. III) | results/stream_s7_l0.3_n0.3_r0.002.json | src/make_numbers.py |
| Error sources: bound violations vs restatements (Tab. IV) | results/stream_s7_*.json (4 settings) | src/make_extra.py |
| Fan-out benchmark (Tab. V) | results/fanout.json | src/make_extra.py |
| Lemma 1 / reduction: no counterexample in randomized search | results/theory_check.json | src/verify_theory.py |
| SEAQ score frozen on dev data before real runs | DEV_LOG.md (D0-D2, SHA-256) | results/acq_synth_dev_lam*.json |
| v1 manuscript and outputs preserved | v1_backup/ | - |

Not verified: bibliography entries were not machine-checked against dblp (dblp blocked automated access); page numbers
removed for the two entries we were least sure of. Public artifact URL: none yet.
| Exact GR planner = brute force (3,000 instances) | console output of src/wcdp.py | src/wcdp.py |
| Exact plan vs heuristics, up to 1.9x at 2% budget (Tab. VI) | results/wc.json | src/run_wc.py, src/make_wc.py |
| Completion/hybrid scores lose to SEAQ (exploratory, post-freeze) | results/acq_real_r3.json, results/fanout_r3.json | src/run_acq.py, src/run_fanout.py |
| v2 manuscript preserved | v2_backup/ | - |
| Value-indexed DP strongly polynomial; = brute force (2,000 real-cost, integer, 300 two-key), ILP = brute force (200) | results/dp_verification.json | src/wcdp_sp.py |
| Roll-ups: ILP optimal 15-30 s; window-DP plan >= 99.4% of optimum; greedy 66-74% (Tab. VII) | results/rollup.json | src/run_rollup.py, src/make_rollup.py |
| v3 manuscript preserved | v3_backup/ | - |
| CQP prototype: 8/8 tests pass in fresh py3.12 venv (pinned) | results/cqp_tests_log.txt | cqp/test_cqp.py |
| Deterministic demo: 5 retractions all contradiction-flagged, 0 certified wrong, inc=full | cqp/demo_out/verification.json, results/cqp_demo_log.txt | cqp/demo.py |
| DuckDB benchmark: 181,412 decisions, inc p50/p95/p99 vs full recompute | results/cqp_bench_120d.json | cqp/bench.py |
| Dockerfile: NOT built/tested | Dockerfile | - |
| v4 manuscript preserved | v4_backup/ | - |
