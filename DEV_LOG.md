# Development log and freeze (ICDE 2027 study)

D0 (2026-11-11 ~21:55 IST). Original design of SEAQ: score = sum_d [P1(d,i) + lam*w_i/W_d]/c_i with lam = 0.5,
chosen a priori. Smoke test on synthetic seed 2027 (1 episode, MCAR) only, file results/acq_synth_smoke.json.
NOTE: seed 2027 synthetic was therefore touched once in a smoke test; confirmatory synthetic runs use seed 3 instead.

D1 (dev, synthetic seed 1, 3 episodes x 3 scenarios). lam sweep {0.25,0.5,1,2,4,8} + lam=inf (P1 removed):
AUC increased monotonically with lam in all three scenarios; lam=inf best (MCAR AUC 0.684 vs 0.642 at lam=0.5).
Decision: the main method SEAQ is frozen as the pure shared-progress score sum_d (w_i/W_d)/c_i.
The original lam=0.5 form is kept as ablation `seaq_myopic`; P1 alone is the baseline `voi1`.
Files: results/acq_synth_dev_lam*.json.

Freeze: after D1, no parameter of any policy is changed. Confirmatory runs: real NSE+BSE panel (never run before
this point), synthetic seed 3, 12 episodes per scenario, unit costs; plus a heterogeneous-cost run on real data.
147ba26c388a60a3b2705c888eeeb24cfa6203681f17e0f9121a0f021940753b *acquire.py
0d875255f6c0d3cbacde47bfd76bfb1e7647f133c3f38e51510f79764f9eb021 *cert.py
6c78d7f92426b3496c2744d40e147d422e04aa0b71a157ae560aea784c100dd9 *workload.py
87e6077a15bfd2c850d5728973404afcdd636c814223ee2473910f7cdb905944 *run_acq.py
027b363a7138345d86a8dc3140679a29def131005f581d506baacf164e52be93 *analyze.py

D2 (post-freeze bug fix, no parameter change). workload.load_real crashed (pandas DataFrame.isin attribute clash: ind.isin -> ind['isin']). Fixed before any real-data result existed. New workload.py sha256: $(sha256sum workload.py)
1c2d3840ba4be218eff10ef91b986188c043edbefab204ad1342db2071ae711e *workload.py

D3 (2026-11-11 ~22:12-22:16 IST, round 2). v1 manuscript/results backed up to v1_backup/. New experiments, all post-hoc and labelled as such:
fan-out benchmark (src/run_fanout.py, results/fanout.json): refutes "sharing gain grows with fan-out";
soundness ablation (run_stream.py with P_RESTATE=0 and/or CAP_MULT=10): certified errors need restatements or bound violations;
theory check (src/verify_theory.py): 0 counterexamples. Fixes: AUC definition stated (mean over 51-point budget grid; code unchanged),
\textbf rendering bug in table generator, author block, conditional soundness wording. SEAQ unchanged; no reported v1 number changed.

D4 (2026-11-11 ~22:20-22:25 IST, round 3). v2 backed up to v2_backup/. Added: exact GR planner (src/wcdp.py, verified vs brute force,
0/3000 mismatches), real-panel GR experiment (src/run_wc.py -> results/wc.json), exploratory completion/hybrid policies
(acquire.py 'comp','hybrid'; real episodes reused -> exploratory only): both lose to SEAQ. SEAQ unchanged. No earlier number changed.

D5 (2026-11-11 ~22:30-22:37 IST, round 4). v3 backed up to v3_backup/. Complexity corrected: budget-indexed DP is pseudo-polynomial;
new value-indexed DP (src/wcdp_sp.py) is strongly polynomial, verified vs brute force (results/dp_verification.json, 0 mismatches).
Roll-ups: exact ILP (HiGHS) reference (src/run_rollup.py -> results/rollup.json); complexity of laminar roll-ups left open.

D6 (2026-11-11 ~22:41-22:50 IST, round 5). v4 backed up. Built cqp/ (DuckDB-backed prototype), tests, demo, bench.
DuckDB 2.0.0 (py3.14) raised an internal error ("dict-surviving cache") on the full-panel benchmark; switched to a fresh
py3.12 venv (.venv_clean) with pinned requirements.txt (duckdb 1.1.3), where tests/demo/bench all ran.
