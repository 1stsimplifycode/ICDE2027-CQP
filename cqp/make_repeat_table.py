"""Summarise repeated benchmark runs (results/cqp_bench_60d_<mode>_rep<r>.json) -> paper/tab_cqp.tex + results/bench_repeat_summary.json.
Unit of replication = run (fresh process + fresh DB); batches within a run are NOT treated as replicates."""
import json, os, glob
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tab_cqp.tex")
modes = [("inc_scan", "Incremental, transition scan"), ("inc", "Incremental, affected-only"), ("full", "Full SQL recompute, same semantics")]
runs = {m: [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"cqp_bench_60d_{m}_rep*.json")))] for m, _ in modes}
S = {}
for m, _ in modes:
    rs = runs[m]
    S[m] = {"n_runs": len(rs), "p50": [r["inc_p50_ms"] for r in rs], "p95": [r["inc_p95_ms"] for r in rs],
            "p99": [r["inc_p99_ms"] for r in rs], "mean": [float(np.mean(r["raw_batch_ms"])) for r in rs],
            "total_s": [r["total_wall_s"] for r in rs], "rows_s": [r["inc_rows_per_s"] for r in rs],
            "digest": sorted({str(r["state_digest"]) + str(r["log_counts"]) for r in rs}), "exact": [r["exact"] for r in rs],
            "err": [r["max_incremental_error"] for r in rs]}
all_digests = {d for m, _ in modes for d in S[m]["digest"]}
sp = [f / i for f, i in zip(S["full"]["total_s"], S["inc_scan"]["total_s"])]
sp50 = [f / i for f, i in zip(S["full"]["p50"], S["inc_scan"]["p50"])]
summary = {"modes": S, "identical_semantics_all_runs": len(all_digests) == 1,
           "speedup_total_full_over_incscan_per_rep": sp, "speedup_p50_per_rep": sp50}
json.dump(summary, open(os.path.join(R, "bench_repeat_summary.json"), "w"), indent=1)
r0 = runs["inc"][0]
rng = lambda a: "%.0f [%.0f--%.0f]" % (np.median(a), min(a), max(a))
L = [r"\begin{table*}[t]\centering\caption{Repeated equivalent-functionality benchmark on DuckDB~%s, exact integer arithmetic (%d stocks, %d days, 8 queries, %s decisions, %d daily batches). %d runs per mode; each run is a fresh process and a fresh database, and the mode order is rotated across runs. Entries: median over runs [min--max over runs] of the per-run statistic. All %d runs produced %s state digests and log counts. Speedup of total time, full over incremental-scan: %.2f [%.2f--%.2f].}\label{tab:cqp}" % (
        r0["duckdb"], r0["K"], r0["days"], "{:,}".format(r0["decisions"]), r0["batches"], len(runs["inc"]),
        sum(len(runs[m]) for m, _ in modes), "identical" if summary["identical_semantics_all_runs"] else "DIFFERENT",
        np.median(sp), min(sp), max(sp)),
     r"\scriptsize\setlength{\tabcolsep}{1.5pt}\begin{tabular}{lcccc}\toprule", r"Mode & p50 (ms) & p95 (ms) & p99 (ms) & total (s)\\", r"\midrule"]
for m, n in modes:
    L.append(r"%s & %s & %s & %s & %s\\" % (n, rng(S[m]["p50"]), rng(S[m]["p95"]), rng(S[m]["p99"]),
                                            "%.1f [%.1f--%.1f]" % (np.median(S[m]["total_s"]), min(S[m]["total_s"]), max(S[m]["total_s"]))))
L.append(r"\bottomrule\end{tabular}\end{table*}")
open(P, "w").write("\n".join(L) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k != "modes"}), {m: (np.median(S[m]["p50"]), np.median(S[m]["p95"]), np.median(S[m]["total_s"])) for m, _ in modes})
