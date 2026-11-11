"""paper/tab_cqp.tex from results/cqp_bench_120d_{inc,inc_scan,full}.json"""
import json, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "tab_cqp.tex")
names = [("inc_scan", r"Incremental bounds, transition scan"), ("inc", r"Incremental bounds, affected-only transitions"),
         ("full", r"Full SQL recomputation, same semantics")]
O = {m: json.load(open(os.path.join(R, f"cqp_bench_120d_{m}.json"))) for m, _ in names}
same = len({str(O[m]["state_digest"]) + str(O[m]["log_counts"]) for m, _ in names}) == 1
L = [r"\begin{table}[t]\centering\caption{CQP on DuckDB~%s, real panel (%d stocks, %d days, 8 queries, %s decisions, %d daily batches of about %.0f rows). Per-batch latency of the complete ingest transaction (contradiction check, bound update, transitions, emission/retraction logging). All three modes produce %s final states and logs (state digest and log counts). Bounds-only full recomputation (no transitions or logs): p50 %.0f\,ms.}\label{tab:cqp}" % (
        O["inc"]["duckdb"], O["inc"]["K"], O["inc"]["days"], "{:,}".format(O["inc"]["decisions"]), O["inc"]["batches"], O["inc"]["rows_per_batch_mean"],
        "identical" if same else "DIFFERENT", O["inc"]["boundsonly_p50_ms"]),
     r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lrrrr}\toprule", r"Mode & p50 & p95 & p99 (ms) & rows/s\\", r"\midrule"]
for m, n in names:
    o = O[m]; L.append(r"%s & %.0f & %.0f & %.0f & %.0f\\" % (n, o["inc_p50_ms"], o["inc_p95_ms"], o["inc_p99_ms"], o["inc_rows_per_s"]))
L.append(r"\bottomrule\end{tabular}\end{table}")
open(P, "w").write("\n".join(L) + "\n"); print("\n".join(L)); print("identical:", same)
