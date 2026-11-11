"""paper/tab_rollup.tex from results/rollup.json."""
import json
import numpy as np
R = json.load(open(r"C:\Users\mailt\Downloads\ICDE-2027\results\rollup.json"))
P = [("milp", r"Exact ILP (HiGHS)"), ("dp_stock", r"Window DP, stock only"),
     ("greedy", "Greedy completion"), ("seaq", r"\seaq{} order"), ("widest", "Widest order")]
cap = (r"\caption{Guaranteed-resolution planning with industry roll-ups (real panel, LATE, 40-day backlog, 3 days). "
       r"Entries: stock + industry decisions guaranteed (mean over days), ratio to the exact optimum in parentheses. "
       r"The ILP proved optimality on all 6 solves in %.0f--%.0f\,s; the window DP (Thm.~\ref{thm:dp}) takes %.1f\,s, "
       r"the value-indexed DP with real-valued costs %.2f\,s. Mean decisions: %.0f stock + %.0f industry.}\label{tab:rollup}") % (
    min(min(r["milp_secs"]) for r in R), max(max(r["milp_secs"]) for r in R), np.mean([r["dp_secs"] for r in R]),
    np.mean([r["value_dp_real_secs"] for r in R]), np.mean([r["n_stock"] for r in R]), np.mean([r["n_ind"] for r in R]))
L = [r"\begin{table}[t]\centering" + cap, r"\scriptsize\setlength{\tabcolsep}{3pt}\begin{tabular}{lcc}\toprule",
     r"Plan & 5\% budget & 20\% budget\\", r"\midrule"]
for k, name in P:
    cells = []
    for i in range(2):
        tot = np.mean([sum(r[k][i]) for r in R]); ind = np.mean([r[k][i][1] for r in R])
        opt = np.mean([sum(r[k][i]) / sum(r["milp"][i]) for r in R])
        cells.append("%.0f+%.0f (%.3f)" % (tot - ind, ind, opt))
    L.append("%s & %s\\\\" % (name, " & ".join(cells)))
L.append(r"\bottomrule\end{tabular}\end{table}")
open(r"C:\Users\mailt\Downloads\ICDE-2027\paper\tab_rollup.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
