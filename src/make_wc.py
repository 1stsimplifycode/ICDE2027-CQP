"""paper/tab_wc.tex from results/wc.json (exact guaranteed-resolution planning vs heuristics)."""
import json
import numpy as np
W = json.load(open(r"C:\Users\mailt\Downloads\ICDE-2027\results\wc.json"))
N = {"greedy_complete": "Greedy cheapest-completion", "comp": "Completion score", "seaq": r"\seaq{} order",
     "widest": "Widest order", "random": "Random"}
L = [r"\begin{table}[t]\centering\caption{Guaranteed-resolution planning on the real panel (stock-level windows, 40-day backlog, 5 days per scenario). Row 1: optimum number of decisions guaranteed resolved (exact DP, Thm.~\ref{thm:dp}), mean over days. Other rows: fraction of the optimum achieved by each non-adaptive plan at the same budget (budget = fraction of total uncertain-item cost).}\label{tab:wc}",
     r"\scriptsize\setlength{\tabcolsep}{2.2pt}\begin{tabular}{llccccc}\toprule",
     r"Scen. & Plan & 2\% & 5\% & 10\% & 20\% & 40\%\\", r"\midrule"]
for s in ["late", "mcar"]:
    rs = [r for r in W if r["scen"] == s]
    L.append(r"%s & \textbf{Exact DP (count)} & %s\\" % (s.upper(), " & ".join("%.0f" % np.mean([r["opt"][i] for r in rs]) for i in range(5))))
    for k, v in N.items():
        L.append(r" & %s & %s\\" % (v, " & ".join("%.2f" % np.mean([r[k][i] / r["opt"][i] for r in rs]) for i in range(5))))
    L.append(r" & \multicolumn{6}{l}{decisions: %.0f; exact DP time: %.2f s}\\" % (np.mean([r["n_dec"] for r in rs]), np.mean([r["t_exact"] for r in rs])))
    L.append(r"\midrule")
L[-1] = r"\bottomrule\end{tabular}\end{table}"
open(r"C:\Users\mailt\Downloads\ICDE-2027\paper\tab_wc.tex", "w").write("\n".join(L) + "\n")
print("ok")
