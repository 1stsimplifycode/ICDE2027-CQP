"""Writes paper/tab_acq.tex: full table environment (real and synthetic side by side)."""
import sys
sys.path.insert(0, r"C:\Users\mailt\Downloads\ICDE-2027\src")
from analyze import summarize
R = r"C:\Users\mailt\Downloads\ICDE-2027\results"
NAMES = {"random": "Random", "widest": "Widest (TRAPP-style)", "degree": "Degree", "local": "Local (per-query)",
         "voi1": "Myopic VOI", "seaq": r"\textbf{SEAQ}", "seaq_myopic": "SEAQ+VOI (abl.)", "seaq_max": "SEAQ-noshare (abl.)"}
SC = {"mcar": "MCAR", "late": "LATE", "mnar": "MNAR"}
_, A = summarize(f"{R}/acq_real_main.json"); _, B = summarize(f"{R}/acq_synth_main.json"); _, H = summarize(f"{R}/acq_real_hetero.json")
def cell(r):
    d = "--" if "dAUC" not in r else "%+.3f [%.3f,%.3f]" % (r["dAUC"], r["dAUC_lo"], r["dAUC_hi"])
    return "%.3f & %.3f & %s" % (r["AUC"], r["C90"], d)
L = [r"\begin{table*}[t]\centering", r"\caption{Resolution vs.\ budget (budget = fraction of WAIT). AUC = mean resolved fraction over budgets (51-point grid on) $[0,0.5]$ (higher is better), C90 = budget for 90\% resolution (lower is better), and paired bootstrap 95\% CI of AUC(\seaq)$-$AUC(policy) over episodes. Real: NSE+BSE panel, 8 episodes/scenario; Synthetic: 12 episodes/scenario; Real-het: heterogeneous probe costs (primary-exchange item 3, secondary 1), 12 episodes.}\label{tab:acq}",
     r"\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llccl|ccl|ccl}\toprule",
     r" & & \multicolumn{3}{c|}{Real (unit cost)} & \multicolumn{3}{c|}{Synthetic (unit cost)} & \multicolumn{3}{c}{Real-het (cost 3/1)}\\",
     r"Scen. & Policy & AUC & C90 & $\Delta$AUC [95\% CI] & AUC & C90 & $\Delta$AUC [95\% CI] & AUC & C90 & $\Delta$AUC [95\% CI]\\"]
for s in ["mcar", "late", "mnar"]:
    L.append(r"\midrule")
    for p in NAMES:
        L.append("%s & %s & %s & %s & %s \\\\" % (SC[s], NAMES[p], cell(A[(s, p)]), cell(B[(s, p)]), cell(H[(s, p)])))
L += [r"\bottomrule\end{tabular}\end{table*}"]
open(r"C:\Users\mailt\Downloads\ICDE-2027\paper\tab_acq.tex", "w").write("\n".join(L) + "\n")
for s in ["mcar","late","mnar"]:
    best = max((H[(s,p)]["AUC"],p) for p in NAMES); print("hetero",s,best)
