"""Tables for the soundness ablation (stream) and the fan-out benchmark -> paper/tab_sound.tex, paper/tab_fanout.tex"""
import json, os
import numpy as np
R = r"C:\Users\mailt\Downloads\ICDE-2027\results"; P = r"C:\Users\mailt\Downloads\ICDE-2027\paper"
rows = [(r"$3\times$, restat. (main)", "stream_s7_l0.3_n0.3_r0.002.json"), (r"$3\times$, no restat.", "stream_s7_l0.3_n0.3_r0.0.json"),
        (r"$10\times$, restat.", "stream_s7_l0.3_n0.3_r0.002_cap10.0.json"), (r"$10\times$, no restat.", "stream_s7_l0.3_n0.3_r0.0_cap10.0.json"),
        (r"$1000\times$, no restat.", "stream_s7_l0.3_n0.3_r0.0_cap1000.0.json")]
L = [r"\begin{table}[t]\centering\caption{Where certified errors come from (stream replay, seed 7). Bound = declared domain bound for unreported items (multiple of calibration max). Violating items = items whose true value exceeds the bound.}\label{tab:sound}",
     r"\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{lrrrrr}\toprule", r"Bound, restat. & Viol. & CERT err & on-time \% & delay (d) & WAIT err\\", r"\midrule"]
for lab, f in rows:
    if not os.path.exists(f"{R}/{f}"): continue
    o = json.load(open(f"{R}/{f}")); c = o["cert"]
    L.append("%s & %d & %d & %.1f & %.2f & %d\\\\" % (lab, o["bound_violations"], c["wrong_at_emission"], 100 * c["on_time"] / c["emitted"], c["mean_delay_days"], o["wait"]["wrong_at_emission"]))
L.append(r"\bottomrule\end{tabular}\end{table}")
open(f"{P}/tab_sound.tex", "w").write("\n".join(L) + "\n")

if os.path.exists(f"{R}/fanout.json"):
    F = json.load(open(f"{R}/fanout.json"))
    def auc(rec, p):
        c = np.array(rec["curves"][p]); base = c[0, 1]; need = 1024 - base; W = rec["wait"]
        g = np.linspace(0, .5, 51)
        return np.mean([(c[np.flatnonzero(c[:, 0] <= b * W + 1e-9)[-1], 1] - base) / need for b in g])
    def c90(rec, p):
        c = np.array(rec["curves"][p]); base = c[0, 1]; need = 1024 - base
        i = np.flatnonzero(c[:, 1] >= base + .9 * need - 1e-9); return c[i[0], 0] / rec["wait"] if len(i) else 1.0
    pols = ["random", "widest", "degree", "voi1", "seaq_max", "seaq"]
    L = [r"\begin{table*}[t]\centering\caption{Controlled fan-out benchmark: 1024 decisions of 32 items each, every item feeds exactly $f$ decisions (5 instances per $f$). Mean AUC; last columns: AUC gain of \seaq{} over SEAQ-noshare and over myopic VOI with min--max over instances, \seaq{} C90, and \seaq{} runtime.}\label{tab:fanout}",
         r"\scriptsize\setlength{\tabcolsep}{2pt}\begin{tabular}{rrcccccccccr}\toprule",
         r"$f$ & items & Rand & Widest & Deg & VOI & noshare & \seaq & $\Delta$noshare & $\Delta$VOI & C90 & s\\", r"\midrule"]
    for f in sorted(set(r["f"] for r in F)):
        rs = [r for r in F if r["f"] == f]
        A = {p: np.array([auc(r, p) for r in rs]) for p in pols}
        dn = A["seaq"] - A["seaq_max"]; dv = A["seaq"] - A["voi1"]
        L.append("%d & %d & %s & %+.3f [%+.3f,%+.3f] & %+.3f [%+.3f,%+.3f] & %.3f & %.1f\\\\" % (
            f, rs[0]["n_items"], " & ".join("%.3f" % A[p].mean() for p in pols), dn.mean(), dn.min(), dn.max(), dv.mean(), dv.min(), dv.max(),
            np.mean([c90(r, "seaq") for r in rs]), np.mean([r["secs"]["seaq"] for r in rs])))
    L.append(r"\bottomrule\end{tabular}\end{table*}")
    open(f"{P}/tab_fanout.tex", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
