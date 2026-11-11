"""Generate paper/numbers.tex and paper/tab_acq.tex from results/ (no hand-typed numbers)."""
import json, glob, sys
sys.path.insert(0, r"C:\Users\mailt\Downloads\ICDE-2027\src")
from analyze import summarize
R = r"C:\Users\mailt\Downloads\ICDE-2027\results"
P = r"C:\Users\mailt\Downloads\ICDE-2027\paper"
mac = []
def m(name, val, fmt="{:.1f}"):
    mac.append("\\newcommand{\\%s}{%s}" % (name, fmt.format(val) if not isinstance(val, str) else val))

NAMES = {"random": "Random", "widest": "Widest (TRAPP-style)", "degree": "Degree", "local": "Local (per-query)",
         "voi1": "Myopic VOI", "seaq": "\\textbf{SEAQ}", "seaq_myopic": "SEAQ+VOI (abl.)", "seaq_max": "SEAQ-noshare (abl.)"}
SC = {"mcar": "MCAR", "late": "LATE", "mnar": "MNAR"}
tab = []
for wl, tag, lab in [("real", "main", "Real"), ("synth", "main", "Synth"), ("real", "hetero", "RealHet")]:
    f = f"{R}/acq_{wl}_{tag}.json"
    try:
        o, rows = summarize(f)
    except FileNotFoundError:
        print("missing", f); continue
    m(f"K{lab}", o["K"], "{}"); m(f"Ep{lab}", len(o["days"]), "{}")
    tab.append("\\midrule\\multicolumn{6}{l}{\\emph{%s workload (%s), %d episodes per scenario}}\\\\" % (
        {"Real": "NSE+BSE", "Synth": "synthetic", "RealHet": "NSE+BSE, heterogeneous cost"}[lab], o["costs"], len(o["days"])))
    for s in ["mcar", "late", "mnar"]:
        if (s, "_zero") not in rows: continue
        z = rows[(s, "_zero")]
        for k in ["cert_frac", "impute_err", "impute_err_on_unk", "impute_err_on_cert"]:
            m(f"{k.replace('_','')}{lab}{s}", 100 * z[k])
        m(f"certwrong{lab}{s}", z["cert_wrong_total"], "{}"); m(f"viol{lab}{s}", z["viol_total"], "{}")
        m(f"ndec{lab}", z["n_dec"], "{:.0f}"); m(f"nitems{lab}", z["n_items"], "{:.0f}")
        for p in NAMES:
            if (s, p) not in rows: continue
            r = rows[(s, p)]
            key = p.replace("_", "").replace("1", "One")
            m(f"auc{lab}{s}{key}", r["AUC"], "{:.3f}"); m(f"cn{lab}{s}{key}", r["C90"], "{:.3f}")
            if "dAUC" in r:
                m(f"dauc{lab}{s}{key}", r["dAUC"], "{:.3f}")
                m(f"ci{lab}{s}{key}", "[{:.3f}, {:.3f}]".format(r["dAUC_lo"], r["dAUC_hi"]))
                m(f"wins{lab}{s}{key}", "{}/{}".format(r["wins"], r["n"]))
            tab.append("%s & %s & %.3f & %.3f & %.3f & %s \\\\" % (SC[s], NAMES[p], r["AUC"], r["R10"], r["C90"],
                       "--" if "dAUC" not in r else "%+.3f [%.3f, %.3f]" % (r["dAUC"], r["dAUC_lo"], r["dAUC_hi"])))
            if r["wrong"]: print("WRONG certified", wl, tag, s, p, r["wrong"])
open(f"{P}/tab_acq.tex", "w").write("\n".join(tab) + "\n")

for f in [f"{R}/stream_s7_l0.3_n0.3_r0.002.json"]:
    o = json.load(open(f))
    m("stK", o["K"], "{}"); m("stT", o["T"], "{}"); m("stEvents", o["events"], "{:,}"); m("stRestated", o["restated_items"], "{}")
    m("stViol", o["bound_violations"], "{}")
    for md in ["point", "cert", "recomp"]:
        m(f"tput{md}", o[f"tput_{md}"] / 1000, "{:.1f}")
    for md in ["cert", "point", "wait"]:
        if md not in o: continue
        r = o[md]
        m(f"st{md}Total", r["total"], "{:,}"); m(f"st{md}Emitted", 100 * r["emitted"] / r["total"])
        m(f"st{md}Wrong", r["wrong_at_emission"], "{:,}"); m(f"st{md}WrongPct", 100 * r["wrong_at_emission"] / max(r["emitted"], 1), "{:.2f}")
        m(f"st{md}Delay", r["mean_delay_days"], "{:.2f}"); m(f"st{md}OnTime", 100 * r["on_time"] / max(r["emitted"], 1))
        m(f"st{md}Rev", r["reversals"], "{:,}"); m(f"st{md}Flag", r["flagged"], "{:,}")
    break
open(f"{P}/numbers.tex", "w").write("\n".join(mac) + "\n")
print(len(mac), "macros")
seen = {}
for x in open(f"{P}/numbers.tex").read().splitlines():
    seen.setdefault(x.split("}")[0], x)
open(f"{P}/numbers.tex", "w").write("\n".join(seen.values()) + "\n")
print(len(seen), "unique macros")
