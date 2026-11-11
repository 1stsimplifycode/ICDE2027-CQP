"""Graphs generated directly from saved result files (no reconstructed data).
G1 resolution vs budget, real panel (curves from results/acq_real_main.json, mean over episodes + min/max band)
G2 guaranteed-resolved decisions vs budget: exact DP vs heuristics (results/wc.json)
G3 repeated DuckDB benchmark: total time per run per mode (results/cqp_bench_60d_*_rep*.json)
Output: paper/g1_resolution.pdf, paper/g2_exact.pdf, paper/g3_bench.pdf
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = r"C:\Users\mailt\Downloads\ICDE-2027\results"; P = r"C:\Users\mailt\Downloads\ICDE-2027\paper"
plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.1, "figure.dpi": 200, "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"seaq": "#1f4e79", "voi1": "#c55a11", "widest": "#548235", "local": "#7f6000", "random": "#7f7f7f",
       "greedy_complete": "#c55a11", "comp": "#7f6000", "opt": "#1f4e79"}

# ---------- G1
o = json.load(open(f"{R}/acq_real_main.json")); grid = np.linspace(0, 0.5, 51)
fig, axs = plt.subplots(1, 3, figsize=(3.45, 1.35), sharey=True)
for ax, sc in zip(axs, ["mcar", "late", "mnar"]):
    eps = [e for e in o["episodes"] if e["scen"] == sc]
    for p, lab in [("seaq", "SEAQ"), ("voi1", "Myopic VOI"), ("widest", "Widest"), ("local", "Local"), ("random", "Random")]:
        Y = []
        for e in eps:
            c = np.array(e["curves"][p]); base = c[0, 1]; need = e["n_dec"] - base; W = e["wait_cost"]
            Y.append([(c[np.flatnonzero(c[:, 0] <= b * W + 1e-9)[-1], 1] - base) / need for b in grid])
        Y = np.array(Y)
        ax.plot(grid, Y.mean(0), color=COL[p], label=lab)
        ax.fill_between(grid, Y.min(0), Y.max(0), color=COL[p], alpha=0.15, lw=0)
    ax.set_title(sc.upper() + f" (n={len(eps)})", fontsize=7); ax.set_xlabel("budget / WAIT")
    ax.grid(alpha=0.3, lw=0.4)
axs[0].set_ylabel("resolved fraction")
axs[0].legend(fontsize=4.5, frameon=False, loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig(f"{P}/g1_resolution.pdf"); plt.close(fig)

# ---------- G2
W = json.load(open(f"{R}/wc.json")); fr = [2, 5, 10, 20, 40]
fig, axs = plt.subplots(1, 2, figsize=(3.45, 1.75))
xp = np.arange(len(fr))
for ax, sc in zip(axs, ["late", "mcar"]):
    rs = [r for r in W if r["scen"] == sc]
    for k, lab in [("opt", "Exact DP (Thm. 2)"), ("greedy_complete", "Greedy completion"), ("comp", "Completion score"), ("seaq", "SEAQ order"), ("random", "Random")]:
        Y = np.array([r[k] for r in rs], float)
        ax.errorbar(xp, Y.mean(0), yerr=[Y.mean(0) - Y.min(0), Y.max(0) - Y.mean(0)], color=COL.get(k, "#7f7f7f"),
                    marker="o", ms=2, capsize=1.5, lw=0.9, label=lab)
    ax.set_xticks(xp); ax.set_xticklabels([f"{x}%" for x in fr]); ax.minorticks_off()
    ax.set_xlabel("budget"); ax.set_title(f"{sc.upper()} (n={len(rs)} days)", fontsize=7); ax.grid(alpha=0.3, lw=0.4)
axs[0].set_ylabel("guaranteed decisions"); axs[0].legend(fontsize=5.5, frameon=False, loc="upper left")
fig.tight_layout(pad=0.3); fig.savefig(f"{P}/g2_exact.pdf"); plt.close(fig)

# ---------- G3 (redesigned): dot-and-range per mode with every run visible + zoomed paired speedups
modes = [("inc_scan", "scan"), ("inc", "affected"), ("full", "full")]
plt.rcParams.update({"font.size": 8})
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw={"width_ratios": [1.25, 1]})
tot = {}
for i, (m, lab) in enumerate(modes):
    rs = sorted(glob.glob(f"{R}/cqp_bench_60d_{m}_rep*.json"))
    d = {json.load(open(f))["rep"]: json.load(open(f))["total_wall_s"] for f in rs}; tot[m] = d
    t = np.array([d[k] for k in sorted(d)])
    off = np.linspace(-0.18, 0.18, len(t))                      # deterministic horizontal offsets, one per run
    axs[0].vlines(i, t.min(), t.max(), color="k", lw=2.2)        # observed min-max range
    axs[0].scatter(i + off, t, s=12, color="#1f4e79", zorder=3, edgecolor="white", lw=0.4)
    axs[0].text(i, t.max() + 2.5, "%.1f s" % np.median(t), ha="center", va="bottom", fontsize=7)
axs[0].set_xticks(range(3)); axs[0].set_xticklabels([l for _, l in modes], fontsize=6.5)
axs[0].set_xlim(-0.5, 2.5); axs[0].set_ylim(0, 70); axs[0].set_ylabel("total time per run (s)")
axs[0].set_title("%d runs per mode" % len(tot["inc"]), fontsize=7); axs[0].grid(alpha=0.3, lw=0.4, axis="y")
reps = sorted(set(tot["full"]) & set(tot["inc_scan"]))
sp = np.array([tot["full"][r] / tot["inc_scan"][r] for r in reps])
x = np.arange(1, len(sp) + 1)
axs[1].scatter(x, sp, s=18, color="#c55a11", zorder=3)
axs[1].axhline(np.median(sp), color="#c55a11", lw=0.8, ls="--")
axs[1].text(0.6, np.median(sp) + 0.004, "median %.2fx" % np.median(sp), va="bottom", fontsize=6.5, color="#c55a11")
lo, hi = sp.min(), sp.max(); pad = 0.04
axs[1].set_ylim(lo - pad, hi + pad); axs[1].set_xlim(0.5, len(sp) + 0.5)
axs[1].set_xticks(x); axs[1].set_xlabel("run"); axs[1].set_ylabel("speedup (full / inc.)")
axs[1].set_title("zoomed y-axis", fontsize=7); axs[1].grid(alpha=0.3, lw=0.4, axis="y")
fig.tight_layout(pad=0.3); fig.savefig(f"{P}/g3_bench.pdf"); plt.close(fig)
print("graphs ok; speedups", [round(v, 3) for v in sp], "totals", {m: sorted(round(v, 2) for v in tot[m].values()) for m in tot})
