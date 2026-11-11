"""Summaries of acquisition curves. Metrics per episode, normalised by WAIT cost
(= cost of probing every uncertain item of every undetermined decision):
  R@b  : fraction of initially-undetermined decisions resolved after spending b*WAIT
  AUC  : mean of R over b in [0, 0.5]
  C90/C100 : spend (fraction of WAIT) to resolve 90% / 100%
Paired bootstrap 95% CI over episodes for SEAQ minus a comparator.
"""
import json, sys
import numpy as np


def ep_metrics(e, pol):
    c = np.array(e["curves"][pol]); base = c[0, 1]; need = e["n_dec"] - base; W = e["wait_cost"]
    if need == 0:
        return None
    def res_at(b):
        idx = np.flatnonzero(c[:, 0] <= b * W + 1e-9)
        return (c[idx[-1], 1] - base) / need
    def cost_at(f):
        idx = np.flatnonzero(c[:, 1] >= base + f * need - 1e-9)
        return c[idx[0], 0] / W if len(idx) else 1.0
    grid = np.linspace(0, 0.5, 51)
    return {"R10": res_at(.1), "R25": res_at(.25), "AUC": float(np.mean([res_at(b) for b in grid])),
            "C90": cost_at(.9), "C100": cost_at(1.0), "wrong": int(c[:, 2].max())}


def summarize(path, ref="seaq", B=2000, seed=0):
    o = json.load(open(path))
    pols = list(o["episodes"][0]["curves"].keys())
    rows = {}
    rng = np.random.default_rng(seed)
    for scen in sorted(set(e["scen"] for e in o["episodes"])):
        eps = [e for e in o["episodes"] if e["scen"] == scen]
        mets = {p: [ep_metrics(e, p) for e in eps] for p in pols}
        keep = [j for j in range(len(eps)) if mets[pols[0]][j] is not None]
        for p in pols:
            r = {k: float(np.mean([mets[p][j][k] for j in keep])) for k in ["R10", "R25", "AUC", "C90", "C100"]}
            r["wrong"] = int(sum(mets[p][j]["wrong"] for j in keep))
            r["secs"] = float(np.mean([eps[j]["secs"][p] for j in keep]))
            if ref in pols and p != ref:
                diff = np.array([mets[ref][j]["AUC"] - mets[p][j]["AUC"] for j in keep])
                bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
                r["dAUC"] = float(diff.mean()); r["dAUC_lo"] = float(np.percentile(bs, 2.5)); r["dAUC_hi"] = float(np.percentile(bs, 97.5))
                r["wins"] = int((diff > 0).sum()); r["n"] = len(diff)
            rows[(scen, p)] = r
        z = {k: float(np.mean([e[k] for e in eps])) for k in ["cert_frac", "cert_wrong", "impute_err", "impute_err_on_unk",
                                                            "impute_err_on_cert", "n_dec", "n_items", "n_missing", "n_noisy",
                                                            "bound_violations", "wait_cost", "n_unk"]}
        z["cert_wrong_total"] = int(sum(e["cert_wrong"] for e in eps)); z["viol_total"] = int(sum(e["bound_violations"] for e in eps))
        rows[(scen, "_zero")] = z
    return o, rows


if __name__ == "__main__":
    for path in sys.argv[1:]:
        o, rows = summarize(path)
        print("==", path)
        for (s, p), r in rows.items():
            print(s, p, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
