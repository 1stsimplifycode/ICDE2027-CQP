"""Experiment A: certification at zero budget vs imputation; budgeted acquisition policies.
Usage: python run_acq.py {real|synth} [n_episodes] [--costs hetero] [--lam 0.5] [--tag x]
Writes results/acq_<workload>_<tag>.json
"""
import json, sys, time, argparse
import numpy as np
sys.path.insert(0, __file__.rsplit("\\", 1)[0] if "\\" in __file__ else ".")
from workload import load_real, synthetic, Calib, build_episode, evidence, CAL_DAYS, WINDOWS
from acquire import run_policy, POLICIES
from cert import DecisionSet, UNK

ap = argparse.ArgumentParser()
ap.add_argument("workload"); ap.add_argument("n_ep", type=int, nargs="?", default=10)
ap.add_argument("--costs", default="unit"); ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--tag", default="main"); ap.add_argument("--policies", default=",".join(POLICIES))
ap.add_argument("--scen", default="mcar,late,mnar"); ap.add_argument("--seed", type=int, default=2027)
a = ap.parse_args()

if a.workload == "real":
    V, industry, keys, dates = load_real()
else:
    V, industry = synthetic(seed=a.seed)
cal = Calib(V, industry)
T = V.shape[2]
rng0 = np.random.default_rng(a.seed)
days = np.sort(rng0.choice(np.arange(CAL_DAYS + max(WINDOWS), T), size=a.n_ep, replace=False))
out = {"workload": a.workload, "K": int(V.shape[0]), "T": int(T), "groups": len(cal.groups),
       "days": days.tolist(), "lam": a.lam, "costs": a.costs, "episodes": []}
print("K", V.shape[0], "T", T, "groups", len(cal.groups), flush=True)
for scen in a.scen.split(","):
    for e, t in enumerate(days):
        M, theta, truth, cap, smed, lag, meta = build_episode(V, cal, int(t))
        rng = np.random.default_rng(a.seed * 1000 + e * 7 + {"mcar": 11, "late": 23, "mnar": 37}[scen])
        lo, hi, est, kind = evidence(truth, cap, smed, lag, scen, rng)
        if a.costs == "unit":
            cost = np.ones(len(lo))
        else:   # heterogeneous: primary exchange records cost 3, secondary 1 (declared, illustrative)
            src = (np.arange(len(lo)) // 20) % 2
            cost = np.where(src == 0, 3.0, 1.0)
        tv = (M @ truth) > theta
        ds = DecisionSet(M, theta, lo, hi)
        st = ds.state()
        unk = st == UNK
        imp = (M @ est) > theta
        viol = int(((truth > hi + 1e-9) | (truth < lo - 1e-9)).sum())
        und_items = np.unique(ds.nz_i[unk[ds.nz_d] & ((hi - lo)[ds.nz_i] > 0)])
        wait_cost = float(cost[und_items].sum())
        ep = {"scen": scen, "day": int(t), "n_dec": int(len(theta)), "n_items": int(len(lo)),
              "n_missing": int((kind == 2).sum()), "n_noisy": int((kind == 1).sum()),
              "bound_violations": viol,
              "cert_frac": float(1 - unk.mean()),
              "cert_wrong": int(((st == 1) & ~tv).sum() + ((st == 0) & tv).sum()),
              "impute_err": float((imp != tv).mean()),
              "impute_err_on_unk": float((imp[unk] != tv[unk]).mean()) if unk.any() else 0.0,
              "impute_err_on_cert": float((imp[~unk] != tv[~unk]).mean()) if (~unk).any() else 0.0,
              "n_unk": int(unk.sum()), "wait_cost": wait_cost, "curves": {}, "secs": {}}
        is_stock = np.array([m[0] == "stock" for m in meta])
        ep["cert_frac_stock"] = float(1 - unk[is_stock].mean()); ep["cert_frac_ind"] = float(1 - unk[~is_stock].mean())
        for pol in a.policies.split(","):
            t0 = time.perf_counter()
            c = run_policy(M, theta, lo, hi, truth, cost, pol, budget=wait_cost, lam=a.lam, seed=e)
            ep["secs"][pol] = time.perf_counter() - t0
            ep["curves"][pol] = c.tolist()
        out["episodes"].append(ep)
        print(scen, e, int(t), "dec", len(theta), "unk", int(unk.sum()), "wait", wait_cost,
              {p: round(ep["secs"][p], 1) for p in ep["secs"]}, flush=True)
json.dump(out, open(rf"C:\Users\mailt\Downloads\ICDE-2027\results\acq_{a.workload}_{a.tag}.json", "w"))
print("done")
