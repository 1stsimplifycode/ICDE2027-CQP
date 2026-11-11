"""Experiment D: exact worst-case acquisition (wcdp) vs heuristic plans on the real panel.
Backlog instance at day t: for every stock, windows W in {5,20} ending at each of the last H=40 days,
two threshold levels (stock-level decisions only: interval structure). Items = 2 per stock-day.
Objective: # undetermined decisions GUARANTEED resolved (all uncertain items probed), budget B
as a fraction of the total uncertain-item cost. Unit item cost.
Usage: python run_wc.py -> results/wc.json
"""
import sys, json, time
import numpy as np
sys.path.insert(0, r"C:\Users\mailt\Downloads\ICDE-2027\src")
from workload import load_real, Calib, evidence, CAL_DAYS, WINDOWS, MULTS
from wcdp import key_dp, combine

H = 40
FRACS = [0.02, 0.05, 0.10, 0.20, 0.40]
V, industry, keys, dates = load_real(); cal = Calib(V, industry)
K, S, T = V.shape
rng0 = np.random.default_rng(99)
days = np.sort(rng0.choice(np.arange(CAL_DAYS + H, T), 5, replace=False))
out = []
for scen in ["late", "mcar"]:
    for e, t in enumerate(days):
        rng = np.random.default_rng(5000 + e + (0 if scen == "late" else 77))
        truth = V[:, :, t - H + 1:t + 1].reshape(-1)                       # (K,S,H) flattened
        lag = np.tile(np.arange(H)[::-1], K * S)
        cap = np.repeat(cal.cap[:, :, None], H, 2).reshape(-1)
        smed = np.repeat(cal.src_med[:, :, None], H, 2).reshape(-1)
        lo, hi, est, kind = evidence(truth, cap, smed, lag, scen, rng)
        lo = lo.reshape(K, S, H); hi = hi.reshape(K, S, H)
        unc = (hi - lo) > 0                                                  # (K,S,H)
        dcost = unc.sum(1)                                                   # (K,H) uncertain items per day
        wday = (hi - lo).sum(1)
        Ld, Ud = lo.sum(1), hi.sum(1)
        # decisions per key: (a,b,theta)
        dec = []                                                             # (k, a, b)
        for W in WINDOWS:
            for c in MULTS:
                for b in range(W - 1, H):
                    a = b - W + 1
                    L = Ld[:, a:b + 1].sum(1); U = Ud[:, a:b + 1].sum(1); th = c * W * cal.med
                    und = (L <= th) & (U > th)
                    for k in np.flatnonzero(und):
                        dec.append((k, a, b))
        dec = np.array(dec)
        total_cost = int(dcost.sum()); Bs = [int(f * total_cost) for f in FRACS]; Bmax = max(Bs)
        # ---- exact ----
        t0 = time.perf_counter(); fs = []
        for k in range(K):
            ud = np.flatnonzero(dcost[k] > 0); pos = {d: i + 1 for i, d in enumerate(ud)}
            R = []
            for (_, a, b) in dec[dec[:, 0] == k]:
                inside = ud[(ud >= a) & (ud <= b)]
                R.append((pos[inside[0]], pos[inside[-1]]))
            fs.append(key_dp(dcost[k, ud].astype(int).tolist(), R, min(Bmax, int(dcost[k].sum()))))
        opt = combine(fs, Bmax); t_exact = time.perf_counter() - t0
        # ---- heuristic day orders (non-adaptive) ----
        def guaranteed(probed):                                              # probed (K,H) bool
            unprobed_unc = ((dcost > 0) & ~probed).astype(int)
            pre = np.concatenate([np.zeros((K, 1), int), np.cumsum(unprobed_unc, 1)], 1)
            return int(((pre[dec[:, 0], dec[:, 2] + 1] - pre[dec[:, 0], dec[:, 1]]) == 0).sum())
        WD = {}
        for (k, a, b) in dec:
            pass
        Wdec = (Ud - Ld)
        prefW = np.concatenate([np.zeros((K, 1)), np.cumsum(Wdec, 1)], 1)
        dW = prefW[dec[:, 0], dec[:, 2] + 1] - prefW[dec[:, 0], dec[:, 1]]
        nunc_pref = np.concatenate([np.zeros((K, 1)), np.cumsum(dcost > 0, 1)], 1)
        rdec = nunc_pref[dec[:, 0], dec[:, 2] + 1] - nunc_pref[dec[:, 0], dec[:, 1]]
        seaq = np.zeros((K, H)); comp = np.zeros((K, H))
        for j, (k, a, b) in enumerate(dec):
            seaq[k, a:b + 1] += wday[k, a:b + 1] / dW[j]
            comp[k, a:b + 1] += 1.0 / rdec[j]
        cst = np.maximum(dcost, 1)
        orders = {"random": rng.random((K, H)), "widest": wday / cst, "seaq": seaq / cst, "comp": comp / cst}
        res = {"scen": scen, "day": int(t), "n_dec": int(len(dec)), "total_cost": total_cost, "B": Bs,
               "opt": [int(opt[b]) for b in Bs], "t_exact": t_exact}
        for name, sc in orders.items():
            sc = np.where(dcost > 0, sc, -np.inf); flat = np.argsort(-sc, axis=None, kind="stable")
            cc = np.cumsum(dcost.reshape(-1)[flat]); vals = []
            for B in Bs:
                probed = np.zeros(K * H, bool); probed[flat[cc <= B]] = True
                vals.append(guaranteed(probed.reshape(K, H)))
            res[name] = vals
        # ---- greedy completion (classic: cheapest remaining window first) ----
        t0 = time.perf_counter(); probed = np.zeros((K, H), bool); spent = 0; vals = []; Bi = 0
        rem_cost = lambda: (np.concatenate([np.zeros((K, 1)), np.cumsum(np.where(probed, 0, dcost), 1)], 1))
        while Bi < len(Bs):
            pre = rem_cost(); rc = pre[dec[:, 0], dec[:, 2] + 1] - pre[dec[:, 0], dec[:, 1]]
            rc = np.where(rc > 0, rc, np.inf)
            j = int(np.argmin(rc))
            if not np.isfinite(rc[j]) or spent + rc[j] > Bs[Bi]:
                vals.append(guaranteed(probed)); Bi += 1; continue
            k, a, b = dec[j]; spent += int(rc[j]); probed[k, a:b + 1] |= dcost[k, a:b + 1] > 0
        res["greedy_complete"] = vals; res["t_greedy"] = time.perf_counter() - t0
        out.append(res); print(scen, e, int(t), "dec", len(dec), "cost", total_cost, "opt", res["opt"],
                               {n: res[n] for n in ["greedy_complete", "seaq", "comp", "widest", "random"]}, round(t_exact, 1), flush=True)
json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\wc.json", "w"))
