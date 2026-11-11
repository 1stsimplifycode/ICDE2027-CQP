"""Experiment C: controlled fan-out. D=1024 decisions, each over s=32 items; every item feeds exactly f
decisions (random regular design), so n_items = D*s/f. Decision count, decision size, missingness
(50% missing, interval [0, 4*mean]), value distribution and threshold rule (theta = sum of means * U(0.9,1.1))
are held fixed; only the sharing structure (and hence the item pool) changes.
Usage: python run_fanout.py  -> results/fanout.json
"""
import sys, json, time, tracemalloc
import numpy as np, scipy.sparse as sp
sys.path.insert(0, r"C:\Users\mailt\Downloads\ICDE-2027\src")
from acquire import run_policy
from cert import DecisionSet, UNK
D, S = 1024, 32
import os
POL = os.environ.get("POLS", "random,widest,degree,voi1,seaq,seaq_max").split(",")
out = []
for f in [1, 2, 4, 8, 16, 32]:
    n = D * S // f
    for rep in range(5):
        rng = np.random.default_rng(1000 * f + rep)
        # regular design: f random permutations of item slots
        slots = np.concatenate([rng.permutation(n) for _ in range(f)])        # length D*S
        rows = np.repeat(np.arange(D), S)
        M = sp.csr_matrix((np.ones(D * S), (rows, slots)), shape=(D, n))
        M.data[:] = 1.0; M.sum_duplicates(); M.data[:] = 1.0                  # rare duplicates collapse
        mu = 100.0
        truth = rng.gamma(2.0, mu / 2.0, n)
        theta = np.asarray(M.sum(1)).ravel() * mu * rng.uniform(0.9, 1.1, D)
        miss = rng.random(n) < 0.5
        lo, hi = truth.copy(), truth.copy(); lo[miss] = 0; hi[miss] = 4 * mu
        hi = np.maximum(hi, truth)          # keep bounds sound in this controlled benchmark
        ds = DecisionSet(M, theta, lo, hi); st = ds.state(); unk = st == UNK
        und_items = np.unique(ds.nz_i[unk[ds.nz_d] & ((hi - lo)[ds.nz_i] > 0)])
        cost = np.ones(n); W = float(cost[und_items].sum())
        rec = {"f": f, "rep": rep, "n_items": n, "n_unk": int(unk.sum()), "wait": W, "curves": {}, "secs": {}, "peak_mb": {}}
        for p in POL:
            tracemalloc.start(); t0 = time.perf_counter()
            c = run_policy(M, theta, lo, hi, truth, cost, p, budget=W, seed=rep)
            rec["secs"][p] = time.perf_counter() - t0; rec["peak_mb"][p] = tracemalloc.get_traced_memory()[1] / 2**20
            tracemalloc.stop(); rec["curves"][p] = c.tolist()
        out.append(rec); print(f, rep, n, int(unk.sum()), {p: round(rec["secs"][p], 1) for p in POL}, flush=True)
json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\fanout" + os.environ.get("FTAG", "") + ".json", "w"))
