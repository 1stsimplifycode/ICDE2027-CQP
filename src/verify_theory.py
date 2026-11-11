"""Counterexample search for Lemma 1 and the DkS reduction (results/theory_check.json).
Lemma 1: L<theta<U, P proper subset of non-exact items -> exists values for P leaving d undetermined.
The constructive witness from the proof (shift s chosen in the feasible interval, split greedily) is
checked on random instances, including exact items, zero-width items and ties theta==L (excluded by
hypothesis; reported separately)."""
import json, itertools
import numpy as np
rng = np.random.default_rng(0)
def state(L, U, th): return 1 if L > th else (0 if U <= th else -1)
ok = bad = 0; tie_cases = tie_resolvable = 0
for trial in range(200000):
    n = rng.integers(1, 7)
    lo = rng.integers(0, 5, n).astype(float); w = rng.integers(0, 4, n).astype(float); hi = lo + w
    L, U = lo.sum(), hi.sum()
    th = float(rng.integers(int(L), int(U) + 1)) if U > L else L
    unc = np.flatnonzero(w > 0)
    if len(unc) < 2: continue
    k = rng.integers(1, len(unc)); P = rng.choice(unc, k, replace=False)
    if th == L:   # tie: outside the lemma's hypothesis; check whether the adversary can still keep U
        tie_cases += 1
        # adversary: reveal lo for all of P -> L unchanged == th, U' = U - sum w_P > th ?
        tie_resolvable += int(U - w[P].sum() <= th)
        continue
    if not (L < th < U): continue
    A = w[P].sum(); a, b = max(0.0, th - U + A), min(A, th - L)
    s = (a + b) / 2 if a < b else a                       # inside (th-U+A, th-L] intersect [0,A]
    v = lo.copy(); v[P] = lo[P]; rem = s
    for i in P:                                           # split shift s greedily
        x = min(w[i], rem); v[i] = lo[i] + x; rem -= x
    lo2, hi2 = lo.copy(), hi.copy(); lo2[P] = v[P]; hi2[P] = v[P]
    if state(lo2.sum(), hi2.sum(), th) == -1: ok += 1
    else: bad += 1
# DkS reduction: items [0,1], theta=1, edge decisions; adversary answers 0.5 -> edge resolved iff both probed
red_bad = 0
for trial in range(2000):
    nv = 6; E = [(u, v) for u, v in itertools.combinations(range(nv), 2) if rng.random() < 0.5]
    for Pset in itertools.chain.from_iterable(itertools.combinations(range(nv), r) for r in range(nv + 1)):
        for (u, v) in E:
            lo = {u: 0.0, v: 0.0}; hi = {u: 1.0, v: 1.0}
            for x in (u, v):
                if x in Pset: lo[x] = hi[x] = 0.5
            res = state(lo[u] + lo[v], hi[u] + hi[v], 1.0) != -1
            if res != (u in Pset and v in Pset): red_bad += 1
out = {"lemma_checked": ok, "lemma_counterexamples": bad, "tie_cases": tie_cases,
       "tie_cases_where_adversary_cannot_keep_U": tie_resolvable, "reduction_mismatches": red_bad}
print(out); json.dump(out, open(r"C:\Users\mailt\Downloads\ICDE-2027\results\theory_check.json", "w"))
