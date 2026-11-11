"""Exact worst-case acquisition for window-structured (interval) workloads.

Problem GR (guaranteed resolution): choose items to probe, total cost <= B, maximising the number of
undetermined decisions that are resolved for EVERY consistent outcome. By Lemma 1 a decision is
guaranteed resolved iff all its uncertain items are probed. Items of one key-day are probed together
(a day's cost c_t = sum of its uncertain items' costs; a partially probed day never helps a guarantee).

Interval structure: every decision of key k covers a contiguous day range. With uncertain days
u_1<..<u_m, decision d maps to an index range [p_d, q_d]; it is guaranteed iff no unprobed index in [p_d,q_d].
DP over the set N of UNPROBED uncertain days (sentinels 0 and m+1):
   best[j][b] = max_{i<j} best[i][b - cost(i,j)] + G(i,j)
   cost(i,j) = sum_{i<l<j} c_l          (days strictly between consecutive unprobed days are probed)
   G(i,j)    = #{d : i < p_d and q_d < j}
O(m^2 * B) per key; keys are combined by a max-plus knapsack. Exact (Theorem 3).
"""
import itertools
import numpy as np


def key_dp(c, ranges, Bmax):
    """c: integer costs of uncertain days (len m); ranges: list of (p,q) 1-based index ranges.
    Returns f[b] = max # guaranteed decisions with cost <= b, for b=0..Bmax."""
    m = len(c)
    cnt = np.zeros((m + 2, m + 2), int)
    for p, q in ranges:
        cnt[p, q] += 1
    # G(i,j) = sum_{p>i, q<j, p<=q} cnt[p,q]
    G = np.zeros((m + 2, m + 2), int)
    for i in range(m + 1):
        acc = np.cumsum(cnt[i + 1:, :].sum(axis=0))          # acc[q] = #ranges with p>i and q'<=q
        for j in range(i + 1, m + 2):
            G[i, j] = acc[j - 1] if j - 1 >= 0 else 0
    pc = np.concatenate([[0], np.cumsum(c)])                 # pc[l] = c_1+..+c_l
    NEG = -10**9
    best = np.full((m + 2, Bmax + 1), NEG); best[0, 0] = 0
    for j in range(1, m + 2):
        for i in range(j):
            cost = pc[j - 1] - pc[i]                           # days i+1..j-1 probed
            if cost > Bmax:
                continue
            cand = np.full(Bmax + 1, NEG)
            cand[cost:] = best[i, :Bmax + 1 - cost] + G[i, j]
            best[j] = np.maximum(best[j], cand)
    f = np.maximum.accumulate(best[m + 1])
    return np.maximum(f, 0)


def brute(c, ranges, Bmax):
    m = len(c); f = np.zeros(Bmax + 1, int)
    for r in range(m + 1):
        for P in itertools.combinations(range(1, m + 1), r):
            cost = sum(c[x - 1] for x in P)
            if cost > Bmax: continue
            Ps = set(P)
            val = sum(all(x in Ps for x in range(p, q + 1)) for p, q in ranges)
            f[cost:] = np.maximum(f[cost:], val)
    return f


def combine(fs, Bmax):
    tot = np.zeros(Bmax + 1, int)
    for f in fs:
        L = len(f); new = tot.copy()
        for x in range(1, L):
            if f[x] == f[x - 1]:
                continue
            new[x:] = np.maximum(new[x:], tot[:Bmax + 1 - x] + f[x])
        tot = np.maximum.accumulate(new)
    return tot


if __name__ == "__main__":
    rng = np.random.default_rng(0); bad = 0; n = 0
    for t in range(3000):
        m = int(rng.integers(1, 10)); c = rng.integers(1, 3, m).tolist()
        R = []
        for _ in range(int(rng.integers(1, 12))):
            p = int(rng.integers(1, m + 1)); q = int(rng.integers(p, m + 1)); R.append((p, q))
        B = int(sum(c))
        bad += int((key_dp(c, R, B) != brute(c, R, B)).any()); n += 1
    # combine check: brute over two keys
    bad2 = 0
    for t in range(300):
        fs = [brute([1, 2, 1], [(1, 2), (2, 3), (1, 3)], 4), brute([2, 1], [(1, 1), (1, 2), (2, 2)], 3)]
        tot = combine(fs, 7)
        bf = max(fs[0][a] + fs[1][min(b - a, 3)] for b in [7] for a in range(0, min(4, 7) + 1) if b - a >= 0)
        bad2 += int(tot[7] != bf)
    print({"dp_instances": n, "dp_mismatches": bad, "combine_mismatches": bad2})
